# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched GOAL evaluator for the hook-grasp PPO policy.

Measures the two overnight targets, per EPISODE, over many parallel envs:
  * catch_rate            = episodes with >=1 catch / total episodes            (goal: >= 0.30)
  * hold_success_caught   = episodes that reached a HOLD_STEPS-consecutive-attached
                            hold / episodes with >=1 catch                       (goal: >= 0.80)

Robustness trick: a real "success" terminates+resets the episode in the SAME sim step, so a naive
post-step read of the env's success flag is destroyed by the reset. So we DISABLE the env's own hold
termination (hold_success_steps -> huge) and measure the hold OURSELVES as the running count of
consecutive steps the hook is attached; an episode "succeeds" once that count reaches --hold_steps.
Episodes then only end on time-out or a crash termination, which never coincides with the hold event.

All training curricula are disabled so the numbers reflect the TRUE (curriculum-end) difficulty; pass
--capture_radius to pin the catch geometry explicitly (else the cfg's end value is used).

Example:
    python scripts/hook_grasp/eval_goals.py --headless \
        --checkpoint logs/skrl/ppo_flycrane_hook/<run>/checkpoints/agent_150000.pt \
        --num_envs 512 --steps 4000 --hold_steps 100 --capture_radius 0.10 --out /tmp/eval.json
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batched goal evaluator for the hook-grasp PPO policy.")
parser.add_argument("--num_envs", type=int, default=512, help="Parallel envs.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to skrl checkpoint .pt.")
parser.add_argument("--steps", type=int, default=4000, help="Total sim steps to roll (more -> more episodes).")
parser.add_argument("--hold_steps", type=int, default=100, help="Consecutive attached steps that count as a hold.")
parser.add_argument("--capture_radius", type=float, default=-1.0, help="Override catch radius (m); <0 = use cfg.")
parser.add_argument("--slack_dwell_steps", type=int, default=-1, help="Override detach grace period; <0 = use cfg (base=5).")
parser.add_argument("--detach_p_vertical", type=float, default=-1.0, help="Override vertical detach prob; <0 = use cfg (base=1.0).")
parser.add_argument("--detach_p_horizontal", type=float, default=-1.0, help="Override horizontal detach prob; <0 = use cfg (base=0.2).")
parser.add_argument("--seed", type=int, default=12345, help="Seed.")
parser.add_argument("--out", type=str, default="", help="Optional path to write a JSON result.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # honest eval: disable every training curriculum so difficulty is the true (end) value ...
    for _attr in ("capture_curriculum", "hold_curriculum", "spawn_curriculum", "attached_spawn_curriculum"):
        if hasattr(env_cfg, _attr):
            setattr(env_cfg, _attr, False)
    # ... and DISABLE the env's own hold-success termination so it never resets on the hold event; we measure
    # the hold ourselves (consecutive attached steps). A giant threshold makes `succeeded` effectively unreachable.
    if hasattr(env_cfg, "hold_success_steps"):
        env_cfg.hold_success_steps = 10_000_000
    if args_cli.capture_radius > 0 and hasattr(env_cfg, "attachment"):
        env_cfg.attachment.capture_radius = args_cli.capture_radius
    # optional detach-model overrides for FEASIBILITY BRACKETING (default <0 -> keep base strict model)
    if hasattr(env_cfg, "attachment"):
        if args_cli.slack_dwell_steps >= 0:
            env_cfg.attachment.slack_dwell_steps = args_cli.slack_dwell_steps
        if args_cli.detach_p_vertical >= 0:
            env_cfg.attachment.detach_p_vertical = args_cli.detach_p_vertical
        if args_cli.detach_p_horizontal >= 0:
            env_cfg.attachment.detach_p_horizontal = args_cli.detach_p_horizontal

    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    if isinstance(base, DirectMARLEnv):
        env = multi_agent_to_single_agent(env, state_as_observation=True)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    experiment_cfg["agent"].pop("learning_rate_scheduler", None)
    experiment_cfg["agent"].pop("learning_rate_scheduler_kwargs", None)
    runner = Runner(env, experiment_cfg)
    print(f"[eval-goals] loading checkpoint: {args_cli.checkpoint}")
    runner.agent.load(args_cli.checkpoint)
    runner.agent.set_running_mode("eval")

    E = args_cli.num_envs
    dev = base.device
    hold_req = args_cli.hold_steps

    # per-env, current-episode trackers
    ever_caught = torch.zeros(E, dtype=torch.bool, device=dev)
    ever_held = torch.zeros(E, dtype=torch.bool, device=dev)
    consec_attach = torch.zeros(E, dtype=torch.long, device=dev)

    # cumulative episode counters
    n_ep = torch.zeros((), dtype=torch.long, device=dev)
    n_caught = torch.zeros((), dtype=torch.long, device=dev)
    n_held = torch.zeros((), dtype=torch.long, device=dev)
    attached_steps = torch.zeros((), dtype=torch.long, device=dev)
    total_steps = 0

    obs, _ = env.reset()
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)

        attached = base.attached.any(dim=-1)                      # (E,) hook attached this step
        newly = base._newly_caught.any(dim=-1)                    # (E,) caught this step
        ever_caught |= newly
        consec_attach = torch.where(attached, consec_attach + 1, torch.zeros_like(consec_attach))
        ever_held |= consec_attach >= hold_req
        attached_steps += attached.sum()
        total_steps += E

        done = (terminated | truncated).view(E).bool()
        if done.any():
            n_ep += done.sum()
            n_caught += (ever_caught & done).sum()
            n_held += (ever_held & done).sum()
            keep = ~done
            ever_caught &= keep
            ever_held &= keep
            consec_attach = torch.where(done, torch.zeros_like(consec_attach), consec_attach)

    ep = int(n_ep.item())
    caught = int(n_caught.item())
    held = int(n_held.item())
    catch_rate = caught / ep if ep else 0.0
    hold_given_caught = held / caught if caught else 0.0
    attached_frac = float(attached_steps.item()) / max(1, total_steps)

    result = {
        "checkpoint": args_cli.checkpoint,
        "num_envs": E,
        "sim_steps": args_cli.steps,
        "hold_steps": hold_req,
        "capture_radius": float(base._acfg.capture_radius),
        "episodes": ep,
        "caught_episodes": caught,
        "held_episodes": held,
        "catch_rate": round(catch_rate, 4),
        "hold_success_given_caught": round(hold_given_caught, 4),
        "attached_frac": round(attached_frac, 4),
        "GOAL_catch_0.30": catch_rate >= 0.30,
        "GOAL_hold_0.80": hold_given_caught >= 0.80,
        "GOALS_MET": bool(catch_rate >= 0.30 and hold_given_caught >= 0.80),
    }
    print("\n==================== GOAL EVAL ====================")
    for k, v in result.items():
        print(f"  {k:26s}: {v}")
    print("==================================================")
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        print(f"[eval-goals] wrote {args_cli.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
