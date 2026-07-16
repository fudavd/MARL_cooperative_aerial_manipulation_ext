# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic GOAL evaluator for an rsl_rl recurrent (LSTM) hook-grasp checkpoint.

Mirrors eval_mappo.py's metrics (catch_rate, hold_success_given_caught, jerk) but for the rsl_rl
backend: it loads the RECURRENT hook-blind actor, runs MEAN actions on the 135-dim hook-free obs WITH
LSTM hidden-state carry and reset-on-done, and measures the hold ourselves as consecutive attached
steps. All curricula are DISABLED, the catch radius is pinned to 0.13, hold measured @ --hold_steps.

Success (true task): catch_rate >= 0.30 AND hold_success_given_caught >= 0.80 at hold_success_steps=100.

Example:
    ./.venv/bin/python scripts/hook_grasp/eval_rslrl.py --headless --num_envs 128 --steps 4000 \
        --checkpoint logs/rsl_rl/v26L1_lstm/<run>/model_5000.pt --out /tmp/e.json
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deterministic goal evaluator for an rsl_rl recurrent hook-grasp actor.")
parser.add_argument("--num_envs", type=int, default=128, help="Parallel envs.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to an rsl_rl model_*.pt checkpoint.")
parser.add_argument("--steps", type=int, default=4000, help="Total sim steps to roll.")
parser.add_argument("--hold_steps", type=int, default=100, help="Consecutive attached steps counted as a hold.")
parser.add_argument("--capture_radius", type=float, default=0.13, help="Catch radius (m); FINAL task value = 0.13.")
parser.add_argument("--seed", type=int, default=12345, help="Seed.")
parser.add_argument("--out", type=str, default="", help="Optional path to write a JSON result.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rslrl_adapter import RslRlMarlAdapter, make_runner_cfg  # noqa: E402


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    # ---- honest eval: disable EVERY curriculum / aid so the numbers reflect the TRUE (end) task ----
    for _attr in ("capture_curriculum", "hold_curriculum", "spawn_curriculum", "attached_spawn_curriculum"):
        if hasattr(env_cfg, _attr):
            setattr(env_cfg, _attr, False)
    if hasattr(env_cfg, "attachment") and args_cli.capture_radius > 0:
        env_cfg.attachment.capture_radius = args_cli.capture_radius
    # disable the env's own hold-success termination; we measure the hold ourselves.
    if hasattr(env_cfg, "hold_success_steps"):
        env_cfg.hold_success_steps = 10_000_000

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    assert isinstance(base, DirectMARLEnv), "eval_rslrl expects a DirectMARLEnv (multi-agent) task."
    adapter = RslRlMarlAdapter(base)

    # build the runner just to reconstruct + load the recurrent policy (architecture must match training)
    train_cfg = make_runner_cfg(experiment_name="eval", max_iterations=1, device=device, logger="tensorboard")
    runner = OnPolicyRunner(adapter, train_cfg, log_dir=None, device=device)
    print(f"[eval-rslrl] loading checkpoint: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    runner.eval_mode()
    policy = runner.alg.policy
    obs_normalizer = runner.obs_normalizer

    E = base.num_envs
    dev = base.device
    hold_req = args_cli.hold_steps

    obs, _ = adapter.get_observations()
    actor_dim = int(obs.shape[-1])
    state_dim = int(base.state().shape[-1])
    print(f"[eval-rslrl] agent={adapter.agent}  actor_obs_dim={actor_dim}  critic_state_dim={state_dim}")
    assert actor_dim == 135, f"actor obs must be 135-dim hook-free, got {actor_dim}"
    assert state_dim == 180, f"critic state expected 180-dim, got {state_dim}"

    policy.reset()  # zero the LSTM hidden state for all envs

    ever_caught = torch.zeros(E, dtype=torch.bool, device=dev)
    ever_held = torch.zeros(E, dtype=torch.bool, device=dev)
    consec_attach = torch.zeros(E, dtype=torch.long, device=dev)
    ep_hold_max = torch.zeros(E, dtype=torch.long, device=dev)

    n_ep = torch.zeros((), dtype=torch.long, device=dev)
    n_caught = torch.zeros((), dtype=torch.long, device=dev)
    n_held = torch.zeros((), dtype=torch.long, device=dev)
    attached_steps = torch.zeros((), dtype=torch.long, device=dev)
    held_hold_sum = torch.zeros((), dtype=torch.long, device=dev)
    total_steps = 0
    jerk_sum = torch.zeros((), dtype=torch.float, device=dev)
    jerk_n = torch.zeros((), dtype=torch.long, device=dev)

    for _ in range(args_cli.steps):
        with torch.inference_mode():
            # DETERMINISTIC: act_inference returns the Gaussian MEAN, carrying the LSTM hidden state.
            actions = policy.act_inference(obs_normalizer(obs))
            obs, _, dones, _ = adapter.step(actions)
            policy.reset(dones)  # reset LSTM hidden state on episode boundaries

        attached = base.attached.any(dim=-1)
        newly = base._newly_caught.any(dim=-1)
        ever_caught |= newly
        consec_attach = torch.where(attached, consec_attach + 1, torch.zeros_like(consec_attach))
        ep_hold_max = torch.maximum(ep_hold_max, consec_attach)
        ever_held |= consec_attach >= hold_req
        attached_steps += attached.sum()
        total_steps += E

        if hasattr(base, "_drone_jerk"):
            jm = torch.norm(base._drone_jerk, dim=-1).mean(dim=-1)
            if attached.any():
                jerk_sum += jm[attached].sum()
                jerk_n += attached.sum()

        done = dones.bool()
        if done.any():
            dc = ever_caught & done
            dh = ever_held & done
            n_ep += done.sum()
            n_caught += dc.sum()
            n_held += dh.sum()
            held_hold_sum += ep_hold_max[dc].sum()
            keep = ~done
            ever_caught &= keep
            ever_held &= keep
            consec_attach = torch.where(done, torch.zeros_like(consec_attach), consec_attach)
            ep_hold_max = torch.where(done, torch.zeros_like(ep_hold_max), ep_hold_max)

    ep = int(n_ep.item())
    caught = int(n_caught.item())
    held = int(n_held.item())
    catch_rate = caught / ep if ep else 0.0
    hold_given_caught = held / caught if caught else 0.0
    attached_frac = float(attached_steps.item()) / max(1, total_steps)
    mean_hold_caught = float(held_hold_sum.item()) / caught if caught else 0.0
    mean_jerk = float(jerk_sum.item()) / max(1, int(jerk_n.item())) if int(jerk_n.item()) else float("nan")

    result = {
        "checkpoint": args_cli.checkpoint,
        "algorithm": "rsl_rl-ActorCriticRecurrent",
        "num_envs": E,
        "sim_steps": args_cli.steps,
        "actor_obs_dim": actor_dim,
        "critic_state_dim": state_dim,
        "hold_steps": hold_req,
        "capture_radius": float(base._acfg.capture_radius),
        "episodes": ep,
        "caught_episodes": caught,
        "held_episodes": held,
        "catch_rate": round(catch_rate, 4),
        "hold_success_given_caught": round(hold_given_caught, 4),
        "mean_hold_steps_given_caught": round(mean_hold_caught, 2),
        "attached_frac": round(attached_frac, 4),
        "mean_jerk_attached_N": round(mean_jerk, 3) if mean_jerk == mean_jerk else None,
        "GOAL_catch_0.30": catch_rate >= 0.30,
        "GOAL_hold_0.80": hold_given_caught >= 0.80,
        "GOALS_MET": bool(catch_rate >= 0.30 and hold_given_caught >= 0.80),
    }
    print("JSON " + json.dumps(result))
    print("\n================= RSL_RL RECURRENT GOAL EVAL =================")
    for k, v in result.items():
        print(f"  {k:30s}: {v}")
    print("  gate: catch>=0.30 AND hold_given_caught>=0.80 -> "
          f"{'PASS' if result['GOALS_MET'] else 'FAIL'}")
    print("=============================================================")
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        print(f"[eval-rslrl] wrote {args_cli.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
