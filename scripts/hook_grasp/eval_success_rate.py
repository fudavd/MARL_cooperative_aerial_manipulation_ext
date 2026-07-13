# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched, headless success/catch-rate eval for a trained skrl-PPO hook-grasp policy.

Unlike play_ppo_video.py (1 env, renders a video), this runs many envs for many steps with NO
rendering and aggregates PER-EPISODE statistics: success rate (held to threshold), catch rate
(episodes with >=1 catch), catches/episode, and mean attached_frac. Curriculum is disabled so the
capture radius is pinned at the TRUE target (honest difficulty).

Example:
    python scripts/hook_grasp/eval_success_rate.py --headless \
        --checkpoint logs/skrl/ppo_flycrane_hook/<run>/checkpoints/agent_1010000.pt \
        --num_envs 128 --steps 2500
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batched success/catch-rate eval of a skrl-PPO hook-grasp policy.")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=2500)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
    # honest eval: disable every training curriculum -> true capture radius, full hold, full random spawn range
    for _attr in ("capture_curriculum", "hold_curriculum", "spawn_curriculum"):
        if hasattr(env_cfg, _attr):
            setattr(env_cfg, _attr, False)

    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    if isinstance(base, DirectMARLEnv):
        env = multi_agent_to_single_agent(env, state_as_observation=True)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    # inference only: drop the LR scheduler string (skrl's runner eval()s it and CosineAnnealingLR is not in scope)
    experiment_cfg["agent"].pop("learning_rate_scheduler", None)
    experiment_cfg["agent"].pop("learning_rate_scheduler_kwargs", None)
    runner = Runner(env, experiment_cfg)
    print(f"[eval] loading checkpoint: {args_cli.checkpoint}")
    runner.agent.load(args_cli.checkpoint)
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()
    N = args_cli.num_envs

    episodes = 0
    catches = 0
    attached_step_sum = 0.0
    # per-env "did this episode catch / succeed at least once" flags (accumulated in this loop, since success no
    # longer terminates and the env clears its own latches on reset). Reset when the episode ends.
    ep_had_catch = torch.zeros(N, dtype=torch.bool, device=base.device)
    ep_had_success = torch.zeros(N, dtype=torch.bool, device=base.device)
    episodes_with_catch = 0
    episodes_with_success = 0
    capture_radius = float(base._acfg.capture_radius)

    for _ in range(args_cli.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)

        newly = base._newly_caught.any(dim=-1)          # (N,) any drone caught this step
        ep_had_catch |= newly
        ep_had_success |= base.succeeded.reshape(N)     # reached the hold milestone this step
        catches += int(base._newly_caught.sum().item())
        attached_step_sum += float(base.attached.float().mean().item())

        done = (terminated | truncated).reshape(N).bool()
        n_done = int(done.sum().item())
        if n_done:
            episodes += n_done
            episodes_with_catch += int((ep_had_catch & done).sum().item())
            episodes_with_success += int((ep_had_success & done).sum().item())
            ep_had_catch = ep_had_catch & (~done)       # clear flags for envs that just reset
            ep_had_success = ep_had_success & (~done)

    steps = args_cli.steps
    succ_rate = episodes_with_success / episodes if episodes else float("nan")
    catch_ep_rate = episodes_with_catch / episodes if episodes else float("nan")
    catches_per_ep = catches / episodes if episodes else float("nan")
    mean_attached = attached_step_sum / steps

    lines = [
        "==================== HOOK-GRASP BATCHED EVAL ====================",
        f" checkpoint          : {args_cli.checkpoint}",
        f" envs x steps        : {N} x {steps}   (capture_radius {capture_radius:.3f})",
        f" episodes completed  : {episodes}",
        f" SUCCESS RATE        : {succ_rate:.3f}   ({episodes_with_success} reached-hold-milestone / {episodes} eps)",
        f" CATCH RATE (>=1)    : {catch_ep_rate:.3f}   ({episodes_with_catch} eps caught at least once)",
        f" catches / episode   : {catches_per_ep:.3f}   ({catches} total catch events)",
        f" mean attached_frac  : {mean_attached:.3f}   (fraction of steps attached = how LONG it holds; success ends the episode)",
        "=================================================================",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    out = "scripts/hook_grasp/_render/eval_success_rate.txt"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(summary + "\n")
        f.flush()
        os.fsync(f.fileno())
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
