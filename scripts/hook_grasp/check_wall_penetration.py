# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Numerically verify the surrogate wall collision: run the policy over many envs and report the MAX hook
penetration into the basket wall footprint (Metrics/wall_penetration). Should be ~0 (a few mm at most)."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--ml_framework", type=str, default="torch")
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
    # honest eval: disable every training curriculum
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
    runner = Runner(env, experiment_cfg)
    runner.agent.load(args_cli.checkpoint)
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()
    N = args_cli.num_envs
    SETTLE = 20  # steps after a reset to treat as "spawn transient" (excluded from the settled stats)
    since_reset = torch.full((N,), 1000, dtype=torch.long, device=base.device)
    max_pen = 0.0
    max_pen_settled = 0.0
    pen_hist = torch.zeros(6, device=base.device)   # ALL steps
    n_hook_steps = 0
    settled_hist = torch.zeros(6, device=base.device)  # excluding first SETTLE steps post-reset
    n_settled = 0
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)
        since_reset += 1
        pen = base._hook_wall_penetration().reshape(N, -1)  # (N, D)
        flat = pen.reshape(-1)
        max_pen = max(max_pen, float(flat.max().item()))
        settled_mask = (since_reset >= SETTLE).unsqueeze(1).expand_as(pen).reshape(-1)
        if settled_mask.any():
            max_pen_settled = max(max_pen_settled, float(flat[settled_mask].max().item()))
        n_hook_steps += flat.numel()
        n_settled += int(settled_mask.sum().item())
        for j, thr in enumerate([0.0, 0.001, 0.005, 0.01, 0.02, 0.05]):
            pen_hist[j] += (flat > thr).sum()
            settled_hist[j] += (flat[settled_mask] > thr).sum()
        done = (terminated | truncated).reshape(N).bool()
        since_reset[done] = 0

    frac = (pen_hist / max(1, n_hook_steps)).tolist()
    sfrac = (settled_hist / max(1, n_settled)).tolist()
    lines = [
        "================= WALL PENETRATION CHECK =================",
        f" checkpoint     : {args_cli.checkpoint}",
        f" envs x steps   : {args_cli.num_envs} x {args_cli.steps}",
        f" MAX penetration (all)     : {max_pen*1000:.2f} mm",
        f" MAX penetration (settled) : {max_pen_settled*1000:.2f} mm   (>= {20} steps post-reset)",
        f" ALL      frac >0mm {frac[0]:.4f}  >5mm {frac[2]:.4f}  >1cm {frac[3]:.4f}  >5cm {frac[5]:.4f}",
        f" SETTLED  frac >0mm {sfrac[0]:.4f}  >5mm {sfrac[2]:.4f}  >1cm {sfrac[3]:.4f}  >5cm {sfrac[5]:.4f}",
        "=========================================================",
    ]
    print("\n" + "\n".join(lines))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
