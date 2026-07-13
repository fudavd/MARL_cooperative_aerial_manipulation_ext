# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate / play a trained Hebbian-CMA-ES flycrane policy and report comparison metrics.

Example:
    python scripts/hebbian/play.py --task=Isaac-flycrane-payload-decentralized-hovering-v0 \
        --headless --num_envs=64 --eval_episodes=5 --checkpoint=logs/hebbian/.../checkpoints/best.pt
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a Hebbian/CMA-ES flycrane policy.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of evaluation environments.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-v0", help="Task name.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt checkpoint.")
parser.add_argument("--eval_episodes", type=int, default=5, help="Episodes per env to average over.")
parser.add_argument("--seed", type=int, default=123, help="Eval seed.")
parser.add_argument("--metrics_out", type=str, default=None, help="Optional CSV path to append summary metrics.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video.")
parser.add_argument("--video_length", type=int, default=400, help="Video length (steps).")
parser.add_argument("--hebbian_lr", type=float, default=None,
                    help="Override the online Hebbian update rate (0 = freeze weights at init -> no adaptation).")
parser.add_argument("--attachment_per_env", type=int, default=-1,
                    help="hover_rand only: 1=per-env attachment (replicate_physics=False), 0=per-run shared, "
                         "-1=leave the env cfg default.")

AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import os

import gymnasium as gym
import torch

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from MARL_mav_carry_ext.algorithms.hebbian.adapter import HebbianPolicy


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    # match the env observation config the checkpoint was trained with
    import torch as _torch

    _ckpt = _torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    env_cfg.partial_obs = bool(_ckpt.get("partial_obs", getattr(env_cfg, "partial_obs", True)))
    env_cfg.history_len = int(_ckpt.get("history_len", getattr(env_cfg, "history_len", 1)))
    _obs_dim = (45 * env_cfg.history_len) if env_cfg.partial_obs else 87
    env_cfg.observation_spaces = {a: _obs_dim for a in env_cfg.possible_agents}

    # hover_rand only: optionally force the attachment-randomization scope for this rollout
    if args_cli.attachment_per_env >= 0 and hasattr(env_cfg, "attachment_per_env"):
        env_cfg.attachment_per_env = bool(args_cli.attachment_per_env)
        if args_cli.attachment_per_env == 1:
            env_cfg.scene.replicate_physics = False
        print(f"[INFO] hover_rand attachment_per_env={env_cfg.attachment_per_env} "
              f"(replicate_physics={env_cfg.scene.replicate_physics})")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(os.path.dirname(args_cli.checkpoint), "..", "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    uenv = env.unwrapped
    policy = HebbianPolicy(uenv, args_cli.checkpoint)
    if args_cli.hebbian_lr is not None:
        for _a in policy.actors.values():
            _a.lr = args_cli.hebbian_lr
        print(f"[INFO] overriding hebbian_lr -> {args_cli.hebbian_lr} "
              f"({'frozen weights (no adaptation)' if args_cli.hebbian_lr == 0 else 'custom rate'})")

    num_envs = uenv.num_envs
    agents = list(uenv.cfg.possible_agents)
    goal_range = uenv.cfg.goal_achieved_range

    obs, _ = env.reset()
    policy.reset()

    ep_return = torch.zeros(num_envs, device=uenv.device)
    pos_err_sum = torch.zeros(num_envs, device=uenv.device)
    ori_err_sum = torch.zeros(num_envs, device=uenv.device)
    step_count = torch.zeros(num_envs, device=uenv.device)

    completed_returns, completed_pos_err, completed_ori_err, completed_success = [], [], [], []
    episodes_done = 0
    target_episodes = args_cli.eval_episodes * num_envs

    while episodes_done < target_episodes:
        actions = policy.act(obs)
        obs, rewards, terminated, time_outs, _ = env.step(actions)

        step_r = torch.stack([rewards[a] for a in agents], dim=0).sum(dim=0)
        ep_return += step_r
        pos_err_sum += uenv.metrics["position_error"]
        ori_err_sum += uenv.metrics["orientation_error"]
        step_count += 1

        done = terminated[agents[0]].bool() | time_outs[agents[0]].bool()
        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            for i in done_ids.tolist():
                n = max(step_count[i].item(), 1.0)
                mean_pos = (pos_err_sum[i] / n).item()
                mean_ori = (ori_err_sum[i] / n).item()
                completed_returns.append(ep_return[i].item())
                completed_pos_err.append(mean_pos)
                completed_ori_err.append(mean_ori)
                completed_success.append(1.0 if mean_pos < goal_range else 0.0)
                episodes_done += 1
            ep_return[done_ids] = 0.0
            pos_err_sum[done_ids] = 0.0
            ori_err_sum[done_ids] = 0.0
            step_count[done_ids] = 0.0
            policy.reset(done_ids.tolist())

    import numpy as np

    summary = {
        "checkpoint": args_cli.checkpoint,
        "episodes": len(completed_returns),
        "return_mean": float(np.mean(completed_returns)),
        "return_std": float(np.std(completed_returns)),
        "pos_err_mean": float(np.mean(completed_pos_err)),
        "ori_err_mean": float(np.mean(completed_ori_err)),
        "success_rate": float(np.mean(completed_success)),
    }
    print("[RESULT]", summary)

    if args_cli.metrics_out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.metrics_out)), exist_ok=True)
        write_header = not os.path.exists(args_cli.metrics_out)
        with open(args_cli.metrics_out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()))
            if write_header:
                w.writeheader()
            w.writerow(summary)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
