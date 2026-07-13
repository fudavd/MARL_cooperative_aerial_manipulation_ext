# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a video of a trained Hebbian controller in the hook-grasp env.

Loads a ``best.pt`` genome (from scripts/hebbian/train.py), rolls it out on the hook-grasp task with a
follow-camera, and writes an mp4 + stills. Reuses the camera / frame-capture pipeline of ``play_sanity.py``
but drives the actions with the evolved Hebbian policy instead of a scripted maneuver.

Example:
    python scripts/hook_grasp/play_hook.py --headless --enable_cameras \
        --checkpoint experiments/hook_grasp/<run>/checkpoints/best.pt --steps 600
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Video of a trained Hebbian controller on the hook-grasp env.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs (1 = single clean rollout).")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt.")
parser.add_argument("--steps", type=int, default=600, help="Rollout steps to record.")
parser.add_argument("--frame_every", type=int, default=4, help="Capture a frame every N steps.")
parser.add_argument("--fps", type=int, default=15, help="Output video fps.")
parser.add_argument("--seed", type=int, default=7, help="Seed.")
parser.add_argument("--hebbian_lr", type=float, default=None, help="Override online Hebbian rate (0 = frozen).")
parser.add_argument("--out_dir", type=str, default="scripts/hook_grasp/_render", help="Output dir.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from MARL_mav_carry_ext.algorithms.hebbian.adapter import HebbianPolicy


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # eval honestly at the TRUE capture radius: disable the training-time capture-radius curriculum
    # (else a fresh env starts at the loose bootstrap radius and inflates the catch rate). Mirrors
    # play_ppo_video.py.
    # honest eval: disable every training curriculum
    for _attr in ("capture_curriculum", "hold_curriculum", "spawn_curriculum"):
        if hasattr(env_cfg, _attr):
            setattr(env_cfg, _attr, False)

    # match the observation config the checkpoint was trained with (hook env: obs_dim_per_frame * history_len)
    ckpt = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    env_cfg.history_len = int(ckpt.get("history_len", getattr(env_cfg, "history_len", 1)))
    per_frame = getattr(env_cfg, "obs_dim_per_frame", None)
    obs_dim = per_frame * env_cfg.history_len if per_frame is not None else int(ckpt["obs_dim"])
    env_cfg.observation_spaces = {a: obs_dim for a in env_cfg.possible_agents}
    print(f"[play] history_len={env_cfg.history_len} obs_dim={obs_dim} (ckpt obs_dim={ckpt['obs_dim']})")

    # follow-camera on the drone body so the wire + hook + approach stay framed as it flies
    env_cfg.viewer.origin_type = "asset_body"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.body_name = "Falcon_base_link_inertia"
    env_cfg.viewer.eye = (2.2, 2.2, 0.6)
    env_cfg.viewer.lookat = (0.0, 0.0, -0.4)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    uenv = env.unwrapped
    agents = list(uenv.cfg.possible_agents)
    print(f"[play] agents={agents} num_envs={uenv.num_envs} device={uenv.device}")

    policy = HebbianPolicy(uenv, args_cli.checkpoint)
    if args_cli.hebbian_lr is not None:
        for _a in policy.actors.values():
            _a.lr = args_cli.hebbian_lr
        print(f"[play] hebbian_lr overridden -> {args_cli.hebbian_lr}")
    print(f"[play] best_fitness in ckpt = {ckpt.get('best_fitness')}")

    obs, _ = env.reset()
    policy.reset()

    frames, frame_steps = [], []
    nan_steps = 0
    caught_total, detached_total, dones_total = 0, 0, 0
    min_hook_bar = float("inf")
    attached_steps = 0
    success_any = False

    for step in range(args_cli.steps):
        with torch.no_grad():
            actions = policy.act(obs)
        obs, rewards, terminated, time_outs, _ = env.step(actions)

        if any(torch.isnan(obs[a]).any() for a in agents):
            nan_steps += 1
        caught_total += int(uenv._newly_caught.any().item())
        detached_total += int(uenv._just_detached.any().item())
        attached_steps += int(uenv.attached[0].any().item())
        min_hook_bar = min(min_hook_bar, float(uenv.metrics["hook_bar_dist"][0].item()))
        if bool(uenv.succeeded.any().item()):
            success_any = True

        done = terminated[agents[0]].bool() | time_outs[agents[0]].bool()
        done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        dones_total += int(done.any().item())
        if done_ids.numel() > 0:
            policy.reset(done_ids.tolist())

        if step % args_cli.frame_every == 0:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
                frame_steps.append(step)

    # save video + stills
    from PIL import Image

    if frames:
        Image.fromarray(frames[len(frames) // 2]).save(os.path.join(args_cli.out_dir, "hook_policy_hero.png"))
        Image.fromarray(frames[-1]).save(os.path.join(args_cli.out_dir, "hook_policy_last.png"))
        try:
            import imageio.v2 as imageio

            mp4 = os.path.join(args_cli.out_dir, "hook_policy.mp4")
            imageio.mimsave(mp4, frames, fps=args_cli.fps)
            print(f"[play] wrote {mp4} ({len(frames)} frames @ {args_cli.fps}fps)")
        except Exception as e:  # noqa: BLE001
            print(f"[play] mp4 save skipped: {e}")

    lines = [
        "==================== HOOK POLICY ROLLOUT ====================",
        f" checkpoint        : {args_cli.checkpoint}",
        f" best_fitness(ckpt): {ckpt.get('best_fitness')}",
        f" steps / frames    : {args_cli.steps} / {len(frames)}",
        f" NaN steps         : {nan_steps}",
        f" catches / detaches: {caught_total} / {detached_total}",
        f" min hook<->bar (m): {min_hook_bar:.3f}   (capture radius {uenv._acfg.capture_radius})",
        f" attached steps    : {attached_steps} / {args_cli.steps}",
        f" episode resets     : {dones_total}",
        f" success (held)    : {success_any}",
        "============================================================",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(args_cli.out_dir, "hook_policy_summary.txt"), "w") as f:
        f.write(summary + "\n")
        f.flush()
        os.fsync(f.fileno())
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
    simulation_app.close()
