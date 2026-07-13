# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a video of a trained skrl-PPO controller on the hook-grasp env.

Uses the RELIABLE manual-render pipeline (env.render() -> frames -> imageio -> os._exit) that
play_sanity.py/play_hook.py use, instead of gym RecordVideo (which hangs on this env in headless due to
the display teardown). Loads the skrl PPO agent exactly like scripts/skrl/play.py.

Example:
    python scripts/hook_grasp/play_ppo_video.py --headless --enable_cameras \
        --checkpoint logs/skrl/ppo_flycrane_hook/<run>/checkpoints/best_agent.pt --steps 600
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Video of a trained skrl-PPO controller on the hook-grasp env.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to skrl best_agent.pt.")
parser.add_argument("--steps", type=int, default=600, help="Rollout steps to record.")
parser.add_argument("--frame_every", type=int, default=4, help="Capture a frame every N steps.")
parser.add_argument("--fps", type=int, default=15, help="Output video fps.")
parser.add_argument("--seed", type=int, default=7, help="Seed.")
parser.add_argument("--out_dir", type=str, default="scripts/hook_grasp/_render", help="Output dir.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
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

from skrl.utils.runner.torch import Runner

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # eval honestly at the TRUE capture radius: disable the training-time capture-radius curriculum
    # (otherwise a fresh env would start at the loose bootstrap radius and inflate the catch rate).
    # honest eval: disable every training curriculum
    for _attr in ("capture_curriculum", "hold_curriculum", "spawn_curriculum"):
        if hasattr(env_cfg, _attr):
            setattr(env_cfg, _attr, False)
    # follow-camera on the drone body (same framing as play_sanity/play_hook)
    env_cfg.viewer.origin_type = "asset_body"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.body_name = "Falcon_base_link_inertia"
    env_cfg.viewer.eye = (2.2, 2.2, 0.6)
    env_cfg.viewer.lookat = (0.0, 0.0, -0.4)

    experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base = env.unwrapped  # MARLHookGraspEnv; render() from here reflects live physics
    # PPO is single-agent: wrap exactly like scripts/skrl/play.py so the agent gets the right obs/action
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
    print(f"[ppo-video] loading checkpoint: {args_cli.checkpoint}")
    runner.agent.load(args_cli.checkpoint)
    runner.agent.set_running_mode("eval")

    obs, _ = env.reset()
    frames, frame_steps = [], []
    caught_total, detached_total, attached_steps = 0, 0, 0
    min_hook_bar = float("inf")
    success_any = False

    for step in range(args_cli.steps):
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)

        caught_total += int(base._newly_caught.any().item())
        detached_total += int(base._just_detached.any().item())
        attached_steps += int(base.attached[0].any().item())
        min_hook_bar = min(min_hook_bar, float(base.metrics["hook_bar_dist"][0].item()))
        if bool(base.succeeded.any().item()):
            success_any = True

        if step % args_cli.frame_every == 0:
            frame = base.render()
            if frame is not None:
                frames.append(np.asarray(frame))
                frame_steps.append(step)

    from PIL import Image

    if frames:
        Image.fromarray(frames[len(frames) // 2]).save(os.path.join(args_cli.out_dir, "ppo_policy_hero.png"))
        Image.fromarray(frames[-1]).save(os.path.join(args_cli.out_dir, "ppo_policy_last.png"))
        try:
            import imageio.v2 as imageio

            mp4 = os.path.join(args_cli.out_dir, "ppo_policy.mp4")
            imageio.mimsave(mp4, frames, fps=args_cli.fps)
            print(f"[ppo-video] wrote {mp4} ({len(frames)} frames @ {args_cli.fps}fps)")
        except Exception as e:  # noqa: BLE001
            print(f"[ppo-video] mp4 save skipped: {e}")

    lines = [
        "==================== PPO POLICY ROLLOUT ====================",
        f" checkpoint        : {args_cli.checkpoint}",
        f" steps / frames    : {args_cli.steps} / {len(frames)}",
        f" catches / detaches: {caught_total} / {detached_total}",
        f" min hook<->bar (m): {min_hook_bar:.3f}   (capture radius {base._acfg.capture_radius})",
        f" attached steps    : {attached_steps} / {args_cli.steps}",
        f" success (held)    : {success_any}",
        "===========================================================",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(args_cli.out_dir, "ppo_policy_summary.txt"), "w") as f:
        f.write(summary + "\n")
        f.flush()
        os.fsync(f.fileno())
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
