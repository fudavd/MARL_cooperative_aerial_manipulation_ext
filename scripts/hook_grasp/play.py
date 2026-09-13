# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out a trained hook-grasp checkpoint on the FINAL task and record a video (follow-camera on the drone).

    python scripts/hook_grasp/play.py --headless --checkpoint logs/rsl_rl/hook_grasp/<run>/model_19999.pt

Writes ``<out_dir>/<tag>.mp4`` plus a short text summary. Rendering needs a GPU with a display driver
(``--headless`` still renders offscreen).
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Video of a trained hook-grasp PPO policy.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="rsl_rl model_<iter>.pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel environments (the camera follows env 0).")
parser.add_argument("--steps", type=int, default=900, help="Control steps to record.")
parser.add_argument("--frame_every", type=int, default=3, help="Capture a frame every N steps.")
parser.add_argument("--fps", type=int, default=20, help="Video fps.")
parser.add_argument("--seed", type=int, default=7, help="Seed.")
parser.add_argument("--out_dir", type=str, default="videos", help="Output directory.")
parser.add_argument("--tag", type=str, default="hook_grasp", help="Output file basename.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True   # offscreen rendering (needed for env.render() under --headless)
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import MARL_mav_carry_ext.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rsl_rl_env import HookGraspVecEnv  # noqa: E402


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.disable_curricula()
    env_cfg.hold_success_steps = 10_000_000   # keep holding until the episode times out (nicer to watch)

    # follow-camera on the drone body
    env_cfg.viewer.origin_type = "asset_body"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.body_name = "Falcon_base_link_inertia"
    env_cfg.viewer.eye = (2.2, 2.2, 0.6)
    env_cfg.viewer.lookat = (0.0, 0.0, -0.4)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base = env.unwrapped
    vec_env = HookGraspVecEnv(base)

    agent_cfg: dict = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg["device"] = device
    runner = OnPolicyRunner(vec_env, agent_cfg, log_dir=None, device=device)
    print(f"[play] loading {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)

    obs, _ = vec_env.get_observations()
    frames = []
    catches = detaches = attached_steps = 0
    for step in range(args_cli.steps):
        with torch.inference_mode():
            obs, _, _, _ = vec_env.step(policy(obs))
        catches += int(base._newly_caught[0].any())
        detaches += int(base._just_detached[0].any())
        attached_steps += int(base.attached[0].any())
        if step % args_cli.frame_every == 0:
            frame = base.render()
            if frame is not None:
                frames.append(np.asarray(frame))

    summary = "\n".join([
        f"checkpoint     : {args_cli.checkpoint}",
        f"steps / frames : {args_cli.steps} / {len(frames)}",
        f"catches        : {catches}",
        f"detaches       : {detaches}",
        f"attached steps : {attached_steps} / {args_cli.steps}",
    ])
    print("\n" + summary)
    with open(os.path.join(args_cli.out_dir, f"{args_cli.tag}_summary.txt"), "w") as f:
        f.write(summary + "\n")
    if frames:
        import imageio.v2 as imageio
        mp4 = os.path.join(args_cli.out_dir, f"{args_cli.tag}.mp4")
        imageio.mimsave(mp4, frames, fps=args_cli.fps)
        print(f"[play] wrote {mp4}")
    else:
        print("[play] no frames captured (rendering unavailable on this machine?)")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(0)   # skip the headless Isaac Sim teardown (can hang)
