# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a video of a trained skrl-MAPPO (CTDE, HOOK-BLIND actor) controller on the hook-grasp env.

Mirrors play_ppo_video.py's reliable manual-render pipeline (env.render() -> frames -> imageio ->
os._exit), but loads the MAPPO checkpoint the way eval_mappo.py does: keep the env MULTI-AGENT
(DirectMARLEnv, NO multi_agent_to_single_agent -> the actor is fed the 135-dim HOOK-FREE obs, never
the 180-dim privileged state), and act DETERMINISTICALLY on the per-agent mean actions.

Example:
    python scripts/hook_grasp/play_mappo_video.py --headless --enable_cameras \
        --checkpoint experiments/best_checkpoints/v25F3_maxstack_95k_..._MAPPO_hookblind.pt --steps 900
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Video of a trained skrl-MAPPO hook-blind controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to skrl MAPPO checkpoint .pt.")
parser.add_argument("--steps", type=int, default=900, help="Rollout steps to record.")
parser.add_argument("--frame_every", type=int, default=3, help="Capture a frame every N steps.")
parser.add_argument("--fps", type=int, default=20, help="Output video fps.")
parser.add_argument("--seed", type=int, default=7, help="Seed.")
parser.add_argument("--capture_radius", type=float, default=0.13, help="Catch radius (m); FINAL task=0.13.")
parser.add_argument("--env_index", type=int, default=0, help="Which env the follow-camera tracks.")
parser.add_argument("--scan", action="store_true", default=False, help="No render; just report the best-holding env index.")
parser.add_argument("--out_dir", type=str, default="scripts/hook_grasp/_render_mappo", help="Output dir.")
parser.add_argument("--tag", type=str, default="mappo_policy", help="Output file basename.")
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
from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    # honest render at the TRUE task: disable every training curriculum / aid (same list as eval_mappo.py)
    for _attr in ("capture_curriculum", "hold_curriculum", "spawn_curriculum", "attached_spawn_curriculum"):
        if hasattr(env_cfg, _attr):
            setattr(env_cfg, _attr, False)
    if hasattr(env_cfg, "attachment") and args_cli.capture_radius > 0:
        env_cfg.attachment.capture_radius = args_cli.capture_radius

    # follow-camera on the drone body (same framing as play_ppo_video/play_sanity)
    env_cfg.viewer.origin_type = "asset_body"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.body_name = "Falcon_base_link_inertia"
    env_cfg.viewer.eye = (2.2, 2.2, 0.6)
    env_cfg.viewer.lookat = (0.0, 0.0, -0.4)
    if hasattr(env_cfg.viewer, "env_index"):
        env_cfg.viewer.env_index = args_cli.env_index

    # ---- MAPPO agent cfg; keep env MULTI-AGENT (hook-blind actor). ----
    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_mappo_cfg_entry_point")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base = env.unwrapped  # MARLHookGraspEnv; render() reflects live physics
    assert isinstance(base, DirectMARLEnv), "play_mappo_video expects a DirectMARLEnv (multi-agent) task."
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    for _k in ("learning_rate_scheduler_actor", "learning_rate_scheduler_critic"):
        agent_cfg["agent"][_k] = None

    runner = Runner(env, agent_cfg)
    print(f"[mappo-video] loading checkpoint: {args_cli.checkpoint}")
    runner.agent.load(args_cli.checkpoint)
    runner.agent.set_running_mode("eval")

    agents = list(base.possible_agents)

    obs, _ = env.reset()
    actor_dim = int(obs[agents[0]].shape[-1])
    print(f"[mappo-video] agents={agents}  actor_obs_dim={actor_dim} (must be 135 hook-free)")
    assert actor_dim == 135, f"actor obs must be 135-dim (45*3) hook-free, got {actor_dim}"

    frames, frame_steps = [], []
    caught_total, detached_total, attached_steps = 0, 0, 0
    min_hook_bar = float("inf")
    success_any = False
    E = base.num_envs
    dev = base.device
    consec = torch.zeros(E, dtype=torch.long, device=dev)      # current consecutive-attached run per env
    best_hold = torch.zeros(E, dtype=torch.long, device=dev)   # best consecutive-attached run per env
    ei = args_cli.env_index

    for step in range(args_cli.steps):
        with torch.inference_mode():
            sampled, _, outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = {uid: (outputs[uid]["mean_actions"] if "mean_actions" in outputs[uid] else sampled[uid])
                       for uid in agents}
            obs, _, _, _, _ = env.step(actions)

        att = base.attached.any(dim=-1)                        # (E,) bool
        consec = torch.where(att, consec + 1, torch.zeros_like(consec))
        best_hold = torch.maximum(best_hold, consec)
        caught_total += int(base._newly_caught.any().item())
        detached_total += int(base._just_detached.any().item())
        attached_steps += int(att[ei].item())
        if hasattr(base, "metrics") and "hook_bar_dist" in base.metrics:
            min_hook_bar = min(min_hook_bar, float(base.metrics["hook_bar_dist"][ei].item()))
        if hasattr(base, "succeeded") and bool(base.succeeded.any().item()):
            success_any = True

        if not args_cli.scan and step % args_cli.frame_every == 0:
            frame = base.render()
            if frame is not None:
                frames.append(np.asarray(frame))
                frame_steps.append(step)

    # report the best-holding envs (for choosing --env_index in a render pass)
    topk = torch.topk(best_hold, k=min(6, E))
    bh = "best-hold envs (idx:steps): " + " ".join(
        f"{int(i)}:{int(v)}" for v, i in zip(topk.values.tolist(), topk.indices.tolist()))
    print("[mappo-video] " + bh, flush=True)
    with open(os.path.join(args_cli.out_dir, f"{args_cli.tag}_besthold.txt"), "w") as _f:
        _f.write(bh + "\n")
        _f.flush()

    from PIL import Image

    if frames:
        Image.fromarray(frames[len(frames) // 2]).save(os.path.join(args_cli.out_dir, f"{args_cli.tag}_hero.png"))
        Image.fromarray(frames[-1]).save(os.path.join(args_cli.out_dir, f"{args_cli.tag}_last.png"))
        try:
            import imageio.v2 as imageio

            mp4 = os.path.join(args_cli.out_dir, f"{args_cli.tag}.mp4")
            imageio.mimsave(mp4, frames, fps=args_cli.fps)
            print(f"[mappo-video] wrote {mp4} ({len(frames)} frames @ {args_cli.fps}fps)")
        except Exception as e:  # noqa: BLE001
            print(f"[mappo-video] mp4 save skipped: {e}")

    lines = [
        "==================== MAPPO POLICY ROLLOUT ====================",
        f" checkpoint        : {args_cli.checkpoint}",
        f" steps / frames    : {args_cli.steps} / {len(frames)}",
        f" catches / detaches: {caught_total} / {detached_total}",
        f" min hook<->bar (m): {min_hook_bar:.3f}   (capture radius {base._acfg.capture_radius})",
        f" attached steps    : {attached_steps} / {args_cli.steps}",
        f" success (held)    : {success_any}",
        "=============================================================",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    with open(os.path.join(args_cli.out_dir, f"{args_cli.tag}_summary.txt"), "w") as f:
        f.write(summary + "\n")
        f.flush()

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        # os._exit avoids the headless display-teardown hang (same pattern as play_ppo_video.py)
        os._exit(0)
