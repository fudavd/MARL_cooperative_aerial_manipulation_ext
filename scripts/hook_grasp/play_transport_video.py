# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a video of the v86 PPO-finetuned CARRY policy on the 3-drone transport hand-off env.

Env = the v86 training env verbatim (hover_rand + actor_indi_residual 153-dim hook-layout obs, wall
directive, deterministic horizontal spawn, per-env attachment DR, mass pinned); policy = one shared
feedforward ActorCritic run on all 3 drones (agent-major batch), deterministic (act_inference).
debug_vis stays ON so the commanded goal pose frame is visible in the video. Manual-render pipeline
(env.render() -> frames -> imageio -> os._exit), static overview camera (the follow-cam viewport
controller kills the hybrid-GPU laptop).
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Video of the PPO-finetuned transport carry policy.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-rand-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="v86 train_rslrl_transport model_*.pt.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=1800)
parser.add_argument("--frame_every", type=int, default=3)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--env_index", type=int, default=0, help="Env framed by the camera (env origins are spaced).")
parser.add_argument("--out_dir", type=str, default="scripts/hook_grasp/_render_transport")
parser.add_argument("--tag", type=str, default="v86_carry")
parser.add_argument("--hidden_dim", type=int, default=256, help="Actor/critic hidden width (must match ckpt).")
parser.add_argument("--dump_traj", type=str, default="",
                    help="Write a per-step trajectory .npz (drones/load/goal/cables) instead of RTX rendering — "
                         "for plot_transport_traj.py matplotlib animation (no cameras; survives broken drivers).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = not bool(args_cli.dump_traj)   # dump mode is pure headless (no RTX at all)
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.modules import ActorCritic, EmpiricalNormalization

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import parse_env_cfg

ACTOR_OBS = 153
CRITIC_OBS = 84
ACT_DIM = 5


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # the v86 hand-off env, verbatim (see train_rslrl_transport.py)
    env_cfg.partial_obs = True
    env_cfg.history_len = 3
    env_cfg.actor_indi_residual = True
    env_cfg.observation_spaces = {a: ACTOR_OBS for a in env_cfg.possible_agents}
    env_cfg.debug_vis = not bool(args_cli.dump_traj)   # goal-pose marker in RTX video; off in headless dump
    env_cfg.randomize_attachment = True
    env_cfg.attachment_per_env = True
    env_cfg.scene.replicate_physics = False
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (1.0, 1.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    env_cfg.events.randomize_mass.params["mass_distribution_params"] = (1.4, 1.4)
    # static overview framing env_index's origin (env spacing 8 m)
    spacing = float(env_cfg.scene.env_spacing)
    env_cfg.viewer.eye = (4.0, 4.0, 3.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 1.2)
    if hasattr(env_cfg.viewer, "env_index"):
        env_cfg.viewer.env_index = args_cli.env_index
        env_cfg.viewer.origin_type = "env"

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base = env.unwrapped
    assert isinstance(base, DirectMARLEnv)
    agents = list(env_cfg.possible_agents)
    E, dev = base.num_envs, base.device

    ck = torch.load(args_cli.checkpoint, map_location=dev, weights_only=False)
    hd = args_cli.hidden_dim
    policy = ActorCritic(num_actor_obs=ACTOR_OBS, num_critic_obs=CRITIC_OBS, num_actions=ACT_DIM,
                         actor_hidden_dims=[hd, hd], critic_hidden_dims=[hd, hd], activation="elu").to(dev)
    policy.load_state_dict(ck["model_state_dict"])
    policy.eval()
    norm = EmpiricalNormalization(shape=[ACTOR_OBS]).to(dev)
    norm.load_state_dict(ck["obs_norm_state_dict"])
    norm.eval()
    print(f"[transport-video] loaded {args_cli.checkpoint} (iter {ck.get('iter')})")

    frames = []
    surv = torch.zeros(E, device=dev)
    ep_lens, pos_errs = [], []
    crashes = 0
    dump = bool(args_cli.dump_traj)
    rec = {k: [] for k in ("drones", "load_pos", "load_mat", "goal", "wall_dir", "done")} if dump else None

    with torch.inference_mode():
        obs_dict, _ = base.reset()
        for step in range(args_cli.steps):
            o3 = torch.nan_to_num(torch.cat([obs_dict[a] for a in agents], dim=0)).clamp_(-1000, 1000)
            act = policy.act_inference(norm(o3))
            actions = {a: act[k * E:(k + 1) * E] for k, a in enumerate(agents)}
            obs_dict, _, term, trunc, _ = base.step(actions)
            d = (term[agents[0]] | trunc[agents[0]]).bool()
            surv += 1
            if d.any():
                ep_lens += surv[d].tolist()
                crashes += int(term[agents[0]][d].sum())
                surv[d] = 0
            pos_errs.append(float(base.metrics["position_error"].mean()))
            if dump:
                wall_w = torch.einsum("nij,nkj->nki", base.current_load_matrix, base._attach_wall_center_bf)
                rec["drones"].append(base.drone_positions.cpu().numpy().copy())        # (E, 3, 3)
                rec["load_pos"].append(base.load_position.cpu().numpy().copy())        # (E, 3)
                rec["load_mat"].append(base.current_load_matrix.cpu().numpy().copy())  # (E, 3, 3)
                rec["goal"].append(base.pose_command_w.cpu().numpy().copy())           # (E, 7)
                rec["wall_dir"].append(wall_w.cpu().numpy().copy())                    # (E, 3, 3) cable anchors rel load
                rec["done"].append(d.cpu().numpy().copy())                             # (E,)
            elif step % args_cli.frame_every == 0:
                frame = base.render()
                if frame is not None:
                    frames.append(np.asarray(frame))

    if dump:
        np.savez(args_cli.dump_traj, **{k: np.stack(v) for k, v in rec.items()})
        print(f"[transport-video] wrote trajectory dump {args_cli.dump_traj}")

    if frames:
        from PIL import Image
        Image.fromarray(frames[len(frames) // 2]).save(os.path.join(args_cli.out_dir, f"{args_cli.tag}_hero.png"))
        try:
            import imageio.v2 as imageio
            mp4 = os.path.join(args_cli.out_dir, f"{args_cli.tag}.mp4")
            imageio.mimsave(mp4, frames, fps=args_cli.fps)
            print(f"[transport-video] wrote {mp4} ({len(frames)} frames @ {args_cli.fps}fps)")
        except Exception as e:  # noqa: BLE001
            print(f"[transport-video] mp4 save skipped: {e}")

    ep = np.array(ep_lens) if ep_lens else np.array([float(args_cli.steps)])
    summary = "\n".join([
        "==================== v86 CARRY ROLLOUT ====================",
        f" checkpoint     : {args_cli.checkpoint}",
        f" steps / frames : {args_cli.steps} / {len(frames)}",
        f" episodes ended : {len(ep_lens)} (crash-terminated: {crashes})",
        f" ep_len         : mean {ep.mean():.0f}  median {np.median(ep):.0f}  max {ep.max():.0f} (timeout=2000)",
        f" pos_error mean : {np.mean(pos_errs):.3f} m (last quarter: {np.mean(pos_errs[-len(pos_errs)//4:]):.3f})",
        "===========================================================",
    ])
    print("\n" + summary)
    with open(os.path.join(args_cli.out_dir, f"{args_cli.tag}_summary.txt"), "w") as f:
        f.write(summary + "\n")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        os._exit(0)
