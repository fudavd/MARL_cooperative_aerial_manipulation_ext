# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""M0 sanity + rendering for the hook-grasp env.

Spawns ``Isaac-flycrane-hook-grasp-v0``, drives it with a hover (zero) action for a few hundred steps,
asserts there are no NaNs, prints wire-swing / attachment stats, and saves a rendered PNG (+ mp4) of the
scene. A drone crash is fine -- the goal is to confirm the asset loads, the wire hangs & swings, the
control cascade runs, and the surrogate attachment plumbing executes without errors.

Run (headless, needs Isaac Sim):
    python scripts/hook_grasp/play_sanity.py --headless --enable_cameras --num_envs 1 --steps 300
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hook-grasp M0 sanity + render.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--steps", type=int, default=300, help="Steps to simulate.")
parser.add_argument("--seed", type=int, default=0, help="Seed.")
parser.add_argument("--out_dir", type=str, default="scripts/hook_grasp/_render", help="Where to save frames.")
parser.add_argument("--maneuver", action="store_true", default=False,
                    help="Drive an oscillating lateral accel so the drone flies and the wire swings (vs hover).")
parser.add_argument("--accel", type=float, default=3.0, help="Maneuver lateral accel amplitude (m/s^2).")
parser.add_argument("--freq", type=float, default=0.5, help="Maneuver oscillation frequency (Hz).")
parser.add_argument("--episode_len", type=float, default=None,
                    help="Override episode_length_s (short -> frequent resets, to probe the reset transient).")
parser.add_argument("--impulse", action="store_true", default=False,
                    help="Maneuver as a single sustained lateral push then release (excites a free swing).")
parser.add_argument("--debug_trace", action="store_true", default=False,
                    help="Print per-step commanded accel vs actual drone position (diagnose action response).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True  # needed for rgb_array render
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""

import os

import gymnasium as gym
import numpy as np
import torch

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    if args_cli.episode_len is not None:
        env_cfg.episode_length_s = args_cli.episode_len
    # camera follows the drone body so the hanging wire + hook stay in frame regardless of drift
    env_cfg.viewer.origin_type = "asset_body"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.body_name = "Falcon_base_link_inertia"
    env_cfg.viewer.eye = (1.4, 1.4, 0.3)
    env_cfg.viewer.lookat = (0.0, 0.0, -0.35)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    uenv = env.unwrapped
    agents = list(uenv.cfg.possible_agents)
    print(f"[sanity] agents={agents}  num_envs={uenv.num_envs}  device={uenv.device}")
    print(f"[sanity] body idx: falcon={uenv._falcon_idx} rotor={uenv._falcon_rotor_idx} hook={uenv._hook_idx}")

    from isaaclab.utils.math import euler_xyz_from_quat

    obs, _ = env.reset()
    if args_cli.debug_trace:
        uenv._dbg_ctrl = True
    act = {a: torch.zeros(uenv.num_envs, uenv.cfg.action_dim, device=uenv.device) for a in agents}

    frames, frame_steps = [], []
    nan_steps = 0
    hook_rel_z, swing_deg, caught_total, detached_total = [], [], 0, 0
    dones_total = 0
    roll_l, pitch_l, drone_z_l = [], [], []  # attitude diagnostics ("is the drone flipped?")
    ep_step_l, term_events, postreset_relz = [], [], []  # reset-transient probe

    dt = uenv.step_dt
    z_hold = uenv.drone_positions[0, 0, 2].item()  # altitude to hold during the maneuver
    ax = 0.0
    for step in range(args_cli.steps):
        # ACCBR action = [lin_acc(3), body_rates(2)]. In --maneuver mode command an oscillating lateral
        # acceleration (planar, near the pendulum resonance) so the wire visibly swings, plus a simple
        # P altitude-hold on the vertical accel so the open-loop drone stays airborne instead of drifting
        # into the ground. Zeros = hover.
        if args_cli.maneuver:
            t = step * dt
            if args_cli.impulse:
                # sustained lateral push (tilt settles to ~atan(a/g)), then release -> wire swings freely
                ax = args_cli.accel if (20 <= step < 60) else 0.0
            else:
                ax = args_cli.accel * float(np.sin(2 * np.pi * args_cli.freq * t))
            az = float(np.clip(2.5 * (z_hold - uenv.drone_positions[0, 0, 2].item()), -3.0, 3.0))
            for a in agents:
                act[a][:, 0] = ax
                act[a][:, 1] = 0.0
                act[a][:, 2] = az
        obs, rewards, terminated, time_outs, _ = env.step(act)

        # --- NaN checks ---
        bad = False
        for a in agents:
            if torch.isnan(obs[a]).any() or torch.isnan(rewards[a]).any():
                bad = True
        if torch.isnan(uenv.hook_positions).any() or torch.isnan(uenv.drone_positions).any():
            bad = True
        nan_steps += int(bad)

        # --- wire-swing stats (env 0, drone 0) ---
        hrel = (uenv.hook_positions[0, 0] - uenv.drone_positions[0, 0])
        hook_rel_z.append(hrel[2].item())
        wire_dir = hrel / (hrel.norm() + 1e-9)
        # angle from straight-down (0 deg = hanging, 90 = horizontal)
        cos_down = (-wire_dir[2]).clamp(-1, 1).item()
        swing_deg.append(float(np.degrees(np.arccos(cos_down))))
        caught_total += int(uenv._newly_caught.any().item())
        detached_total += int(uenv._just_detached.any().item())
        dones_total += int(terminated[agents[0]].any().item())

        # --- drone attitude (wrap to [-180,180] deg); flags a flipped/unstable drone ---
        r, p, _y = euler_xyz_from_quat(uenv.drone_orientations[0, 0].unsqueeze(0))
        wrap = lambda a: float((np.degrees(a.item()) + 180) % 360 - 180)  # noqa: E731
        roll_l.append(wrap(r[0]))
        pitch_l.append(wrap(p[0]))
        drone_z_l.append(uenv.drone_positions[0, 0, 2].item())

        # --- reset-transient probe: episode-step index, termination reason, post-reset hook rel-z ---
        ep_step = int(uenv.episode_length_buf[0].item())
        ep_step_l.append(ep_step)
        if args_cli.debug_trace and step < 80 and step % 4 == 0:
            dp = uenv.drone_positions[0, 0]
            sp = uenv.actions[agents[0]][0]  # what the env actually stored as the action
            qt = getattr(uenv, "_dbg_qtilt", float("nan"))
            al = getattr(uenv, "_dbg_alpha", float("nan"))
            th = getattr(uenv, "_dbg_thrusts", None)
            th_s = "[" + ",".join(f"{t:.2f}" for t in th) + "]" if th else "?"
            print(f"[dbg] s={step:3d} ax={ax:+.2f} drone_xy=[{dp[0]:+.2f},{dp[1]:+.2f}] z={dp[2]:.2f} "
                  f"pitch={pitch_l[-1]:+.1f} | qcmd_tilt={qt:.1f}deg alpha={al:.2f} thrusts={th_s}", flush=True)
        if ep_step <= 2:
            postreset_relz.append((ep_step, hrel[2].item()))
        if bool(terminated[agents[0]][0]) or bool(time_outs[agents[0]][0]):
            reason = ("entangled" if bool(uenv.entangled[0]) else "fly_low" if bool(uenv.drone_fly_low[0])
                      else "bounds" if bool(uenv.body_pos_outside[0]) else "success" if bool(uenv.succeeded[0])
                      else "timeout")
            term_events.append((ep_step, reason))

        if step % 10 == 0:
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame))
                frame_steps.append(step)

    # --- save render ---
    from PIL import Image

    png_path = os.path.join(args_cli.out_dir, "hook_grasp_env.png")
    if frames:
        # "hero" still: the captured frame nearest the max-swing step (best shows the wire swinging)
        max_swing_step = int(np.argmax(swing_deg))
        hero = int(np.argmin([abs(fs - max_swing_step) for fs in frame_steps]))
        Image.fromarray(frames[hero]).save(png_path)
        Image.fromarray(frames[-1]).save(os.path.join(args_cli.out_dir, "hook_grasp_env_last.png"))
        # tight center crop (follow-cam holds the drone near image center) for a body/propeller close-up
        fr = Image.fromarray(frames[hero])
        W, H = fr.size
        fr.crop((int(W * 0.40), int(H * 0.28), int(W * 0.60), int(H * 0.62))).resize((640, 640)).save(
            os.path.join(args_cli.out_dir, "drone_closeup.png"))
        try:
            import imageio.v2 as imageio

            imageio.mimsave(os.path.join(args_cli.out_dir, "hook_grasp_env.mp4"), frames, fps=8)
        except Exception as e:  # noqa: BLE001
            print(f"[sanity] mp4 save skipped: {e}")
        print(f"[sanity] saved {len(frames)} frames; hero=step {frame_steps[hero]} -> {png_path}")

    # --- summary ---
    z = np.array(hook_rel_z)
    s = np.array(swing_deg)
    roll = np.array(roll_l)
    pitch = np.array(pitch_l)
    dz = np.array(drone_z_l)
    # body up-axis in world (rotation-matrix 3rd column). ~[0,0,+1] => body upright; [~,~,-1] => flipped.
    body_up = uenv.drone_rot_matrices[0, 0, :, 2].detach().cpu().numpy()
    upright = (np.abs(roll).max() < 90.0) and (np.abs(pitch).max() < 90.0)
    swung = s.max() > (25.0 if args_cli.maneuver else 1.0)
    ok = (nan_steps == 0) and (z.mean() < -0.2) and swung and upright and bool(frames)
    lines = [
        "==================== M0 SANITY SUMMARY ====================",
        f" mode                 : {'MANEUVER (fly+swing)' if args_cli.maneuver else 'HOVER'}",
        f" steps / frames       : {args_cli.steps} / {len(frames)}",
        f" NaN steps            : {nan_steps}   {'OK' if nan_steps == 0 else 'FAIL <<<'}",
        f" drone roll (deg)     : min={roll.min():+6.1f}  max={roll.max():+6.1f}",
        f" drone pitch (deg)    : min={pitch.min():+6.1f}  max={pitch.max():+6.1f}   "
        f"({'UPRIGHT ok' if upright else 'FLIPPED <<<'})",
        f" body up-axis (world) : [{body_up[0]:+.3f}, {body_up[1]:+.3f}, {body_up[2]:+.3f}]   "
        f"(z=+1 => body upright dynamically)",
        f" drone z (m)          : start={dz[0]:.2f}  min={dz.min():.2f}  max={dz.max():.2f}",
        f" hook rel-z (m)       : mean={z.mean():+.3f}  min={z.min():+.3f}  max={z.max():+.3f}   (taut ~ -0.55)",
        f" wire swing angle(deg): mean={s.mean():5.1f}  min={s.min():5.1f}  max={s.max():5.1f}   (0=hang,90=horiz)",
        f" catches / detaches   : {caught_total} / {detached_total}",
        f" env-0 terminations   : {dones_total}",
        "--- reset-transient probe (env 0) ---",
        f" post-reset hook rel-z: " + ", ".join(f"s{s}:{z:+.2f}" for s, z in postreset_relz[:8])
        + ("  (taut ~ -0.60; near 0 => wire not settled)" if postreset_relz else " (none)"),
        f" terminations by (ep_step,reason): {term_events[:12]}",
        f" EARLY terms (ep_step<=3): {sum(1 for s, _ in term_events if s <= 3)} / {len(term_events)}"
        f"   (>0 => reset transient trips a termination <<<)",
        "==========================================================",
        f" RESULT: {'PASS' if ok else 'CHECK - see above'}"
        f"  [upright={upright}, swung={swung}, no-nan={nan_steps == 0}]",
    ]
    summary = "\n".join(lines)
    print("\n" + summary)
    # write to a file and force-exit: Isaac's app teardown can hang/SIGABRT and would otherwise
    # (a) strand the process and (b) swallow this unflushed stdout. os._exit after an explicit flush.
    with open(os.path.join(args_cli.out_dir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
        f.flush()
        os.fsync(f.fileno())
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 3)


if __name__ == "__main__":
    main()
    simulation_app.close()
