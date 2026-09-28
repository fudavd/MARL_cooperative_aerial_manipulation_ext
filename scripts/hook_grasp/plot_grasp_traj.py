# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Animate hook-grasp trajectory dumps (from eval_behavior.py --dump_traj) — pure matplotlib, NO Isaac.

Renders an MP4 (ffmpeg) of the WORLD-frame scene per env: the basket box (4 walls, yawed), the target bar
point (red), the drone (marker), the wire (line drone->hook) and the hook bob — side-by-side TOP-DOWN (x-y)
and SIDE (dist-z) views. Wire turns green while ATTACHED. This is the render-pipeline-free "video" of the
behavior (both the ALC and local Isaac render paths are unreliable).

Usage:
    ./.venv/bin/python scripts/hook_grasp/plot_grasp_traj.py --npz /tmp/v64a_traj.npz \
        --out experiments/v64_render/v64a_scene.mp4 --env 0 --fps 25
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation  # noqa: E402


def _quat_yaw(q):
    """Yaw from a wxyz quaternion (basket only yaws)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _box_corners(bpos, yaw, hx, hy):
    """(4,2) world x-y corners of the basket footprint."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    local = np.array([[hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy]])
    return bpos[:2] + local @ R.T


def main():
    ap = argparse.ArgumentParser(description="Animate a hook-grasp trajectory dump (top-down + side views).")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True, help="Output .mp4 path")
    ap.add_argument("--env", type=int, default=0, help="Which dumped env to animate.")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--every", type=int, default=2, help="Use every Nth step as a frame.")
    ap.add_argument("--title", type=str, default="")
    args = ap.parse_args()

    d = np.load(args.npz)
    k = args.env
    drone, hook, bar = d["drone"][:, k], d["hook"][:, k], d["bar"][:, k]        # (T,3)
    att, done = d["attached"][:, k], d["done"][:, k]                            # (T,)
    bpos, bquat = d["basket_pos"][:, k], d["basket_quat"][:, k]                 # (T,3/4)
    hx, hy, wall_h = float(d["half_x"]), float(d["half_y"]), float(d["wall_h"])
    T = drone.shape[0]
    frames = list(range(0, T, args.every))

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(11, 5.2))
    ttl = args.title or f"env {k} — catch_rate {float(d['catch_rate']):.2f}, hold {float(d['hold']):.2f}"
    fig.suptitle(ttl, fontsize=11)

    # --- top-down (x-y) ---
    box_line, = ax_top.plot([], [], "-", color="saddlebrown", lw=2)
    bar_pt_top, = ax_top.plot([], [], "s", color="red", ms=7)
    wire_top, = ax_top.plot([], [], "-", color="0.4", lw=1.2)
    drone_top, = ax_top.plot([], [], "o", color="tab:blue", ms=9)
    hook_top, = ax_top.plot([], [], "o", color="tab:orange", ms=6)
    trail_top, = ax_top.plot([], [], "-", color="tab:orange", lw=0.7, alpha=0.5)
    ax_top.set_title("top-down (x–y)", fontsize=9); ax_top.set_aspect("equal")
    ax_top.set_xlim(-3, 3); ax_top.set_ylim(-3, 3); ax_top.tick_params(labelsize=7)

    # --- side (x-z; project on world x for simplicity) ---
    wall_side, = ax_side.plot([], [], "-", color="saddlebrown", lw=3)
    bar_pt_side, = ax_side.plot([], [], "s", color="red", ms=7)
    wire_side, = ax_side.plot([], [], "-", color="0.4", lw=1.2)
    drone_side, = ax_side.plot([], [], "o", color="tab:blue", ms=9)
    hook_side, = ax_side.plot([], [], "o", color="tab:orange", ms=6)
    trail_side, = ax_side.plot([], [], "-", color="tab:orange", lw=0.7, alpha=0.5)
    status = ax_side.text(0.02, 0.96, "", transform=ax_side.transAxes, fontsize=9, va="top")
    ax_side.set_title("side (x–z)", fontsize=9)
    ax_side.set_xlim(-3, 3); ax_side.set_ylim(0, 3); ax_side.tick_params(labelsize=7)

    tail = 90

    def update(fi):
        t = frames[fi]
        a = max(0, t - tail)
        yaw = _quat_yaw(bquat[t])
        corners = _box_corners(bpos[t], yaw, hx, hy)
        cc = np.vstack([corners, corners[:1]])
        box_line.set_data(cc[:, 0], cc[:, 1])
        bar_pt_top.set_data([bar[t, 0]], [bar[t, 1]])
        wire_top.set_data([drone[t, 0], hook[t, 0]], [drone[t, 1], hook[t, 1]])
        drone_top.set_data([drone[t, 0]], [drone[t, 1]])
        hook_top.set_data([hook[t, 0]], [hook[t, 1]])
        trail_top.set_data(hook[a:t + 1, 0], hook[a:t + 1, 1])

        # side view: draw the box as a wall band at its x extent (approximate: box x-range at basket yaw)
        xs = [corners[:, 0].min(), corners[:, 0].max()]
        wall_side.set_data([xs[0], xs[0], xs[1], xs[1]], [bpos[t, 2], bpos[t, 2] + wall_h, bpos[t, 2] + wall_h, bpos[t, 2]])
        bar_pt_side.set_data([bar[t, 0]], [bar[t, 2]])
        wire_side.set_data([drone[t, 0], hook[t, 0]], [drone[t, 2], hook[t, 2]])
        drone_side.set_data([drone[t, 0]], [drone[t, 2]])
        hook_side.set_data([hook[t, 0]], [hook[t, 2]])
        trail_side.set_data(hook[a:t + 1, 0], hook[a:t + 1, 2])
        col = "tab:green" if att[t] > 0.5 else "0.4"
        wire_top.set_color(col); wire_side.set_color(col)
        status.set_text(f"t={t}  {'ATTACHED' if att[t] > 0.5 else ''}{'  |RESET' if done[t] > 0.5 else ''}")
        return [box_line, bar_pt_top, wire_top, drone_top, hook_top, trail_top,
                wall_side, bar_pt_side, wire_side, drone_side, hook_side, trail_side, status]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // args.fps, blit=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    anim.save(args.out, writer=FFMpegWriter(fps=args.fps, bitrate=1800))
    print(f"[plot_grasp] wrote {args.out} ({len(frames)} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
