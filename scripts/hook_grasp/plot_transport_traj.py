# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Animate transport-carry trajectory dumps (play_transport_video.py --dump_traj) — pure matplotlib, NO Isaac.

3D world-frame scene per env: the basket as an oriented box, the 3 drones (markers), the 3 cables
(drone -> its wall-anchor on the basket), and the commanded goal pose (RGB axis triad). Cables flash red on
the step an episode terminates. This is the render-pipeline-free "video" of the carry behaviour (the local
RTX render stack segfaults under the 595.84 open-kernel driver, 2026-08-17).

Usage:
    ./.venv/bin/python scripts/hook_grasp/plot_transport_traj.py --npz /tmp/v86_traj.npz \
        --out experiments/v86_video/v86_carry_env0.mp4 --env 0 --fps 25 --every 4
"""

import argparse
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--npz", type=str, required=True)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--env", type=int, default=0)
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--every", type=int, default=4, help="Use every Nth step as a frame.")
args = parser.parse_args()

HX, HY, HZ_LO, HZ_HI = 0.275, 0.225, -0.06, 0.08   # basket half-extents around the load CoM (z: bottom/rim)


def box_edges(pos, R):
    """12 edges of the oriented basket box, world frame -> list of (2,3) segments."""
    corners = np.array([[sx * HX, sy * HY, z] for z in (HZ_LO, HZ_HI) for sx in (-1, 1) for sy in (-1, 1)])
    w = (R @ corners.T).T + pos
    idx = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return [w[[a, b]] for a, b in idx]


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


d = np.load(args.npz)
e = args.env
drones = d["drones"][:, e]      # (T, 3, 3)
load_p = d["load_pos"][:, e]    # (T, 3)
load_R = d["load_mat"][:, e]    # (T, 3, 3)
goal = d["goal"][:, e]          # (T, 7)
wall = d["wall_dir"][:, e]      # (T, 3, 3) anchors rel load CoM (world axes)
done = d["done"][:, e]          # (T,)
T = drones.shape[0]
steps = list(range(0, T, args.every))
print(f"[plot-transport] env {e}: {T} steps -> {len(steps)} frames; episodes ended at {np.where(done)[0].tolist()[:12]}")

fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
writer = FFMpegWriter(fps=args.fps)

with writer.saving(fig, args.out, dpi=110):
    for t in steps:
        ax.cla()
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2); ax.set_zlim(0, 3.0)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
        recent_done = done[max(0, t - args.every + 1): t + 1].any()
        # basket
        for seg in box_edges(load_p[t], load_R[t]):
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="saddlebrown", lw=1.4)
        # goal pose triad
        gp, gR = goal[t, :3], quat_to_mat(goal[t, 3:7])
        for k, c in enumerate("rgb"):
            axis = gR[:, k] * 0.35
            ax.plot([gp[0], gp[0] + axis[0]], [gp[1], gp[1] + axis[1]], [gp[2], gp[2] + axis[2]], c, lw=2)
        # drones + cables to their wall anchors
        anch = load_p[t] + wall[t]                                  # (3, 3)
        for i in range(3):
            dp = drones[t, i]
            ax.plot([dp[0]], [dp[1]], [dp[2]], marker="o", ms=9, color=f"C{i}")
            ax.plot([dp[0], anch[i, 0]], [dp[1], anch[i, 1]], [dp[2], anch[i, 2]],
                    color=("red" if recent_done else f"C{i}"), lw=1.2, alpha=0.85)
        perr = float(np.linalg.norm(goal[t, :3] - load_p[t]))
        ax.set_title(f"v86 carry — t={t * 0.01:.2f}s   |goal−load|={perr:.2f} m"
                     + ("   [EPISODE RESET]" if recent_done else ""))
        ax.view_init(elev=22, azim=(35 + t * 0.02))
        writer.grab_frame()

print(f"[plot-transport] wrote {args.out} ({len(steps)} frames @ {args.fps} fps)")
