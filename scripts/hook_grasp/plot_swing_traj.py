# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Plot swing-test trajectory dumps (from eval_swing.py --dump_traj) — pure numpy + matplotlib, NO Isaac.

Visualizes whether the hook swings as a PLANAR pendulum ALIGNED to the commanded azimuth. For each env it
plots the TOP-DOWN (x-y) trajectory of the hook RELATIVE to the drone (the pendulum bob about the wire pivot),
colored by time, with the commanded swing axis u_hat drawn as a dashed line. A tight arc along the dashed line
= a directed planar swing; a filled disk / circle = a conical whirl; a dot at the origin = a dead-hang.

Runs anywhere with matplotlib (sidesteps the ALC render segfault + the local Isaac-init hang).

Usage:
    ./.venv/bin/python scripts/hook_grasp/plot_swing_traj.py \
        --npz experiments/v63/traj_blind.npz experiments/v63/traj_true.npz experiments/v63/traj_obs.npz \
        --out experiments/v63/swing_arcs.png --gif experiments/v63/swing_true.gif --gif_npz experiments/v63/traj_true.npz
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _last_segment(done_k):
    """Indices of the LAST contiguous episode segment (no done) for one env's done flags (T,)."""
    T = len(done_k)
    end = T - 1
    # walk back over the final segment: it ends at the last step, starts after the previous done
    starts = np.where(done_k[:end] > 0.5)[0]
    start = (starts[-1] + 1) if len(starts) else 0
    return np.arange(start, T)


def _panel(ax, rel_k, phi_k, seg, title):
    xy = rel_k[seg, :2]                                   # (S,2) horizontal hook-drone
    phi = float(np.median(phi_k[seg]))                   # commanded azimuth (constant within an episode)
    R = 0.6                                               # ~wire length, for axis scale
    # commanded axis u_hat (dashed) and off-plane axis w_hat (faint dotted)
    ux, uy = np.cos(phi), np.sin(phi)
    ax.plot([-R * ux, R * ux], [-R * uy, R * uy], "--", color="0.4", lw=1.2, zorder=1, label="commanded plane")
    ax.plot([R * uy, -R * uy], [-R * ux, R * ux], ":", color="0.8", lw=1.0, zorder=1)
    # trajectory colored by time
    t = np.linspace(0, 1, len(seg))
    ax.scatter(xy[:, 0], xy[:, 1], c=t, cmap="viridis", s=6, zorder=3)
    ax.plot(xy[:, 0], xy[:, 1], color="0.6", lw=0.4, alpha=0.5, zorder=2)
    ax.scatter([0], [0], marker="+", color="red", s=60, zorder=4)   # pivot (straight-below)
    ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal")
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def main():
    ap = argparse.ArgumentParser(description="Plot swing-test trajectory dumps (top-down arcs vs commanded plane).")
    ap.add_argument("--npz", nargs="+", required=True, help="One or more traj .npz files (one per arm).")
    ap.add_argument("--out", type=str, default="swing_arcs.png", help="Output PNG (arms x envs grid).")
    ap.add_argument("--envs", type=int, default=4, help="How many envs to show per arm.")
    ap.add_argument("--gif", type=str, default="", help="Optional animated GIF path.")
    ap.add_argument("--gif_npz", type=str, default="", help="Which npz to animate (defaults to the first --npz).")
    args = ap.parse_args()

    data = [(os.path.basename(p), np.load(p, allow_pickle=True)) for p in args.npz]
    nrows = len(data)
    ncols = args.envs

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 3.0 * nrows), squeeze=False)
    for r, (name, d) in enumerate(data):
        rel, phi, done = d["rel"], d["phi"], d["done"]           # (T,K,3) (T,K) (T,K)
        arm = str(d["arm"]) if "arm" in d else name
        planarity = float(d["planarity"]) if "planarity" in d else float("nan")
        aim = float(d["aim_deg"]) if "aim_deg" in d else float("nan")
        whirl = float(d["whirl_sin"]) if "whirl_sin" in d else float("nan")
        K = min(ncols, rel.shape[1])
        for k in range(K):
            seg = _last_segment(done[:, k])
            _panel(axes[r][k], rel[:, k, :], phi[:, k], seg, f"env {k}")
        axes[r][0].set_ylabel(f"{arm}\nplan={planarity:.2f} aim={aim:.0f}° whirl={whirl:.2f}", fontsize=8)
    fig.suptitle("Swing plane vs commanded axis (top-down: hook relative to drone)  — dashed = commanded plane",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out, dpi=130)
    print(f"[plot] wrote {args.out}")

    if args.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter
        p = args.gif_npz or args.npz[0]
        d = np.load(p, allow_pickle=True)
        rel, phi, done = d["rel"], d["phi"], d["done"]
        arm = str(d["arm"]) if "arm" in d else os.path.basename(p)
        K = min(ncols, rel.shape[1])
        segs = [_last_segment(done[:, k]) for k in range(K)]
        L = min(len(s) for s in segs)
        segs = [s[-L:] for s in segs]                            # align lengths for a shared frame index
        R = 0.6
        gfig, gaxes = plt.subplots(1, K, figsize=(3.0 * K, 3.2), squeeze=False)
        gaxes = gaxes[0]
        arts = []
        for k in range(K):
            ax = gaxes[k]
            ph = float(np.median(phi[segs[k], k]))
            ux, uy = np.cos(ph), np.sin(ph)
            ax.plot([-R * ux, R * ux], [-R * uy, R * uy], "--", color="0.4", lw=1.2)
            ax.plot([R * uy, -R * uy], [-R * ux, R * ux], ":", color="0.85", lw=1.0)
            ax.scatter([0], [0], marker="+", color="red", s=60)
            ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal"); ax.set_title(f"env {k}", fontsize=8)
            ax.tick_params(labelsize=6)
            trail, = ax.plot([], [], color="tab:blue", lw=1.0, alpha=0.6)
            bob, = ax.plot([], [], "o", color="tab:blue", ms=7)
            arts.append((trail, bob))
        gfig.suptitle(f"{arm}: hook swing (top-down, relative to drone) vs commanded plane (dashed)", fontsize=10)
        gfig.tight_layout(rect=[0, 0, 1, 0.93])

        tail = 40

        def update(f):
            out = []
            for k in range(K):
                xy = rel[segs[k], k, :2]
                a = max(0, f - tail)
                arts[k][0].set_data(xy[a:f + 1, 0], xy[a:f + 1, 1])
                arts[k][1].set_data([xy[f, 0]], [xy[f, 1]])
                out += [arts[k][0], arts[k][1]]
            return out

        anim = FuncAnimation(gfig, update, frames=L, interval=50, blit=True)
        anim.save(args.gif, writer=PillowWriter(fps=20))
        print(f"[plot] wrote {args.gif} ({L} frames)")


if __name__ == "__main__":
    main()
