# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Quantify, for one Hebbian checkpoint, how much the plastic weights drift from their init and
how much the action (output) distribution shifts over an episode.

Anchors each actor's weights at reset (W0 == evolved init), then runs a fixed horizon WITHOUT
mid-episode resets, accumulating per-step stats over the still-alive envs:

  weight divergence vs init:
    rel_l2   = ||W_t - W0|| / ||W0||           (relative magnitude of drift)
    cos      = cosine(W_t, W0)                 (direction preserved? 1.0 = unchanged)
    max|dW|  = largest single-synapse change
    frac>1e-2 = fraction of synapses moved more than 1e-2
  output distribution shift (per action dim, across envs; baseline = steps [5,15)):
    drift    = |mean_t - mean_0| / std_0       (mean shift in baseline-std units)
    std_ratio= std_t / std_0                   (spread inflation/collapse)

    python scripts/hebbian/analyze_plasticity.py --headless --num_envs 64 --steps 300 \
        --checkpoint logs/hebbian/_p128_ewd_r8/checkpoints/best.pt
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300, help="horizon (no mid-episode resets).")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--out", type=str, default=None, help="optional CSV of the per-step trace.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv

import gymnasium as gym
import numpy as np
import torch

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from MARL_mav_carry_ext.algorithms.hebbian.adapter import HebbianPolicy


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    ckpt = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    env_cfg.partial_obs = bool(ckpt.get("partial_obs", True))
    env_cfg.history_len = int(ckpt.get("history_len", 1))
    obs_dim = (45 * env_cfg.history_len) if env_cfg.partial_obs else 87
    env_cfg.observation_spaces = {a: obs_dim for a in env_cfg.possible_agents}

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    dev = uenv.device
    agents = list(uenv.cfg.possible_agents)
    N = uenv.num_envs
    ui = int(ckpt.get("cfg", {}).get("update_interval", 1))

    obs, _ = env.reset()
    policy = HebbianPolicy(uenv, args_cli.checkpoint)
    policy.reset()

    # anchor: W0 per agent (== evolved init), captured right after reset
    W0 = {a: policy.actors[a].weights.clone() for a in agents}
    W0_norm = {a: W0[a].norm(dim=1).clamp(min=1e-8) for a in agents}  # [N]

    alive = torch.ones(N, dtype=torch.bool, device=dev)
    # trace[a] = list over t of dict(rel_l2, cos, maxabs, frac, act_mean[ndim], act_std[ndim], n_alive)
    trace = {a: [] for a in agents}

    for t in range(args_cli.steps):
        actions = policy.act(obs)  # applies the Hebbian update internally
        for a in agents:
            W = policy.actors[a].weights
            dW = W - W0[a]
            am = alive
            if am.sum() == 0:
                continue
            rel_l2 = (dW[am].norm(dim=1) / W0_norm[a][am])
            cos = torch.nn.functional.cosine_similarity(W[am], W0[a][am], dim=1)
            maxabs = dW[am].abs().max(dim=1).values
            frac = (dW[am].abs() > 1e-2).float().mean(dim=1)
            act = actions[a][am]  # [n_alive, ndim]
            trace[a].append(dict(
                rel_l2=rel_l2.mean().item(), cos=cos.mean().item(),
                maxabs=maxabs.mean().item(), frac=frac.mean().item(),
                act_mean=act.mean(dim=0).cpu().numpy(), act_std=act.std(dim=0).cpu().numpy(),
                n=int(am.sum().item()),
            ))
        obs, _, terminated, time_outs, _ = env.step(actions)
        done = terminated[agents[0]].bool() | time_outs[agents[0]].bool()
        alive = alive & ~done  # once an env ends, drop it from the within-episode stats

    # ---- summarize ----
    a0 = agents[0]
    ndim = len(trace[a0][0]["act_mean"])
    # baseline window [5,15) for the output distribution
    def baseline(a):
        lo, hi = 5, min(15, len(trace[a]))
        ms = np.stack([trace[a][i]["act_mean"] for i in range(lo, hi)]).mean(0)
        ss = np.stack([trace[a][i]["act_std"] for i in range(lo, hi)]).mean(0)
        return ms, np.clip(ss, 1e-6, None)

    print(f"\n=== plasticity / output drift :: {args_cli.checkpoint} ===")
    print(f"agents={agents}  num_envs={N}  horizon={args_cli.steps}  update_interval={ui}  "
          f"n_weights/agent={policy.actors[a0].n_weights}")
    milestones = [m for m in (1, 5, 10, 25, 50, 100, 150, 200, 250, args_cli.steps - 1) if m < len(trace[a0])]

    for a in agents:
        ms0, ss0 = baseline(a)
        print(f"\n-- {a} --")
        print(f"  {'t':>4} {'alive':>5} | {'relL2%':>7} {'cos':>6} {'max|dW|':>8} {'frac>1e-2%':>10} "
              f"| {'outDrift(σ0)':>12} {'stdRatio':>8}")
        for m in milestones:
            r = trace[a][m]
            drift = np.abs(r["act_mean"] - ms0) / ss0
            std_ratio = r["act_std"] / ss0
            print(f"  {m:>4} {r['n']:>5} | {100*r['rel_l2']:>7.2f} {r['cos']:>6.3f} {r['maxabs']:>8.3f} "
                  f"{100*r['frac']:>10.2f} | {drift.mean():>12.3f} {std_ratio.mean():>8.3f}")
        # heuristics
        last = trace[a][milestones[-1]]
        half = trace[a][milestones[len(milestones) // 2]]
        driftL = (np.abs(last["act_mean"] - ms0) / ss0).mean()
        print(f"  heuristic: weights drift to ~{100*last['rel_l2']:.1f}% of ||W0|| "
              f"(cos {last['cos']:.3f}); ~{100*last['frac']:.1f}% of synapses move >1e-2. "
              f"actions shift ~{driftL:.2f}σ from the early baseline by t={milestones[-1]}.")

    if args_cli.out:
        with open(args_cli.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["agent", "t", "n_alive", "rel_l2", "cos", "maxabs", "frac_gt_1e-2",
                        *[f"act_mean_{i}" for i in range(ndim)], *[f"act_std_{i}" for i in range(ndim)]])
            for a in agents:
                for t, r in enumerate(trace[a]):
                    w.writerow([a, t, r["n"], r["rel_l2"], r["cos"], r["maxabs"], r["frac"],
                                *r["act_mean"].tolist(), *r["act_std"].tolist()])
        print(f"\n[INFO] wrote per-step trace to {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
