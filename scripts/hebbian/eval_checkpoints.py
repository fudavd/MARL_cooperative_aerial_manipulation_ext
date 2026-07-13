# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate several Hebbian checkpoints in ONE Isaac Sim session and rank by true eval return.

The per-run ``best.pt`` stores the *training* best fitness, which is selection-biased on noisy
fitness. This re-evaluates each checkpoint over many fresh episodes to get its true mean return,
success rate, and tracking error, then ranks them -- so we can pick the genuinely best controller.

    python scripts/hebbian/eval_checkpoints.py --headless --num_envs 64 --episodes 3 \
        --checkpoints a/best.pt b/best.pt ...
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoints", nargs="+", required=True)
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=3, help="episodes per env per checkpoint")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--out", type=str, default="/tmp/cand/eval_ranking.csv", help="results CSV (incremental).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
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


def evaluate(env, uenv, policy, agents, goal_range, n_target):
    num_envs = uenv.num_envs
    obs, _ = env.reset()
    policy.reset()
    ep_ret = torch.zeros(num_envs, device=uenv.device)
    pos_sum = torch.zeros(num_envs, device=uenv.device)
    ori_sum = torch.zeros(num_envs, device=uenv.device)
    steps = torch.zeros(num_envs, device=uenv.device)
    rets, poss, oris, succ = [], [], [], []
    done_count = 0
    while done_count < n_target:
        actions = policy.act(obs)
        obs, rewards, terminated, time_outs, _ = env.step(actions)
        ep_ret += torch.stack([rewards[a] for a in agents], 0).sum(0)
        pos_sum += uenv.metrics["position_error"]
        ori_sum += uenv.metrics["orientation_error"]
        steps += 1
        done = terminated[agents[0]].bool() | time_outs[agents[0]].bool()
        ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        for i in ids.tolist():
            n = max(steps[i].item(), 1.0)
            rets.append(ep_ret[i].item())
            poss.append((pos_sum[i] / n).item())
            oris.append((ori_sum[i] / n).item())
            succ.append(1.0 if (pos_sum[i] / n).item() < goal_range else 0.0)
            done_count += 1
        ep_ret[ids] = 0.0; pos_sum[ids] = 0.0; ori_sum[ids] = 0.0; steps[ids] = 0.0
        if ids.numel() > 0:
            policy.reset(ids.tolist())
    return dict(ret=float(np.mean(rets)), ret_std=float(np.std(rets)),
                pos=float(np.mean(poss)), ori=float(np.mean(oris)),
                succ=float(np.mean(succ)), n=len(rets))


def main():
    # all candidates here share obs config (partial_obs, history_len) -> read from first ckpt
    c0 = torch.load(args_cli.checkpoints[0], map_location="cpu", weights_only=False)
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.partial_obs = bool(c0.get("partial_obs", True))
    env_cfg.history_len = int(c0.get("history_len", 1))
    obs_dim = (45 * env_cfg.history_len) if env_cfg.partial_obs else 87
    env_cfg.observation_spaces = {a: obs_dim for a in env_cfg.possible_agents}

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    agents = list(uenv.cfg.possible_agents)
    goal_range = uenv.cfg.goal_achieved_range
    n_target = args_cli.episodes * uenv.num_envs

    import csv
    out_csv = args_cli.out
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerow(["ckpt", "train_best", "eval_ret", "eval_ret_std", "pos_err", "ori_err", "success", "n"])

    results = []
    for ck in args_cli.checkpoints:
        bf = torch.load(ck, map_location="cpu", weights_only=False).get("best_fitness", float("nan"))
        policy = HebbianPolicy(uenv, ck)
        m = evaluate(env, uenv, policy, agents, goal_range, n_target)
        m["ckpt"] = ck
        m["train_best"] = float(bf)
        results.append(m)
        # append immediately (robust to a later crash) + flush stdout
        with open(out_csv, "a", newline="") as f:
            csv.writer(f).writerow([ck, f"{bf:.2f}", f"{m['ret']:.2f}", f"{m['ret_std']:.2f}",
                                    f"{m['pos']:.3f}", f"{m['ori']:.3f}", f"{m['succ']:.4f}", m["n"]])
        print(f"[EVAL] {ck}  train_best={bf:.1f}  eval_ret={m['ret']:.1f}+-{m['ret_std']:.0f}  "
              f"pos_err={m['pos']:.2f}  success={m['succ']*100:.1f}%  (n={m['n']})", flush=True)

    results.sort(key=lambda r: r["ret"], reverse=True)
    print("\n===== RANKING by true eval return =====")
    for r in results:
        print(f"  ret={r['ret']:6.1f}  succ={r['succ']*100:4.1f}%  pos={r['pos']:.2f}  "
              f"train_best={r['train_best']:6.1f}  {r['ckpt']}")
    print(f"\nBEST_BY_RETURN: {results[0]['ckpt']}")
    best_succ = max(results, key=lambda r: (r["succ"], -r["pos"]))
    print(f"BEST_BY_SUCCESS: {best_succ['ckpt']} (succ={best_succ['succ']*100:.1f}%, pos={best_succ['pos']:.2f})")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
