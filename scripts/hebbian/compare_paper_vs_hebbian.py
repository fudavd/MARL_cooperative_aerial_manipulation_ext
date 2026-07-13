# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Head-to-head eval of the PAPER MAPPO policy (HuggingFace 3_mav.pt) vs our best Hebbian policy,
under ONE identical protocol in the same hover env (same reward, same seeded goals, same metric).

Both policies are driven through the exact same ``evaluate()`` loop used by eval_checkpoints.py, so
episode return / position error / orientation error / success rate are directly comparable. The only
difference is the observation encoding each policy was trained with:
  * paper  MAPPO : partial_obs, history_len=3 -> 135-dim policy obs (matches 3_mav.pt)
  * hebbian      : partial_obs, history_len read from the checkpoint (=1 -> 45-dim)

    python scripts/hebbian/compare_paper_vs_hebbian.py --headless --model paper \
        --checkpoint /path/3_mav.pt --num_envs 128 --episodes 10 --seed 123 --out /tmp/cmp.csv
    python scripts/hebbian/compare_paper_vs_hebbian.py --headless --model hebbian \
        --checkpoint logs/hebbian/_overall_best/checkpoints/best.pt --num_envs 128 --episodes 10 --seed 123 --out /tmp/cmp.csv
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["paper", "hebbian"], required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--attachment_per_env", type=int, default=-1,
                    help="hover_rand only: 1=per-env attachment DR (replicate_physics=False), "
                         "0=per-run, -1=leave cfg default.")
parser.add_argument("--episodes", type=int, default=10, help="episodes per env")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--out", type=str, default="/tmp/cmp.csv", help="results CSV (appended).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from MARL_mav_carry_ext.algorithms.hebbian.adapter import HebbianPolicy


def evaluate(env, uenv, policy, agents, goal_range, n_target):
    """Identical to scripts/hebbian/eval_checkpoints.py:evaluate (episode return + tracking + success)."""
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


class PaperMappoPolicy:
    """Deterministic (mean-action) forward pass of the paper's per-agent MAPPO actor.

    Reproduces skrl GaussianMixin inference: obs -> RunningStandardScaler -> ELU-MLP mean. Matches
    scripts/skrl/play.py which feeds ``mean_actions`` straight to env.step (clip_actions=False)."""

    def __init__(self, uenv, ckpt_path, agents, obs_dim=135, act_dim=5):
        from skrl.resources.preprocessors.torch import RunningStandardScaler

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.agents = list(agents)
        self.device = uenv.device
        self.nets, self.pre = {}, {}
        for a in self.agents:
            pol = ck[a]["policy"]
            # skrl_mappo_cfg.yaml: layers [1024,512,256,128], activations elu, output ACTIONS (no final act)
            net = nn.Sequential(
                nn.Linear(obs_dim, 1024), nn.ELU(),
                nn.Linear(1024, 512), nn.ELU(),
                nn.Linear(512, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, act_dim),
            )
            sd = {k[len("net_container."):]: v for k, v in pol.items() if k.startswith("net_container.")}
            net.load_state_dict(sd)
            net.eval().to(self.device)
            self.nets[a] = net
            sp = RunningStandardScaler(size=obs_dim, device=self.device)
            sp.load_state_dict(ck[a]["state_preprocessor"])
            sp.eval()
            self.pre[a] = sp

    def reset(self, ids=None):
        pass

    @torch.no_grad()
    def act(self, obs):
        out = {}
        for a in self.agents:
            x = obs[a].to(self.device)
            xn = self.pre[a](x, train=False)   # (x - mean)/sqrt(var+eps), skrl-exact
            out[a] = self.nets[a](xn)
        return out


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.partial_obs = True

    if args_cli.model == "paper":
        env_cfg.history_len = 3
        obs_dim = 45 * 3
    else:
        c0 = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
        env_cfg.history_len = int(c0.get("history_len", 1))
        obs_dim = 45 * env_cfg.history_len
    env_cfg.observation_spaces = {a: obs_dim for a in env_cfg.possible_agents}

    # perturbation env (hover_rand): attachment-randomization scope. per-env needs replicate_physics
    # off; __post_init__ already ran, so set both flags manually (mirror scripts/hebbian/train.py).
    if args_cli.attachment_per_env >= 0 and hasattr(env_cfg, "attachment_per_env"):
        env_cfg.attachment_per_env = bool(args_cli.attachment_per_env)
        if args_cli.attachment_per_env == 1:
            env_cfg.scene.replicate_physics = False
        print(f"[INFO] attachment_per_env={env_cfg.attachment_per_env} "
              f"(replicate_physics={env_cfg.scene.replicate_physics})", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    uenv = env.unwrapped
    agents = list(uenv.cfg.possible_agents)
    goal_range = uenv.cfg.goal_achieved_range
    n_target = args_cli.episodes * uenv.num_envs

    if args_cli.model == "paper":
        policy = PaperMappoPolicy(uenv, args_cli.checkpoint, agents, obs_dim=obs_dim)
        train_best = float("nan")
    else:
        policy = HebbianPolicy(uenv, args_cli.checkpoint)
        train_best = float(torch.load(args_cli.checkpoint, map_location="cpu",
                                      weights_only=False).get("best_fitness", float("nan")))

    m = evaluate(env, uenv, policy, agents, goal_range, n_target)

    import csv
    write_header = not os.path.exists(args_cli.out) or os.path.getsize(args_cli.out) == 0
    with open(args_cli.out, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["model", "checkpoint", "obs_dim", "train_best", "eval_ret", "eval_ret_std",
                        "pos_err_m", "ori_err_rad", "success", "n_episodes", "seed"])
        w.writerow([args_cli.model, os.path.basename(args_cli.checkpoint), obs_dim,
                    f"{train_best:.2f}", f"{m['ret']:.3f}", f"{m['ret_std']:.3f}",
                    f"{m['pos']:.4f}", f"{m['ori']:.4f}", f"{m['succ']:.4f}", m["n"], args_cli.seed])

    print(f"\n[COMPARE] model={args_cli.model}  obs_dim={obs_dim}  "
          f"eval_ret={m['ret']:.2f}+-{m['ret_std']:.1f}  pos_err={m['pos']:.3f} m  "
          f"ori_err={m['ori']:.3f} rad  success={m['succ']*100:.1f}%  (n={m['n']} episodes)", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
