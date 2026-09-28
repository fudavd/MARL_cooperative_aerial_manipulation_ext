# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evolve a HEBBIAN-PLASTIC final layer on top of the FROZEN v74 PPO backbone (user direction, 2026-08-10).

Both pure-ES attempts (v79 CMA-ES, v80 OpenAI-ES at the full July scale) scored honest catch 0.0: under the
strict 25°/5 cm gates a near-catch earns nothing, so evolution from scratch has no fitness gradient. This
hybrid removes that problem by STARTING AT THE PPO SOLUTION:

    obs -> [frozen] empirical normalizer -> LSTM (memory_a) -> actor MLP hidden (actor[0..3]) -> phi (256)
        -> [plastic head] a = phi @ w_e^T + b0,   w_e(t=0) = W0 (the trained PPO head, actor.4)

GENOME (per the user's choice): the ABCD Hebbian plasticity coefficients of the head only —
A,B,C,D each (5,256) -> 5120 params. With genome = 0 the head never moves off W0, so generation 0 IS the
PPO policy (honest catch 0.787). Evolution shapes how the head ADAPTS ONLINE within an episode:

    every `ui` control steps:  dw = A*(post ⊗ pre) + B*pre + C*post + D          (per-connection)
                               w += hebbian_lr * dw - anchor * (w - W0)          (elastic anchor to W0)
                               w  = W0 + clamp(w - W0, ±delta_bound)             (bounded drift)
    pre = mean phi over the window (256), post = tanh(mean action) (5); w resets to W0 on episode reset.

OPTIMIZER: OpenAI-ES (antithetic pairs), CRN: env.crn=true + _crn_repeats/_spawn_generation wired here exactly
like the Hebbian adapter, so every genome in a generation faces identical episode conditions. Fitness = per-step
mean reward over the window.

v82 (user: retry with baseline-subtracted fitness + anchor; fixes the v81 failure mode):
v81 (centered ranks + Adam) completed 300 gens but strictly degraded the policy: theta_norm drifted
monotonically to 25 while fitness collapsed by gen ~28 and never recovered. Root cause: rank shaping is
offset/scale-INVARIANT — it hands a no-signal generation (fitness spread = pure episode noise) the exact same
gradient magnitude as a strong-signal one, and Adam's per-parameter normalization then converts that into a
constant-rate random walk. v82 changes:
  1. GRADIENT = RAW baseline-subtracted antithetic differences, NOT ranks: u_j = clip(0.5*(f+_j - f-_j)/u_scale,
     ±3), grad = mean(u_j * eps_j)/sigma, plain SGD (no Adam). The common episode condition cancels exactly
     within each CRN pair; a no-signal generation now yields a near-zero gradient instead of a full-size step.
  2. Two reserved (non-perturbed) population slots evaluated every generation under the SAME CRN conditions:
     genome=0 (pure PPO baseline) and genome=theta (current center). improvement = fit_center - fit_base is the
     honest cross-generation progress metric (raw fitness is dominated by the per-gen spawn condition).
  3. THETA ANCHOR: theta <- (1 - theta_anchor) * theta after each step (elastic decay toward the PPO solution;
     v81 anchored w to W0 within episodes but nothing anchored theta).
  4. TRUST-REGION REVERT: if improvement < -revert_margin for revert_patience consecutive generations, theta
     reverts to the best-improvement genome seen (initially 0 = PPO). theta_best.npy tracks that genome.

Run (ALC):
    ./.venv/bin/python scripts/hook_grasp/evolve_head.py --headless --device cuda:0 \
        --checkpoint <v74.pt> --popsize 256 --n_repeats 2 --generations 300 \
        --wandb --wandb_project hook-grasp-es-head --env_kv "<v74 recipe>"
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evolve a Hebbian-plastic head on the frozen PPO backbone.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="v74 rsl_rl checkpoint (frozen backbone + head init).")
parser.add_argument("--popsize", type=int, default=256, help="ES population (must be even; antithetic pairs).")
parser.add_argument("--n_repeats", type=int, default=2, help="Envs per genome (num_envs = popsize * n_repeats).")
parser.add_argument("--generations", type=int, default=300)
parser.add_argument("--steps_per_gen", type=int, default=2000, help="Control steps rolled per generation (~1 episode).")
parser.add_argument("--sigma", type=float, default=0.05, help="ES perturbation std on the ABCD genome.")
parser.add_argument("--es_lr", type=float, default=0.001, help="SGD lr on theta (v82: raw-diff gradient, no Adam).")
parser.add_argument("--u_scale", type=float, default=0.5, help="Fixed fitness-difference scale: u=clip(diff/u_scale, ±3).")
parser.add_argument("--theta_anchor", type=float, default=0.01, help="Elastic decay of theta toward 0 (PPO) per generation.")
parser.add_argument("--revert_margin", type=float, default=0.5, help="improvement < -margin counts as a bad generation.")
parser.add_argument("--revert_patience", type=int, default=10, help="Consecutive bad generations before reverting theta to best.")
parser.add_argument("--hebbian_lr", type=float, default=0.05, help="Online head update rate.")
parser.add_argument("--update_interval", type=int, default=25, help="Control steps between Hebbian head updates.")
parser.add_argument("--anchor", type=float, default=0.01, help="Elastic decay toward W0 per update (weight anchoring).")
parser.add_argument("--delta_bound", type=float, default=0.5, help="Max |w - W0| per connection.")
parser.add_argument("--genome_bound", type=float, default=2.0, help="Clamp on ABCD coefficients.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_hours", type=float, default=20.0)
parser.add_argument("--env_kv", type=str, default="", help="env cfg overrides (v74 recipe; same syntax as train_rslrl).")
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--wandb_project", type=str, default="hook-grasp-es-head")
parser.add_argument("--exp_name", type=str, default="es_head")
parser.add_argument("--log_root", type=str, default="experiments")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rslrl_adapter import RslRlMarlAdapter, apply_indi_residual_cfg, make_runner_cfg  # noqa: E402


def _apply_env_kv(env_cfg, s: str):
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.partition("=")
        lo = v.strip().lower()
        val = (lo == "true") if lo in ("true", "false") else (
            int(v) if v.strip().lstrip("-").isdigit() else float(v))
        obj, attr = (env_cfg, k.strip())
        if "." in k:
            head, attr = k.strip().split(".", 1)
            obj = getattr(env_cfg, head)
        setattr(obj, attr, val)
        print(f"[es-head] env override: {k.strip()} = {val}")


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    assert args_cli.popsize % 2 == 0, "popsize must be even (antithetic pairs)"
    num_envs = args_cli.popsize * args_cli.n_repeats
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=num_envs)
    env_cfg.seed = args_cli.seed
    apply_indi_residual_cfg(env_cfg, actor_history_len=3, residual=True)   # v74 actor layout (153)
    if args_cli.env_kv:
        _apply_env_kv(env_cfg, args_cli.env_kv)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    assert isinstance(base, DirectMARLEnv)
    adapter = RslRlMarlAdapter(base)
    # CRN wiring (mirrors the Hebbian adapter): scenario = env % n_repeats, redrawn per generation
    base._crn_repeats = int(args_cli.n_repeats)

    # ---- frozen PPO backbone --------------------------------------------------------------------
    train_cfg = make_runner_cfg(experiment_name="es_head", max_iterations=1, device=device, logger="tensorboard")
    runner = OnPolicyRunner(adapter, train_cfg, log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    runner.eval_mode()
    policy = runner.alg.policy
    normalizer = runner.obs_normalizer
    for p in policy.parameters():
        p.requires_grad_(False)
    head = policy.actor[-1]
    assert isinstance(head, torch.nn.Linear) and head.out_features == base.cfg.action_dim, \
        f"expected the actor's last module to be Linear(->{base.cfg.action_dim}), got {head}"
    trunk = policy.actor[:-1]                      # frozen feature extractor phi (256)
    W0 = head.weight.detach().clone()              # (5, 256)
    b0 = head.bias.detach().clone()                # (5,)
    A_DIM = W0.numel()                             # 1280
    GENOME = 4 * A_DIM                             # ABCD -> 5120
    E, dev = num_envs, base.device
    print(f"[es-head] frozen backbone loaded; head {tuple(W0.shape)}, genome dim {GENOME}, envs {E}")

    # ---- log dir / wandb ------------------------------------------------------------------------
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args_cli.log_root, args_cli.exp_name,
                           f"eshead_p{args_cli.popsize}_s{args_cli.sigma}_hlr{args_cli.hebbian_lr}"
                           f"_ui{args_cli.update_interval}_seed{args_cli.seed}_{ts}")
    os.makedirs(os.path.join(log_dir, "gens"), exist_ok=True)
    with open(os.path.join(log_dir, "settings.json"), "w") as f:
        json.dump(vars(args_cli), f, indent=2, default=str)
    print(f"[es-head] logging to {log_dir}")
    wb = None
    if args_cli.wandb:
        try:
            import wandb
            wb = wandb.init(project=args_cli.wandb_project, name=os.path.basename(log_dir),
                            config=vars(args_cli), dir=log_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[es-head] wandb disabled: {e}")

    # ---- ES state --------------------------------------------------------------------------------
    theta = torch.zeros(GENOME, device=dev)                      # ABCD = 0 -> generation 0 == PPO
    n_pert = args_cli.popsize - 2                                # last 2 slots: theta=0 baseline, theta center
    assert n_pert >= 2 and n_pert % 2 == 0, "popsize must be even and >= 4 (2 slots reserved for baseline/center)"
    best_theta = torch.zeros_like(theta)                         # best-improvement genome (init = PPO)
    best_impr = 0.0                                              # theta=0 improves on itself by 0 by definition
    bad_streak = 0
    n_reverts = 0
    gen_rng = torch.Generator(device=dev); gen_rng.manual_seed(args_cli.seed)

    def head_batch(genomes: torch.Tensor):
        """genomes (P, GENOME) -> per-env ABCD tensors (E, 5, 256) each (env e -> genome e // n_repeats)."""
        g = genomes.clamp_(-args_cli.genome_bound, args_cli.genome_bound)
        per_env = g.repeat_interleave(args_cli.n_repeats, dim=0)               # (E, GENOME)
        A, B, C, D = per_env.split(A_DIM, dim=1)
        shape = (E,) + tuple(W0.shape)
        return A.view(shape), B.view(shape), C.view(shape), D.view(shape)

    t0 = time.time()
    best_f = -1e9
    for gen in range(args_cli.generations):
        if (time.time() - t0) / 3600.0 > args_cli.max_hours:
            print(f"[es-head] max_hours reached at generation {gen}")
            break
        base._spawn_generation = gen                                            # generation-driven curricula + CRN redraw
        theta_eval = theta.clone()                                              # what the center slot measures this gen
        # antithetic perturbations on slots [0, n_pert): pair i -> genomes 2i (+eps) and 2i+1 (-eps);
        # slot n_pert = theta=0 (pure PPO baseline), slot n_pert+1 = unperturbed theta (center)
        eps = torch.randn(n_pert // 2, GENOME, device=dev, generator=gen_rng)
        genomes = torch.empty(args_cli.popsize, GENOME, device=dev)
        genomes[0:n_pert:2] = theta + args_cli.sigma * eps
        genomes[1:n_pert:2] = theta - args_cli.sigma * eps
        genomes[n_pert] = 0.0
        genomes[n_pert + 1] = theta
        A, B, C, D = head_batch(genomes)

        # The WHOLE generation (env reset + rollout + plastic-head updates) runs inside ONE inference_mode
        # block: env.step/reset write env buffers in-place, and mixing inference/normal tensors across the
        # boundary raises "Inplace update to inference tensor outside InferenceMode". Fitness is .clone()d
        # out at the end (the documented escape hatch to a normal tensor).
        with torch.inference_mode():
            # per-env plastic head state
            w = W0.unsqueeze(0).repeat(E, 1, 1)                                 # (E, 5, 256)
            pre_acc = torch.zeros(E, W0.shape[1], device=dev)                   # mean phi over window
            post_acc = torch.zeros(E, W0.shape[0], device=dev)
            acc_n = 0
            rew_sum = torch.zeros(E, device=dev)

            obs, _ = adapter.reset()
            policy.reset()
            for t in range(args_cli.steps_per_gen):
                phi = trunk(policy.memory_a(normalizer(obs)).squeeze(0))        # (E, 256)
                actions = torch.bmm(w, phi.unsqueeze(-1)).squeeze(-1) + b0      # per-env plastic head
                obs, rewards, dones, _ = adapter.step(actions)
                policy.reset(dones)
                rew_sum += rewards
                pre_acc += phi; post_acc += torch.tanh(actions); acc_n += 1
                if (t + 1) % args_cli.update_interval == 0:
                    pre = (pre_acc / acc_n)                                      # (E, 256)
                    post = (post_acc / acc_n)                                    # (E, 5)
                    outer = post.unsqueeze(-1) * pre.unsqueeze(1)                # (E, 5, 256)
                    dw = A * outer + B * pre.unsqueeze(1) + C * post.unsqueeze(-1) + D
                    w = w + args_cli.hebbian_lr * dw - args_cli.anchor * (w - W0)
                    w = W0 + (w - W0).clamp_(-args_cli.delta_bound, args_cli.delta_bound)
                    pre_acc.zero_(); post_acc.zero_(); acc_n = 0
                if dones.any():                                                  # plasticity is intra-episode
                    m = dones.bool()
                    w[m] = W0

            fit_env = (rew_sum / args_cli.steps_per_gen).clone()                 # escape inference mode

        fit = fit_env.clone().view(args_cli.popsize, args_cli.n_repeats).mean(dim=1)   # per-genome (normal tensor)
        fit_p = fit[:n_pert]                                                    # perturbed genomes only
        fit_base = float(fit[n_pert])                                           # pure PPO, this gen's CRN conditions
        fit_center = float(fit[n_pert + 1])                                     # current theta, same conditions
        improvement = fit_center - fit_base

        # v82 gradient: RAW baseline-subtracted antithetic differences on a FIXED scale (not ranks — rank
        # shaping is offset/scale-invariant, so a no-signal generation gets a full-size gradient; raw diffs
        # make gradient magnitude proportional to actual signal). Plain SGD; no Adam (its per-parameter
        # normalization turned v81's noise into a constant-rate drift, theta_norm -> 25).
        diff = 0.5 * (fit_p[0::2] - fit_p[1::2])                                # common CRN condition cancels
        u = (diff / args_cli.u_scale).clamp(-3.0, 3.0)
        grad = (u.unsqueeze(1) * eps).mean(dim=0) / args_cli.sigma
        theta = theta + args_cli.es_lr * grad
        theta = ((1.0 - args_cli.theta_anchor) * theta).clamp_(-args_cli.genome_bound, args_cli.genome_bound)

        # trust region: track the best-improvement genome; revert after sustained regression vs the PPO baseline
        if improvement > best_impr:
            best_impr = improvement
            best_theta = theta_eval.clone()
            np.save(os.path.join(log_dir, "theta_best.npy"), best_theta.cpu().numpy())
        if improvement < -args_cli.revert_margin:
            bad_streak += 1
        else:
            bad_streak = 0
        if bad_streak >= args_cli.revert_patience:
            theta = best_theta.clone()
            n_reverts += 1
            bad_streak = 0
            print(f"[es-head] REVERT #{n_reverts}: improvement < -{args_cli.revert_margin} for "
                  f"{args_cli.revert_patience} gens -> theta reset to best (impr={best_impr:.3f})", flush=True)

        gbest = float(fit_p.max()); gmean = float(fit_p.mean())
        af = float(base.attached.any(dim=-1).float().mean())
        if gbest > best_f:
            best_f = gbest
        np.save(os.path.join(log_dir, "gens", f"theta_{gen}.npy"), theta.cpu().numpy())
        print(f"Generation {gen}:\tbest={gbest:.4f}\tmean={gmean:.4f}\tbase={fit_base:.4f}"
              f"\tcenter={fit_center:.4f}\timpr={improvement:.4f}\taf={af:.4f}", flush=True)
        if wb:
            wb.log({"gen": gen, "fit_best": gbest, "fit_mean": gmean, "fit_best_so_far": best_f,
                    "fit_base": fit_base, "fit_center": fit_center, "improvement": improvement,
                    "best_improvement": best_impr, "diff_abs_mean": float(diff.abs().mean()),
                    "n_reverts": n_reverts, "attached_frac": af, "theta_norm": float(theta.norm())}, step=gen)

    np.save(os.path.join(log_dir, "theta_final.npy"), theta.cpu().numpy())
    print(f"[es-head] done; theta_final.npy in {log_dir}")
    if wb:
        wb.finish()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        os._exit(0)
