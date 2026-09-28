# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ATTACH -> TRANSPORT HAND-OFF: evolve a Hebbian-plastic head on the frozen v74 attach policy, in the
3-drone transport env (user direction, 2026-08-10).

The v74 hook-grasp policy (hook-blind, 51x3=153 obs, LSTM-256) learned to swing-attach a single drone to a
basket wall. This experiment runs THAT SAME POLICY on each of the 3 drones of the hover_rand transport env —
"as if the drones just did the attachment one by one" — and evolves ONLY the ABCD Hebbian plasticity
coefficients of its 5x256 action head (v82 recipe), so online plasticity is the mechanism that adapts the
attach policy into a cooperative carrier.

Obs match BY DESIGN: the hook env's 45-dim actor frame was built to mirror the hover partial obs layout
(basket=load, goal terms identical), and BOTH envs replace the ego one-hot id with the same wall-position
directive (centre of the drone's assigned/attached wall, relative to the load CoM, world axes). The only gap
was the 6-dim INDI disturbance residual, now appended per frame via cfg.actor_indi_residual (45 -> 51).

Episode start = the post-attachment state: deterministic horizontal spawn (roll/pitch/yaw = 0, fixed z
offset), cables taut, plastic head w = W0 (each drone begins with the pristine attach head and adapts online
during the carry). Task variability comes from the CRN'd random basket goal poses (pos + orientation) and the
per-env ATTACHMENT randomization (imperfect one-by-one attach points); basket mass is pinned at nominal so
the ES instrument stays clean.

ES = the v82 honest instrument (see scripts/hook_grasp/evolve_head.py):
  - gradient from RAW baseline-subtracted antithetic pair differences on a fixed scale (no ranks, no Adam),
  - two reserved population slots per generation (genome=0 = pure v74, unperturbed theta = center) ->
    improvement = fit_center - fit_base is the honest cross-generation metric,
  - theta anchor toward 0 + trust-region revert to the best-improvement genome.
Condition matching without a global CRN on attachments (they are baked per env at startup): antithetic pair
members and the baseline/center slots get IDENTICAL attachment draws via cfg.attachment_condition_slots, and
goal poses are CRN'd per scenario (env % n_repeats) — so every within-pair and center-vs-base fitness
difference compares like against like.

Run (ALC):
    ./.venv/bin/python scripts/hebbian/evolve_head_transport.py --headless --device cuda:0 \
        --checkpoint logs/rsl_rl/v74_noobs/2026-08-05_22-03-33_v74_noobs/model_19999.pt \
        --popsize 128 --n_repeats 2 --generations 300 --wandb --exp_name v83_handoff
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evolve a Hebbian-plastic head on the frozen v74 policy, 3-drone transport env.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-rand-v0")
parser.add_argument("--checkpoint", type=str, required=True, help="v74 rsl_rl checkpoint (frozen backbone + head init).")
parser.add_argument("--popsize", type=int, default=128, help="ES population incl. 2 reserved slots (even, >= 4).")
parser.add_argument("--n_repeats", type=int, default=2, help="Envs per genome (num_envs = popsize * n_repeats).")
parser.add_argument("--generations", type=int, default=300)
parser.add_argument("--steps_per_gen", type=int, default=2000, help="Control steps rolled per generation (= 1 episode).")
parser.add_argument("--sigma", type=float, default=0.05, help="ES perturbation std on the ABCD genome.")
parser.add_argument("--es_lr", type=float, default=0.001, help="SGD lr on theta (raw-diff gradient, no Adam).")
parser.add_argument("--u_scale", type=float, default=0.5, help="Fixed fitness-difference scale: u=clip(diff/u_scale, ±3).")
parser.add_argument("--theta_anchor", type=float, default=0.01, help="Elastic decay of theta toward 0 (pure v74) per generation.")
parser.add_argument("--revert_margin", type=float, default=0.5, help="improvement < -margin counts as a bad generation.")
parser.add_argument("--revert_patience", type=int, default=10, help="Consecutive bad generations before reverting theta to best.")
parser.add_argument("--hebbian_lr", type=float, default=0.05, help="Online head update rate.")
parser.add_argument("--update_interval", type=int, default=25, help="Control steps between Hebbian head updates.")
parser.add_argument("--anchor", type=float, default=0.01, help="Elastic decay of w toward W0 per update (weight anchoring).")
parser.add_argument("--delta_bound", type=float, default=0.5, help="Max |w - W0| per connection.")
parser.add_argument("--genome_bound", type=float, default=2.0, help="Clamp on ABCD coefficients.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_hours", type=float, default=18.0)
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--wandb_project", type=str, default="hebbian-transport-eshead")
parser.add_argument("--exp_name", type=str, default="es_head_transport")
parser.add_argument("--log_root", type=str, default="experiments")
AppLauncher.add_app_launcher_args(parser)
args_cli, env_override_args = parser.parse_known_args()
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

from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, EmpiricalNormalization

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import parse_env_cfg

_OBS_BOUND = 1000.0
_REW_BOUND = 200.0
ACTOR_OBS = 153      # (45 base + 6 INDI residual) x H3 — the v74 actor layout
CRITIC_OBS = 180     # v74 critic LSTM input (needed only to reconstruct the module for load_state_dict)
ACT_DIM = 5


def _coerce(text: str):
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def apply_env_overrides(env_cfg, tokens):
    """`env.<dotted.path>=<value>` CLI overrides, scripts/hebbian/train.py convention (strict)."""
    for tok in tokens:
        if not tok.startswith("env.") or "=" not in tok:
            raise SystemExit(f"[ERROR] unrecognized argument: {tok!r} (use env.<dotted.path>=<value>)")
        path, _, raw = tok.partition("=")
        parts = path.split(".")[1:]
        target = env_cfg
        for p in parts[:-1]:
            if not hasattr(target, p):
                raise SystemExit(f"[ERROR] no such env config group: {path!r} (missing {p!r})")
            target = getattr(target, p)
        if not hasattr(target, parts[-1]):
            raise SystemExit(f"[ERROR] no such env config field: {path!r}")
        value = _coerce(raw)
        setattr(target, parts[-1], value)
        print(f"[es-transport] env override: {path} = {value!r}")


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    assert args_cli.popsize % 2 == 0 and args_cli.popsize >= 4, "popsize must be even and >= 4"
    P, R = args_cli.popsize, args_cli.n_repeats
    n_pert = P - 2                       # last 2 slots: theta=0 baseline, theta center
    assert n_pert % 2 == 0
    num_envs = P * R
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    # condition-slot map: env e -> genome g=e//R, repeat r=e%R. Antithetic pair members (g=2i, 2i+1) share
    # slot (i*R + r); the baseline and center slots share slot ((n_pert//2)*R + r). Attachment draws are
    # identical within a slot -> attachment differences cancel out of every fitness comparison that matters.
    def slot_of_env(e: int) -> int:
        g, r = e // R, e % R
        pair = min(g, n_pert) // 2 if g < n_pert else n_pert // 2
        return pair * R + r

    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=num_envs)
    env_cfg.seed = args_cli.seed
    # v74 actor layout on the transport env: partial obs + 6-dim INDI residual per frame, H3 -> 153
    env_cfg.partial_obs = True
    env_cfg.history_len = 3
    env_cfg.actor_indi_residual = True
    env_cfg.observation_spaces = {a: ACTOR_OBS for a in env_cfg.possible_agents}
    env_cfg.debug_vis = False
    # attachment randomization: per-env anchors, condition-slot matched for the ES instrument
    env_cfg.randomize_attachment = True
    env_cfg.attachment_per_env = True
    env_cfg.scene.replicate_physics = False
    env_cfg.attachment_condition_slots = [slot_of_env(e) for e in range(num_envs)]
    # goal-pose CRN (env._crn_repeats set after construction)
    env_cfg.crn = True
    env_cfg.crn_seed = args_cli.seed
    # post-attachment start: deterministic horizontal spawn (basket level, fixed height, no initial velocity)
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (1.0, 1.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    # pin basket mass at nominal — the user asked for ATTACHMENT randomization; a random mass would add
    # un-CRN'd condition noise to the pair differences the instrument depends on
    env_cfg.events.randomize_mass.params["mass_distribution_params"] = (1.4, 1.4)
    if env_override_args:
        apply_env_overrides(env_cfg, env_override_args)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    assert isinstance(base, DirectMARLEnv)
    base._crn_repeats = R
    agents = list(env_cfg.possible_agents)
    n_ag = len(agents)
    E, dev = num_envs, base.device
    B = n_ag * E                          # policy batch: agent-major blocks [f1 envs | f2 envs | f3 envs]

    # ---- frozen backbone (v74 LSTM or v84 feedforward MLP; auto-detected) + head init -----------
    ckpt = torch.load(args_cli.checkpoint, map_location=dev, weights_only=False)
    is_recurrent = any(k.startswith("memory_a") for k in ckpt["model_state_dict"])
    if is_recurrent:
        policy = ActorCriticRecurrent(
            num_actor_obs=ACTOR_OBS, num_critic_obs=CRITIC_OBS, num_actions=ACT_DIM,
            actor_hidden_dims=[256, 256], critic_hidden_dims=[256, 256], activation="elu",
            rnn_type="lstm", rnn_hidden_dim=256, rnn_num_layers=1, init_noise_std=1.0,
        ).to(dev)
    else:
        policy = ActorCritic(
            num_actor_obs=ACTOR_OBS, num_critic_obs=CRITIC_OBS, num_actions=ACT_DIM,
            actor_hidden_dims=[256, 256], critic_hidden_dims=[256, 256], activation="elu",
            init_noise_std=1.0,
        ).to(dev)
    print(f"[es-transport] backbone: {'LSTM (recurrent)' if is_recurrent else 'feedforward MLP'}")
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    normalizer = EmpiricalNormalization(shape=[ACTOR_OBS]).to(dev)
    normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
    normalizer.eval()
    head = policy.actor[-1]
    assert isinstance(head, torch.nn.Linear) and head.out_features == ACT_DIM
    trunk = policy.actor[:-1]
    W0 = head.weight.detach().clone()     # (5, 256)
    b0 = head.bias.detach().clone()
    A_DIM = W0.numel()
    GENOME = 4 * A_DIM                    # ABCD -> 5120
    print(f"[es-transport] frozen v74 loaded (iter {ckpt.get('iter')}); head {tuple(W0.shape)}, "
          f"genome {GENOME}, envs {E} x {n_ag} drones = batch {B}")

    # ---- log dir / wandb -------------------------------------------------------------------------
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(args_cli.log_root, args_cli.exp_name,
                           f"eshead_p{P}_s{args_cli.sigma}_hlr{args_cli.hebbian_lr}"
                           f"_ui{args_cli.update_interval}_seed{args_cli.seed}_{ts}")
    os.makedirs(os.path.join(log_dir, "gens"), exist_ok=True)
    with open(os.path.join(log_dir, "settings.json"), "w") as f:
        json.dump(vars(args_cli), f, indent=2, default=str)
    print(f"[es-transport] logging to {log_dir}")
    wb = None
    if args_cli.wandb:
        try:
            import wandb
            wb = wandb.init(project=args_cli.wandb_project, name=os.path.basename(log_dir),
                            config=vars(args_cli), dir=log_dir)
        except Exception as e:  # noqa: BLE001
            print(f"[es-transport] wandb disabled: {e}")

    # ---- ES state --------------------------------------------------------------------------------
    theta = torch.zeros(GENOME, device=dev)          # ABCD = 0 -> the pure v74 policy on every drone
    best_theta = torch.zeros_like(theta)
    best_impr = 0.0
    best_f = -1e9
    bad_streak = 0
    n_reverts = 0
    gen_rng = torch.Generator(device=dev); gen_rng.manual_seed(args_cli.seed)

    def head_batch(genomes: torch.Tensor):
        """genomes (P, GENOME) -> per-batch ABCD (B, 5, 256); env e -> genome e//R, tiled over the 3 drones."""
        g = genomes.clamp_(-args_cli.genome_bound, args_cli.genome_bound)
        per_env = g.repeat_interleave(R, dim=0)                     # (E, GENOME)
        A, B_, C, D = per_env.split(A_DIM, dim=1)
        shape = (E,) + tuple(W0.shape)
        return tuple(x.view(shape).repeat(n_ag, 1, 1) for x in (A, B_, C, D))   # each (B, 5, 256)

    def sanitize(x: torch.Tensor, bound: float) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=bound, neginf=-bound).clamp_(-bound, bound)

    t0 = time.time()
    for gen in range(args_cli.generations):
        if (time.time() - t0) / 3600.0 > args_cli.max_hours:
            print(f"[es-transport] max_hours reached at generation {gen}")
            break
        base._spawn_generation = gen
        theta_eval = theta.clone()
        eps = torch.randn(n_pert // 2, GENOME, device=dev, generator=gen_rng)
        genomes = torch.empty(P, GENOME, device=dev)
        genomes[0:n_pert:2] = theta + args_cli.sigma * eps
        genomes[1:n_pert:2] = theta - args_cli.sigma * eps
        genomes[n_pert] = 0.0                    # pure v74 baseline
        genomes[n_pert + 1] = theta              # unperturbed center
        A, B_, C, D = head_batch(genomes)

        with torch.inference_mode():
            w = W0.unsqueeze(0).repeat(B, 1, 1)                     # per-drone plastic head state
            pre_acc = torch.zeros(B, W0.shape[1], device=dev)
            post_acc = torch.zeros(B, W0.shape[0], device=dev)
            acc_n = 0
            rew_sum = torch.zeros(E, device=dev)
            done_any = torch.zeros(E, device=dev, dtype=torch.bool)

            obs_dict, _ = base.reset()
            policy.reset()
            for t in range(args_cli.steps_per_gen):
                obs3 = torch.cat([sanitize(obs_dict[a], _OBS_BOUND) for a in agents], dim=0)   # (B, 153)
                feat = policy.memory_a(normalizer(obs3)).squeeze(0) if is_recurrent else normalizer(obs3)
                phi = trunk(feat)                                                              # (B, 256)
                act3 = torch.bmm(w, phi.unsqueeze(-1)).squeeze(-1) + b0                        # (B, 5)
                actions = {a: act3[k * E:(k + 1) * E] for k, a in enumerate(agents)}
                obs_dict, rewards, terminated, truncated, _ = base.step(actions)
                dones_env = (terminated[agents[0]] | truncated[agents[0]]).bool()              # (E,)
                dones3 = dones_env.repeat(n_ag)
                policy.reset(dones3)
                rew_sum += sanitize(rewards[agents[0]], _REW_BOUND)
                done_any |= terminated[agents[0]].bool()     # early termination = crash (timeout is not)
                pre_acc += phi; post_acc += torch.tanh(act3); acc_n += 1
                if (t + 1) % args_cli.update_interval == 0:
                    pre = pre_acc / acc_n
                    post = post_acc / acc_n
                    outer = post.unsqueeze(-1) * pre.unsqueeze(1)
                    dw = A * outer + B_ * pre.unsqueeze(1) + C * post.unsqueeze(-1) + D
                    w = w + args_cli.hebbian_lr * dw - args_cli.anchor * (w - W0)
                    w = W0 + (w - W0).clamp_(-args_cli.delta_bound, args_cli.delta_bound)
                    pre_acc.zero_(); post_acc.zero_(); acc_n = 0
                if dones3.any():                                     # plasticity is intra-episode
                    w[dones3] = W0

            fit_env = (rew_sum / args_cli.steps_per_gen).clone()
            crash_frac = float(done_any.float().mean())              # episode = steps_per_gen; any done = failure
            pos_err = float(base.metrics["position_error"].mean())
            ori_err = float(base.metrics["orientation_error"].mean())

        fit = fit_env.clone().view(P, R).mean(dim=1)
        fit_p = fit[:n_pert]
        fit_base = float(fit[n_pert])
        fit_center = float(fit[n_pert + 1])
        improvement = fit_center - fit_base

        # v82 gradient: raw baseline-subtracted antithetic differences, fixed scale, plain SGD
        diff = 0.5 * (fit_p[0::2] - fit_p[1::2])
        u = (diff / args_cli.u_scale).clamp(-3.0, 3.0)
        grad = (u.unsqueeze(1) * eps).mean(dim=0) / args_cli.sigma
        theta = theta + args_cli.es_lr * grad
        theta = ((1.0 - args_cli.theta_anchor) * theta).clamp_(-args_cli.genome_bound, args_cli.genome_bound)

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
            print(f"[es-transport] REVERT #{n_reverts}: improvement < -{args_cli.revert_margin} for "
                  f"{args_cli.revert_patience} gens -> theta reset to best (impr={best_impr:.3f})", flush=True)

        gbest = float(fit_p.max()); gmean = float(fit_p.mean())
        if gbest > best_f:
            best_f = gbest
        np.save(os.path.join(log_dir, "gens", f"theta_{gen}.npy"), theta.cpu().numpy())
        print(f"Generation {gen}:\tbest={gbest:.4f}\tmean={gmean:.4f}\tbase={fit_base:.4f}"
              f"\tcenter={fit_center:.4f}\timpr={improvement:.4f}\tcrash={crash_frac:.3f}"
              f"\tpos_err={pos_err:.3f}", flush=True)
        if wb:
            wb.log({"gen": gen, "fit_best": gbest, "fit_mean": gmean, "fit_best_so_far": best_f,
                    "fit_base": fit_base, "fit_center": fit_center, "improvement": improvement,
                    "best_improvement": best_impr, "diff_abs_mean": float(diff.abs().mean()),
                    "n_reverts": n_reverts, "crash_frac": crash_frac,
                    "pos_error": pos_err, "ori_error": ori_err,
                    "theta_norm": float(theta.norm())}, step=gen)

    np.save(os.path.join(log_dir, "theta_final.npy"), theta.cpu().numpy())
    print(f"[es-transport] done; theta_final.npy in {log_dir}")
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
