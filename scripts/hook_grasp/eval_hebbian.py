# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic GOAL evaluator for a Hebbian/ES hook-grasp genome.

The Hebbian twin of ``eval_mappo.py``: identical env setup, curriculum-disable list, hold-measurement
trick and reported metrics, so a Hebbian/ES result is directly comparable to a MAPPO one (F3 max-stack
= catch 0.11 / hold|caught 0.41). The only difference is what produces the actions.

WHICH GENOME: under a curriculum the EA's ``x_best_so_far`` (what ``checkpoints/best.pt`` stores) is
pinned to whatever scored highest on the EASIEST early task -- for es_f3 that is generation 0, where
half the envs spawned already attached and banked the success bonus for free. That genome is NOT the
best controller. So this evaluates a PER-GENERATION snapshot instead:
    <run>/openai_es/<gen>/theta.npy   -- the ES centre (the policy of record; default)
    <run>/openai_es/<gen>/x_best.npy  -- best sampled perturbation at that generation (--x_best)
``best.pt`` is still needed for the cfg / obs layout / obs-normalizer state.

NOTE: the obs normalizer is only stored in best.pt (latest), not per generation, so an older genome is
evaluated against slightly newer running stats. zscore stats are broadly stable, but it is a known
approximation -- reported as `rms_source` in the JSON.

Success (all curricula disabled, true task difficulty):
    catch_rate >= 0.30  AND  hold_success_given_caught >= 0.80   at hold_steps = 100.

Example (ALC):
    ./.venv/bin/python scripts/hook_grasp/eval_hebbian.py --headless --device cuda:0 --num_envs 256 \
        --run experiments/es_f3/<run_dir> --generation 60 --steps 4000 --out experiments/eval_es_f3_g60.json
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deterministic goal evaluator for a Hebbian/ES hook-grasp genome.")
parser.add_argument("--num_envs", type=int, default=128, help="Parallel envs (local 8GB: keep modest, e.g. 128).")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--run", type=str, required=True, help="Hebbian run dir (holds checkpoints/best.pt + openai_es/<gen>/).")
parser.add_argument("--generation", type=int, default=-1, help="Generation snapshot to evaluate. -1 = latest available.")
parser.add_argument("--x_best", action="store_true", default=False,
                    help="Evaluate x_best.npy (best sampled perturbation) instead of theta.npy (ES centre).")
parser.add_argument("--steps", type=int, default=4000, help="Total sim steps to roll (more -> more episodes).")
parser.add_argument("--hold_steps", type=int, default=100, help="Consecutive attached steps that count as a hold (final=100).")
parser.add_argument("--capture_radius", type=float, default=0.13, help="Catch radius (m); FINAL task value = 0.13.")
parser.add_argument("--history_len", type=int, default=1, help="Obs history; MUST match training (es_f3 = 1).")
parser.add_argument("--seed", type=int, default=12345, help="Seed.")
parser.add_argument("--out", type=str, default="", help="Optional path to write a JSON result.")
parser.add_argument("--env_kv", type=str, default="",
                    help="env_cfg overrides applied at eval (e.g. 'attachment.min_chord_angle_deg=25,"
                         "attachment.chord_angle_perp_only=true,basket_wall_z_hi=0.22') — match the TRUE task.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os
import tempfile

import gymnasium as gym
import numpy as np
import torch

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import parse_env_cfg

from MARL_mav_carry_ext.algorithms.hebbian.adapter import HebbianPolicy

# body-frame INDI disturbance-residual width appended per actor frame when cfg.actor_indi_residual
# is on (source of truth: scripts/hook_grasp/rslrl_adapter.INDI_RESIDUAL_DIM).
INDI_RESIDUAL_DIM = 6


def _resolve_genome(run_dir: str, generation: int, use_x_best: bool):
    """Locate the requested per-generation snapshot; return (genome, gen, path)."""
    ea_root = None
    for cand in ("openai_es", "cmaes", "revde", "pgpe"):
        p = os.path.join(run_dir, cand)
        if os.path.isdir(p):
            ea_root = p
            break
    if ea_root is None:
        raise SystemExit(f"[ERROR] no EA snapshot dir (openai_es/ etc.) under {run_dir!r}")
    gens = sorted(int(d) for d in os.listdir(ea_root) if d.isdigit())
    if not gens:
        raise SystemExit(f"[ERROR] no per-generation snapshots under {ea_root!r}")
    gen = gens[-1] if generation < 0 else generation
    if gen not in gens:
        raise SystemExit(f"[ERROR] generation {gen} not saved (have {gens[0]}..{gens[-1]})")
    path = os.path.join(ea_root, str(gen), "x_best.npy" if use_x_best else "theta.npy")
    if not os.path.isfile(path):
        raise SystemExit(f"[ERROR] missing {path!r}")
    return np.load(path), gen, path


def main():
    run_dir = args_cli.run
    best_pt = os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.isfile(best_pt):
        raise SystemExit(f"[ERROR] no checkpoints/best.pt under {run_dir!r} (needed for cfg + obs normalizer)")
    genome, gen, gpath = _resolve_genome(run_dir, args_cli.generation, args_cli.x_best)
    print(f"[eval-hebbian] genome: {gpath}  (generation {gen}, dim {genome.shape})")

    # load the checkpoint BEFORE the env: its obs_dim tells us which ACTOR layout the genome was trained with.
    payload = torch.load(best_pt, map_location=args_cli.device, weights_only=False)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    # ---- honest eval: DISABLE every training curriculum / aid (identical list to eval_mappo.py) ----
    # attached_spawn_curriculum is CRITICAL: left on, half the envs spawn already-attached and
    # hold_given_caught comes out > 1. spawn_curriculum=False also drives _spawn_frac()->1.0, which
    # zeroes the swing assist (assist = v0*(1-frac)) -> no spawn-velocity aid, and gives the full
    # honest spawn spread.
    for _attr in ("capture_curriculum", "hold_curriculum", "spawn_curriculum", "attached_spawn_curriculum",
                  "chord_angle_curriculum"):
        if hasattr(env_cfg, _attr):
            setattr(env_cfg, _attr, False)
    # pin the FINAL catch geometry (0.13) explicitly so no annealed value leaks in
    if hasattr(env_cfg, "attachment") and args_cli.capture_radius > 0:
        env_cfg.attachment.capture_radius = args_cli.capture_radius
    # task-gate overrides (same parser as eval_behavior; pass the TRUE task gates, e.g.
    # 'attachment.min_chord_angle_deg=25,attachment.chord_angle_perp_only=true,basket_wall_z_hi=0.22')
    for tok in getattr(args_cli, "env_kv", "").split(","):
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
        print(f"[eval-hebbian] env override: {k.strip()} = {val}")
    # DISABLE the env's own hold-success termination so it never resets on the hold event; we measure
    # the hold ourselves (consecutive attached steps) and count a success at --hold_steps.
    if hasattr(env_cfg, "hold_success_steps"):
        env_cfg.hold_success_steps = 10_000_000

    # ---- ACTOR layout must match what the genome was trained with -------------------------------------
    # INFERRED from the checkpoint, not from a flag: the actor frame is 45 (hook-free base) or 51 when the
    # 6-dim INDI disturbance residual is appended (cfg.actor_indi_residual). Guessing wrong would build the
    # policy net to the wrong input width, so derive it and fail loudly on anything unexpected.
    indi = False
    if hasattr(env_cfg, "obs_dim_per_frame"):
        env_cfg.history_len = args_cli.history_len
        base_dim = int(env_cfg.obs_dim_per_frame)
        want = int(payload["obs_dim"]) // max(1, args_cli.history_len)
        if want == base_dim + INDI_RESIDUAL_DIM:
            indi = True
        elif want != base_dim:
            raise SystemExit(
                f"[ERROR] cannot match trained obs {payload['obs_dim']} with --history_len "
                f"{args_cli.history_len}: implies per_frame={want}, expected {base_dim} or "
                f"{base_dim + INDI_RESIDUAL_DIM}."
            )
        env_cfg.actor_indi_residual = indi
        env_cfg.actor_history_len = args_cli.history_len
        env_cfg.observation_spaces = {a: want * args_cli.history_len for a in env_cfg.possible_agents}
        print(f"[eval-hebbian] actor layout: per_frame={want}"
              f"{f' (={base_dim} base + {INDI_RESIDUAL_DIM} INDI residual)' if indi else ''}, "
              f"history={args_cli.history_len} -> obs_dim={want * args_cli.history_len}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    base = env.unwrapped
    assert isinstance(base, DirectMARLEnv), "eval_hebbian expects a DirectMARLEnv (multi-agent) task."

    # ---- rebuild the policy: best.pt supplies cfg/obs_dim/normalizer, the snapshot supplies the genome ----
    if payload["best_genome"] is not None and genome.shape != np.asarray(payload["best_genome"]).shape:
        raise SystemExit(f"[ERROR] genome dim {genome.shape} != best.pt genome dim "
                         f"{np.asarray(payload['best_genome']).shape}")
    payload["best_genome"] = genome
    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(payload, tmp.name)
    tmp.close()
    policy = HebbianPolicy(base, tmp.name)
    os.unlink(tmp.name)

    agents = list(base.possible_agents)
    E = base.num_envs
    dev = base.device
    hold_req = args_cli.hold_steps

    obs, _ = base.reset()
    policy.reset()
    actor_dim = int(obs[agents[0]].shape[-1])
    print(f"[eval-hebbian] agents={agents}  actor_obs_dim={actor_dim}  (hook-free)")
    assert actor_dim == payload["obs_dim"], \
        f"env obs {actor_dim} != trained obs {payload['obs_dim']} (check --history_len)"

    # per-env, current-episode trackers
    ever_caught = torch.zeros(E, dtype=torch.bool, device=dev)
    ever_held = torch.zeros(E, dtype=torch.bool, device=dev)
    consec_attach = torch.zeros(E, dtype=torch.long, device=dev)
    ep_hold_max = torch.zeros(E, dtype=torch.long, device=dev)

    # cumulative counters
    n_ep = torch.zeros((), dtype=torch.long, device=dev)
    n_caught = torch.zeros((), dtype=torch.long, device=dev)
    n_held = torch.zeros((), dtype=torch.long, device=dev)
    attached_steps = torch.zeros((), dtype=torch.long, device=dev)
    held_hold_sum = torch.zeros((), dtype=torch.long, device=dev)
    total_steps = 0
    jerk_sum = torch.zeros((), dtype=torch.float, device=dev)
    jerk_n = torch.zeros((), dtype=torch.long, device=dev)

    for _ in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy.act(obs)                       # Hebbian actor is deterministic (no sampling)
            obs, _, terminated, time_outs, _ = base.step(actions)

        attached = base.attached.any(dim=-1)
        newly = base._newly_caught.any(dim=-1)
        ever_caught |= newly
        consec_attach = torch.where(attached, consec_attach + 1, torch.zeros_like(consec_attach))
        ep_hold_max = torch.maximum(ep_hold_max, consec_attach)
        ever_held |= consec_attach >= hold_req
        attached_steps += attached.sum()
        total_steps += E

        if hasattr(base, "_drone_jerk"):
            jm = torch.norm(base._drone_jerk, dim=-1).mean(dim=-1)
            if attached.any():
                jerk_sum += jm[attached].sum()
                jerk_n += attached.sum()

        def _reduce(d):
            if isinstance(d, dict):
                return torch.stack([d[a].view(E) for a in agents], dim=0).any(dim=0).bool()
            return d.view(E).bool()

        done = _reduce(terminated) | _reduce(time_outs)
        if done.any():
            dc = ever_caught & done
            n_ep += done.sum()
            n_caught += dc.sum()
            n_held += (ever_held & done).sum()
            held_hold_sum += ep_hold_max[dc].sum()
            keep = ~done
            ever_caught &= keep
            ever_held &= keep
            consec_attach = torch.where(done, torch.zeros_like(consec_attach), consec_attach)
            ep_hold_max = torch.where(done, torch.zeros_like(ep_hold_max), ep_hold_max)
            # plastic weights are per-episode: reset the finished envs' networks (mirrors the trainer)
            policy.reset(torch.nonzero(done, as_tuple=False).view(-1).tolist())

    ep = int(n_ep.item())
    caught = int(n_caught.item())
    held = int(n_held.item())
    catch_rate = caught / ep if ep else 0.0
    hold_given_caught = held / caught if caught else 0.0
    mean_jerk = float(jerk_sum.item()) / max(1, int(jerk_n.item())) if int(jerk_n.item()) else float("nan")

    result = {
        "run": run_dir,
        "algorithm": "Hebbian-openai_es",
        "genome": gpath,
        "generation": gen,
        "genome_kind": "x_best" if args_cli.x_best else "theta(ES centre)",
        "rms_source": "checkpoints/best.pt (latest generation; genome may be older)",
        "num_envs": E,
        "sim_steps": args_cli.steps,
        "actor_obs_dim": actor_dim,
        "history_len": args_cli.history_len,
        "hold_steps": hold_req,
        "capture_radius": float(base._acfg.capture_radius),
        "episodes": ep,
        "caught_episodes": caught,
        "held_episodes": held,
        "catch_rate": round(catch_rate, 4),
        "hold_success_given_caught": round(hold_given_caught, 4),
        "mean_hold_steps_given_caught": round(float(held_hold_sum.item()) / caught if caught else 0.0, 2),
        "attached_frac": round(float(attached_steps.item()) / max(1, total_steps), 4),
        "mean_jerk_attached_N": round(mean_jerk, 3) if mean_jerk == mean_jerk else None,
        "GOAL_catch_0.30": catch_rate >= 0.30,
        "GOAL_hold_0.80": hold_given_caught >= 0.80,
        "GOALS_MET": bool(catch_rate >= 0.30 and hold_given_caught >= 0.80),
    }
    print("JSON " + json.dumps(result))
    print("\n================= HEBBIAN/ES GOAL EVAL =================")
    for k, v in result.items():
        print(f"  {k:30s}: {v}")
    print("  reference (MAPPO v25F3_maxstack): catch 0.110 / hold|caught 0.41")
    print("  gate: catch>=0.30 AND hold_given_caught>=0.80 -> "
          f"{'PASS' if result['GOALS_MET'] else 'FAIL'}")
    print("========================================================")
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        print(f"[eval-hebbian] wrote {args_cli.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
