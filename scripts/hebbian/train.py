# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a Hebbian / CMA-ES policy on a flycrane DirectMARL task.

The plasticity rule (A/B/C/D coefficients) is evolved with CMA-ES; the network weights adapt
online via the Hebbian rule.  Each parallel IsaacLab environment evaluates one CMA-ES candidate.

Example:
    python scripts/hebbian/train.py --task=Isaac-flycrane-payload-decentralized-hovering-v0 \
        --headless --num_envs=90 --popsize=30 --n_repeats=3 --num_generations=100 --seed=42

Any trailing `env.<dotted.path>=<value>` token overrides that field on the task cfg, matching the
`scripts/skrl/train.py` (Hydra) convention -- so an env/reward/curriculum recipe tuned for one
algorithm can be replayed verbatim here, e.g.
    ... env.attachment.capture_radius=0.13 env.wall_pull_weight=0.0 env.attached_spawn_v0=0.4
"""

import argparse
import sys

from isaaclab.app import AppLauncher


def str2bool(v):
    """Parse a boolean from the CLI. Lets store_true-style flags also be swept as 0/1 values
    (so a W&B sweep can pass `--keep_d 0` / `--keep_d 1` while `--keep_d` alone stays True)."""
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def _coerce(text: str):
    """Parse an override value the way Hydra would: bool > None > int > float > str."""
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
    """Apply `env.<dotted.path>=<value>` CLI tokens onto the task cfg.

    Strict on purpose: an unknown field or a stray token raises instead of being dropped, because a
    silently-ignored override would invalidate the experiment it was meant to define.
    """
    for tok in tokens:
        if not tok.startswith("env.") or "=" not in tok:
            raise SystemExit(
                f"[ERROR] unrecognized argument: {tok!r} "
                "(env overrides must look like env.<dotted.path>=<value>)"
            )
        path, _, raw = tok.partition("=")
        parts = path.split(".")[1:]  # drop the leading 'env'
        target = env_cfg
        for p in parts[:-1]:
            if not hasattr(target, p):
                raise SystemExit(f"[ERROR] no such env config group: {path!r} (missing {p!r})")
            target = getattr(target, p)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise SystemExit(f"[ERROR] no such env config field: {path!r}")
        value = _coerce(raw)
        setattr(target, leaf, value)
        print(f"[INFO] env override: {path} = {value!r}")


parser = argparse.ArgumentParser(description="Train a Hebbian/CMA-ES agent on a flycrane DirectMARL task.")
parser.add_argument("--num_envs", type=int, default=-1,
                    help="Number of environments. -1 = auto (popsize*n_repeats, x3 for revde).")
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-v0", help="Task name.")
parser.add_argument("--seed", type=int, default=42, help="Seed (use -1 for random).")
parser.add_argument("--num_generations", type=int, default=100, help="Number of CMA-ES generations.")
parser.add_argument("--max_env_steps", type=int, default=None, help="Optional hard cap on total env steps.")
# observability / memory
parser.add_argument("--history_len", type=int, default=None,
                    help="Observation frames stacked as memory (1 = no memory). None = use the task "
                         "cfg default (hover envs fall back to 1).")
parser.add_argument("--full_obs", action="store_true", default=False,
                    help="Use full 87-dim observation instead of the 45-dim partial (local) observation.")
# CMA-ES / Hebbian hyper-parameters
parser.add_argument("--popsize", type=int, default=30, help="CMA-ES population size.")
parser.add_argument("--n_repeats", type=int, default=3, help="Episodes evaluated per genome (fitness = mean).")
parser.add_argument("--latent_size", type=int, default=4, help="Single hidden layer size of the Hebbian net.")
parser.add_argument("--sigma", type=float, default=0.3, help="CMA-ES initial step size.")
parser.add_argument("--bound", type=float, default=1.0, help="ABCD coefficient bound [-bound, bound].")
parser.add_argument("--action_scale", type=float, default=2.0, help="Tanh output -> setpoint range scale.")
parser.add_argument("--hebbian_lr", type=float, default=0.1, help="Online Hebbian update rate.")
parser.add_argument("--weight_init_scale", type=float, default=0.01,
                    help="Std of the plastic weight init (randn*scale). 0 = blank-slate weights.")
parser.add_argument("--evolve_init_weights", action="store_true", default=False,
                    help="Add an evolved initial-weights genome block (set at episode reset).")
parser.add_argument("--keep_d", type=str2bool, nargs="?", const=True, default=False,
                    help="Keep the D term as well (with --evolve_init_weights -> ABCD+W, 5 blocks). "
                         "Bare flag = True; sweepable as --keep_d 0 / --keep_d 1.")
parser.add_argument("--reward_curriculum", type=str2bool, nargs="?", const=True, default=False,
                    help="Enable task/reward scheduling: widen the goal range to the harder ranges "
                         "after range_curriculum_steps. Bare flag = True; sweepable as 0 / 1.")
parser.add_argument("--weight_anchor", type=float, default=0.0,
                    help="Weight anchoring / relative weight decay toward the init weights: each "
                         "Hebbian update gets an elastic term -weight_anchor*(w - w_init) that bounds "
                         "weight growth. 0 = off. Intended with --evolve_init_weights (anchor = init_W).")
parser.add_argument("--hebbian_update_interval", type=int, default=1,
                    help="Plasticity timescale: average activations over N inference steps per weight update (1=per-step).")
parser.add_argument("--max_hours", type=float, default=None,
                    help="Optional wall-clock cap (hours): stop after the current generation past this.")
parser.add_argument("--diagonal", action="store_true", default=False,
                    help="Use diagonal CMA-ES (faster for large genomes). Upstream uses full aCMA-ES (default).")
parser.add_argument("--optimizer", type=str, default="cmaes",
                    choices=["cmaes", "revde", "snes", "openai_es", "pgpe"],
                    help="Optimizer: cmaes | revde (thirdparty/revde) | snes | openai_es | pgpe (EvoTorch+Adam).")
parser.add_argument("--de_type", type=str, default="revde", choices=["de", "ade", "revde", "dex3"],
                    help="RevDE recombination variant (only used when --optimizer=revde).")
parser.add_argument("--de_F", type=float, default=0.5, help="DE differential weight in [0,2].")
parser.add_argument("--de_CR", type=float, default=0.9, help="DE crossover rate in (0,1].")
parser.add_argument("--es_sigma", type=float, default=0.1, help="OpenAI-ES perturbation std.")
parser.add_argument("--es_lr", type=float, default=0.02, help="OpenAI-ES Adam learning rate.")
parser.add_argument("--pgpe_center_lr", type=float, default=0.15, help="PGPE (EvoTorch) Adam center lr.")
parser.add_argument("--pgpe_stdev_lr", type=float, default=0.1, help="PGPE std learning rate.")
parser.add_argument("--pgpe_stdev_max_change", type=float, default=0.2, help="PGPE relative max std change/step.")
parser.add_argument("--fitness", type=str, default="return", choices=["return", "mean_reward"],
                    help="Per-episode fitness: 'return' (sum of rewards) or 'mean_reward' (return/episode length). "
                         "Use mean_reward when success terminates the episode early.")
parser.add_argument("--pgpe_ranking_method", type=str, default="centered",
                    choices=["centered", "normalized", "raw"], help="PGPE fitness shaping.")
# Weights & Biases logging
parser.add_argument("--wandb", action="store_true", default=False, help="Log to Weights & Biases.")
parser.add_argument("--wandb_project", type=str, default="hebbian-flycrane", help="W&B project.")
parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity/team (optional).")
parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"],
                    help="W&B mode.")
parser.add_argument("--obs_norm", type=str, default="zscore", choices=["zscore", "minmax", "none"],
                    help="Observation normalization: zscore (robust, default), minmax ([-1,1]), or none.")
# domain-randomization env (hover_rand) + run-organization
parser.add_argument("--attachment_per_env", type=int, default=-1,
                    help="hover_rand only: 1=per-env attachment DR (replicate_physics=False), "
                         "0=per-run shared, -1=leave the env cfg default.")
parser.add_argument("--log_root", type=str, default="logs/hebbian",
                    help="Base directory for run logs (e.g. 'experiments' for a named experiment).")
parser.add_argument("--exp_name", type=str, default="",
                    help="If set, use a flat named folder convention: "
                         "<log_root>/<exp_name>/<config-slug>_<timestamp>/ instead of the default "
                         "<log_root>/<task>/<optimizer>/<timestamp>_seed<seed>/.")

AppLauncher.add_app_launcher_args(parser)
args_cli, env_override_args = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import json
import os
import random
from datetime import datetime

import gymnasium as gym

import MARL_mav_carry_ext.tasks  # noqa: F401  (registers gym envs)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from MARL_mav_carry_ext.algorithms.hebbian.adapter import HebbianConfig, HebbianTrainer


def main():
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # auto-derive num_envs from popsize when not explicitly set (handy for W&B sweeps)
    if args_cli.num_envs is None or args_cli.num_envs <= 0:
        mult = 3 if args_cli.optimizer == "revde" else 1
        args_cli.num_envs = mult * args_cli.popsize * args_cli.n_repeats
        print(f"[INFO] auto num_envs = {args_cli.num_envs}")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    # `env.<path>=<value>` CLI overrides (skrl/Hydra convention); applied before the obs-space
    # recompute below so an overridden history_len stays consistent with observation_spaces.
    if env_override_args:
        apply_env_overrides(env_cfg, env_override_args)

    # observability / memory override (observation_spaces is computed at class-definition time,
    # so recompute it to match the requested partial_obs/history_len).
    # Some tasks (e.g. hook_grasp) declare a fixed per-frame obs layout via `obs_dim_per_frame` and a
    # baked history_len; honor the cfg for those and skip the hover-style partial/full 45/87 surgery.
    if hasattr(env_cfg, "obs_dim_per_frame"):
        if args_cli.full_obs:
            print("[WARN] --full_obs ignored: this task defines a fixed obs layout (obs_dim_per_frame).")
        # honor an explicit --history_len (e.g. 1 = no memory -> obs_dim = obs_dim_per_frame); None keeps cfg
        if args_cli.history_len is not None:
            env_cfg.history_len = args_cli.history_len
        # Mirror the env's ACTOR layout exactly (marl_hook_env: _drone_obs_frame + _actor_hist), because the
        # actor frame is NOT simply obs_dim_per_frame * history_len:
        #   * actor_indi_residual appends the 6-dim body-frame INDI disturbance residual per frame, and
        #   * actor_history_len (0 -> history_len) sets the ACTOR-only stack depth, decoupled from the critic.
        # Getting this wrong builds the policy net to the wrong input size. Source of truth for the residual
        # width: scripts/hook_grasp/rslrl_adapter.INDI_RESIDUAL_DIM.
        _indi = bool(getattr(env_cfg, "actor_indi_residual", False))
        per_frame = env_cfg.obs_dim_per_frame + (6 if _indi else 0)
        actor_hist = int(getattr(env_cfg, "actor_history_len", 0)) or int(env_cfg.history_len)
        obs_dim = per_frame * actor_hist
        env_cfg.observation_spaces = {a: obs_dim for a in env_cfg.possible_agents}
        print(f"[INFO] task-defined obs: per_frame={per_frame}"
              f"{f' (={env_cfg.obs_dim_per_frame} base + 6 INDI residual)' if _indi else ''}, "
              f"actor_history_len={actor_hist}, obs_dim={obs_dim}")
    else:
        env_cfg.partial_obs = not args_cli.full_obs
        env_cfg.history_len = args_cli.history_len if args_cli.history_len is not None else 1
        if env_cfg.partial_obs:
            obs_dim = 45 * env_cfg.history_len  # 45-dim local per-frame obs (see env._get_observations)
        else:
            obs_dim = 87
        env_cfg.observation_spaces = {a: obs_dim for a in env_cfg.possible_agents}
        print(f"[INFO] partial_obs={env_cfg.partial_obs}, history_len={env_cfg.history_len}, obs_dim={obs_dim}")

    # reward/task scheduling toggle (widens goal_range mid-training; hover envs only)
    if hasattr(env_cfg, "range_curriculum"):
        env_cfg.range_curriculum = args_cli.reward_curriculum
        print(f"[INFO] reward_curriculum={env_cfg.range_curriculum}")
    elif args_cli.reward_curriculum:
        print("[WARN] --reward_curriculum ignored: task cfg has no range_curriculum.")

    # perturbation env (hover_rand): attachment-randomization scope. per-env needs replicate_physics
    # off (each env parses its own anchors); set it here since __post_init__ already ran at parse.
    if args_cli.attachment_per_env >= 0 and hasattr(env_cfg, "attachment_per_env"):
        env_cfg.attachment_per_env = bool(args_cli.attachment_per_env)
        if args_cli.attachment_per_env == 1:
            env_cfg.scene.replicate_physics = False
        print(f"[INFO] attachment_per_env={env_cfg.attachment_per_env} "
              f"(replicate_physics={env_cfg.scene.replicate_physics})")

    # EA subfolder so runs are grouped by optimizer (DE variant name for revde, else optimizer)
    ea_label = args_cli.de_type if args_cli.optimizer == "revde" else args_cli.optimizer
    _ew = ("_ewd" if args_cli.keep_d else "_ew") if args_cli.evolve_init_weights else ""
    _ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args_cli.exp_name:
        # flat, descriptive convention for a named experiment (distinct from the default sweeps)
        _scope = ""
        if getattr(env_cfg, "attachment_per_env", None) is not None:
            _scope = "_perenv" if env_cfg.attachment_per_env else "_perrun"
        _wa = f"_wa{args_cli.weight_anchor}" if args_cli.weight_anchor > 0 else ""
        run_slug = (f"{ea_label}_p{args_cli.popsize}{_ew}_b{args_cli.bound}_hlr{args_cli.hebbian_lr}"
                    f"_ui{args_cli.hebbian_update_interval}{_wa}{_scope}_seed{args_cli.seed}")
        log_dir = os.path.join(args_cli.log_root, args_cli.exp_name, f"{run_slug}_{_ts}")
    else:
        log_dir = os.path.join(
            args_cli.log_root, os.path.basename(args_cli.task), ea_label, f"{_ts}_seed{args_cli.seed}"
        )
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging Hebbian experiment to: {os.path.abspath(log_dir)}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    cfg = HebbianConfig(
        latent_size=args_cli.latent_size,
        action_scale=args_cli.action_scale,
        hebbian_lr=args_cli.hebbian_lr,
        weight_init_scale=args_cli.weight_init_scale,
        evolve_init_weights=args_cli.evolve_init_weights,
        keep_d=args_cli.keep_d,
        update_interval=args_cli.hebbian_update_interval,
        weight_anchor=args_cli.weight_anchor,
        obs_norm=args_cli.obs_norm,
        popsize=args_cli.popsize,
        n_repeats=args_cli.n_repeats,
        sigma=args_cli.sigma,
        bound=args_cli.bound,
        diagonal=args_cli.diagonal,
        optimizer=args_cli.optimizer,
        de_type=args_cli.de_type,
        de_F=args_cli.de_F,
        de_CR=args_cli.de_CR,
        es_sigma=args_cli.es_sigma,
        es_lr=args_cli.es_lr,
        pgpe_center_lr=args_cli.pgpe_center_lr,
        pgpe_stdev_lr=args_cli.pgpe_stdev_lr,
        pgpe_stdev_max_change=args_cli.pgpe_stdev_max_change,
        pgpe_ranking_method=args_cli.pgpe_ranking_method,
        fitness=args_cli.fitness,
        num_generations=args_cli.num_generations,
        seed=args_cli.seed,
        log_dir=log_dir,
    )

    trainer = HebbianTrainer(env.unwrapped, cfg)
    print(
        f"[INFO] genome dim={trainer.genome_dim} (n_weights/agent={trainer.n_weights}), "
        f"num_envs={trainer.num_envs}, popsize={cfg.popsize}, n_repeats={cfg.n_repeats}"
    )

    # --- dump a full settings file (all hyper-params + derived dims + env config) ---
    if hasattr(env_cfg, "approach_weight"):  # hook_grasp task reward terms (continuous per-step; no one-off bonuses)
        rw = ["approach_weight", "proximity_weight", "impact_perp_weight", "impact_up_weight",
              "lock_weight", "tension_weight",
              "action_smoothness_weight", "body_rate_penalty_weight", "force_penalty_weight",
              "downwash_rew_weight", "swing_limit_weight"]
    else:  # hover / hover_rand task reward terms
        rw = ["pos_track_weight", "ori_track_weight", "action_smoothness_weight",
              "body_rate_penalty_weight", "force_penalty_weight", "downwash_rew_weight"]
    settings = {
        "cli_args": {k: v for k, v in vars(args_cli).items()},
        "hebbian_config": dict(cfg.__dict__),
        "derived": {
            "obs_dim": trainer.obs_dim, "act_dim": trainer.act_dim,
            "n_weights_per_agent": trainer.n_weights, "genome_dim": trainer.genome_dim,
            "num_envs": trainer.num_envs, "agents": trainer.agents,
        },
        "env": {
            "task": args_cli.task,
            "partial_obs": getattr(env_cfg, "partial_obs", None),
            "history_len": getattr(env_cfg, "history_len", None),
            "control_mode": getattr(env_cfg, "control_mode", None),
            "episode_length_s": getattr(env_cfg, "episode_length_s", None),
            "decimation": getattr(env_cfg, "decimation", None),
            "sim_dt": getattr(env_cfg.sim, "dt", None),
            "reward_weights": {k: getattr(env_cfg, k, None) for k in rw},
        },
    }
    with open(os.path.join(log_dir, "settings.json"), "w") as f:
        json.dump(settings, f, indent=2, default=str)
    print(f"[INFO] wrote {os.path.join(log_dir, 'settings.json')}")

    # --- optional Weights & Biases logging (never fatal: a bad key must not kill the run) ---
    wandb_run = None
    if args_cli.wandb:
        try:
            import wandb

            ew_tag = ("_ewd" if args_cli.keep_d else "_ew") if args_cli.evolve_init_weights else ""
            extra = ""
            if args_cli.optimizer == "pgpe":
                extra = f"_clr{args_cli.pgpe_center_lr}_slr{args_cli.pgpe_stdev_lr}"
            elif args_cli.optimizer == "openai_es":
                extra = f"_es{args_cli.es_sigma}_elr{args_cli.es_lr}"
            elif args_cli.optimizer == "revde":
                extra = f"_F{args_cli.de_F}CR{args_cli.de_CR}"
            # encode the swept knobs so distinct configs get distinct display names
            extra += f"_b{args_cli.bound}_hlr{args_cli.hebbian_lr}_ui{args_cli.hebbian_update_interval}"
            if args_cli.weight_anchor > 0:
                extra += f"_wa{args_cli.weight_anchor}"
            if args_cli.latent_size != 4:
                extra += f"_l{args_cli.latent_size}"
            if args_cli.reward_curriculum:
                extra += "_rc"
            run_name = f"{ea_label}_p{args_cli.popsize}_i{args_cli.weight_init_scale}{ew_tag}{extra}_s{args_cli.seed}"
            wandb_run = wandb.init(
                project=args_cli.wandb_project, entity=args_cli.wandb_entity, name=run_name,
                group=ea_label, tags=[ea_label, f"pop{args_cli.popsize}", f"init{args_cli.weight_init_scale}"],
                config=settings, mode=args_cli.wandb_mode, dir=log_dir,
            )
            print(f"[INFO] W&B logging to project '{args_cli.wandb_project}' as run '{run_name}'")
        except Exception as e:
            print(f"[WARN] W&B init failed ({type(e).__name__}: {e}); continuing WITHOUT wandb.")
            wandb_run = None

    results_file = os.path.join(log_dir, "results.csv")
    with open(results_file, "w", newline="") as f:
        csv.writer(f).writerow(["generation", "env_steps", "best", "mean", "std", "best_so_far"])
    components_file = os.path.join(log_dir, "reward_components.csv")
    _comp_header = {"written": False}

    def log_fn(rec):
        with open(results_file, "a", newline="") as f:
            csv.writer(f).writerow(
                [rec["generation"], rec["env_steps"], rec["best"], rec["mean"], rec["std"], rec["best_so_far"]]
            )
        comps = rec.get("components", {})
        if comps:
            keys = list(comps.keys())
            with open(components_file, "a", newline="") as f:
                w = csv.writer(f)
                if not _comp_header["written"]:
                    w.writerow(["generation", "env_steps", "total_mean"] + keys)
                    _comp_header["written"] = True
                w.writerow([rec["generation"], rec["env_steps"], rec["mean"]] + [comps[k] for k in keys])
        if wandb_run is not None:
            payload = {"best": rec["best"], "mean": rec["mean"], "std": rec["std"],
                       "best_so_far": rec["best_so_far"], "generation": rec["generation"],
                       "env_steps": rec["env_steps"]}
            payload.update({f"reward/{k}": v for k, v in comps.items()})
            wandb_run.log(payload, step=rec["env_steps"])

    trainer.train(num_generations=args_cli.num_generations, max_env_steps=args_cli.max_env_steps,
                  max_hours=args_cli.max_hours, log_fn=log_fn)
    print(f"[INFO] Training done. Best fitness: {trainer.ea.f_best_so_far:.3f}")
    if wandb_run is not None:
        wandb_run.summary["best_fitness"] = float(trainer.ea.f_best_so_far)
        wandb_run.finish()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
