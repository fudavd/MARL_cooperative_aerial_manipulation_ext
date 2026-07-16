# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the single-drone hook-grasp task with rsl_rl (asymmetric + recurrent LSTM).

Backend pivot from skrl/MAPPO -> rsl_rl: the recurrent actor is HOOK-BLIND (135-dim obs) and infers
the latent phase (approach / attached / missed) from observation HISTORY via an LSTM; the critic is
PRIVILEGED (180-dim state incl. hook rel/vel, hook->bar err, attached, tension, hold_progress). This
asymmetric actor-critic replaces the MAPPO/CTDE machinery -- rsl_rl reads the privileged obs natively
from ``extras['observations']['critic']``.

Env stays UNCHANGED (DirectMARLEnv, num_drones=1). A thin adapter (rslrl_adapter.py) presents its
single agent to rsl_rl. The F3 recipe (env-level) is applied via apply_f3_env_cfg.

Local smoke test (8 GB GPU):
    ./.venv/bin/python scripts/hook_grasp/train_rslrl.py --headless --num_envs 32 --max_iterations 3

ALC full run:
    ./.venv/bin/python scripts/hook_grasp/train_rslrl.py --headless --device cuda:0 --num_envs 256 \
        --max_iterations 6000 --wandb --wandb_project hook-grasp-rnn --experiment_name v26L1_lstm
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the hook-grasp task with rsl_rl (recurrent + asymmetric).")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel envs.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--max_iterations", type=int, default=6000, help="rsl_rl learning iterations.")
parser.add_argument("--num_steps_per_env", type=int, default=24, help="Rollout length per env per update.")
parser.add_argument("--seed", type=int, default=42, help="Seed.")
parser.add_argument("--experiment_name", type=str, default="v26L1_lstm", help="Experiment / run name.")
parser.add_argument("--wandb", action="store_true", default=False, help="Log to Weights & Biases.")
parser.add_argument("--wandb_project", type=str, default="hook-grasp-rnn", help="W&B project name.")
parser.add_argument("--resume", type=str, default="", help="Path to a checkpoint to resume from.")
parser.add_argument("--indi_residual", action="store_true", default=False,
                    help="Feed the 6-dim body-frame INDI disturbance residual (hook/wire wrench) to the LSTM actor.")
parser.add_argument("--actor_history_len", type=int, default=0,
                    help="ACTOR-only frame-stack depth H (0 -> use env history_len). Critic stays 180-dim.")
parser.add_argument("--reward_norm", action="store_true", default=False,
                    help="Pathak reward normalization: divide reward by running discounted-return std so the "
                         "value function sees a stationary return scale across curriculum level shifts.")
parser.add_argument("--env_kv", type=str, default="",
                    help="Comma-sep env_cfg overrides applied AFTER the F3 recipe, e.g. "
                         "'reward_pump=true,pump_lift_weight=3.0,attachment.upsweep_v_min=0.06'. Rewards / "
                         "curriculum / gates only -- NEVER obs_dim/history/task-defining fields.")
parser.add_argument("--policy_kv", type=str, default="",
                    help="Comma-sep ActorCriticRecurrent overrides, e.g. 'rnn_hidden_dim=512,rnn_num_layers=2'.")
parser.add_argument("--algo_kv", type=str, default="",
                    help="Comma-sep PPO overrides, e.g. 'entropy_coef=0.01,learning_rate=0.0005,desired_kl=0.02'.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
from datetime import datetime

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import parse_env_cfg

# local imports (this script's dir is on sys.path when run directly)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rslrl_adapter import (  # noqa: E402
    RslRlMarlAdapter,
    apply_f3_env_cfg,
    apply_indi_residual_cfg,
    make_runner_cfg,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# fields that MUST NOT be touched by --env_kv (they change the obs space or the task itself). The overnight
# search tunes rewards / curriculum / gates / HP / architecture only; these are hard-blocked so a stray key
# can never silently alter the locked observation space or the fundamental task.
_ENV_KV_BLOCKLIST = {
    "history_len", "actor_history_len", "actor_indi_residual", "obs_dim_per_frame", "observation_spaces",
    "state_space", "action_dim", "action_spaces", "num_drones", "possible_agents", "partial_obs",
}


def _coerce(v: str):
    """Infer bool/int/float/str from a string override value."""
    lo = v.strip().lower()
    if lo in ("true", "false"):
        return lo == "true"
    if lo in ("none", "null"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _parse_kv(s: str) -> dict:
    """Parse 'a=1,b=2.0,c.d=true' -> {'a':1,'b':2.0,'c.d':True} with type inference."""
    out = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.partition("=")
        out[k.strip()] = _coerce(v)
    return out


def _apply_env_kv(env_cfg, kv: dict):
    """Set env_cfg fields (supports one level of nesting: 'attachment.upsweep_v_min'). Blocklisted keys raise."""
    for k, v in kv.items():
        root = k.split(".", 1)[0]
        if root in _ENV_KV_BLOCKLIST:
            raise ValueError(f"--env_kv key '{k}' is BLOCKED (changes obs space / task); refusing.")
        if "." in k:
            obj_name, attr = k.split(".", 1)
            obj = getattr(env_cfg, obj_name)
            assert hasattr(obj, attr), f"env_cfg.{obj_name} has no attribute '{attr}'"
            setattr(obj, attr, v)
        else:
            assert hasattr(env_cfg, k), f"env_cfg has no attribute '{k}'"
            setattr(env_cfg, k, v)
        print(f"[train-rslrl] env override: {k} = {v}")


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    apply_f3_env_cfg(env_cfg)
    if args_cli.indi_residual or args_cli.actor_history_len:
        H = args_cli.actor_history_len or env_cfg.history_len
        apply_indi_residual_cfg(env_cfg, actor_history_len=H, residual=args_cli.indi_residual)
        print(f"[train-rslrl] INDI-residual actor obs: residual={args_cli.indi_residual}  H={H}  "
              f"actor_obs_dim={next(iter(env_cfg.observation_spaces.values()))}")
    if args_cli.env_kv:
        _apply_env_kv(env_cfg, _parse_kv(args_cli.env_kv))

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    assert isinstance(base, DirectMARLEnv), "train_rslrl expects a DirectMARLEnv (multi-agent) task."
    adapter = RslRlMarlAdapter(base, reward_norm=args_cli.reward_norm)
    if args_cli.reward_norm:
        print("[train-rslrl] reward normalization ON (Pathak discounted-return-std): value fn sees a "
              "stationary return scale across curriculum shifts")

    train_cfg = make_runner_cfg(
        experiment_name=args_cli.experiment_name,
        max_iterations=args_cli.max_iterations,
        num_steps_per_env=args_cli.num_steps_per_env,
        seed=args_cli.seed,
        device=device,
        logger="wandb" if args_cli.wandb else "tensorboard",
        wandb_project=args_cli.wandb_project,
        policy_overrides=_parse_kv(args_cli.policy_kv) if args_cli.policy_kv else None,
        algo_overrides=_parse_kv(args_cli.algo_kv) if args_cli.algo_kv else None,
    )

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.experiment_name}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train-rslrl] logging to {log_dir}")

    runner = OnPolicyRunner(adapter, train_cfg, log_dir=log_dir, device=device)
    if args_cli.resume:
        print(f"[train-rslrl] resuming from {args_cli.resume}")
        runner.load(args_cli.resume)

    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    # os._exit(0) avoids the Isaac headless teardown hang (simulation_app.close() lingers and holds the
    # GPU, leaving a zombie after every run — bad on the shared ALC card). Checkpoints + wandb per-iter
    # logs are already flushed inside runner.learn() before main() returns. Same pattern as eval_rslrl.py.
    # BUT os._exit skips wandb's atexit -> the run shows "crashed" on the dashboard even though it finished.
    # So finish wandb explicitly first, then hard-exit past the Isaac teardown.
    try:
        main()
    finally:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass
        os._exit(0)
