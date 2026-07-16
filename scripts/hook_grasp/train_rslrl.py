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
from rslrl_adapter import RslRlMarlAdapter, apply_f3_env_cfg, make_runner_cfg  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    apply_f3_env_cfg(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    assert isinstance(base, DirectMARLEnv), "train_rslrl expects a DirectMARLEnv (multi-agent) task."
    adapter = RslRlMarlAdapter(base)

    train_cfg = make_runner_cfg(
        experiment_name=args_cli.experiment_name,
        max_iterations=args_cli.max_iterations,
        num_steps_per_env=args_cli.num_steps_per_env,
        seed=args_cli.seed,
        device=device,
        logger="wandb" if args_cli.wandb else "tensorboard",
        wandb_project=args_cli.wandb_project,
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
