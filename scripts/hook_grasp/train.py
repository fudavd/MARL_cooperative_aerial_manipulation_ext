# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the hook-grasp task with PPO (rsl_rl).

The task, its curricula and the PPO hyper-parameters are fully specified by the registered configs
(``MARLHookGraspEnvCfg`` and ``agents/rsl_rl_ppo_cfg.py``); this script only sets the run size.

Full recipe (~7 h on an RTX 4090; checkpoints in logs/rsl_rl/<run_name>/<timestamp>/model_<iter>.pt):
    python scripts/hook_grasp/train.py --headless

Smoke test:
    python scripts/hook_grasp/train.py --headless --num_envs 32 --max_iterations 3
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the hook-grasp task with PPO (rsl_rl).")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--num_envs", type=int, default=512, help="Parallel environments.")
parser.add_argument("--max_iterations", type=int, default=20000, help="PPO iterations (24 env steps each).")
parser.add_argument("--seed", type=int, default=42, help="Seed.")
parser.add_argument("--run_name", type=str, default="hook_grasp", help="Run name (log folder).")
parser.add_argument("--resume", type=str, default="", help="Checkpoint to resume from (model_<iter>.pt).")
parser.add_argument("--wandb", action="store_true", default=False, help="Log to Weights & Biases.")
parser.add_argument("--wandb_project", type=str, default="hook-grasp", help="W&B project.")
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
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rsl_rl_env import HookGraspVecEnv  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    agent_cfg: dict = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg.update(
        seed=args_cli.seed,
        device=device,
        max_iterations=args_cli.max_iterations,
        experiment_name=args_cli.run_name,
        run_name=args_cli.run_name,
        logger="wandb" if args_cli.wandb else "tensorboard",
        wandb_project=args_cli.wandb_project,
    )

    log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.run_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    os.makedirs(log_dir, exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    print(f"[train] logging to {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    vec_env = HookGraspVecEnv(env.unwrapped)

    runner = OnPolicyRunner(vec_env, agent_cfg, log_dir=log_dir, device=device)
    if args_cli.resume:
        print(f"[train] resuming from {args_cli.resume}")
        runner.load(args_cli.resume)

    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        # Checkpoints and logs are flushed inside runner.learn(). Hard-exit past the headless Isaac Sim teardown,
        # which can hang and keep the GPU allocated; finish wandb first so the run is not reported as crashed.
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass
        os._exit(0)
