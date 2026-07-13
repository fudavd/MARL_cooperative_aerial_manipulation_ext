# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to the checkpoint to resume training.")
parser.add_argument(
    "--attachment_per_env",
    type=int,
    default=-1,
    choices=[-1, 0, 1],
    help="Per-env attachment domain randomization: -1 use cfg default, 0 off, 1 on "
    "(forces replicate_physics=False -> full DR, each env its own cable attachment).",
)
parser.add_argument("--wandb", action="store_true", default=False, help="Log the run to Weights & Biases.")
parser.add_argument(
    "--wandb_project", type=str, default="mappo-flycrane-rand", help="W&B project name (used when --wandb)."
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import MARL_mav_carry_ext.tasks  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # per-env attachment domain randomization (full DR). __post_init__ already ran with the cfg default, so when
    # enabling it here we must also flip replicate_physics off ourselves (mirrors scripts/hebbian/train.py).
    if args_cli.attachment_per_env >= 0 and hasattr(env_cfg, "attachment_per_env"):
        env_cfg.attachment_per_env = bool(args_cli.attachment_per_env)
        if args_cli.attachment_per_env == 1:
            env_cfg.scene.replicate_physics = False
        print(
            f"[INFO] attachment_per_env={env_cfg.attachment_per_env} "
            f"(replicate_physics={env_cfg.scene.replicate_physics})"
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # cosine LR annealing: resolve the torch scheduler class and auto-size T_max from the run length. skrl steps
    # the scheduler once per learning epoch per update, so total steps = (timesteps / rollouts) * learning_epochs.
    # Passing the class object (not a string) makes the runner skip its str-eval and use it directly.
    _sched = agent_cfg["agent"].get("learning_rate_scheduler")
    if isinstance(_sched, str) and _sched.lower() in ("cosineannealinglr", "cosine"):
        from torch.optim.lr_scheduler import CosineAnnealingLR

        _ts = int(agent_cfg["trainer"]["timesteps"])
        _updates = max(1, _ts // int(agent_cfg["agent"]["rollouts"]))
        _kwargs = agent_cfg["agent"].get("learning_rate_scheduler_kwargs") or {}
        _kwargs.setdefault("eta_min", 1.0e-5)
        _kwargs["T_max"] = _updates * int(agent_cfg["agent"]["learning_epochs"])
        agent_cfg["agent"]["learning_rate_scheduler"] = CosineAnnealingLR
        agent_cfg["agent"]["learning_rate_scheduler_kwargs"] = _kwargs
        print(f"[INFO] CosineAnnealingLR: T_max={_kwargs['T_max']} (updates={_updates}), eta_min={_kwargs['eta_min']}")
    elif isinstance(_sched, str) and _sched.lower() in ("kladaptivelr", "kladaptive", "kl"):
        # KL-adaptive LR: shrink/grow the LR to track a target policy-update KL. Robust against the
        # "peak-then-collapse" (it slows down when the policy lurches). No T_max needed.
        from skrl.resources.schedulers.torch import KLAdaptiveLR

        _kwargs = agent_cfg["agent"].get("learning_rate_scheduler_kwargs") or {}
        _kwargs.setdefault("kl_threshold", 0.008)
        agent_cfg["agent"]["learning_rate_scheduler"] = KLAdaptiveLR
        agent_cfg["agent"]["learning_rate_scheduler_kwargs"] = _kwargs
        print(f"[INFO] KLAdaptiveLR: kl_threshold={_kwargs['kl_threshold']}")
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir

    # optional Weights & Biases logging (skrl reads experiment.wandb + experiment.wandb_kwargs; it syncs the
    # tensorboard writer to W&B and names the run after the experiment dir).
    if args_cli.wandb:
        agent_cfg["agent"]["experiment"]["wandb"] = True
        _wk = agent_cfg["agent"]["experiment"].get("wandb_kwargs") or {}
        _wk.setdefault("project", args_cli.wandb_project)
        _wk.setdefault("name", log_dir)
        agent_cfg["agent"]["experiment"]["wandb_kwargs"] = _wk
        print(f"[INFO] W&B logging enabled: project={_wk['project']} name={log_dir}")

    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # load checkpoint
    if args_cli.resume:
        if args_cli.checkpoint:
            resume_path = args_cli.checkpoint
        else:
            resume_path = get_checkpoint_path(
                log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
            )

        print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env, state_as_observation=True)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    if args_cli.resume:
        runner.agent.load(resume_path)

    # run training
    runner.run()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    print("running main")
    main()
    simulation_app.close()
