# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train a RECURRENT (LSTM) MAPPO / CTDE hook-grasp policy.

ROUTE 1 (CTDE preserved): a RECURRENT, HOOK-BLIND actor + a PRIVILEGED critic.
- Actor: Gaussian LSTM over the per-agent 135-dim HOOK-FREE observation (``_drone_obs_frame``,
  45*3). Recurrence adds MEMORY only -- it lets the actor INFER the latent phase
  (approaching / attached / just-missed) from the HISTORY of observable dynamics, WITHOUT adding
  any hook/attached/tension state to the observation.
- Critic: memoryless MLP over the 180-dim PRIVILEGED shared/global state (``env.state()``, which
  includes hook truth). This keeps standard MAPPO/CTDE: the critic is privileged, the actor is not.

The env is kept MULTI-AGENT (DirectMARLEnv); we do NOT convert to single-agent and we NEVER feed the
180-dim state to the actor. The agent (``skrl.multi_agents.torch.mappo.MAPPO_RNN``) reads the shared
state from ``infos["shared_states"]`` exactly like stock MAPPO. Models + agent are built MANUALLY
(the skrl yaml Runner cannot build recurrent models); the existing MLP MAPPO path (train.py) is
untouched.

Smoke test (local, 8 GB):
    ./.venv/bin/python scripts/skrl/train_mappo_rnn.py --headless --num_envs 32 --max_iterations 3 \
        --rollouts 32 --experiment_name smoke
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a recurrent (LSTM) MAPPO/CTDE hook-grasp agent.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None,
                    help="Number of policy updates (timesteps = max_iterations * rollouts).")
parser.add_argument("--experiment_name", type=str, default="", help="Suffix appended to the experiment run directory.")
parser.add_argument("--rollouts", type=int, default=128, help="Rollout length / memory size (must be a multiple of sequence_length).")
parser.add_argument("--mini_batches", type=int, default=4, help="Mini-batches per learning epoch.")
parser.add_argument("--learning_epochs", type=int, default=5, help="Learning epochs per update.")
parser.add_argument("--hidden_size", type=int, default=256, help="LSTM hidden size.")
parser.add_argument("--sequence_length", type=int, default=16, help="LSTM training sequence length (divides rollouts).")
parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers.")
parser.add_argument("--wandb", action="store_true", default=False, help="Log the run to Weights & Biases.")
parser.add_argument("--wandb_project", type=str, default="hook-grasp-rnn", help="W&B project name (used when --wandb).")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import random
from datetime import datetime

import skrl
from packaging import version

SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}. Need >= {SKRL_VERSION}")
    exit()

from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch.mappo import MAPPO_RNN, MAPPO_RNN_DEFAULT_CONFIG
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import parse_env_cfg

# shared model definitions (identical structure on the eval side -> checkpoint loads cleanly)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rnn_models import RecurrentPolicy, MLPValue  # noqa: E402


def main():
    seed = args_cli.seed
    if seed == -1:
        seed = random.randint(0, 10000)
    if seed is None:
        seed = 42
    set_seed(seed)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = seed

    # ---- proven "C recipe" task-shaping overrides (catch already solved with these). Guard with hasattr. ----
    if hasattr(env_cfg, "attachment") and hasattr(env_cfg.attachment, "capture_radius"):
        env_cfg.attachment.capture_radius = 0.13
    if hasattr(env_cfg, "capture_radius_start"):
        env_cfg.capture_radius_start = 0.25
    if hasattr(env_cfg, "capture_curriculum_steps"):
        env_cfg.capture_curriculum_steps = 200000
    if hasattr(env_cfg, "hold_success_steps"):
        env_cfg.hold_success_steps = 100
    if hasattr(env_cfg, "hold_success_steps_start"):
        env_cfg.hold_success_steps_start = 30
    if hasattr(env_cfg, "hold_curriculum_steps"):
        env_cfg.hold_curriculum_steps = 150000
    if hasattr(env_cfg, "alive_weight"):
        env_cfg.alive_weight = 1.0
    if hasattr(env_cfg, "proximity_weight"):
        env_cfg.proximity_weight = 2.0
    if hasattr(env_cfg, "spawn_dist_mu"):
        env_cfg.spawn_dist_mu = 0.35

    # ---- keep the env MULTI-AGENT (CTDE): do NOT convert to single-agent; do NOT feed the state to the actor ----
    env = gym.make(args_cli.task, cfg=env_cfg)
    assert isinstance(env.unwrapped, DirectMARLEnv), "train_mappo_rnn expects a DirectMARLEnv (multi-agent) task."
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device = env.device
    possible_agents = list(env.possible_agents)
    obs_spaces = env.observation_spaces      # per-agent 135-dim HOOK-FREE actor obs
    state_spaces = env.state_spaces          # per-agent 180-dim PRIVILEGED critic state
    action_spaces = env.action_spaces

    actor_obs_dim = int(gym.spaces.flatdim(obs_spaces[possible_agents[0]]))
    critic_state_dim = int(gym.spaces.flatdim(state_spaces[possible_agents[0]]))
    print(f"[train-mappo-rnn] agents={possible_agents}  actor_obs_dim={actor_obs_dim}  "
          f"critic_state_dim={critic_state_dim}")
    assert actor_obs_dim == 135, f"actor obs must be 135-dim (45*3) HOOK-FREE, got {actor_obs_dim}"
    assert critic_state_dim == 180, f"critic state expected 180-dim (60*3), got {critic_state_dim}"

    assert args_cli.rollouts % args_cli.sequence_length == 0, \
        f"rollouts ({args_cli.rollouts}) must be a multiple of sequence_length ({args_cli.sequence_length})"

    rollouts = args_cli.rollouts
    max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else 8000
    timesteps = max_iterations * rollouts

    # ---- build shared recurrent actor + privileged MLP critic (single agent: num_drones == 1) ----
    models, memories = {}, {}
    policy = RecurrentPolicy(
        obs_spaces[possible_agents[0]], action_spaces[possible_agents[0]], device,
        num_envs=env.num_envs, num_layers=args_cli.num_layers,
        hidden_size=args_cli.hidden_size, sequence_length=args_cli.sequence_length,
    )
    value = MLPValue(state_spaces[possible_agents[0]], action_spaces[possible_agents[0]], device)
    for uid in possible_agents:
        models[uid] = {"policy": policy, "value": value}
        memories[uid] = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device)

    # experiment directory: {time-stamp}_MAPPO_RNN_torch[_{name}]
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", "mappo_flycrane_hook_rnn"))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_MAPPO_RNN_torch"
    if args_cli.experiment_name:
        log_dir += f"_{args_cli.experiment_name}"
    print(f"[train-mappo-rnn] Logging experiment in: {os.path.join(log_root_path, log_dir)}")

    cfg = MAPPO_RNN_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = rollouts
    cfg["learning_epochs"] = args_cli.learning_epochs
    cfg["mini_batches"] = args_cli.mini_batches
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate_actor"] = 5.0e-5
    cfg["learning_rate_critic"] = 1.0e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.012}
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.004
    cfg["value_loss_scale"] = 1.0
    cfg["kl_threshold"] = 0.0
    cfg["time_limit_bootstrap"] = True
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": obs_spaces[possible_agents[0]], "device": device}
    cfg["shared_state_preprocessor"] = RunningStandardScaler
    cfg["shared_state_preprocessor_kwargs"] = {"size": state_spaces[possible_agents[0]], "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg["experiment"]["write_interval"] = 100
    cfg["experiment"]["checkpoint_interval"] = 2500
    cfg["experiment"]["directory"] = log_root_path
    cfg["experiment"]["experiment_name"] = log_dir
    if args_cli.wandb:
        cfg["experiment"]["wandb"] = True
        cfg["experiment"]["wandb_kwargs"] = {"project": args_cli.wandb_project, "name": log_dir}

    agent = MAPPO_RNN(
        possible_agents=possible_agents,
        models=models,
        memories=memories,
        cfg=cfg,
        observation_spaces=obs_spaces,
        action_spaces=action_spaces,
        device=device,
        shared_observation_spaces=state_spaces,
    )

    cfg_trainer = {"timesteps": timesteps, "headless": True, "close_environment_at_exit": False,
                   "environment_info": "log"}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    trainer.train()
    env.close()


if __name__ == "__main__":
    print("running main")
    main()
    simulation_app.close()
