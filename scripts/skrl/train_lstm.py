# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train a RECURRENT (LSTM) RL agent with skrl's PPO_RNN.

LSTM variant of ``scripts/skrl/train.py`` for the ``Isaac-flycrane-hook-grasp-v0`` task.
The agent is built MANUALLY (not via the skrl Runner, whose yaml path cannot build RNN models).
The Policy/Value LSTM models are copied from skrl's canonical PPO+LSTM example:
``thirdparty/skrl/docs/source/examples/gymnasium/torch_gymnasium_pendulumnovel_ppo_lstm.py``.

Visit the skrl documentation (https://skrl.readthedocs.io) for more details.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train a recurrent (LSTM) RL agent with skrl PPO_RNN.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--max_iterations", type=int, default=None, help="RL Policy training iterations (timesteps = max_iterations * rollouts)."
)
parser.add_argument("--experiment_name", type=str, default="", help="Suffix appended to the experiment run directory.")
parser.add_argument("--hidden_size", type=int, default=256, help="LSTM hidden size.")
parser.add_argument("--sequence_length", type=int, default=16, help="LSTM training sequence length (must divide rollouts).")
parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers.")

# append AppLauncher cli args (adds --headless, --device, ...)
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

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

import torch
import torch.nn as nn

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

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO_RNN as PPO
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import MARL_mav_carry_ext.tasks  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg


# define models (stochastic and deterministic models) using mixins.
# copied VERBATIM from skrl's PPO+LSTM example (compute()/get_specification() logic is subtle);
# only adaptation: Policy.compute() returns the RAW net output (no 2*tanh action scaling) because
# our env uses raw actions (clip_actions=False).
class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-3.0, max_log_std=0.0, reduction="sum",
                 num_envs=1, num_layers=1, hidden_size=256, sequence_length=16):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        # raw actions (our env uses clip_actions=False; no tanh scaling)
        return self.net(rnn_output), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, num_layers=1, hidden_size=256, sequence_length=16):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)  # batch_first -> (batch, sequence, features)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))

    def get_specification(self):
        # batch size (N) is the number of envs
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])  # (N, L, Hin): N=batch_size, L=sequence_length

            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[:,:,0,:].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:,:-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:,i0:i1,:], (hidden_states, cell_states))
                    hidden_states[:, (terminated[:,i1-1]), :] = 0
                    cell_states[:, (terminated[:,i1-1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])  # (N, L, Hin): N=num_envs, L=1
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        return self.net(rnn_output), {"rnn": [rnn_states[0], rnn_states[1]]}


def main():
    """Train with a recurrent (LSTM) skrl PPO_RNN agent."""
    # randomly sample a seed if seed = -1
    seed = args_cli.seed
    if seed == -1:
        seed = random.randint(0, 10000)
    if seed is None:
        seed = 42
    # seed for reproducibility (skrl global seed)
    set_seed(seed)

    # parse the environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = seed

    # ---- proven "C recipe" task-shaping overrides (catch is already solved with these) ----
    # guard each with hasattr so an absent field won't crash the run.
    if hasattr(env_cfg, "attachment") and hasattr(env_cfg.attachment, "capture_radius"):
        env_cfg.attachment.capture_radius = 0.13
    if hasattr(env_cfg, "capture_radius_start"):
        env_cfg.capture_radius_start = 0.15
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

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required (state_as_observation -> ~180-dim STATE obs)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env, state_as_observation=True)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework="torch")  # same as: `wrap_env(env, wrapper="auto")`

    device = env.device

    # rollout / trainer sizing
    rollouts = 32
    max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else 6250  # 200k timesteps default
    timesteps = max_iterations * rollouts

    # instantiate a memory as rollout buffer
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device)

    # instantiate the agent's models (function approximators)
    models = {}
    models["policy"] = Policy(
        env.observation_space, env.action_space, device,
        num_envs=env.num_envs, num_layers=args_cli.num_layers,
        hidden_size=args_cli.hidden_size, sequence_length=args_cli.sequence_length,
    )
    models["value"] = Value(
        env.observation_space, env.action_space, device,
        num_envs=env.num_envs, num_layers=args_cli.num_layers,
        hidden_size=args_cli.hidden_size, sequence_length=args_cli.sequence_length,
    )

    # experiment directory: {time-stamp}_PPO_RNN_torch[_{experiment_name}]
    log_root_path = os.path.abspath(os.path.join("logs", "skrl", "ppo_flycrane_hook"))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_PPO_RNN_torch"
    if args_cli.experiment_name:
        log_dir += f"_{args_cli.experiment_name}"

    # configure and instantiate the agent (mirrors skrl_ppo_cfg.yaml agent: block)
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = rollouts  # memory_size
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 32
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1e-3
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.003
    cfg["value_loss_scale"] = 1.0
    cfg["kl_threshold"] = 0.0
    cfg["rewards_shaper_scale"] = 1.0
    cfg["time_limit_bootstrap"] = False
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging and checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 100
    cfg["experiment"]["checkpoint_interval"] = 2500
    cfg["experiment"]["directory"] = log_root_path
    cfg["experiment"]["experiment_name"] = log_dir
    cfg["experiment"]["wandb"] = True
    cfg["experiment"]["wandb_kwargs"] = {"project": "hook-grasp-flycrane", "name": log_dir}

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    # configure and instantiate the RL trainer
    # NOTE: environment_info="log" is REQUIRED so the SequentialTrainer surfaces the env's
    # extras["log"] dict (Metrics/attached_frac, Episode_Reward/*, Episode_Termination/*) to
    # TensorBoard. The skrl default is "episode", which our env does not populate -> without this
    # only reward scalars are logged and the peak-then-collapse attached_frac curve is invisible.
    cfg_trainer = {"timesteps": timesteps, "headless": True, "close_environment_at_exit": False,
                   "environment_info": "log"}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

    # start training
    trainer.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    print("running main")
    main()
    simulation_app.close()
