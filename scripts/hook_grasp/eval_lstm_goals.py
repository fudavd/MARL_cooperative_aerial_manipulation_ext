# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched GOAL evaluator for RECURRENT (LSTM) hook-grasp PPO_RNN checkpoints.

LSTM counterpart of ``scripts/hook_grasp/eval_goals.py``. The MLP evaluator builds the agent via the
skrl ``Runner`` (yaml path) which CANNOT construct RNN models, so here we build the PPO_RNN agent
MANUALLY, reusing the exact Policy/Value LSTM classes and agent config from
``scripts/skrl/train_lstm.py``. Everything else (per-episode catch/hold measurement, curriculum
disabling, detach overrides) mirrors eval_goals.py.

CRITICAL — RNN hidden-state propagation: PPO_RNN only carries the LSTM hidden state across steps
inside ``record_transition()`` (training only). In an ``agent.act()`` eval loop we must replicate that
manually or the LSTM would run with ZERO memory every step. See the block after env.step() below.

Example:
    .venv/bin/python scripts/hook_grasp/eval_lstm_goals.py --headless \
        --checkpoint experiments/lstm_ckpts/L1_h256s16_77500.pt \
        --num_envs 256 --steps 3000 --hold_steps 100 --capture_radius 0.13 \
        --hidden_size 256 --sequence_length 16
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Batched goal evaluator for the hook-grasp PPO_RNN (LSTM) policy.")
parser.add_argument("--num_envs", type=int, default=512, help="Parallel envs.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to skrl PPO_RNN checkpoint .pt.")
parser.add_argument("--steps", type=int, default=4000, help="Total sim steps to roll (more -> more episodes).")
parser.add_argument("--hold_steps", type=int, default=100, help="Consecutive attached steps that count as a hold.")
parser.add_argument("--capture_radius", type=float, default=-1.0, help="Override catch radius (m); <0 = use cfg.")
parser.add_argument("--slack_dwell_steps", type=int, default=-1, help="Override detach grace period; <0 = use cfg (base=5).")
parser.add_argument("--detach_p_vertical", type=float, default=-1.0, help="Override vertical detach prob; <0 = use cfg (base=1.0).")
parser.add_argument("--detach_p_horizontal", type=float, default=-1.0, help="Override horizontal detach prob; <0 = use cfg (base=0.2).")
parser.add_argument("--seed", type=int, default=12345, help="Seed.")
parser.add_argument("--out", type=str, default="", help="Optional path to write a JSON result.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
# LSTM architecture args -- MUST match the checkpoint being evaluated or weights won't load.
parser.add_argument("--hidden_size", type=int, default=256, help="LSTM hidden size (match checkpoint).")
parser.add_argument("--sequence_length", type=int, default=16, help="LSTM sequence length (match checkpoint).")
parser.add_argument("--num_layers", type=int, default=1, help="Number of LSTM layers (match checkpoint).")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os

import gymnasium as gym
import torch
import torch.nn as nn

from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO_RNN as PPO
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import parse_env_cfg


# ---- LSTM models: copied VERBATIM from scripts/skrl/train_lstm.py ----
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
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # honest eval: disable every training curriculum so difficulty is the true (end) value ...
    for _attr in ("capture_curriculum", "hold_curriculum", "spawn_curriculum"):
        if hasattr(env_cfg, _attr):
            setattr(env_cfg, _attr, False)
    # ... and DISABLE the env's own hold-success termination so it never resets on the hold event; we measure
    # the hold ourselves (consecutive attached steps). A giant threshold makes `succeeded` effectively unreachable.
    if hasattr(env_cfg, "hold_success_steps"):
        env_cfg.hold_success_steps = 10_000_000
    if args_cli.capture_radius > 0 and hasattr(env_cfg, "attachment"):
        env_cfg.attachment.capture_radius = args_cli.capture_radius
    # optional detach-model overrides for FEASIBILITY BRACKETING (default <0 -> keep base strict model)
    if hasattr(env_cfg, "attachment"):
        if args_cli.slack_dwell_steps >= 0:
            env_cfg.attachment.slack_dwell_steps = args_cli.slack_dwell_steps
        if args_cli.detach_p_vertical >= 0:
            env_cfg.attachment.detach_p_vertical = args_cli.detach_p_vertical
        if args_cli.detach_p_horizontal >= 0:
            env_cfg.attachment.detach_p_horizontal = args_cli.detach_p_horizontal

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    if isinstance(base, DirectMARLEnv):
        env = multi_agent_to_single_agent(env, state_as_observation=True)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    device = env.device

    # ---- build the PPO_RNN agent MANUALLY (mirrors scripts/skrl/train_lstm.py) ----
    rollouts = 32
    memory = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=device)

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

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = rollouts  # memory_size
    cfg["learning_epochs"] = 5
    cfg["mini_batches"] = 32
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1e-3
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
    # no logging / checkpointing during eval
    cfg["experiment"]["write_interval"] = 0
    cfg["experiment"]["checkpoint_interval"] = 0
    cfg["experiment"]["wandb"] = False

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )
    # init() sets up _rnn / _rnn_initial_states / _rnn_final_states from the models' get_specification()
    agent.init()
    print(f"[eval-lstm-goals] loading checkpoint: {args_cli.checkpoint}")
    agent.load(args_cli.checkpoint)
    agent.set_running_mode("eval")

    E = args_cli.num_envs
    dev = base.device
    hold_req = args_cli.hold_steps

    # per-env, current-episode trackers
    ever_caught = torch.zeros(E, dtype=torch.bool, device=dev)
    ever_held = torch.zeros(E, dtype=torch.bool, device=dev)
    consec_attach = torch.zeros(E, dtype=torch.long, device=dev)

    # cumulative episode counters
    n_ep = torch.zeros((), dtype=torch.long, device=dev)
    n_caught = torch.zeros((), dtype=torch.long, device=dev)
    n_held = torch.zeros((), dtype=torch.long, device=dev)
    attached_steps = torch.zeros((), dtype=torch.long, device=dev)
    total_steps = 0

    obs, _ = env.reset()
    for _ in range(args_cli.steps):
        with torch.inference_mode():
            outputs = agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, terminated, truncated, _ = env.step(actions)

            # --- CRITICAL: propagate the LSTM hidden state across steps (replicates the RNN-state
            # bookkeeping that PPO_RNN.record_transition() does only during training). agent.act() reads
            # _rnn_initial_states["policy"] and writes _rnn_final_states["policy"]; here we zero the states
            # of finished episodes and roll final->initial so memory carries forward within an episode and
            # resets at its boundary. Without this the LSTM would run with ZERO memory every step.
            if agent._rnn:
                finished = (terminated | truncated).nonzero(as_tuple=False)
                if finished.numel():
                    for s in agent._rnn_final_states["policy"]:
                        s[:, finished[:, 0]] = 0
                agent._rnn_initial_states = agent._rnn_final_states

        attached = base.attached.any(dim=-1)                      # (E,) hook attached this step
        newly = base._newly_caught.any(dim=-1)                    # (E,) caught this step
        ever_caught |= newly
        consec_attach = torch.where(attached, consec_attach + 1, torch.zeros_like(consec_attach))
        ever_held |= consec_attach >= hold_req
        attached_steps += attached.sum()
        total_steps += E

        done = (terminated | truncated).view(E).bool()
        if done.any():
            n_ep += done.sum()
            n_caught += (ever_caught & done).sum()
            n_held += (ever_held & done).sum()
            keep = ~done
            ever_caught &= keep
            ever_held &= keep
            consec_attach = torch.where(done, torch.zeros_like(consec_attach), consec_attach)

    ep = int(n_ep.item())
    caught = int(n_caught.item())
    held = int(n_held.item())
    catch_rate = caught / ep if ep else 0.0
    hold_given_caught = held / caught if caught else 0.0
    attached_frac = float(attached_steps.item()) / max(1, total_steps)

    result = {
        "checkpoint": args_cli.checkpoint,
        "hidden_size": args_cli.hidden_size,
        "sequence_length": args_cli.sequence_length,
        "num_layers": args_cli.num_layers,
        "num_envs": E,
        "sim_steps": args_cli.steps,
        "hold_steps": hold_req,
        "capture_radius": float(base._acfg.capture_radius),
        "episodes": ep,
        "caught_episodes": caught,
        "held_episodes": held,
        "catch_rate": round(catch_rate, 4),
        "hold_success_given_caught": round(hold_given_caught, 4),
        "attached_frac": round(attached_frac, 4),
        "GOAL_catch_0.30": catch_rate >= 0.30,
        "GOAL_hold_0.80": hold_given_caught >= 0.80,
        "GOALS_MET": bool(catch_rate >= 0.30 and hold_given_caught >= 0.80),
    }
    print("\n==================== GOAL EVAL (LSTM) ====================")
    for k, v in result.items():
        print(f"  {k:26s}: {v}")
    print("=========================================================")
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        print(f"[eval-lstm-goals] wrote {args_cli.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
