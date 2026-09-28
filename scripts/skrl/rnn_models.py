# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recurrent CTDE models for the hook-grasp MAPPO_RNN agent.

Shared by ``scripts/skrl/train_mappo_rnn.py`` (training) and
``scripts/hook_grasp/eval_mappo_rnn.py`` (evaluation) so the ``nn.Module`` structure --
and therefore the checkpoint state_dict keys -- is IDENTICAL on both sides.

- ``RecurrentPolicy``: a Gaussian LSTM actor. It consumes ONLY the per-agent, HOOK-FREE 135-dim
  observation. Recurrence adds MEMORY (latent-phase inference from the history of observable
  dynamics), never new inputs. The LSTM hidden-state rollout/reset logic is copied verbatim from
  skrl's canonical PPO+LSTM example (compute()/get_specification() is subtle).
- ``MLPValue``: a plain MLP critic. It consumes the PRIVILEGED 180-dim shared/global state
  (``env.state()``), which DOES include hook/tension truth -- allowed for the critic under CTDE.
  Kept memoryless (the actor recurrence is the point); this is the simplest stable critic.
"""

import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class RecurrentPolicy(GaussianMixin, Model):
    """Gaussian LSTM actor over the hook-free per-agent observation (135-dim)."""

    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-3.0, max_log_std=0.0, reduction="sum",
                 num_envs=1, num_layers=1, hidden_size=256, sequence_length=16):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.lstm = nn.LSTM(input_size=self.num_observations,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

        self.net = nn.Sequential(nn.Linear(self.hidden_size, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def get_specification(self):
        return {"rnn": {"sequence_length": self.sequence_length,
                        "sizes": [(self.num_layers, self.num_envs, self.hidden_size),
                                  (self.num_layers, self.num_envs, self.hidden_size)]}}

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        # training
        if self.training:
            rnn_input = states.view(-1, self.sequence_length, states.shape[-1])
            hidden_states = hidden_states.view(self.num_layers, -1, self.sequence_length, hidden_states.shape[-1])
            cell_states = cell_states.view(self.num_layers, -1, self.sequence_length, cell_states.shape[-1])
            hidden_states = hidden_states[:, :, 0, :].contiguous()
            cell_states = cell_states[:, :, 0, :].contiguous()

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = [0] + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() \
                    + [self.sequence_length]
                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(rnn_input[:, i0:i1, :],
                                                                         (hidden_states, cell_states))
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    cell_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)
                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))
        # rollout
        else:
            rnn_input = states.view(-1, 1, states.shape[-1])
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)
        # raw actions (env uses clip_actions=False; no tanh scaling)
        return self.net(rnn_output), self.log_std_parameter, {"rnn": [rnn_states[0], rnn_states[1]]}


class MLPValue(DeterministicMixin, Model):
    """Memoryless MLP critic over the privileged 180-dim shared/global state."""

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 1024),
                                 nn.ELU(),
                                 nn.Linear(1024, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
