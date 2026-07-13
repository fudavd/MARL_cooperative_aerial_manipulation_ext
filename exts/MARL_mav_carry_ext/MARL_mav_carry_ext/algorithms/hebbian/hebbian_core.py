# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dependency-light vendoring of the Hebbian actor and CMA-ES optimizer.

Faithful (behaviour-preserving) copy of the framework-agnostic core of
``thirdparty/hebbian_marl/src/hebbian.py`` (:class:`HebbianActor`) and
``thirdparty/hebbian_marl/src/CMAES.py`` (:class:`CMAES`), with the ``agilerl`` /
``pettingzoo`` couplings removed and the ``src.utils.search_file_list`` helper inlined.

Network: a single-hidden-layer, bias-free MLP whose weights are *not* trained by gradient
descent.  Instead, a set of Hebbian plasticity coefficients (A, B, C, D) -- the "genome" --
drives an online weight update every forward pass.  The genome is evolved by CMA-ES, with the
episode return as fitness.  A separate set of weights is kept per parallel environment (the
leading ``n_envs`` dimension), so each environment can evaluate a different genome.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

import cma

__all__ = ["CMAES", "HebbianActor"]


# ---------------------------------------------------------------------------
# Hebbian actor
# ---------------------------------------------------------------------------
class HebbianActor:
    """Online-plastic single-hidden-layer policy network (one weight set per env).

    :param observation_space: Per-agent observation space (Box).
    :param action_space: Per-agent action space (Box).
    :param num_envs: Number of parallel weight sets to maintain (= number of envs).
    :param net_config: dict with keys ``latent_size``, ``hidden_activation``,
        ``output_activation``, ``clip_actions``.
    :param device: torch device.
    """

    supported_spaces = (spaces.Box, spaces.Discrete)

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        num_envs: int,
        net_config: Optional[Dict] = None,
        device: str | torch.device = "cpu",
        lr: float = 0.1,
        init_scale: float = 0.01,
    ):
        net_config = net_config or {}
        self.device = device
        self.action_space = action_space

        if isinstance(action_space, spaces.Box):
            self.action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=self.device)
            self.action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=self.device)
        else:
            self.action_low = None
            self.action_high = None

        latent_dim = net_config.get("latent_size", 16)
        hidden_activation = net_config.get("hidden_activation", "Tanh")
        output_activation = net_config.get("output_activation", "Tanh")
        self.output_function = output_activation

        if output_activation == "Tanh":
            self.output_activation = nn.Tanh()
        elif output_activation == "Softmax":
            self.output_activation = nn.Softmax(dim=-1)
        else:
            self.output_activation = nn.Identity()

        if hidden_activation == "Tanh":
            self.hidden_activation = nn.Tanh()
        elif hidden_activation == "Sigmoid":
            self.hidden_activation = nn.Sigmoid()
        elif hidden_activation == "ReLu":
            self.hidden_activation = nn.ReLU()
        else:
            self.hidden_activation = nn.Identity()

        self.clip_actions = net_config.get("clip_actions", False)

        self.n_input = int(np.prod(observation_space.shape))
        self.n_hidden = latent_dim
        self.n_output = action_space.shape[0]

        self.n_envs = num_envs
        self.n_weights = (self.n_output + self.n_input) * self.n_hidden
        self.n_l1_weights = self.n_input * self.n_hidden
        self.n_out_weights = self.n_hidden * self.n_output
        self.lr = lr
        self.init_scale = init_scale
        # Genome layout: blocks [A, B, C] (+ [D] if D term used) (+ [init_W] if evolving init weights).
        #   evolve_init_weights=False                 -> ABCD          (4 blocks; random init weights)
        #   evolve_init_weights=True,  keep_d=False    -> ABC + W       (4 blocks; D=0, evolved init)
        #   evolve_init_weights=True,  keep_d=True     -> ABCD + W      (5 blocks; full rule, evolved init)
        self.evolve_init_weights = bool(net_config.get("evolve_init_weights", False))
        self.keep_d = bool(net_config.get("keep_d", False))
        self.use_d = (not self.evolve_init_weights) or self.keep_d
        self.n_blocks = 3 + (1 if self.use_d else 0) + (1 if self.evolve_init_weights else 0)
        self.init_W = None
        self.li = 0
        self.n_gen = 0
        # Lower-timescale plasticity: accumulate pre/post activations over `update_interval`
        # inference steps, average them, and apply ONE Hebbian update from those averages.
        # update_interval=1 reproduces the original per-step update exactly.
        self.update_interval = max(1, int(net_config.get("update_interval", 1)))
        # Weight anchoring (relative weight decay): add an elastic restoring term
        #   -weight_anchor * (w - w_init)
        # to each Hebbian weight update, pulling the plastic weights back toward their per-episode
        # initial values w_init (the evolved init_W when evolve_init_weights, else the reset draw).
        # This makes the plastic dynamics a leaky/elastic integrator with a stable fixed point
        #   w* = w_init + (Hebbian drive) / weight_anchor,
        # bounding |w - w_init| and preventing Hebbian weight explosion. 0 disables it (original
        # behaviour). The anchor w_init is (re)captured per environment in reset_network().
        self.weight_anchor = float(net_config.get("weight_anchor", 0.0))

        with torch.no_grad():
            self.weights = torch.randn((self.n_envs, self.n_weights), device=self.device) * self.init_scale
            self.weights_hid = self.weights[:, : self.n_l1_weights].reshape(self.n_envs, self.n_hidden, self.n_input)
            self.weights_out = self.weights[:, -self.n_out_weights :].reshape(self.n_envs, self.n_output, self.n_hidden)

            self.A = torch.zeros_like(self.weights, device=self.device)
            self.B = torch.zeros_like(self.weights, device=self.device)
            self.C = torch.zeros_like(self.weights, device=self.device)
            self.D = torch.zeros_like(self.weights, device=self.device)

            # running sums of per-synapse pre/post activations + per-env step counts since last update
            self._pre_accum = torch.zeros_like(self.weights, device=self.device)
            self._pos_accum = torch.zeros_like(self.weights, device=self.device)
            self._accum_steps = torch.zeros((self.n_envs, 1), device=self.device)
            self._global_step = 0

            # anchor for weight-anchoring = the per-episode initial weights (w_init); refreshed on
            # every reset_network so the decay always pulls toward the current episode's init.
            self.w_anchor = self.weights.detach().clone()

    def hebbian_update(self, weights_delta: torch.Tensor) -> None:
        """Apply an additive update to the per-env weight matrices."""
        self.weights += weights_delta
        self.weights_hid = self.weights[:, : self.n_l1_weights].reshape(self.n_envs, self.n_hidden, self.n_input)
        self.weights_out = self.weights[:, -self.n_out_weights :].reshape(self.n_envs, self.n_output, self.n_hidden)
        self.li += 1

    @staticmethod
    def rescale_action(
        action: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
        output_activation: Optional[str] = None,
    ) -> torch.Tensor:
        """Rescale a network output to the environment action range."""
        if output_activation in ["Tanh", "Softsign"]:
            prescaled_min, prescaled_max = -1.0, 1.0
        elif output_activation in ["Sigmoid", "Softmax", "GumbelSoftmax"]:
            prescaled_min, prescaled_max = 0.0, 1.0
        else:
            return action

        if low.isinf().any() or high.isinf().any():
            rescaled_action = action
        else:
            rescaled_action = low + (high - low) * (action - prescaled_min) / (prescaled_max - prescaled_min)
        return rescaled_action

    @torch.no_grad()
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass + online Hebbian weight update.

        :param states: ``[n_envs, n_input]`` observation tensor.
        :return: ``[n_envs, n_output]`` action tensor (raw network output, in activation range).
        """
        hidden = self.hidden_activation(torch.einsum("ki,kji->kj", states, self.weights_hid))
        output = self.output_activation(torch.einsum("ki,kji->kj", hidden, self.weights_out))

        # pre = source-neuron activation (tile); post = target-neuron activation (repeat_interleave).
        pre_synaptic = torch.hstack((states.tile(1, self.n_hidden), hidden.tile(1, self.n_output)))
        pos_synaptic = torch.hstack(
            (hidden.repeat_interleave(self.n_input, dim=1), output.repeat_interleave(self.n_hidden, dim=1))
        )

        if self.update_interval <= 1:
            # original per-step update
            weights_delta = self.lr * (
                self.A * pre_synaptic + self.B * pos_synaptic + self.C * pre_synaptic * pos_synaptic + self.D
            )
            if self.weight_anchor > 0.0:
                weights_delta = weights_delta - self.weight_anchor * (self.weights - self.w_anchor)
            self.hebbian_update(weights_delta)
            return output

        # lower-timescale update: accumulate activations, apply rule once per `update_interval` steps
        self._pre_accum += pre_synaptic
        self._pos_accum += pos_synaptic
        self._accum_steps += 1.0
        self._global_step += 1
        if self._global_step % self.update_interval == 0:
            cnt = self._accum_steps.clamp(min=1.0)  # per-env -> robust to mid-batch resets
            pre_avg = self._pre_accum / cnt
            pos_avg = self._pos_accum / cnt
            weights_delta = self.lr * (
                self.A * pre_avg + self.B * pos_avg + self.C * pre_avg * pos_avg + self.D
            )
            if self.weight_anchor > 0.0:
                weights_delta = weights_delta - self.weight_anchor * (self.weights - self.w_anchor)
            self.hebbian_update(weights_delta)
            self._pre_accum.zero_()
            self._pos_accum.zero_()
            self._accum_steps.zero_()
        return output

    @torch.no_grad()
    def act(self, states: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: forward + rescale to the action space range."""
        output = self.forward(states)
        if self.action_low is not None:
            output = self.rescale_action(output, self.action_low, self.action_high, self.output_function)
        return output

    def reset_network(self, env_ids: List[int] | torch.Tensor) -> None:
        """Re-initialize the plastic weights for the given environments.

        Default: random init (randn * init_scale). If ``evolve_init_weights`` is set, the weights
        are reset to the genome's evolved initial weights (``init_W``) instead -- requires
        set_hebbian_rules() to have been called first.
        """
        if self.evolve_init_weights and self.init_W is not None:
            for env_id in env_ids:
                self.weights[env_id, :] = self.init_W[env_id]
        else:
            for env_id in env_ids:
                self.weights[env_id, :] = torch.randn(self.n_weights, device=self.device) * self.init_scale
        self.weights_hid = self.weights[:, : self.n_l1_weights].reshape(self.n_envs, self.n_hidden, self.n_input)
        self.weights_out = self.weights[:, -self.n_out_weights :].reshape(self.n_envs, self.n_output, self.n_hidden)
        # re-anchor weight-anchoring to THIS episode's fresh initial weights (w_init) for reset envs
        if self.weight_anchor > 0.0:
            for env_id in env_ids:
                self.w_anchor[env_id, :] = self.weights[env_id, :]
        # drop any partially-accumulated activations for the reset envs (fresh episode -> fresh batch)
        for env_id in env_ids:
            self._pre_accum[env_id, :] = 0.0
            self._pos_accum[env_id, :] = 0.0
            self._accum_steps[env_id] = 0.0

    def set_hebbian_rules(self, abcd: torch.Tensor) -> None:
        """Set per-env plasticity coefficients from a ``[n_envs, n_blocks, n_weights]`` genome.

        Block order: A, B, C, then D (if ``use_d``), then init_W (if ``evolve_init_weights``).
        """
        assert abcd[:, 0, :].shape == self.weights.shape, "genome block size does not match weights"
        self.A = abcd[:, 0, :].detach().clone()
        self.B = abcd[:, 1, :].detach().clone()
        self.C = abcd[:, 2, :].detach().clone()
        idx = 3
        if self.use_d:
            self.D = abcd[:, idx, :].detach().clone()
            idx += 1
        else:
            self.D = torch.zeros_like(self.A)
        if self.evolve_init_weights:
            self.init_W = abcd[:, idx, :].detach().clone()


# ---------------------------------------------------------------------------
# CMA-ES optimizer
# ---------------------------------------------------------------------------
def _atoi(text: str):
    return int(text) if text.isdigit() else text


def _natural_keys(text: str):
    return [_atoi(c) for c in re.split(r"(\d+)", text)]


def _search_file_list(rootname: str, file_name: str) -> List[str]:
    file_list = []
    for root, _dirs, files in os.walk(rootname):
        for file in files:
            if file_name in file:
                file_list.append(os.path.join(root, file))
    file_list.sort(key=_natural_keys)
    return file_list


CMAES_opts = {
    "min": -1.0,
    "max": 1.0,
    "num_generations": 1000,
    "mutation_sigma": 0.3,
}


class CMAES:
    """Thin wrapper around ``cma.CMAEvolutionStrategy`` with bounds, clipping and checkpointing.

    Note: CMA-ES *minimizes*, so fitnesses (which we maximize) are negated in :meth:`tell`.
    """

    def __init__(self, n_pop: int, n_params: int, opts: Dict = CMAES_opts, output_dir: str = "./results/CMAES",
                 diagonal: bool = False, seed: Optional[int] = None):
        self.n_params = n_params
        self.n_pop = n_pop
        self.n_gen = opts["num_generations"]
        self.min = opts["min"]
        self.max = opts["max"]

        self.current_gen = 0
        self.initial_pop = self.initialise_x0(n_params)
        self.current_sigma = opts["mutation_sigma"]
        self.diagonal = diagonal
        self.seed = seed
        self.f_new = np.empty(self.n_pop)

        self.cmaes = self.load_cmaes()

        # bookkeeping
        self.directory_name = output_dir
        self.full_x = []
        self.full_fitness = []
        self.x_best_so_far = None
        self.f_best_so_far = -np.inf
        self.x = [None] * self.n_pop
        self.f = [-np.inf] * self.n_pop

    def load_cmaes(self):
        params = {
            "popsize": self.n_pop,
            "bounds": ([self.min] * self.n_params, [self.max] * self.n_params),
        }
        if self.diagonal:
            params["CMA_diagonal"] = True
        if self.seed is not None:
            params["seed"] = int(self.seed)
        return cma.CMAEvolutionStrategy(self.initial_pop, self.current_sigma, inopts=params)

    def ask(self):
        new_population = self.cmaes.ask()
        new_population = np.clip(new_population, self.min, self.max)
        return new_population

    def tell(self, solutions, function_values, save_checkpoint: bool = True):
        self.cmaes.tell(solutions, -np.asarray(function_values))

        self.full_fitness.append(function_values)
        self.full_x.append(solutions)
        self.f = function_values
        self.x = solutions

        if np.max(function_values) > self.f_best_so_far:
            best_index = int(np.argmax(function_values))
            self.f_best_so_far = function_values[best_index]
            self.x_best_so_far = solutions[best_index]

        print(
            f"Generation {self.current_gen}:\tbest={self.f_best_so_far}\n"
            f"Mean fitness:\t{np.mean(self.f)} +- {np.std(self.f)}\n"
        )

        if save_checkpoint:
            self.save_checkpoint()
        self.current_gen += 1

    def initialise_x0(self, num_parameters):
        return np.random.uniform(low=self.min, high=self.max, size=num_parameters)

    def save_checkpoint(self):
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))
        os.makedirs(curr_gen_path, exist_ok=True)
        np.save(os.path.join(self.directory_name, "full_f"), np.array(self.full_fitness, dtype=object), allow_pickle=True)
        np.save(os.path.join(self.directory_name, "full_x"), np.array(self.full_x, dtype=object), allow_pickle=True)
        np.save(os.path.join(curr_gen_path, "f_best"), np.array(self.f_best_so_far))
        if self.x_best_so_far is not None:
            np.save(os.path.join(curr_gen_path, "x_best"), np.array(self.x_best_so_far))
        np.save(os.path.join(curr_gen_path, "x"), np.array(self.x))
        np.save(os.path.join(curr_gen_path, "f"), np.array(self.f))

    def load_checkpoint(self):
        dir_path = _search_file_list(self.directory_name, "f_best.npy")
        assert len(dir_path) > 0, "No checkpoint files found, check the directory_name!"

        self.current_gen = int(dir_path[-1].split("/")[-2])
        curr_gen_path = os.path.join(self.directory_name, str(self.current_gen))
        print(f"Loading from: {curr_gen_path}")
        self.full_fitness = list(np.load(os.path.join(self.directory_name, "full_f.npy"), allow_pickle=True))
        self.full_x = list(np.load(os.path.join(self.directory_name, "full_x.npy"), allow_pickle=True))
        self.f_best_so_far = np.load(os.path.join(curr_gen_path, "f_best.npy"))
        self.x_best_so_far = np.load(os.path.join(curr_gen_path, "x_best.npy"))
        self.x = np.load(os.path.join(curr_gen_path, "x.npy"))
        self.f = np.load(os.path.join(curr_gen_path, "f.npy"))

        self.cmaes = self.load_cmaes()
        for x, f in zip(self.full_x, self.full_fitness):
            self.cmaes.tell(x, -np.asarray(f))
