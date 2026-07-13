# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PGPE (Policy Gradients with Parameter-based Exploration) + Adam, via EvoTorch.

Built on EvoTorch's ``SymmetricSeparableGaussian`` -- the exact distribution and rank-based
parameter-gradient computation used by ``evotorch.algorithms.PGPE`` -- driven in an ask()/tell()
loop so it matches the CMAES/RevDE/SNES/OpenAI-ES interface. The distribution center (mu) is
updated with **Adam** (this is the ``optimizer="adam"`` variant of PGPE); the per-dimension std
(sigma) follows its rank gradient with a learning rate and a relative max-change clip.

Maximizes reward. Distribution/sampling run on CPU (tiny relative to the sim) to avoid GPU
contention with the Isaac Sim rollouts.

Ref: Sehnke et al., "Parameter-exploring policy gradients" (2010); EvoTorch (NNAISENSE).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch

from evotorch.distributions import SymmetricSeparableGaussian

__all__ = ["PGPEOptimizer"]


class PGPEOptimizer:
    def __init__(self, popsize: int, n_params: int, opts: Dict, output_dir: str = "./results/pgpe",
                 center_lr: float = 0.15, stdev_lr: float = 0.1, stdev_init: Optional[float] = None,
                 stdev_max_change: float = 0.2, ranking_method: str = "centered",
                 seed: Optional[int] = None):
        self.popsize = popsize if popsize % 2 == 0 else popsize + 1  # symmetric sampling needs even
        self.n = n_params
        self.min = opts["min"]
        self.max = opts["max"]
        self.stdev_init = float(stdev_init if stdev_init is not None else opts.get("mutation_sigma", 0.3))
        self.center_lr = center_lr
        self.stdev_lr = stdev_lr
        self.stdev_max_change = stdev_max_change
        self.ranking_method = ranking_method

        self.dtype = torch.float32
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(int(seed))

        self.mu = torch.zeros(n_params, dtype=self.dtype)              # box-centered start
        self.sigma = torch.full((n_params,), self.stdev_init, dtype=self.dtype)
        self._build_dist()

        # Adam state for the center update
        self.m = torch.zeros(n_params, dtype=self.dtype)
        self.v = torch.zeros(n_params, dtype=self.dtype)
        self.beta1, self.beta2, self.eps, self.t = 0.9, 0.999, 1e-8, 0

        self.directory_name = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.current_gen = 0
        self.f_best_so_far = -np.inf
        self.x_best_so_far: Optional[np.ndarray] = None
        self.full_fitness = []
        self._samples: Optional[torch.Tensor] = None

    def _build_dist(self):
        self.dist = SymmetricSeparableGaussian(
            parameters={"mu": self.mu.clone(), "sigma": self.sigma.clone()},
            solution_length=self.n, dtype=self.dtype,
        )

    def ask(self) -> np.ndarray:
        self._samples = self.dist.sample(self.popsize, generator=self.gen)
        return np.clip(self._samples.cpu().numpy(), self.min, self.max)

    def tell(self, solutions, function_values, save_checkpoint: bool = True):
        f = torch.as_tensor(np.asarray(function_values), dtype=self.dtype)
        grads = self.dist.compute_gradients(self._samples, f, objective_sense="max",
                                            ranking_method=self.ranking_method)
        gmu, gsigma = grads["mu"], grads["sigma"]

        # Adam ascent on the center
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gmu
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gmu ** 2)
        mhat = self.m / (1 - self.beta1 ** self.t)
        vhat = self.v / (1 - self.beta2 ** self.t)
        self.mu = torch.clamp(self.mu + self.center_lr * mhat / (torch.sqrt(vhat) + self.eps),
                              self.min, self.max)

        # std follows its rank gradient with a relative max-change clip
        step = self.stdev_lr * gsigma
        max_change = self.stdev_max_change * self.sigma
        step = torch.clamp(step, -max_change, max_change)
        self.sigma = torch.clamp(self.sigma + step, min=1e-6)

        self._build_dist()

        fr = np.asarray(function_values)
        sols = np.asarray(solutions)
        bi = int(np.argmax(fr))
        if fr[bi] > self.f_best_so_far:
            self.f_best_so_far = float(fr[bi])
            self.x_best_so_far = sols[bi].copy()

        self.full_fitness.append(fr.tolist())
        print(
            f"Generation {self.current_gen}:\tbest={self.f_best_so_far}\n"
            f"Pop reward mean:\t{float(np.mean(fr))} +- {float(np.std(fr))} | sigma~{float(self.sigma.mean()):.4f}\n"
        )
        self.current_gen += 1
        if save_checkpoint:
            self.save_checkpoint()

    def save_checkpoint(self):
        g = os.path.join(self.directory_name, str(self.current_gen))
        os.makedirs(g, exist_ok=True)
        np.save(os.path.join(g, "f_best"), np.array(self.f_best_so_far))
        if self.x_best_so_far is not None:
            np.save(os.path.join(g, "x_best"), np.array(self.x_best_so_far))
        np.save(os.path.join(g, "mu"), self.mu.cpu().numpy())
        np.save(os.path.join(g, "sigma"), self.sigma.cpu().numpy())
        np.save(os.path.join(self.directory_name, "full_f"), np.array(self.full_fitness, dtype=object),
                allow_pickle=True)
