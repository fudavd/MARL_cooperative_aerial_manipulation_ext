# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OpenAI-ES (Salimans et al., 2017) with a CMAES-compatible ask()/tell() interface.

A single search point ``theta`` is perturbed with isotropic Gaussian noise (antithetic pairs).
The expected-fitness gradient is estimated from rank-normalized returns and applied with Adam.
O(popsize * n), no covariance -- ideal for massively-parallel, noisy, high-dim gradient-free
optimization (the standard large-scale neuroevolution method). Maximizes reward.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

__all__ = ["OpenAIESOptimizer"]


def _centered_ranks(x: np.ndarray) -> np.ndarray:
    """Rank-transform to [-0.5, 0.5] (OpenAI-ES fitness shaping)."""
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[np.argsort(x)] = np.arange(len(x), dtype=np.float64)
    if len(x) > 1:
        ranks /= (len(x) - 1)
    return ranks - 0.5


class OpenAIESOptimizer:
    def __init__(self, popsize: int, n_params: int, opts: Dict, output_dir: str = "./results/openai_es",
                 sigma: float = 0.1, lr: float = 0.02, l2: float = 0.005, seed: Optional[int] = None):
        # antithetic sampling needs an even population
        self.popsize = popsize if popsize % 2 == 0 else popsize + 1
        self.n = n_params
        self.min = opts["min"]
        self.max = opts["max"]
        self.sigma = sigma
        self.lr = lr
        self.l2 = l2

        self.rng = np.random.RandomState(seed)
        self.theta = np.zeros(n_params, dtype=np.float64)  # box-centered start

        # Adam state
        self.m = np.zeros(n_params, dtype=np.float64)
        self.v = np.zeros(n_params, dtype=np.float64)
        self.beta1, self.beta2, self.eps, self.t = 0.9, 0.999, 1e-8, 0

        self.directory_name = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.current_gen = 0
        self.f_best_so_far = -np.inf
        self.x_best_so_far: Optional[np.ndarray] = None
        self.full_fitness = []
        self._eps: Optional[np.ndarray] = None

    def ask(self) -> np.ndarray:
        half = self.popsize // 2
        e = self.rng.randn(half, self.n)
        self._eps = np.concatenate([e, -e], axis=0)  # antithetic pairs
        z = self.theta[None, :] + self.sigma * self._eps
        return np.clip(z, self.min, self.max)

    def tell(self, solutions, function_values, save_checkpoint: bool = True):
        solutions = np.asarray(solutions)
        f = np.asarray(function_values, dtype=np.float64)
        u = _centered_ranks(f)  # [-0.5, 0.5], higher reward -> higher weight

        # estimated gradient of expected reward w.r.t. theta
        grad = (u[:, None] * self._eps).sum(axis=0) / (self.popsize * self.sigma)
        grad = grad - self.l2 * self.theta  # weight decay

        # Adam ascent (maximize)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        mhat = self.m / (1 - self.beta1 ** self.t)
        vhat = self.v / (1 - self.beta2 ** self.t)
        self.theta = np.clip(self.theta + self.lr * mhat / (np.sqrt(vhat) + self.eps), self.min, self.max)

        bi = int(np.argmax(f))
        if f[bi] > self.f_best_so_far:
            self.f_best_so_far = float(f[bi])
            self.x_best_so_far = solutions[bi].copy()

        self.full_fitness.append(f.tolist())
        print(
            f"Generation {self.current_gen}:\tbest={self.f_best_so_far}\n"
            f"Pop reward mean:\t{float(np.mean(f))} +- {float(np.std(f))}\n"
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
        np.save(os.path.join(g, "theta"), self.theta)
        np.save(os.path.join(self.directory_name, "full_f"), np.array(self.full_fitness, dtype=object),
                allow_pickle=True)
