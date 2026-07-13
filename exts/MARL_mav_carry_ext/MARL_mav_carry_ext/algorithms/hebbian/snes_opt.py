# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Separable Natural Evolution Strategy (SNES) with a CMAES-compatible ask()/tell() interface.

SNES (Schaul, Glasmachers & Schmidhuber, 2011) maintains a Gaussian search distribution with a
mean and *per-dimension* std (no covariance matrix), updated along the natural gradient using
rank-based fitness utilities. Cost is O(popsize * n) -- it scales to high-dim genomes and large
populations as cheaply as DE, and the rank-based update makes it robust to the noisy fitness of
plastic-network evaluation. This is the recommended optimizer for the massively-parallel, noisy,
gradient-free regime here (cf. OpenAI-ES / neuroevolution practice).

Maximizes reward (we negate nothing -- utilities are assigned by descending fitness).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

__all__ = ["SNESOptimizer"]


class SNESOptimizer:
    def __init__(self, popsize: int, n_params: int, opts: Dict, output_dir: str = "./results/snes",
                 eta_mu: float = 1.0, eta_sigma: Optional[float] = None, sigma_init: Optional[float] = None,
                 seed: Optional[int] = None):
        self.popsize = popsize
        self.n = n_params
        self.min = opts["min"]
        self.max = opts["max"]
        self.sigma_init = sigma_init if sigma_init is not None else opts.get("mutation_sigma", 0.3)

        self.rng = np.random.RandomState(seed)
        self.mu = np.zeros(n_params, dtype=np.float64)            # distribution center (box-centered)
        self.sigma = np.full(n_params, float(self.sigma_init), dtype=np.float64)  # per-dim std
        self.eta_mu = eta_mu
        # standard SNES std learning rate (independent of popsize)
        self.eta_sigma = eta_sigma if eta_sigma is not None else (3.0 + np.log(n_params)) / (5.0 * np.sqrt(n_params))

        self.utilities = self._utilities(popsize)  # index 0 = best
        self.directory_name = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.current_gen = 0
        self.f_best_so_far = -np.inf   # reward (maximize)
        self.x_best_so_far: Optional[np.ndarray] = None
        self.full_fitness = []
        self._s: Optional[np.ndarray] = None  # noise that generated the last asked population

    @staticmethod
    def _utilities(lam: int) -> np.ndarray:
        ranks = np.arange(1, lam + 1)
        u = np.maximum(0.0, np.log(lam / 2.0 + 1.0) - np.log(ranks))
        s = u.sum()
        if s > 0:
            u = u / s
        return u - 1.0 / lam

    def ask(self) -> np.ndarray:
        self._s = self.rng.randn(self.popsize, self.n)
        z = self.mu[None, :] + self.sigma[None, :] * self._s
        return np.clip(z, self.min, self.max)

    def tell(self, solutions, function_values, save_checkpoint: bool = True):
        solutions = np.asarray(solutions)
        f = np.asarray(function_values, dtype=np.float64)
        order = np.argsort(-f)                       # descending: best first
        u = np.zeros(self.popsize, dtype=np.float64)
        u[order] = self.utilities                     # best-ranked sample gets the top utility

        grad_mu = (u[:, None] * self._s).sum(axis=0)
        grad_sigma = (u[:, None] * (self._s ** 2 - 1.0)).sum(axis=0)

        self.mu = np.clip(self.mu + self.eta_mu * self.sigma * grad_mu, self.min, self.max)
        self.sigma = self.sigma * np.exp(0.5 * self.eta_sigma * grad_sigma)

        bi = int(np.argmax(f))
        if f[bi] > self.f_best_so_far:
            self.f_best_so_far = float(f[bi])
            self.x_best_so_far = solutions[bi].copy()

        self.full_fitness.append(f.tolist())
        print(
            f"Generation {self.current_gen}:\tbest={self.f_best_so_far}\n"
            f"Pop reward mean:\t{float(np.mean(f))} +- {float(np.std(f))} | sigma~{float(np.mean(self.sigma)):.4f}\n"
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
        np.save(os.path.join(g, "mu"), self.mu)
        np.save(os.path.join(g, "sigma"), self.sigma)
        np.save(os.path.join(self.directory_name, "full_f"), np.array(self.full_fitness, dtype=object),
                allow_pickle=True)
