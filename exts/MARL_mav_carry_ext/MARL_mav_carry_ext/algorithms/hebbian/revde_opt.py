# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ask()/tell() wrapper around the RevDE differential-evolution optimizer.

Uses the upstream implementation connected at ``thirdparty/revde``
(https://github.com/fudavd/revde, Tomczak et al., "Differential Evolution with Reversible Linear
Transformations", GECCO 2020). Exposes the same ``ask()``/``tell()``/``f_best_so_far``/
``x_best_so_far``/``current_gen`` surface as :class:`CMAES`, so :class:`HebbianTrainer` can switch
optimizers with a single config flag.

RevDE *minimizes*; our fitness is episode return (to maximize), so we negate internally.
Generation 0 evaluates the random initial population (``popsize`` candidates); each later
generation evaluates the recombination offspring (``3*popsize`` for ``revde``/``ade``/``dex3``,
``popsize`` for plain ``de``) and selects the best ``popsize`` from offspring + current population.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import numpy as np

__all__ = ["RevDEOptimizer"]


def _ensure_revde_importable() -> bool:
    """Put the repo's ``thirdparty`` dir on sys.path so ``import revde`` resolves (the package
    uses relative imports, so it must be imported under the name ``revde``)."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "thirdparty", "revde")):
            tp = os.path.join(d, "thirdparty")
            if tp not in sys.path:
                sys.path.insert(0, tp)
            return True
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return False


_ensure_revde_importable()
from revde.algorithms.recombination import DifferentialRecombination  # noqa: E402
from revde.algorithms.selections import SelectBest  # noqa: E402


class RevDEOptimizer:
    """Differential-evolution optimizer with a CMAES-compatible ask()/tell() interface.

    :param popsize: population size kept between generations.
    :param n_params: genome dimensionality.
    :param opts: dict with ``min``/``max`` (coefficient bounds); other keys ignored.
    :param de_type: one of ``de``/``ade``/``revde``/``dex3``.
    :param F: differential weight (must be in [0, 2]).
    :param CR: crossover rate (must be in (0, 1]).
    """

    def __init__(self, popsize: int, n_params: int, opts: Dict, output_dir: str = "./results/revde",
                 de_type: str = "revde", F: float = 0.5, CR: float = 0.9, seed: Optional[int] = None):
        self.popsize = popsize
        self.n_params = n_params
        self.min = opts["min"]
        self.max = opts["max"]
        self.de_type = de_type
        self.directory_name = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # recombination uses the global numpy RNG (permutations, bernoulli) -> seed it for repro
        if seed is not None:
            np.random.seed(int(seed))
        rng = np.random.RandomState(seed)
        self.x = rng.uniform(self.min, self.max, size=(popsize, n_params))
        self.f: Optional[np.ndarray] = None  # minimized objective (= -reward); None until gen 0 told

        self.differential = DifferentialRecombination(type=de_type, bounds=(self.min, self.max),
                                                      params={"F": F, "CR": CR})
        self.selection = SelectBest()

        self._pending: Optional[np.ndarray] = None
        self._mode: Optional[str] = None
        self.current_gen = 0
        self.f_best_so_far = -np.inf   # in reward terms (maximize)
        self.x_best_so_far: Optional[np.ndarray] = None
        self.full_fitness = []

    def ask(self) -> np.ndarray:
        if self.f is None:
            self._pending = self.x.copy()
            self._mode = "init"
        else:
            out, _ = self.differential.recombination(self.x)
            x_new = np.asarray(out) if self.de_type == "de" else np.concatenate(out, 0)
            self._pending = np.clip(x_new, self.min, self.max)
            self._mode = "step"
        return self._pending

    def tell(self, solutions, function_values, save_checkpoint: bool = True):
        solutions = np.asarray(solutions)
        neg = -np.asarray(function_values, dtype=np.float64)  # RevDE minimizes -> negate reward
        if self._mode == "init":
            self.x = solutions
            self.f = neg
        else:
            x_all = np.concatenate((solutions, self.x), 0)
            f_all = np.concatenate((neg, self.f))
            self.x, self.f = self.selection.select(x_all, f_all, population_size=self.popsize)

        best_idx = int(np.argmin(self.f))
        best_reward = float(-self.f[best_idx])
        if best_reward > self.f_best_so_far:
            self.f_best_so_far = best_reward
            self.x_best_so_far = self.x[best_idx].copy()

        self.full_fitness.append((-self.f).tolist())
        print(
            f"Generation {self.current_gen}:\tbest={self.f_best_so_far}\n"
            f"Pop reward mean:\t{float(np.mean(-self.f))} +- {float(np.std(-self.f))}\n"
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
        np.save(os.path.join(g, "x"), np.array(self.x))
        np.save(os.path.join(g, "f"), np.array(-self.f))  # stored as reward
        np.save(os.path.join(self.directory_name, "full_f"), np.array(self.full_fitness, dtype=object),
                allow_pickle=True)
