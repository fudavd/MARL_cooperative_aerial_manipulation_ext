# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hebbian / CMA-ES learning for the flycrane DirectMARL tasks.

A dependency-light implementation (only ``torch``/``numpy``/``gymnasium``/``cma``) of
evolved Hebbian plasticity: the plasticity rule (A/B/C/D coefficients) is evolved with
CMA-ES while the network weights adapt online, during the episode, via that rule.

* :mod:`hebbian_core` -- the framework-agnostic pieces: the online-plastic
  :class:`HebbianActor` and the :class:`CMAES` optimizer wrapper.
* :mod:`adapter` -- the bridge to a GPU-vectorized IsaacLab :class:`DirectMARLEnv`
  (:class:`HebbianTrainer` for training, :class:`HebbianPolicy` for evaluation).
"""

from .hebbian_core import CMAES, HebbianActor

__all__ = ["CMAES", "HebbianActor"]
