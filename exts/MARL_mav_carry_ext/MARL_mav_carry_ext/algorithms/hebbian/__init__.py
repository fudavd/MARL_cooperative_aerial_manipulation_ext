# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hebbian / CMA-ES learning for the flycrane DirectMARL tasks.

This is a thin, dependency-light (only ``torch``/``numpy``/``gymnasium``/``cma``) vendoring of the
algorithm core from the ``hebbian-marl`` project (symlinked at ``thirdparty/hebbian_marl``).

Only the two framework-agnostic pieces are vendored here -- the online-plastic
:class:`HebbianActor` and the :class:`CMAES` optimizer -- so that the heavy ``agilerl`` /
``pettingzoo`` dependency stack of the upstream project does not have to be installed into the
Isaac Sim virtual environment (which would risk moving ``torch``/``gymnasium`` away from the
versions Isaac Sim 4.5 is built against).

Upstream source of truth: ``thirdparty/hebbian_marl/src/hebbian.py`` and
``thirdparty/hebbian_marl/src/CMAES.py``.
"""

from .hebbian_core import CMAES, HebbianActor

__all__ = ["CMAES", "HebbianActor"]
