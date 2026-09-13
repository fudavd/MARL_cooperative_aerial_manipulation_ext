# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-drone hook-grasp task: swing a weighted wire + hook onto a basket wall bar and hold it taut.

Trained with PPO (rsl_rl) through ``scripts/hook_grasp/train.py``.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-flycrane-hook-grasp-v0",
    entry_point=f"{__name__}.marl_hook_env:MARLHookGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.marl_hook_env_cfg:MARLHookGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:hook_grasp_ppo_runner_cfg",
    },
)
