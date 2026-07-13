# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-drone hook-grasping task: swing a weighted wire + hook onto a basket wall bar.

Building block for the multi-agent attach->transport->release pipeline (see
``agent/hook-based-grasping/workplan.md``).
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-flycrane-hook-grasp-v0",
    entry_point=f"{__name__}.marl_hook_env:MARLHookGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.marl_hook_env_cfg:MARLHookGraspEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
