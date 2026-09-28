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

# Clean, confound-free directed-swing CONTROLLABILITY test env (same plant, no basket/catch/curriculum).
# See marl_swing_env.py / marl_swing_env_cfg.py. Reuses the rsl_rl adapter's true_hook / hook_observer arms.
gym.register(
    id="Isaac-flycrane-swing-test-v0",
    entry_point=f"{__name__}.marl_swing_env:MARLSwingTestEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.marl_swing_env_cfg:MARLSwingTestEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)
