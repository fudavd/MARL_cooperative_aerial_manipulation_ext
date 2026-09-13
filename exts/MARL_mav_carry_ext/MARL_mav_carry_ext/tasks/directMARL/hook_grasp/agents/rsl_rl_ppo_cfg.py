# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO (rsl_rl) runner configuration for the hook-grasp task.

Returned as the plain dict ``rsl_rl.runners.OnPolicyRunner`` consumes. Asymmetric actor-critic: the actor sees the
153-dim hook-blind observation, the critic the 180-dim privileged state (``extras["observations"]["critic"]``).
Both networks are plain MLPs; the 3-frame observation history provides the short-term memory the swing needs.
"""


def hook_grasp_ppo_runner_cfg() -> dict:
    return {
        "seed": 42,
        "device": "cuda:0",
        "num_steps_per_env": 24,
        "max_iterations": 20000,
        "empirical_normalization": True,
        "save_interval": 100,
        "experiment_name": "hook_grasp",
        "run_name": "hook_grasp",
        "logger": "tensorboard",
        "wandb_project": "hook-grasp",
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.004,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",   # adaptive-KL learning rate
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }
