# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter: the single-agent ``DirectMARLEnv`` hook-grasp task -> an rsl_rl ``VecEnv`` with a privileged critic.

``OnPolicyRunner`` wants a single-agent ``VecEnv`` whose ``get_observations()`` / ``step()`` return the ACTOR
observation tensor and expose the CRITIC observation under ``extras["observations"]["critic"]``. The task is a
``DirectMARLEnv`` with exactly one agent (``falcon1``), so this shim only unwraps the per-agent dicts:

    * actor obs  = env.observation_dict["falcon1"]  -> 153-dim (hook-blind, 3 frames)
    * critic obs = env.state()                      -> 180-dim (privileged, 3 frames)

Nothing about the task (physics, rewards, observations) is touched here.
"""

from __future__ import annotations

import torch

from rsl_rl.env import VecEnv

from isaaclab.envs import DirectMARLEnv


class HookGraspVecEnv(VecEnv):
    """Expose the single-agent DirectMARLEnv to rsl_rl with an asymmetric (privileged) critic observation."""

    # Everything entering rsl_rl is bounded here: a rare PhysX wire-tension blow-up can emit huge-but-finite
    # values whose value-target explosion would NaN the policy std. Bounds sit far outside the legitimate ranges.
    _OBS_BOUND = 1000.0
    _REW_BOUND = 200.0

    def __init__(self, env: DirectMARLEnv):
        assert isinstance(env, DirectMARLEnv), "HookGraspVecEnv expects a DirectMARLEnv."
        self.env = env
        self.agents = list(env.possible_agents)
        assert len(self.agents) == 1, f"single-agent adapter; got possible_agents={self.agents}"
        self.agent = self.agents[0]

        # rsl_rl VecEnv contract
        self.num_envs = int(env.num_envs)
        self.num_actions = int(env.cfg.action_dim)
        self.device = env.device
        self.cfg = env.cfg
        self.max_episode_length = env.max_episode_length

        # prime the env so get_observations() is valid before the first step (the runner reads it at init)
        obs, self._extras = self._reset_env()
        self._obs = self._sanitize(obs)

        actor_dim = int(self._obs.shape[-1])
        critic_dim = int(self._extras["observations"]["critic"].shape[-1])
        expected_actor = int(next(iter(env.cfg.observation_spaces.values())))
        expected_critic = int(env.cfg.state_space)
        assert actor_dim == expected_actor, f"actor obs {actor_dim} != declared observation_space {expected_actor}"
        assert critic_dim == expected_critic, f"critic obs {critic_dim} != declared state_space {expected_critic}"
        print(f"[HookGraspVecEnv] agent={self.agent}  actor_obs={actor_dim}  critic_obs={critic_dim}  "
              f"num_envs={self.num_envs}  num_actions={self.num_actions}")

    # episode_length_buf must be a live proxy: rsl_rl overwrites it (init_at_random_ep_len)
    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    # ------------------------------------------------------------------ helpers
    def _sanitize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-self._OBS_BOUND, self._OBS_BOUND)

    def _build_extras(self, time_outs: torch.Tensor, log: dict | None = None) -> dict:
        extras = {
            "observations": {"critic": self._sanitize(self.env.state())},
            "time_outs": time_outs.to(self.device).float(),
        }
        if log:
            extras["log"] = log
        return extras

    def _reset_env(self):
        obs_dict, _ = self.env.reset()
        time_outs = torch.zeros(self.num_envs, device=self.device)
        return obs_dict[self.agent], self._build_extras(time_outs)

    # ------------------------------------------------------------------ VecEnv API
    def get_observations(self):
        return self._obs, self._extras

    def reset(self):
        obs, self._extras = self._reset_env()
        self._obs = self._sanitize(obs)
        return self._obs, self._extras

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.env.device)
        obs_dict, rew_dict, term_dict, tout_dict, _ = self.env.step({self.agent: actions})

        obs = self._sanitize(obs_dict[self.agent])
        rewards = rew_dict[self.agent].view(self.num_envs)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-self._REW_BOUND, self._REW_BOUND)
        terminated = term_dict[self.agent].view(self.num_envs)
        time_outs = tout_dict[self.agent].view(self.num_envs)
        dones = (terminated | time_outs).float()

        base = self.env
        log = {
            "Metrics/attached_frac": base.attached.any(dim=-1).float().mean(),
            "Metrics/newly_caught": base._newly_caught.any(dim=-1).float().mean(),
            "Metrics/capture_radius": torch.tensor(float(base._acfg.capture_radius), device=self.device),
            "Metrics/hold_req": torch.tensor(float(base._current_hold_req), device=self.device),
        }
        self._obs, self._extras = obs, self._build_extras(time_outs, log)
        return self._obs, rewards, dones, self._extras
