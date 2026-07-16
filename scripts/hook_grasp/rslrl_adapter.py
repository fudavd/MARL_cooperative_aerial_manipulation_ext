# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin adapter: DirectMARLEnv (num_drones=1 hook-grasp) -> rsl_rl single-agent asymmetric VecEnv.

rsl_rl's ``OnPolicyRunner`` wants a single-agent ``VecEnv`` whose ``get_observations()`` /
``step()`` return the ACTOR observation tensor and expose the PRIVILEGED critic observation under
``extras["observations"]["critic"]``. Our env is a multi-agent ``DirectMARLEnv`` with exactly one
agent (``falcon1``). This adapter unwraps the single-agent dicts to plain tensors:

    * actor obs   = env.observation_dict['falcon1']  -> 135-dim (45 x 3 stacked, HOOK-FREE)
    * critic obs  = env.state()                      -> 180-dim (60 x 3, PRIVILEGED)

Nothing about the env (task, physics, rewards, obs/state construction) is touched -- this is a pure
interface shim. The asymmetry (135 actor / 180 critic) replaces the skrl MAPPO/CTDE machinery: the
recurrent actor never sees the hook; the critic does, plus tension / attached / hold_progress.
"""

from __future__ import annotations

import torch

from rsl_rl.env import VecEnv
from rsl_rl.modules.normalizer import EmpiricalDiscountedVariationNormalization

from isaaclab.envs import DirectMARLEnv

ACTOR_OBS_DIM = 135   # default (45 x history_len=3, hook-free); overridden when the INDI-residual obs is enabled
CRITIC_OBS_DIM = 180
INDI_RESIDUAL_DIM = 6  # body-frame INDI disturbance residual appended per actor frame when enabled


class RslRlMarlAdapter(VecEnv):
    """Expose a single-agent DirectMARLEnv to rsl_rl with an asymmetric (privileged) critic obs."""

    def __init__(self, env: DirectMARLEnv, *, reward_norm: bool = False, gamma: float = 0.99):
        assert isinstance(env, DirectMARLEnv), "RslRlMarlAdapter expects a DirectMARLEnv."
        self.env = env
        # Optional Pathak reward normalization: divide the per-step reward by a running std of the DISCOUNTED
        # return, so the value function sees a ~stationary return SCALE even as the curriculum shifts reward
        # reachability (capture radius tightening, hold_req ramping, attached-spawn decaying). Addresses the
        # "value function chases a moving return scale at each curriculum level shift" failure. Training-only;
        # eval/play never train the critic so they leave this off (the policy net is unaffected).
        self._rew_norm = None
        if reward_norm:
            self._rew_norm = EmpiricalDiscountedVariationNormalization(shape=[], gamma=gamma).to(env.device)
            self._rew_norm.train()
        self.agents = list(env.possible_agents)
        assert len(self.agents) == 1, f"adapter is single-agent; got possible_agents={self.agents}"
        self.agent = self.agents[0]

        # rsl_rl VecEnv contract
        self.num_envs = int(env.num_envs)
        self.num_actions = int(env.cfg.action_dim)
        self.device = env.device
        self.cfg = env.cfg
        self.max_episode_length = env.max_episode_length

        # prime the env so get_observations() is valid before the first step (runner reads it at init)
        obs, self._extras = self._reset_env()

        # hard asymmetry assertions -- the whole point of this backend. The actor dim is whatever the env cfg
        # declares (default 135 hook-free; larger when the INDI-residual obs / a different actor H is enabled),
        # so assert the live obs matches the DECLARED observation space rather than a fixed constant. The critic
        # stays privileged 180-dim regardless (its history_len is untouched by the actor-side knobs).
        actor_dim = int(obs.shape[-1])
        critic_dim = int(self._extras["observations"]["critic"].shape[-1])
        expected_actor = int(next(iter(env.cfg.observation_spaces.values())))
        assert actor_dim == expected_actor, (
            f"actor obs {actor_dim} != declared observation_space {expected_actor} "
            f"(indi_residual={getattr(env.cfg, 'actor_indi_residual', False)}, "
            f"actor_history_len={getattr(env.cfg, 'actor_history_len', 0) or env.cfg.history_len})"
        )
        assert critic_dim == CRITIC_OBS_DIM, f"critic obs must be {CRITIC_OBS_DIM}-dim privileged, got {critic_dim}"
        print(f"[rslrl-adapter] agent={self.agent}  actor_obs={actor_dim}  critic_obs={critic_dim}  "
              f"num_envs={self.num_envs}  num_actions={self.num_actions}")
        self._obs = obs

    # -- episode_length_buf must be a live proxy: rsl_rl may overwrite it (init_at_random_ep_len) --
    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    # ------------------------------------------------------------------ helpers
    def _build_extras(self, marl_extras: dict, time_outs: torch.Tensor, log: dict | None = None) -> dict:
        critic = self.env.state()  # (num_envs, 180) privileged frame-stacked state
        extras = {
            "observations": {"critic": critic},
            "time_outs": time_outs.to(self.device).float(),
        }
        if log:
            extras["log"] = log
        return extras

    def _reset_env(self):
        obs_dict, marl_extras = self.env.reset()
        obs = obs_dict[self.agent]
        time_outs = torch.zeros(self.num_envs, device=self.device)
        return obs, self._build_extras(marl_extras, time_outs)

    # ------------------------------------------------------------------ VecEnv API
    def get_observations(self):
        return self._obs, self._extras

    def reset(self):
        self._obs, self._extras = self._reset_env()
        return self._obs, self._extras

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.env.device)
        obs_dict, rew_dict, term_dict, tout_dict, marl_extras = self.env.step({self.agent: actions})

        obs = obs_dict[self.agent]
        rewards = rew_dict[self.agent].view(self.num_envs)
        if self._rew_norm is not None:
            rewards = self._rew_norm(rewards)  # scale-only: keeps the return scale stationary across curriculum shifts
        terminated = term_dict[self.agent].view(self.num_envs)
        time_outs = tout_dict[self.agent].view(self.num_envs)
        dones = (terminated | time_outs).float()

        # lightweight logging (per-step means; averaged over the iteration by the runner)
        base = self.env
        log = {
            "Metrics/attached_frac": base.attached.any(dim=-1).float().mean(),
            "Metrics/newly_caught": base._newly_caught.any(dim=-1).float().mean(),
            "Metrics/capture_radius": torch.tensor(float(base._acfg.capture_radius), device=self.device),
            "Metrics/hold_req": torch.tensor(float(base._current_hold_req), device=self.device),
        }
        extras = self._build_extras(marl_extras, time_outs, log)

        self._obs, self._extras = obs, extras
        return obs, rewards, dones, extras


def make_runner_cfg(
    *,
    experiment_name: str,
    max_iterations: int,
    num_steps_per_env: int = 24,
    seed: int = 42,
    device: str = "cuda:0",
    logger: str = "tensorboard",
    wandb_project: str = "hook-grasp-rnn",
    save_interval: int = 100,
    policy_overrides: dict | None = None,
    algo_overrides: dict | None = None,
) -> dict:
    """rsl_rl OnPolicyRunner train_cfg: ActorCriticRecurrent (LSTM) + PPO with adaptive-KL LR.

    Built as a plain dict (rather than the Isaac-Lab configclasses) so the recurrent RNN kwargs pass
    straight through to ``ActorCriticRecurrent.__init__`` without dataclass-field friction.

    ``policy_overrides`` / ``algo_overrides`` merge into the policy / algorithm sub-dicts (the overnight
    search tunes architecture + PPO HP here). ``class_name`` cannot be overridden (guarded).
    """
    cfg = {
        "seed": seed,
        "device": device,
        "num_steps_per_env": num_steps_per_env,
        "max_iterations": max_iterations,
        "empirical_normalization": True,
        "save_interval": save_interval,
        "experiment_name": experiment_name,
        "run_name": experiment_name,
        "logger": logger,
        "wandb_project": wandb_project,
        "policy": {
            "class_name": "ActorCriticRecurrent",
            "init_noise_std": 1.0,
            "actor_hidden_dims": [256, 256],
            "critic_hidden_dims": [256, 256],
            "activation": "elu",
            "rnn_type": "lstm",
            "rnn_hidden_dim": 256,
            "rnn_num_layers": 1,
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
            "schedule": "adaptive",   # adaptive-KL LR (rsl_rl native) -> replaces skrl kl_threshold hack
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }
    if policy_overrides:
        assert "class_name" not in policy_overrides, "cannot override policy class_name (must stay ActorCriticRecurrent)"
        cfg["policy"].update(policy_overrides)
        print(f"[make_runner_cfg] policy overrides: {policy_overrides}")
    if algo_overrides:
        assert "class_name" not in algo_overrides, "cannot override algorithm class_name (must stay PPO)"
        cfg["algorithm"].update(algo_overrides)
        print(f"[make_runner_cfg] algorithm overrides: {algo_overrides}")
    return cfg


def apply_f3_env_cfg(env_cfg) -> None:
    """Apply the winning 'F3' recipe (env-level, backend-independent) in place.

    These are the same knobs the skrl arms pass as ``env.*`` overrides; centralised here so the rsl_rl
    train script matches the overnight best recipe. Curricula are left ON (defaults) for training.
    """
    env_cfg.attachment.capture_radius = 0.13
    env_cfg.attachment.upsweep_v_min = 0.08
    env_cfg.capture_radius_start = 0.25
    env_cfg.capture_curriculum_steps = 100000
    env_cfg.hold_success_steps = 100
    env_cfg.hold_success_steps_start = 30
    env_cfg.hold_curriculum_steps = 85000
    env_cfg.attached_spawn_steps = 65000
    env_cfg.attached_spawn_v0 = 0.4
    env_cfg.wall_pull_weight = 0.0
    env_cfg.alive_weight = 1.0
    env_cfg.proximity_weight = 2.0
    env_cfg.spawn_dist_mu = 0.35
    env_cfg.swing_assist_v0 = 2.2
    env_cfg.impact_toward_weight = 6.0
    env_cfg.anti_sag_weight = 6.0
    env_cfg.slack_penalty_weight = 20.0


def apply_indi_residual_cfg(env_cfg, *, actor_history_len: int, residual: bool = True) -> None:
    """Enable the INDI-residual actor obs and set the ACTOR-only frame-stack depth H, recomputing dims.

    The actor obs becomes ``(45 + 6*residual) * actor_history_len`` (45 hook-free base + the 6-dim body-frame
    INDI disturbance residual per frame). The PRIVILEGED critic is left untouched: it keeps ``history_len`` (3)
    frames -> ``state_space`` stays 180-dim, so only the actor input changes (a clean ablation). Call AFTER
    ``apply_f3_env_cfg`` and BEFORE ``gym.make`` (DirectMARLEnv reads observation_spaces at construction).
    """
    per_frame = 45 + (INDI_RESIDUAL_DIM if residual else 0)
    env_cfg.actor_indi_residual = bool(residual)
    env_cfg.actor_history_len = int(actor_history_len)
    env_cfg.observation_spaces = {a: per_frame * int(actor_history_len) for a in env_cfg.possible_agents}
    # critic (state_space) is deliberately NOT changed here -- it stays _STATE_DIM * history_len = 180.
