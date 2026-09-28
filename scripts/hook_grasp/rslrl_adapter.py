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
HOOK_OBSERVER_DIM = 3  # predicted hook position RELATIVE to the drone (world axes) appended to the actor obs


class HookObserver(torch.nn.Module):
    """A learned hook-state ESTIMATOR (user idea, 2026-07-23). The actor is hook-blind (no hook pos/vel in obs),
    but the hook state IS inferable from the observable history (INDI wrench residual + drone dynamics). This MLP
    predicts the hook's position RELATIVE to the drone from the (hook-free) actor obs, is trained ONLINE by a
    supervised MSE against the TRUE relative hook position (privileged, from the env), and its (detached)
    prediction is APPENDED to the actor obs -> the actor gains a self-computed 'hook belief' without ever seeing
    ground-truth hook state. Lives in the adapter so rsl_rl's ActorCritic/PPO are untouched.
    """

    # ROBUSTNESS (v64 fix, 2026-08-03): in the violent near-wall grasp regime the RAW actor obs contains
    # unbounded spikes (INDI residual during wire-tension transients) that DIVERGED the MLP (observer_mse
    # climbed 0.16 -> 4.0, i.e. 30x worse than predicting zero; the hook is always within ~0.6 m). Fixes:
    # clamp the inputs (train AND inference), clamp the INFERENCE output to the physical wire sphere (the
    # training loss stays on the unclamped output so gradients never saturate), and a gentler lr.
    IN_CLAMP = 20.0     # raw-obs clamp (units are mixed m / m/s / rad/s / residual accels; 20 is far out-of-band)
    OUT_CLAMP = 0.8     # m: prediction clamp at inference (wire is 0.55 m; 0.8 = generous sphere)

    def __init__(self, in_dim: int, out_dim: int = HOOK_OBSERVER_DIM, hidden: int = 128, lr: float = 3.0e-4,
                 device: str = "cuda:0"):
        super().__init__()
        self.out_dim = out_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.ELU(),
            torch.nn.Linear(hidden, out_dim),
        ).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.last_loss = torch.zeros((), device=device)

    @torch.no_grad()
    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.clamp(-self.IN_CLAMP, self.IN_CLAMP)).clamp(-self.OUT_CLAMP, self.OUT_CLAMP)

    @torch.no_grad()
    def predict_frozen(self, obs: torch.Tensor) -> torch.Tensor:
        """Inference-safe frozen forward (used to AUGMENT the obs during rollout -> stationary within a rollout)."""
        return self.net(obs.clamp(-self.IN_CLAMP, self.IN_CLAMP)).clamp(-self.OUT_CLAMP, self.OUT_CLAMP).detach()

    def update_batch(self, obs_batch: torch.Tensor, target_batch: torch.Tensor) -> None:
        """One SGD step on an accumulated ROLLOUT batch (called once per rollout, NOT per step -> the observer is
        frozen during collection so PPO sees a stationary obs). Copies out of the inference-mode rollout tensors."""
        with torch.inference_mode(False), torch.enable_grad():
            o = torch.empty(obs_batch.shape, dtype=obs_batch.dtype, device=obs_batch.device).copy_(obs_batch)
            t = torch.empty(target_batch.shape, dtype=target_batch.dtype, device=target_batch.device).copy_(target_batch)
            o.clamp_(-self.IN_CLAMP, self.IN_CLAMP)       # same input clamp as inference (consistent function)
            pred = self.net(o)                             # loss on the UNCLAMPED output (no saturated gradients)
            loss = torch.nn.functional.mse_loss(pred, t)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            self.last_loss = loss.detach()


class RslRlMarlAdapter(VecEnv):
    """Expose a single-agent DirectMARLEnv to rsl_rl with an asymmetric (privileged) critic obs."""

    def __init__(self, env: DirectMARLEnv, *, reward_norm: bool = False, gamma: float = 0.99,
                 hook_observer: bool = False, observer_lr: float = 1.0e-3, observer_update_every: int = 24,
                 true_hook: bool = False):
        assert isinstance(env, DirectMARLEnv), "RslRlMarlAdapter expects a DirectMARLEnv."
        self.env = env
        # true_hook: append the GROUND-TRUTH relative hook position to the actor obs (privileged ablation) instead of
        # the observer ESTIMATE -> isolates observation vs control (does perfect hook info break the perp_ratio wall?).
        self._true_hook = bool(true_hook)
        self._use_observer = bool(hook_observer) and not self._true_hook
        self.observer = None            # built lazily in __init__ once the raw actor dim is known
        self._observer_lr = observer_lr
        self._save_path = None          # set by the trainer -> periodic observer checkpoint (robust to kill)
        self._obs_step = 0
        self.observer_eval = False      # eval sets True -> use the loaded estimator frozen (predict, no online update)
        # observer is FROZEN during a rollout (predict-only) and batch-updated once per rollout -> stationary obs for PPO
        self._observer_update_every = int(observer_update_every)
        self._obs_buf, self._tgt_buf = [], []
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
        # Critic must match the env's DECLARED privileged state_space (180 for the hook task; the swing-test env
        # declares its own, e.g. 78) rather than a fixed constant -> the adapter works for both tasks unchanged.
        expected_critic = int(getattr(env.cfg, "state_space", CRITIC_OBS_DIM))
        assert critic_dim == expected_critic, (
            f"critic obs must be {expected_critic}-dim privileged (env.cfg.state_space), got {critic_dim}"
        )
        # HOOK OBSERVER: build on the RAW actor dim; its (detached) prediction is appended -> policy sees +3 dims.
        self._raw_actor_dim = actor_dim
        if self._use_observer:
            self.observer = HookObserver(in_dim=actor_dim, out_dim=HOOK_OBSERVER_DIM,
                                         lr=self._observer_lr, device=str(self.device))
        if self._use_observer or self._true_hook:
            obs = self._augment(obs)
        _tag = "+%d(true_hook)=%d" % (HOOK_OBSERVER_DIM, obs.shape[-1]) if self._true_hook else (
               "+%d(observer)=%d" % (HOOK_OBSERVER_DIM, obs.shape[-1]) if self._use_observer else "")
        print(f"[rslrl-adapter] agent={self.agent}  actor_obs={actor_dim}{_tag}  "
              f"critic_obs={critic_dim}  num_envs={self.num_envs}  num_actions={self.num_actions}")
        self._obs = obs

    # ------------------------------------------------------------------ hook observer
    def _true_hook_rel(self) -> torch.Tensor:
        """TRUE hook position relative to the drone (world axes, env-local) -> observer supervision target (E, 3)."""
        base = self.env
        return (base.hook_positions[:, 0] - base.drone_positions[:, 0]).detach()

    def _augment(self, obs: torch.Tensor) -> torch.Tensor:
        """Append the observer's (online-trained) hook estimate to the actor obs. No-op if the observer is off."""
        if self._true_hook:
            # nan_to_num: a mid-divergence hook state (reset the same step) must not poison the rollout
            true_rel = torch.nan_to_num(self._true_hook_rel(), nan=0.0, posinf=0.0, neginf=0.0)
            return torch.cat([obs, true_rel], dim=-1)   # PRIVILEGED: append ground-truth relative hook
        if not self._use_observer or self.observer is None:
            return obs
        # nan_to_num AFTER the clamp: clamp(NaN)=NaN, so a poisoned obs would otherwise pass straight through
        pred = torch.nan_to_num(self.observer.predict_frozen(obs), nan=0.0, posinf=0.0, neginf=0.0)
        if not self.observer_eval:
            # accumulate this step; batch-update the observer ONCE per rollout (num_steps_per_env), then clear.
            self._obs_buf.append(obs.detach())
            self._tgt_buf.append(self._true_hook_rel())
            if len(self._obs_buf) >= self._observer_update_every:
                self.observer.update_batch(torch.cat(self._obs_buf, 0), torch.cat(self._tgt_buf, 0))
                self._obs_buf.clear()
                self._tgt_buf.clear()
        return torch.cat([obs, pred], dim=-1)

    def save_observer(self, path: str) -> None:
        if self.observer is not None:
            torch.save(self.observer.state_dict(), path)

    def load_observer(self, path: str) -> None:
        if self.observer is not None:
            self.observer.load_state_dict(torch.load(path, map_location=self.device))

    # -- episode_length_buf must be a live proxy: rsl_rl may overwrite it (init_at_random_ep_len) --
    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value

    # ------------------------------------------------------------------ helpers
    # NUMERICAL HARDENING (2026-08-04, after two stochastic NaN-std PPO crashes): bound EVERYTHING that enters
    # rsl_rl at this single choke point. nan_to_num alone is not enough — a PhysX wire-tension blow-up can emit
    # HUGE-BUT-FINITE obs (hook flung far -> unclamped approach delta) whose value-target explosion NaNs the
    # policy log_std ("normal expects all elements of std >= 0.0"). Bounds are far outside the legit ranges
    # (obs ~O(10), per-step reward ~O(10), success one-off 80), so training signals are untouched.
    _OBS_BOUND = 1000.0
    _REW_BOUND = 200.0

    def _sanitize_obs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-self._OBS_BOUND, self._OBS_BOUND)

    def _build_extras(self, marl_extras: dict, time_outs: torch.Tensor, log: dict | None = None) -> dict:
        critic = self._sanitize_obs(self.env.state())  # (num_envs, 180) privileged frame-stacked state
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
        obs, self._extras = self._reset_env()
        self._obs = self._sanitize_obs(self._augment(obs))
        return self._obs, self._extras

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.env.device)
        obs_dict, rew_dict, term_dict, tout_dict, marl_extras = self.env.step({self.agent: actions})

        obs = self._sanitize_obs(self._augment(obs_dict[self.agent]))
        rewards = rew_dict[self.agent].view(self.num_envs)
        rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-self._REW_BOUND, self._REW_BOUND)
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
        if self._use_observer and self.observer is not None:
            log["Metrics/observer_mse"] = self.observer.last_loss   # -> falling = the estimator is learning the hook
            self._obs_step += 1
            if self._save_path and self._obs_step % 3000 == 0:
                self.save_observer(self._save_path)
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
    feedforward: bool = False,
) -> dict:
    """rsl_rl OnPolicyRunner train_cfg: ActorCriticRecurrent (LSTM) + PPO with adaptive-KL LR.

    Built as a plain dict (rather than the Isaac-Lab configclasses) so the recurrent RNN kwargs pass
    straight through to ``ActorCriticRecurrent.__init__`` without dataclass-field friction.

    ``policy_overrides`` / ``algo_overrides`` merge into the policy / algorithm sub-dicts (the overnight
    search tunes architecture + PPO HP here). ``class_name`` cannot be overridden (guarded), EXCEPT via
    ``feedforward=True`` which selects the memoryless ``ActorCritic`` (plain MLP, rnn_* keys dropped) —
    the architecture-reduction grid (v84) uses this. ``policy_overrides`` also accepts the pseudo-key
    ``hidden_dim`` (int) as shorthand for actor_hidden_dims = critic_hidden_dims = [n, n].
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
    if feedforward:
        cfg["policy"]["class_name"] = "ActorCritic"
        for k in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
            cfg["policy"].pop(k, None)
        print("[make_runner_cfg] FEEDFORWARD policy (ActorCritic, no RNN)")
    if policy_overrides:
        policy_overrides = dict(policy_overrides)
        assert "class_name" not in policy_overrides, "cannot override policy class_name (use feedforward=True)"
        n = policy_overrides.pop("hidden_dim", None)
        if n is not None:
            policy_overrides["actor_hidden_dims"] = [int(n), int(n)]
            policy_overrides["critic_hidden_dims"] = [int(n), int(n)]
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
