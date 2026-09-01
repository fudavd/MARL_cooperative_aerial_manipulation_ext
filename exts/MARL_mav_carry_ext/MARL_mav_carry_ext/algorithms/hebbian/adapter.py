# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bridge between the GPU-vectorized IsaacLab :class:`DirectMARLEnv` and the CMA-ES / Hebbian core.

Design:

* The IsaacLab env runs ``num_envs`` rollouts in parallel on the GPU.  Each environment evaluates
  one CMA-ES candidate ("genome"); a genome is the flat A/B/C/D plasticity rule (length
  ``4 * n_weights``).
* The 3 flycrane drones (``falcon1/2/3``) are homogeneous, so they **share a single genome** (one
  plasticity rule), but each keeps its own plastic weight matrix (which diverges online because the
  per-agent observations differ).
* A generation = ``popsize`` genomes, each evaluated for ``n_repeats`` episodes (fitness = mean
  episode return).  Evaluations are streamed onto the parallel envs via a job queue, so any
  ``num_envs`` works (ideally ``num_envs == popsize * n_repeats`` for a fully-parallel generation).
* Observations are standardized with a running mean/std (mirrors the MAPPO baseline's
  ``RunningStandardScaler``), and the Tanh action output is scaled to setpoint range.
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from gymnasium import spaces

from .hebbian_core import CMAES, HebbianActor


# ---------------------------------------------------------------------------
# Running observation standardizer (Welford), torch, per-feature
# ---------------------------------------------------------------------------
class RunningMeanStd:
    def __init__(self, shape, device, epsilon: float = 1e-4, clip: float = 5.0):
        self.mean = torch.zeros(shape, device=device, dtype=torch.float64)
        self.var = torch.ones(shape, device=device, dtype=torch.float64)
        self.count = epsilon
        self.clip = clip
        self.frozen = False

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if self.frozen:
            return
        x = x.to(torch.float64)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean += delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot
        self.var = m2 / tot
        self.count = tot

    @torch.no_grad()
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        out = (x.to(torch.float64) - self.mean) / torch.sqrt(self.var + 1e-8)
        return out.clamp(-self.clip, self.clip).to(torch.float32)

    def state_dict(self):
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": self.count, "clip": self.clip}

    def load_state_dict(self, sd, device):
        self.mean = sd["mean"].to(device).to(torch.float64)
        self.var = sd["var"].to(device).to(torch.float64)
        self.count = sd["count"]
        self.clip = sd["clip"]


# ---------------------------------------------------------------------------
# Running min-max normalizer -> maps each observation feature into [-1, 1]
# ---------------------------------------------------------------------------
class RunningMinMax:
    """Per-feature running min/max, scaling observations to [-1, 1].

    Caveat: a single outlier (e.g. a crash-induced large velocity) permanently widens the
    range and compresses normal values toward 0. Less robust than z-score for unbounded RL
    observations -- provided because it is the classic "min-max to [-1, 1]" normalization.
    """

    def __init__(self, shape, device, eps: float = 1e-6):
        self.lo = torch.full((shape,), float("inf"), device=device, dtype=torch.float64)
        self.hi = torch.full((shape,), float("-inf"), device=device, dtype=torch.float64)
        self.eps = eps
        self.frozen = False

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        if self.frozen:
            return
        x = x.to(torch.float64)
        self.lo = torch.minimum(self.lo, x.min(dim=0).values)
        self.hi = torch.maximum(self.hi, x.max(dim=0).values)

    @torch.no_grad()
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        lo, hi = self.lo, self.hi
        span = (hi - lo).clamp_min(self.eps)
        # features unseen so far (lo=inf) -> output 0
        seen = torch.isfinite(lo) & torch.isfinite(hi)
        out = 2.0 * (x.to(torch.float64) - lo) / span - 1.0
        out = torch.where(seen, out, torch.zeros_like(out))
        return out.clamp(-1.0, 1.0).to(torch.float32)

    def state_dict(self):
        return {"lo": self.lo.cpu(), "hi": self.hi.cpu(), "eps": self.eps}

    def load_state_dict(self, sd, device):
        self.lo = sd["lo"].to(device).to(torch.float64)
        self.hi = sd["hi"].to(device).to(torch.float64)
        self.eps = sd["eps"]


def make_normalizer(mode: str, shape: int, device):
    """Build an observation normalizer. mode in {"zscore", "minmax", "none"}."""
    if mode == "zscore":
        return RunningMeanStd(shape, device)
    if mode == "minmax":
        return RunningMinMax(shape, device)
    if mode in ("none", None):
        return None
    raise ValueError(f"unknown obs_norm mode: {mode!r} (use zscore|minmax|none)")


@dataclass
class HebbianConfig:
    """Hyper-parameters for the CMA-ES / Hebbian trainer."""

    # network -- a single hidden layer (input -> latent -> output, no bias), matching the
    # upstream HebbianActor.  latent_size=4 keeps the genome (=4*(obs+act)*latent) ~2.2k dims,
    # a scale at which full CMA-ES stays tractable.
    latent_size: int = 4
    hidden_activation: str = "Tanh"
    output_activation: str = "Tanh"
    hebbian_lr: float = 0.1
    weight_init_scale: float = 0.01
    action_scale: float = 2.0  # Tanh output (+-1) -> setpoint range (+-action_scale)
    obs_norm: str = "zscore"  # observation normalization: zscore | minmax | none
    shared_genome: bool = True  # one plasticity rule shared by all agents
    evolve_init_weights: bool = False  # add an evolved initial-weights genome block (set at reset)
    keep_d: bool = False               # keep the D term too (with evolve_init -> ABCD+W, 5 blocks)
    update_interval: int = 1           # plasticity timescale: avg activations over N steps per ΔW
    weight_anchor: float = 0.0         # elastic decay toward init weights: ΔW -= weight_anchor*(w-w_init)

    # CMA-ES (popsize 30, sigma 0.3, bounds +-1, n_params = n_blocks*n_weights, n_repeats 3,
    # full aCMA-ES).
    popsize: int = 30
    n_repeats: int = 3
    sigma: float = 0.3
    bound: float = 1.0  # ABCD coefficients in [-bound, bound]
    diagonal: bool = False  # enable diagonal CMA-ES only for large genomes
    num_generations: int = 100

    # fitness metric per episode: "return" = raw episode return (sum of rewards); "mean_reward" = episode
    # return / episode length (mean reward per timestep). Use mean_reward when success TERMINATES the episode
    # early -- otherwise the raw sum penalizes short successful episodes vs long mediocre ones.
    fitness: str = "return"

    # bookkeeping
    seed: Optional[int] = None
    log_dir: str = "./logs/hebbian/run"


class HebbianTrainer:
    """Runs the CMA-ES / Hebbian optimization loop against an IsaacLab DirectMARLEnv."""

    def __init__(self, env, cfg: HebbianConfig):
        self.env = env
        self.uenv = env.unwrapped if hasattr(env, "unwrapped") else env
        self.cfg = cfg
        self.device = self.uenv.device
        self.num_envs = self.uenv.num_envs
        self.agents: List[str] = list(self.uenv.cfg.possible_agents)
        self.n_agents = len(self.agents)

        # per-agent dims (homogeneous -> identical)
        self.obs_dim = int(self.uenv.cfg.observation_spaces[self.agents[0]])
        self.act_dim = int(self.uenv.cfg.action_spaces[self.agents[0]])

        obs_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)
        act_space = spaces.Box(-cfg.action_scale, cfg.action_scale, (self.act_dim,), dtype=np.float32)
        net_config = {
            "latent_size": cfg.latent_size,
            "hidden_activation": cfg.hidden_activation,
            "output_activation": cfg.output_activation,
            "clip_actions": False,
            "evolve_init_weights": cfg.evolve_init_weights,
            "keep_d": cfg.keep_d,
            "update_interval": cfg.update_interval,
            "weight_anchor": cfg.weight_anchor,
        }
        self.actors: Dict[str, HebbianActor] = {
            agent: HebbianActor(
                obs_space, act_space, self.num_envs, net_config=net_config,
                device=self.device, lr=cfg.hebbian_lr, init_scale=cfg.weight_init_scale,
            )
            for agent in self.agents
        }
        self.n_weights = self.actors[self.agents[0]].n_weights
        self.n_blocks = self.actors[self.agents[0]].n_blocks  # 3..5 (ABC[+D][+init_W])
        self.genome_dim = self.n_blocks * self.n_weights  # shared genome across agents

        # per-env current genome reshaped to [num_envs, n_blocks, n_weights]
        self.curr_abcd = torch.zeros(self.num_envs, self.n_blocks, self.n_weights, device=self.device)

        # obs normalizer (shared across homogeneous agents)
        self.rms = make_normalizer(cfg.obs_norm, self.obs_dim, self.device)

        # evolutionary optimizer
        opts = {"min": -cfg.bound, "max": cfg.bound, "num_generations": cfg.num_generations,
                "mutation_sigma": cfg.sigma}
        os.makedirs(cfg.log_dir, exist_ok=True)
        self.ea = CMAES(cfg.popsize, self.genome_dim, opts, output_dir=os.path.join(cfg.log_dir, "cmaes"),
                        diagonal=cfg.diagonal, seed=cfg.seed)

        # episode-return accumulator + per-episode step counter (for the "mean_reward" = return/length fitness)
        self.ep_return = torch.zeros(self.num_envs, device=self.device)
        self.ep_steps = torch.zeros(self.num_envs, device=self.device)
        self.env_steps_total = 0

        # generation bookkeeping
        self.env_genome = [-1] * self.num_envs
        self.genomes: Optional[np.ndarray] = None
        self.pending: deque = deque()
        self.returns_per_genome: List[List[float]] = []

        self.history = []  # list of dicts per generation
        # decomposed reward-component accumulation (per generation, env-weighted mean of the
        # env's per-episode "Episode_Reward/<term>" values, exposed via step() extras["log"])
        self.component_history = []
        self._gen_comp_sum: Dict[str, float] = {}
        self._gen_comp_n = 0

    # ---- genome <-> per-env plasticity rule --------------------------------
    def _set_env_genome(self, env_id: int, genome_idx: int):
        g = torch.as_tensor(self.genomes[genome_idx], dtype=torch.float32, device=self.device)
        self.curr_abcd[env_id] = g.reshape(self.n_blocks, self.n_weights)

    def _push_rules(self):
        for actor in self.actors.values():
            actor.set_hebbian_rules(self.curr_abcd)

    def _reset_actor_weights(self, env_ids):
        for actor in self.actors.values():
            actor.reset_network(env_ids)

    # ---- generation lifecycle ---------------------------------------------
    def _start_generation(self):
        self.genomes = self.ea.ask()  # [P, genome_dim]
        P = len(self.genomes)
        self.returns_per_genome = [[] for _ in range(P)]
        self.total_jobs = P * self.cfg.n_repeats
        self.completed_jobs = 0
        # job queue: each genome n_repeats times
        self.pending = deque(g for g in range(P) for _ in range(self.cfg.n_repeats))
        # IMPORTANT: reset the whole sim so every genome assigned at the generation boundary
        # starts from a clean episode (otherwise idle envs would begin mid-episode and the
        # fitness measurement would be contaminated). Mid-generation re-assignments instead
        # rely on Isaac Lab's per-env auto-reset on `done`, which is already a clean start.
        obs, _ = self.env.reset()
        all_ids = list(range(self.num_envs))
        for env_id in all_ids:
            self._assign_job(env_id)
        self._push_rules()                  # set A/B/C(/D or init_W) before resetting weights
        self._reset_actor_weights(all_ids)
        self.ep_return.zero_()
        self.ep_steps.zero_()
        self._gen_comp_sum = {}
        self._gen_comp_n = 0
        return obs

    def _assign_job(self, env_id: int):
        if self.pending:
            g = self.pending.popleft()
            self.env_genome[env_id] = g
            self._set_env_genome(env_id, g)
        else:
            self.env_genome[env_id] = -1  # idle: keeps stepping, result discarded

    def _on_dones(self, done_ids: torch.Tensor):
        """Record finished episodes, assign next jobs, reset their plastic weights."""
        if done_ids.numel() == 0:
            return
        done_list = done_ids.tolist()
        for env_id in done_list:
            g = self.env_genome[env_id]
            if g >= 0:
                if self.cfg.fitness == "mean_reward":
                    # mean reward per timestep: return / episode length. Doesn't penalize episodes that end
                    # early on success (they keep their high per-step reward instead of a truncated sum).
                    steps = max(1.0, float(self.ep_steps[env_id].item()))
                    fit = float(self.ep_return[env_id].item()) / steps
                else:
                    fit = float(self.ep_return[env_id].item())
                self.returns_per_genome[g].append(fit)
                self.completed_jobs += 1
            self._assign_job(env_id)
        # push new rules (incl. evolved init_W) before resetting those envs' weights
        self._push_rules()
        self._reset_actor_weights(done_list)
        self.ep_return[done_ids] = 0.0
        self.ep_steps[done_ids] = 0.0

    def _accumulate_components(self, extras, n_done: int):
        """Accumulate per-episode decomposed reward terms for envs that reset this step.

        The env writes ``extras["log"]["Episode_Reward/<term>"]`` in _reset_idx as the mean over
        the just-reset envs (normalized by max_episode_length_s). We pool these env-weighted across
        the generation.
        """
        log = extras.get("log", {}) if isinstance(extras, dict) else {}
        for k, v in log.items():
            if k.startswith("Episode_Reward/"):
                comp = k.split("/", 1)[1]
                self._gen_comp_sum[comp] = self._gen_comp_sum.get(comp, 0.0) + float(v) * n_done
        self._gen_comp_n += n_done

    def _finish_generation(self) -> dict:
        # aggregate mean return per genome (genomes with no completed repeat -> -inf)
        fit = np.array(
            [np.mean(r) if len(r) > 0 else -1e9 for r in self.returns_per_genome],
            dtype=np.float32,
        )
        self.ea.tell(self.genomes, fit, save_checkpoint=True)
        rec = {
            "generation": self.ea.current_gen - 1,
            "env_steps": self.env_steps_total,
            "best": float(np.max(fit)),
            "mean": float(np.mean(fit)),
            "std": float(np.std(fit)),
            "best_so_far": float(self.ea.f_best_so_far),
        }
        # decomposed reward components (env-weighted mean over the generation)
        comps = {
            c: (s / self._gen_comp_n if self._gen_comp_n else 0.0)
            for c, s in sorted(self._gen_comp_sum.items())
        }
        rec["components"] = comps
        self.component_history.append(
            {"generation": rec["generation"], "env_steps": rec["env_steps"], **comps}
        )
        self.history.append(rec)
        return rec

    # ---- action ------------------------------------------------------------
    @torch.no_grad()
    def _act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        actions = {}
        for agent in self.agents:
            o = obs[agent]
            if self.rms is not None:
                self.rms.update(o)
                o = self.rms.normalize(o)
            actions[agent] = self.actors[agent].act(o)
        return actions

    # ---- main training loop ------------------------------------------------
    def train(self, num_generations: Optional[int] = None, max_env_steps: Optional[int] = None,
              max_hours: Optional[float] = None, log_fn=None):
        import time as _time
        n_gen = num_generations if num_generations is not None else self.cfg.num_generations
        deadline = (_time.monotonic() + max_hours * 3600.0) if max_hours else None
        obs = self._start_generation()  # resets the env -> clean start for generation 0
        gens_done = 0
        env_steps = 0
        while gens_done < n_gen:
            actions = self._act(obs)
            obs, rewards, terminated, time_outs, extras = self.env.step(actions)
            env_steps += self.num_envs
            self.env_steps_total = env_steps

            # shared reward: sum across agents (they are identical shared rewards)
            step_r = torch.stack([rewards[a] for a in self.agents], dim=0).sum(dim=0)
            self.ep_return += step_r
            self.ep_steps += 1.0

            a0 = self.agents[0]
            done = (terminated[a0].bool() | time_outs[a0].bool())
            done_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
            if done_ids.numel() > 0:
                self._accumulate_components(extras, int(done_ids.numel()))
            self._on_dones(done_ids)

            if self.completed_jobs >= self.total_jobs:
                rec = self._finish_generation()
                if log_fn is not None:
                    log_fn(rec)
                gens_done += 1
                self._save_best(os.path.join(self.cfg.log_dir, "checkpoints"))
                if deadline is not None and _time.monotonic() >= deadline:
                    print(f"[INFO] max_hours reached after {gens_done} generations; stopping.")
                    break
                if gens_done < n_gen:
                    obs = self._start_generation()  # clean reset for the next generation
            if max_env_steps is not None and env_steps >= max_env_steps:
                break
        return self.history

    # ---- checkpoint --------------------------------------------------------
    def _save_best(self, ckpt_dir: str):
        os.makedirs(ckpt_dir, exist_ok=True)
        payload = {
            "best_genome": None if self.ea.x_best_so_far is None else np.asarray(self.ea.x_best_so_far),
            "best_fitness": float(self.ea.f_best_so_far),
            "cfg": self.cfg.__dict__,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "n_weights": self.n_weights,
            "agents": self.agents,
            "rms": self.rms.state_dict() if self.rms is not None else None,
            # env observation config, so eval can rebuild a matching env
            "partial_obs": bool(getattr(self.uenv.cfg, "partial_obs", True)),
            "history_len": int(getattr(self.uenv.cfg, "history_len", 1)),
        }
        torch.save(payload, os.path.join(ckpt_dir, "best.pt"))


class HebbianPolicy:
    """Deterministic policy for evaluation/play: loads a best genome + normalizer and acts."""

    def __init__(self, env, ckpt_path: str):
        self.env = env
        self.uenv = env.unwrapped if hasattr(env, "unwrapped") else env
        self.device = self.uenv.device
        self.num_envs = self.uenv.num_envs
        payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        cfg = HebbianConfig(**payload["cfg"])
        self.cfg = cfg
        self.agents = payload["agents"]
        self.obs_dim = payload["obs_dim"]
        self.act_dim = payload["act_dim"]

        obs_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)
        act_space = spaces.Box(-cfg.action_scale, cfg.action_scale, (self.act_dim,), dtype=np.float32)
        net_config = {"latent_size": cfg.latent_size, "hidden_activation": cfg.hidden_activation,
                      "output_activation": cfg.output_activation, "clip_actions": False,
                      "evolve_init_weights": cfg.evolve_init_weights, "keep_d": cfg.keep_d,
                      "update_interval": cfg.update_interval, "weight_anchor": cfg.weight_anchor}
        self.actors = {
            a: HebbianActor(obs_space, act_space, self.num_envs, net_config=net_config,
                            device=self.device, lr=cfg.hebbian_lr, init_scale=cfg.weight_init_scale)
            for a in self.agents
        }
        n_weights = self.actors[self.agents[0]].n_weights
        n_blocks = self.actors[self.agents[0]].n_blocks
        genome = torch.as_tensor(payload["best_genome"], dtype=torch.float32, device=self.device)
        abcd = genome.reshape(1, n_blocks, n_weights).repeat(self.num_envs, 1, 1)
        for actor in self.actors.values():
            actor.set_hebbian_rules(abcd)
            actor.reset_network(list(range(self.num_envs)))

        self.rms = None
        if payload["rms"] is not None:
            self.rms = make_normalizer(cfg.obs_norm, self.obs_dim, self.device)
            self.rms.load_state_dict(payload["rms"], self.device)
            self.rms.frozen = True

    @torch.no_grad()
    def act(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        actions = {}
        for a in self.agents:
            o = obs[a]
            if self.rms is not None:
                o = self.rms.normalize(o)
            actions[a] = self.actors[a].act(o)
        return actions

    def reset(self, env_ids=None):
        ids = list(range(self.num_envs)) if env_ids is None else list(env_ids)
        for actor in self.actors.values():
            actor.reset_network(ids)
