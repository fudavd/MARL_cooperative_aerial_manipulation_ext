# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""v86: PPO-FINETUNE the v84 r2 attach policy on the 3-drone TRANSPORT hand-off task (user direction,
2026-08-15) — does gradient-based weight adaptation steer the attach behaviour into cooperative carrying,
where evolved head plasticity (v83/v85) had no causal channel?

Setup mirrors the v83/v85 hand-off env exactly (hover_rand + actor_indi_residual 153-dim obs = the hook
actor layout, wall-directive in place of the ego id, deterministic horizontal post-attachment spawn,
per-env attachment DR, basket mass pinned) minus the ES-only machinery (no CRN, no condition slots — PPO
wants fresh draws).

MARL -> rsl_rl: the three drones run ONE shared policy; each drone is one batch lane (batch = 3E,
agent-major blocks). Rewards/dones are env-shared and tiled. Critic = the env's 84-dim global state
(load + all drones + goal), tiled per lane — the value is env-level (shared reward), so lanes agree.

WARM START: loads the r2 checkpoint's ACTOR weights + action std + actor obs-normalizer. The critic is
FRESH (r2's critic was the hook env's 180-dim privileged head — that state does not exist here).

Run (ALC):
    ./.venv/bin/python scripts/hook_grasp/train_rslrl_transport.py --headless --device cuda:0 \
        --checkpoint logs/rsl_rl/r2_h3_mlp256/.../model_19999.pt --num_envs 512 \
        --max_iterations 10000 --wandb --experiment_name v86_ppo_handoff
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PPO-finetune the r2 attach policy on the 3-drone transport task.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-rand-v0")
parser.add_argument("--checkpoint", type=str, default="", help="r2 (feedforward) checkpoint for the actor warm start; empty = from scratch.")
parser.add_argument("--num_envs", type=int, default=512, help="Parallel envs E (policy batch = 3E drone lanes).")
parser.add_argument("--max_iterations", type=int, default=10000)
parser.add_argument("--num_steps_per_env", type=int, default=24)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--experiment_name", type=str, default="v86_ppo_handoff")
parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--wandb_project", type=str, default="transport-ppo-handoff")
parser.add_argument("--resume", type=str, default="", help="Full runner checkpoint to resume from (this script's own).")
parser.add_argument("--policy_kv", type=str, default="", help="policy overrides (e.g. 'hidden_dim=128'); must match --checkpoint arch.")
AppLauncher.add_app_launcher_args(parser)
args_cli, env_override_args = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
from datetime import datetime

import gymnasium as gym
import torch

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rslrl_adapter import make_runner_cfg  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

ACTOR_OBS = 153


def _coerce(text: str):
    low = text.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def _parse_kv(s: str) -> dict:
    out = {}
    for tok in s.split(","):
        tok = tok.strip()
        if tok:
            k, _, v = tok.partition("=")
            out[k.strip()] = _coerce(v)
    return out


def apply_env_overrides(env_cfg, tokens):
    for tok in tokens:
        if not tok.startswith("env.") or "=" not in tok:
            raise SystemExit(f"[ERROR] unrecognized argument: {tok!r} (use env.<dotted.path>=<value>)")
        path, _, raw = tok.partition("=")
        parts = path.split(".")[1:]
        target = env_cfg
        for p in parts[:-1]:
            target = getattr(target, p)
        if not hasattr(target, parts[-1]):
            raise SystemExit(f"[ERROR] no such env config field: {path!r}")
        setattr(target, parts[-1], _coerce(raw))
        print(f"[transport-ppo] env override: {path} = {_coerce(raw)!r}")


class TransportSharedAdapter(VecEnv):
    """3-drone DirectMARLEnv -> rsl_rl VecEnv: one shared policy, each drone = one batch lane (3E)."""

    _OBS_BOUND = 1000.0
    _REW_BOUND = 200.0

    def __init__(self, env: DirectMARLEnv):
        self.env = env
        self.agents = list(env.possible_agents)
        self.n_ag = len(self.agents)
        self.E = int(env.num_envs)
        self.num_envs = self.n_ag * self.E
        self.num_actions = 5
        self.device = env.device
        self.cfg = env.cfg
        self.max_episode_length = env.max_episode_length
        obs, self._extras = self._reset_env()
        self._obs = obs
        print(f"[transport-adapter] E={self.E} envs x {self.n_ag} drones = batch {self.num_envs}  "
              f"actor_obs={obs.shape[-1]}  critic_obs={self._extras['observations']['critic'].shape[-1]}")

    # rsl_rl writes this for init_at_random_ep_len; keep the three lanes of one env in lockstep
    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.env.episode_length_buf.repeat(self.n_ag)

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.env.episode_length_buf = value[: self.E]

    def _san(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_(-self._OBS_BOUND, self._OBS_BOUND)

    def _cat_obs(self, obs_dict) -> torch.Tensor:
        return torch.cat([self._san(obs_dict[a]) for a in self.agents], dim=0)

    def _extras_from(self, time_outs_env: torch.Tensor, log: dict | None = None) -> dict:
        critic = self._san(self.env.state()).repeat(self.n_ag, 1)          # (3E, state_space)
        extras = {"observations": {"critic": critic},
                  "time_outs": time_outs_env.repeat(self.n_ag).to(self.device).float()}
        if log:
            extras["log"] = log
        return extras

    def _reset_env(self):
        obs_dict, _ = self.env.reset()
        return self._cat_obs(obs_dict), self._extras_from(torch.zeros(self.E, device=self.device))

    def get_observations(self):
        return self._obs, self._extras

    def reset(self):
        self._obs, self._extras = self._reset_env()
        return self._obs, self._extras

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.env.device)
        act = {a: actions[k * self.E:(k + 1) * self.E] for k, a in enumerate(self.agents)}
        obs_dict, rew_dict, term_dict, tout_dict, _ = self.env.step(act)

        a0 = self.agents[0]
        rew_env = torch.nan_to_num(rew_dict[a0].view(self.E), nan=0.0, posinf=0.0, neginf=0.0).clamp_(
            -self._REW_BOUND, self._REW_BOUND)
        term_env = term_dict[a0].view(self.E)
        tout_env = tout_dict[a0].view(self.E)
        base = self.env
        log = {
            "Metrics/pos_error": base.metrics["position_error"].mean(),
            "Metrics/ori_error": base.metrics["orientation_error"].mean(),
            "Metrics/crash_term": term_env.float().mean(),
        }
        self._obs = self._cat_obs(obs_dict)
        self._extras = self._extras_from(tout_env, log)
        rewards = rew_env.repeat(self.n_ag)
        dones = (term_env | tout_env).float().repeat(self.n_ag)
        return self._obs, rewards, dones, self._extras


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    # the hand-off env (identical to v83/v85 minus ES-only CRN/slots)
    env_cfg.partial_obs = True
    env_cfg.history_len = 3
    env_cfg.actor_indi_residual = True
    env_cfg.observation_spaces = {a: ACTOR_OBS for a in env_cfg.possible_agents}
    env_cfg.debug_vis = False
    env_cfg.randomize_attachment = True
    env_cfg.attachment_per_env = True
    env_cfg.scene.replicate_physics = False
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (1.0, 1.0),
        "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
    }
    env_cfg.events.randomize_mass.params["mass_distribution_params"] = (1.4, 1.4)
    if env_override_args:
        apply_env_overrides(env_cfg, env_override_args)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    assert isinstance(base, DirectMARLEnv)
    adapter = TransportSharedAdapter(base)

    train_cfg = make_runner_cfg(
        experiment_name=args_cli.experiment_name,
        max_iterations=args_cli.max_iterations,
        num_steps_per_env=args_cli.num_steps_per_env,
        seed=args_cli.seed,
        device=device,
        logger="wandb" if args_cli.wandb else "tensorboard",
        wandb_project=args_cli.wandb_project,
        policy_overrides=_parse_kv(args_cli.policy_kv) if args_cli.policy_kv else None,
        feedforward=True,
    )

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.experiment_name))
    log_dir = os.path.join(log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.experiment_name}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"[transport-ppo] logging to {log_dir}")

    runner = OnPolicyRunner(adapter, train_cfg, log_dir=log_dir, device=device)

    if args_cli.resume:
        print(f"[transport-ppo] resuming full runner state from {args_cli.resume}")
        runner.load(args_cli.resume)
    elif args_cli.checkpoint:
        # WARM START: actor + action std + actor obs-normalizer from r2; critic stays fresh (dim mismatch
        # with the hook env's 180-dim privileged critic is intentional — that state does not exist here).
        ck = torch.load(args_cli.checkpoint, map_location=device, weights_only=False)
        sd = ck["model_state_dict"]
        assert not any(k.startswith("memory_a") for k in sd), \
            "checkpoint is recurrent (LSTM); this hand-off trainer expects a v84 feedforward checkpoint"
        actor_sd = {k: v for k, v in sd.items() if k.startswith("actor.") or k == "std"}
        # NB: rsl_rl's ActorCritic overrides load_state_dict to return a bool — no missing/unexpected tuple
        runner.alg.policy.load_state_dict(actor_sd, strict=False)
        # verify the warm start actually landed (guards against silent key mismatches under strict=False)
        w_ck = actor_sd["actor.0.weight"]
        w_live = dict(runner.alg.policy.named_parameters())["actor.0.weight"]
        assert torch.equal(w_live.detach().cpu(), w_ck.cpu()), "warm start failed: actor.0.weight not loaded"
        runner.obs_normalizer.load_state_dict(ck["obs_norm_state_dict"])
        print(f"[transport-ppo] WARM START from {args_cli.checkpoint}\n"
              f"  loaded + verified: {sorted(actor_sd.keys())}\n  critic-side stays fresh")

    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass
        os._exit(0)