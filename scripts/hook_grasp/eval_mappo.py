# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deterministic GOAL evaluator for a MAPPO / CTDE hook-grasp checkpoint.

Judges the HOOK-BLIND actor (NOT the privileged critic). Unlike the legacy PPO evaluators
(eval_goals.py / diag_catch.py) this does NOT convert the env to single-agent and does NOT feed
the 180-dim privileged STATE to the policy. It keeps the env multi-agent (DirectMARLEnv), loads
ONLY the MAPPO policy (actor) from the checkpoint, and runs it DETERMINISTICALLY (mean actions) on
the per-agent HOOK-FREE actor observation the env returns (135-dim = 45*3). The critic and its
state preprocessor are loaded too (they live in the same checkpoint) but never touch the rollout.

Success (all curricula disabled, true task difficulty):
    catch_rate >= 0.30  AND  hold_success_given_caught >= 0.80   at hold_success_steps = 100.

Robustness trick (identical to eval_goals.py): a true hold-success terminates + resets the episode
in the SAME sim step, which would destroy a naive post-step read of the env flag. So we DISABLE the
env's own hold termination (hold_success_steps -> huge) and measure the hold OURSELVES as the running
count of consecutive attached steps; an episode "succeeds" once that count reaches --hold_steps.

Runs locally (8 GB GPU -> modest --num_envs) and on ALC. Example (local):
    python scripts/hook_grasp/eval_mappo.py --headless --num_envs 128 --steps 4000 \
        --checkpoint logs/skrl/mappo_flycrane_hover/<run>/checkpoints/agent_55000.pt --out /tmp/e.json
Example (ALC):
    ./.venv/bin/python scripts/hook_grasp/eval_mappo.py --headless --device cuda:0 --num_envs 256 \
        --checkpoint logs/skrl/mappo_flycrane_hover/<run>/checkpoints/agent_55000.pt
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Deterministic goal evaluator for a MAPPO hook-grasp actor.")
parser.add_argument("--num_envs", type=int, default=128, help="Parallel envs (local 8GB: keep modest, e.g. 128).")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to skrl MAPPO checkpoint .pt.")
parser.add_argument("--steps", type=int, default=4000, help="Total sim steps to roll (more -> more episodes).")
parser.add_argument("--hold_steps", type=int, default=100, help="Consecutive attached steps that count as a hold (final=100).")
parser.add_argument("--capture_radius", type=float, default=0.13, help="Catch radius (m); FINAL task value = 0.13.")
parser.add_argument("--seed", type=int, default=12345, help="Seed.")
parser.add_argument("--out", type=str, default="", help="Optional path to write a JSON result.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os

import gymnasium as gym
import torch

from skrl.utils.runner.torch import Runner

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    # ---- honest eval: DISABLE every training curriculum / aid so the numbers reflect the TRUE (end) task ----
    # Mirrors the disable-list in eval_goals.py / diag_catch.py. attached_spawn_curriculum is CRITICAL: if left on,
    # half the envs spawn already-attached and hold_given_caught comes out > 1 (known bug). spawn_curriculum=False
    # also drives _spawn_frac()->1.0, which zeroes swing-assist (assist = v0*(1-frac)) -> no spawn-velocity aid.
    for _attr in ("capture_curriculum", "hold_curriculum", "spawn_curriculum", "attached_spawn_curriculum"):
        if hasattr(env_cfg, _attr):
            setattr(env_cfg, _attr, False)
    # pin the FINAL catch geometry (0.13) explicitly so no annealed value leaks in
    if hasattr(env_cfg, "attachment") and args_cli.capture_radius > 0:
        env_cfg.attachment.capture_radius = args_cli.capture_radius
    # DISABLE the env's own hold-success termination so it never resets on the hold event; we measure the hold
    # ourselves (consecutive attached steps) and count a success at --hold_steps (final = 100).
    if hasattr(env_cfg, "hold_success_steps"):
        env_cfg.hold_success_steps = 10_000_000

    # ---- MAPPO agent cfg (separate hook-blind actor + privileged critic); silence writers/schedulers for eval ----
    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_mappo_cfg_entry_point")

    # ---- build the env MULTI-AGENT: keep DirectMARLEnv, do NOT convert to single-agent (that path feeds the
    #      hook STATE to the policy -- forbidden). SkrlVecEnvWrapper auto-detects MARL and yields per-agent dicts. ----
    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    assert isinstance(base, DirectMARLEnv), "eval_mappo expects a DirectMARLEnv (multi-agent) task."
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    # disable LR schedulers for eval (keep the keys present; MAPPO_DEFAULT_CONFIG omits them so the yaml must supply)
    for _k in ("learning_rate_scheduler_actor", "learning_rate_scheduler_critic"):
        agent_cfg["agent"][_k] = None

    runner = Runner(env, agent_cfg)
    print(f"[eval-mappo] loading checkpoint: {args_cli.checkpoint}")
    runner.agent.load(args_cli.checkpoint)
    runner.agent.set_running_mode("eval")

    agents = list(base.possible_agents)
    E = base.num_envs
    dev = base.device
    hold_req = args_cli.hold_steps

    # ---- confirm the ACTOR input is the 135-dim HOOK-FREE obs, never the 180-dim privileged state ----
    obs, _ = env.reset()
    actor_dim = int(obs[agents[0]].shape[-1])
    state_dim = int(base.state().shape[-1]) if hasattr(base, "state") else -1
    print(f"[eval-mappo] agents={agents}  actor_obs_dim={actor_dim}  critic_state_dim={state_dim}")
    assert actor_dim == 135, f"actor obs must be 135-dim (45*3) hook-free, got {actor_dim}"
    assert state_dim in (-1, 180), f"critic state expected 180-dim, got {state_dim}"

    # per-env, current-episode trackers
    ever_caught = torch.zeros(E, dtype=torch.bool, device=dev)
    ever_held = torch.zeros(E, dtype=torch.bool, device=dev)
    consec_attach = torch.zeros(E, dtype=torch.long, device=dev)
    ep_hold_max = torch.zeros(E, dtype=torch.long, device=dev)  # best consecutive-hold run in the current episode

    # cumulative counters
    n_ep = torch.zeros((), dtype=torch.long, device=dev)
    n_caught = torch.zeros((), dtype=torch.long, device=dev)
    n_held = torch.zeros((), dtype=torch.long, device=dev)
    attached_steps = torch.zeros((), dtype=torch.long, device=dev)
    held_hold_sum = torch.zeros((), dtype=torch.long, device=dev)  # sum of per-episode max-hold over caught episodes
    total_steps = 0
    # cheap slack/jerk diagnostics (accumulated over attached steps)
    slack_steps = torch.zeros((), dtype=torch.long, device=dev)
    jerk_sum = torch.zeros((), dtype=torch.float, device=dev)
    jerk_n = torch.zeros((), dtype=torch.long, device=dev)

    for _ in range(args_cli.steps):
        with torch.inference_mode():
            # MAPPO.act -> (actions, log_prob, outputs); outputs[uid]["mean_actions"] is the DETERMINISTIC action
            sampled, _, outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # DETERMINISTIC: use the Gaussian mean, not the sampled (exploration-noised) action
            actions = {uid: (outputs[uid]["mean_actions"] if "mean_actions" in outputs[uid] else sampled[uid])
                       for uid in agents}
            obs, _, terminated, truncated, _ = env.step(actions)

        attached = base.attached.any(dim=-1)                       # (E,) hook attached this step
        newly = base._newly_caught.any(dim=-1)                     # (E,) newly caught this step
        ever_caught |= newly
        consec_attach = torch.where(attached, consec_attach + 1, torch.zeros_like(consec_attach))
        ep_hold_max = torch.maximum(ep_hold_max, consec_attach)
        ever_held |= consec_attach >= hold_req
        attached_steps += attached.sum()
        total_steps += E

        # slack/jerk diagnostics (only meaningful while attached)
        if hasattr(base, "_drone_jerk"):
            jm = torch.norm(base._drone_jerk, dim=-1).mean(dim=-1)  # (E,) per-env mean jerk magnitude
            att_mask = attached
            if att_mask.any():
                jerk_sum += jm[att_mask].sum()
                jerk_n += att_mask.sum()

        # done handling: MARL wrapper returns per-agent dicts; reduce across agents
        def _reduce(d):
            if isinstance(d, dict):
                t = torch.stack([d[a].view(E) for a in agents], dim=0).any(dim=0)
                return t.bool()
            return d.view(E).bool()

        done = (_reduce(terminated) | _reduce(truncated))
        if done.any():
            dc = ever_caught & done
            dh = ever_held & done
            n_ep += done.sum()
            n_caught += dc.sum()
            n_held += dh.sum()
            held_hold_sum += ep_hold_max[dc].sum()
            keep = ~done
            ever_caught &= keep
            ever_held &= keep
            consec_attach = torch.where(done, torch.zeros_like(consec_attach), consec_attach)
            ep_hold_max = torch.where(done, torch.zeros_like(ep_hold_max), ep_hold_max)

    ep = int(n_ep.item())
    caught = int(n_caught.item())
    held = int(n_held.item())
    catch_rate = caught / ep if ep else 0.0
    hold_given_caught = held / caught if caught else 0.0
    attached_frac = float(attached_steps.item()) / max(1, total_steps)
    mean_hold_caught = float(held_hold_sum.item()) / caught if caught else 0.0
    mean_jerk = float(jerk_sum.item()) / max(1, int(jerk_n.item())) if int(jerk_n.item()) else float("nan")

    result = {
        "checkpoint": args_cli.checkpoint,
        "algorithm": "MAPPO",
        "num_envs": E,
        "sim_steps": args_cli.steps,
        "actor_obs_dim": actor_dim,
        "critic_state_dim": state_dim,
        "hold_steps": hold_req,
        "capture_radius": float(base._acfg.capture_radius),
        "episodes": ep,
        "caught_episodes": caught,
        "held_episodes": held,
        "catch_rate": round(catch_rate, 4),
        "hold_success_given_caught": round(hold_given_caught, 4),
        "mean_hold_steps_given_caught": round(mean_hold_caught, 2),
        "attached_frac": round(attached_frac, 4),
        "mean_jerk_attached_N": round(mean_jerk, 3) if mean_jerk == mean_jerk else None,
        "GOAL_catch_0.30": catch_rate >= 0.30,
        "GOAL_hold_0.80": hold_given_caught >= 0.80,
        "GOALS_MET": bool(catch_rate >= 0.30 and hold_given_caught >= 0.80),
    }
    print("JSON " + json.dumps(result))
    print("\n==================== MAPPO GOAL EVAL ====================")
    for k, v in result.items():
        print(f"  {k:30s}: {v}")
    print("  gate: catch>=0.30 AND hold_given_caught>=0.80 -> "
          f"{'PASS' if result['GOALS_MET'] else 'FAIL'}")
    print("========================================================")
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        print(f"[eval-mappo] wrote {args_cli.out}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
