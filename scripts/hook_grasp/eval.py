# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained hook-grasp checkpoint on the FINAL task (all curricula off, deterministic policy).

Metrics per episode: did the hook catch the bar, and, given a catch, was it held for ``--hold_steps``
consecutive control steps. Goal: catch_rate >= 0.30 and hold_success_given_caught >= 0.80.

    python scripts/hook_grasp/eval.py --headless --checkpoint logs/rsl_rl/hook_grasp/<run>/model_19999.pt
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a hook-grasp PPO checkpoint on the final task.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="rsl_rl model_<iter>.pt checkpoint.")
parser.add_argument("--num_envs", type=int, default=256, help="Parallel environments.")
parser.add_argument("--steps", type=int, default=4000, help="Control steps to roll out.")
parser.add_argument("--hold_steps", type=int, default=100, help="Consecutive attached steps that count as a hold.")
parser.add_argument("--seed", type=int, default=12345, help="Seed.")
parser.add_argument("--out", type=str, default="", help="Optional JSON output path.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

import MARL_mav_carry_ext.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rsl_rl_env import HookGraspVecEnv  # noqa: E402


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.disable_curricula()
    # the hold is measured here, so the env must not end an episode on its own hold milestone
    env_cfg.hold_success_steps = 10_000_000

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    vec_env = HookGraspVecEnv(base)

    agent_cfg: dict = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    agent_cfg["device"] = device
    runner = OnPolicyRunner(vec_env, agent_cfg, log_dir=None, device=device)
    print(f"[eval] loading {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)   # deterministic (Gaussian mean), normalizer included

    E, dev = base.num_envs, base.device
    obs, _ = vec_env.get_observations()
    ever_caught = torch.zeros(E, dtype=torch.bool, device=dev)
    ever_held = torch.zeros(E, dtype=torch.bool, device=dev)
    consec_attached = torch.zeros(E, dtype=torch.long, device=dev)
    n_ep = n_caught = n_held = 0
    attached_steps = 0
    jerk_sum, jerk_n = 0.0, 0

    for _ in range(args_cli.steps):
        with torch.inference_mode():
            obs, _, dones, _ = vec_env.step(policy(obs))
        attached = base.attached.any(dim=-1)
        ever_caught |= base._newly_caught.any(dim=-1)
        consec_attached = torch.where(attached, consec_attached + 1, torch.zeros_like(consec_attached))
        ever_held |= consec_attached >= args_cli.hold_steps
        attached_steps += int(attached.sum())
        if attached.any():
            jerk = torch.norm(base._drone_jerk, dim=-1).mean(dim=-1)
            jerk_sum += float(jerk[attached].sum())
            jerk_n += int(attached.sum())
        done = dones.bool()
        if done.any():
            n_ep += int(done.sum())
            n_caught += int((ever_caught & done).sum())
            n_held += int((ever_held & done).sum())
            ever_caught &= ~done
            ever_held &= ~done
            consec_attached[done] = 0

    catch_rate = n_caught / n_ep if n_ep else 0.0
    hold_given_caught = n_held / n_caught if n_caught else 0.0
    result = {
        "checkpoint": args_cli.checkpoint,
        "num_envs": E,
        "steps": args_cli.steps,
        "episodes": n_ep,
        "caught_episodes": n_caught,
        "held_episodes": n_held,
        "hold_steps": args_cli.hold_steps,
        "capture_radius": float(base._acfg.capture_radius),
        "min_chord_angle_deg": float(base._acfg.min_chord_angle_deg),
        "catch_rate": round(catch_rate, 4),
        "hold_success_given_caught": round(hold_given_caught, 4),
        "attached_frac": round(attached_steps / max(1, args_cli.steps * E), 4),
        "mean_jerk_attached": round(jerk_sum / jerk_n, 3) if jerk_n else None,
        "GOALS_MET": bool(catch_rate >= 0.30 and hold_given_caught >= 0.80),
    }
    print("\n==================== HOOK-GRASP EVAL ====================")
    for k, v in result.items():
        print(f"  {k:28s}: {v}")
    print("  goal: catch_rate >= 0.30 AND hold_success_given_caught >= 0.80 -> "
          f"{'PASS' if result['GOALS_MET'] else 'FAIL'}")
    print("=========================================================")
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[eval] wrote {args_cli.out}")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(0)   # skip the headless Isaac Sim teardown (can hang)
