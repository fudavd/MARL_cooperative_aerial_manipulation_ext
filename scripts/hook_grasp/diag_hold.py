# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Diagnose the ATTACHED-phase behaviour: does the drone pull AWAY from the wall (correct, taut) or
TOWARD it (slack-inducing / wall-pressing)? Aggregates over attached steps across many envs.

Reports, over all attached steps:
  * drone wall-normal velocity  (>0 = moving AWAY from wall = pulling taut; <0 = INTO the wall)
  * drone->attach-point distance vs the taut length (wire_length - slack_margin); slack if below
  * hook tension, wall penetration
"""
import argparse, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--capture_radius", type=float, default=0.13)
parser.add_argument("--slack_dwell_steps", type=int, default=-1)
parser.add_argument("--detach_p_vertical", type=float, default=-1.0)
parser.add_argument("--detach_p_horizontal", type=float, default=-1.0)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--ml_framework", type=str, default="torch")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]
app_launcher = AppLauncher(args_cli); simulation_app = app_launcher.app

import torch, gymnasium as gym
from skrl.utils.runner.torch import Runner
import MARL_mav_carry_ext.tasks  # noqa
import isaaclab_tasks  # noqa
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
env_cfg.seed = args_cli.seed
for a in ("capture_curriculum", "hold_curriculum", "spawn_curriculum"):
    if hasattr(env_cfg, a): setattr(env_cfg, a, False)
env_cfg.attachment.capture_radius = args_cli.capture_radius
if args_cli.slack_dwell_steps >= 0: env_cfg.attachment.slack_dwell_steps = args_cli.slack_dwell_steps
if args_cli.detach_p_vertical >= 0: env_cfg.attachment.detach_p_vertical = args_cli.detach_p_vertical
if args_cli.detach_p_horizontal >= 0: env_cfg.attachment.detach_p_horizontal = args_cli.detach_p_horizontal
env_cfg.hold_success_steps = 10_000_000  # never self-terminate on hold

exp = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
env = gym.make(args_cli.task, cfg=env_cfg); base = env.unwrapped
if isinstance(base, DirectMARLEnv): env = multi_agent_to_single_agent(env, state_as_observation=True)
env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
exp["trainer"]["close_environment_at_exit"] = False
exp["agent"]["experiment"]["write_interval"] = 0; exp["agent"]["experiment"]["checkpoint_interval"] = 0
exp["agent"].pop("learning_rate_scheduler", None); exp["agent"].pop("learning_rate_scheduler_kwargs", None)
runner = Runner(env, exp); runner.agent.load(args_cli.checkpoint); runner.agent.set_running_mode("eval")

taut = env_cfg.attachment.wire_length - env_cfg.attachment.slack_margin
dev = base.device
# accumulators over attached steps
n = torch.zeros((), device=dev)
sum_vn = torch.zeros((), device=dev)          # drone wall-normal velocity (signed, + = away from wall)
n_into = torch.zeros((), device=dev)          # attached steps moving INTO wall (vn<0)
sum_dist = torch.zeros((), device=dev)        # drone->attach distance
n_slack = torch.zeros((), device=dev)         # attached steps where wire is slack (dist<taut)
sum_pen = torch.zeros((), device=dev)         # wall penetration
sum_tens = torch.zeros((), device=dev)

obs, _ = env.reset()
for _ in range(args_cli.steps):
    with torch.inference_mode():
        out = runner.agent.act(obs, timestep=0, timesteps=0)
        obs, _, _, _, _ = env.step(out[-1].get("mean_actions", out[0]))
    att = base.attached[:, 0]                                        # (E,)
    if not att.any(): continue
    drone = base.drone_positions[:, 0]                              # (E,3)
    dvel = base.drone_linear_velocities[:, 0]                       # (E,3)
    nrm = base.wall_normal_w                                        # (E,3) outward wall normal
    attach = base.attach_point_w[:, 0]                             # (E,3)
    vn = (dvel * nrm).sum(-1)                                       # + = moving away from wall (outward)
    dist = torch.norm(drone - attach, dim=-1)
    m = att
    n += m.sum()
    sum_vn += (vn * m).sum(); n_into += ((vn < 0) & m).sum()
    sum_dist += (dist * m).sum(); n_slack += ((dist < taut) & m).sum()
    sum_pen += (base.metrics["wall_penetration"] * m.float()).sum()
    sum_tens += (base._hook_tension[:, 0] * m.float()).sum()

N = max(1.0, float(n.item()))
print("\n==================== ATTACHED-PHASE DIAGNOSIS ====================")
print(f" checkpoint            : {args_cli.checkpoint}")
print(f" attached steps sampled: {int(n.item())}")
print(f" taut length (m)       : {taut:.3f}  (wire {env_cfg.attachment.wire_length} - margin {env_cfg.attachment.slack_margin})")
print(f" drone->attach dist (m): mean {sum_dist.item()/N:.3f}   (>= {taut:.3f} = taut/away ; < = slack/toward)")
print(f"   frac slack           : {n_slack.item()/N:.3f}")
print(f" drone wall-normal vel : mean {sum_vn.item()/N:+.3f} m/s  (+ = pulling AWAY from wall ; - = INTO wall)")
print(f"   frac moving INTO wall: {n_into.item()/N:.3f}")
print(f" wall penetration (m)  : mean {sum_pen.item()/N:.4f}")
print(f" hook tension (N)      : mean {sum_tens.item()/N:.2f}")
print("==================================================================")
sys.stdout.flush()
import os; os._exit(0)
