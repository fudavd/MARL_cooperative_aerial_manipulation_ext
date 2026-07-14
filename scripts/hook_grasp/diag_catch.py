# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Diagnose the CATCH mechanism: does the policy seat the hook via a genuine PENDULUM UP-SWING, or via an
unphysical horizontal "creep" into the bar?  Measures, ONLY at the catch instant (the step where a drone's
hook newly latches), the hook's velocity decomposed into the wall's up-axis and into-wall components, plus a
swing-amplitude proxy (the hook's vertical travel over the ~60 steps leading into the catch).

Reports, over all sampled catches:
  * v_up_catch    -- hook velocity along the wall-up axis (>0 = sweeping UP; a physical J-hook needs this)
  * v_into_catch  -- into-wall closing speed (-hook_vel . outward_wall_normal)
  * speed_catch   -- total hook speed
  * swing amplitude -- max-min of hook height over the last 60 steps before the catch (a swing arcs; a creep ~0)
  * fraction of catches classified GENUINE UP-SWEEP (v_up > 0.2 m/s) vs CREEP-like (v_up <= 0.05 m/s)
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
parser.add_argument("--require_upsweep", type=int, default=-1)
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
if args_cli.require_upsweep >= 0: env_cfg.attachment.require_upsweep = bool(args_cli.require_upsweep)
env_cfg.hold_success_steps = 10_000_000  # never self-terminate on hold

exp = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
env = gym.make(args_cli.task, cfg=env_cfg); base = env.unwrapped
if isinstance(base, DirectMARLEnv): env = multi_agent_to_single_agent(env, state_as_observation=True)
env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
exp["trainer"]["close_environment_at_exit"] = False
exp["agent"]["experiment"]["write_interval"] = 0; exp["agent"]["experiment"]["checkpoint_interval"] = 0
exp["agent"].pop("learning_rate_scheduler", None); exp["agent"].pop("learning_rate_scheduler_kwargs", None)
runner = Runner(env, exp); runner.agent.load(args_cli.checkpoint); runner.agent.set_running_mode("eval")

dev = base.device
E = base.num_envs
WIN = 60                                            # rolling window for the swing-amplitude proxy
UP_THRESH = 0.2                                     # v_up above this = GENUINE up-sweep
CREEP_THRESH = 0.05                                 # v_up at/below this = CREEP-like

# ring buffer of hook height (E, WIN); seeded to the current height so early catches aren't spuriously large
hbuf = torch.zeros((E, WIN), device=dev)
buf_filled = False

# per-catch samples (collected only at the catch instant)
v_up_all, v_into_all, speed_all, amp_all = [], [], [], []

obs, _ = env.reset()
hbuf[:] = base.hook_positions[:, 0, 2].unsqueeze(-1)   # seed window with the initial height
for t in range(args_cli.steps):
    with torch.inference_mode():
        out = runner.agent.act(obs, timestep=0, timesteps=0)
        obs, _, _, _, _ = env.step(out[-1].get("mean_actions", out[0]))
    # roll the height window forward (drop oldest, append newest)
    hbuf = torch.roll(hbuf, shifts=-1, dims=1)
    hbuf[:, -1] = base.hook_positions[:, 0, 2]
    if t + 1 >= WIN: buf_filled = True

    caught = base._newly_caught[:, 0]                               # (E,) bool
    if not caught.any(): continue
    hook_vel = base.hook_velocities[:, 0]                          # (E, 3)
    # outward wall normal for the assigned wall at the current hook position
    _, _, wall_normal = base._assigned_wall_point(base.hook_positions[:, 0], 0)  # (E, 3)
    v_up = (hook_vel * base.wall_up_w).sum(-1)                     # (E,) + = sweeping up
    v_into = -(hook_vel * wall_normal).sum(-1)                     # (E,) + = closing into wall
    speed = torch.norm(hook_vel, dim=-1)                          # (E,)
    # swing amplitude over the window (only meaningful once the buffer is full; still recorded either way)
    amp = hbuf.max(dim=1).values - hbuf.min(dim=1).values          # (E,)

    idx = caught.nonzero(as_tuple=False).squeeze(-1)
    v_up_all.append(v_up[idx].detach().cpu())
    v_into_all.append(v_into[idx].detach().cpu())
    speed_all.append(speed[idx].detach().cpu())
    amp_all.append(amp[idx].detach().cpu())

v_up_c = torch.cat(v_up_all) if v_up_all else torch.zeros(0)
v_into_c = torch.cat(v_into_all) if v_into_all else torch.zeros(0)
speed_c = torch.cat(speed_all) if speed_all else torch.zeros(0)
amp_c = torch.cat(amp_all) if amp_all else torch.zeros(0)
NC = v_up_c.numel()

def _mean(x): return float(x.mean().item()) if x.numel() else float("nan")
def _median(x): return float(x.median().item()) if x.numel() else float("nan")
frac_up = float((v_up_c > UP_THRESH).float().mean().item()) if NC else float("nan")
frac_creep = float((v_up_c <= CREEP_THRESH).float().mean().item()) if NC else float("nan")

print("\n==================== CATCH-MECHANISM DIAGNOSIS ====================")
print(f" checkpoint            : {args_cli.checkpoint}")
print(f" require_upsweep       : {env_cfg.attachment.require_upsweep}")
print(f" catches sampled       : {NC}")
if not buf_filled:
    print(f" (warning: fewer than {WIN} steps ran; swing amplitude window not full)")
print(f" v_up_catch (m/s)      : mean {_mean(v_up_c):+.3f}   median {_median(v_up_c):+.3f}   (+ = sweeping UP)")
print(f"   frac GENUINE UP-SWEEP (v_up > {UP_THRESH:.2f}) : {frac_up:.3f}")
print(f"   frac CREEP-like      (v_up <= {CREEP_THRESH:.2f}): {frac_creep:.3f}")
print(f" v_into_catch (m/s)    : mean {_mean(v_into_c):+.3f}   (+ = closing INTO wall)")
print(f" speed_catch (m/s)     : mean {_mean(speed_c):.3f}")
print(f" swing amplitude (m)   : mean {_mean(amp_c):.3f}   (last {WIN} steps of hook height; ~0 = creep, large = arc)")
print("==================================================================")
print(" VERDICT: physical up-swing policy -> mean v_up well above 0.2, most catches UP-SWEEP.")
print("          creep policy             -> v_up ~0, most catches CREEP-like, tiny swing amplitude.")
sys.stdout.flush()
import os; os._exit(0)
