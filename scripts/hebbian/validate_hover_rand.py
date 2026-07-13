# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validation + micro-benchmark for the hover_rand domain-randomization env.

Checks, for a given N and attachment scope:
  (A) cable-attachment randomization applied, on the basket surface, and varies per env (per-env)
      or is shared (per-run);
  (B) the physics actually relocated each rope base to the randomized anchor (FK from the moved
      joint frame);
  (C) per-env COM / inertia perturbation round-trips through the physics sim (set -> get) and is
      dynamically active (perturbed envs' load spins differently under identical zero actions).
Also reports env-build wall time and VRAM (the per-N cost of replicate_physics True vs False).

Example (on ALC):
    .venv/bin/python scripts/hebbian/validate_hover_rand.py --num_envs 256 --attachment_per_env 0 --headless
    .venv/bin/python scripts/hebbian/validate_hover_rand.py --num_envs 256 --attachment_per_env 1 --headless
"""

import argparse
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Validate/benchmark the hover_rand env.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--attachment_per_env", type=int, default=0, help="1 = per-env (replicate_physics=False).")
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-rand-v0")
parser.add_argument("--seed", type=int, default=2)
parser.add_argument("--steps", type=int, default=60, help="Sim steps for the COM/inertia dynamics probe.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]

t_launch = time.time()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import subprocess

import numpy as np
import torch

import gymnasium as gym
import omni.usd

import MARL_mav_carry_ext.tasks  # noqa: F401  (registers gym envs)
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def gpu_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"]
        ).decode().splitlines()[0]
        return out.strip()
    except Exception as e:
        return f"n/a ({e})"


def main():
    per_env = bool(args_cli.attachment_per_env)
    N = args_cli.num_envs

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=N)
    env_cfg.seed = args_cli.seed
    env_cfg.randomize_attachment = True
    env_cfg.attachment_per_env = per_env
    # __post_init__ already ran at parse time with the default flag, so set replicate_physics here
    # to match the runtime override (per-env needs independent physics per env).
    if per_env:
        env_cfg.scene.replicate_physics = False

    print(f"\n===== VALIDATE hover_rand | N={N} per_env={per_env} seed={args_cli.seed} =====")
    print(f"[cfg] replicate_physics={env_cfg.scene.replicate_physics}")

    torch.cuda.reset_peak_memory_stats()
    t_build = time.time()
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()  # through the gym wrapper so OrderEnforcing is satisfied before env.step()
    torch.cuda.synchronize()
    build_s = time.time() - t_build

    print(f"\n[timing] env build+reset = {build_s:.1f}s | total since app launch = {time.time()-t_launch:.1f}s")
    print(f"[vram] nvidia-smi used,free MiB = {gpu_mem()}")
    print(f"[vram] torch reserved = {torch.cuda.memory_reserved()/2**20:.0f} MiB | "
          f"peak alloc = {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")

    stage = omni.usd.get_context().get_stage()
    ropes = ("rope_1", "rope_2", "rope_3")
    hx, hy = env_cfg.attachment_basket_half_x, env_cfg.attachment_basket_half_y
    ztop, zdep = env_cfg.attachment_basket_top_z, env_cfg.attachment_z_depth

    def on_surface(p, tol=1e-4):
        x, y, z = p
        onwall = abs(abs(x) - hx) < tol or abs(abs(y) - hy) < tol
        ontop = abs(z - ztop) < tol
        inside = abs(x) <= hx + tol and abs(y) <= hy + tol and (ztop - zdep - tol) <= z <= (ztop + tol)
        return (onwall or ontop) and inside

    # ---------- (A) attachment localPos0 read-back ----------
    def anchor(i, r):
        prim = stage.GetPrimAtPath(f"/World/envs/env_{i}/flycrane/load_link/{r}_sphere_joint_0_joint_x")
        return tuple(float(v) for v in prim.GetAttribute("physics:localPos0").Get())

    sample = sorted({0, 1, 2, N // 2, N - 1} & set(range(N)))
    print("\n[A] attachment localPos0 read-back (per sampled env):")
    n_off = 0
    for i in sample:
        cells = []
        for r in ropes:
            p = anchor(i, r)
            n_off += 0 if on_surface(p) else 1
            cells.append(f"{r}=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})")
        print(f"   env {i:>5}: " + "  ".join(cells))

    all_r1 = np.array([anchor(i, "rope_1") for i in range(N)])
    print(f"[A] rope_1 anchor spread over {N} envs: "
          f"std=(x {all_r1[:,0].std():.4f}, y {all_r1[:,1].std():.4f}, z {all_r1[:,2].std():.4f})")
    print(f"[A] z range over envs = [{all_r1[:,2].min():.3f}, {all_r1[:,2].max():.3f}]  (z-variation check)")
    print(f"[A] on-surface violations (sampled) = {n_off}  (expect 0)")
    print(f"[A] EXPECT: per_env -> nonzero std; per_run -> ~0 std (all envs share one triangle)")

    # ---------- (B) physics FK: rope base actually at the randomized anchor ----------
    try:
        robot = u.robot
        base_ids, base_names = robot.find_bodies("rope_._sphere_joint_0_link_1")
        load_ids, _ = robot.find_bodies("load_link")
        load_i = load_ids[0]
        pos = robot.data.body_com_state_w[:, :, :3]
        load_pos = pos[:, load_i]
        rel = pos[:, base_ids] - load_pos.unsqueeze(1)  # (N, 3 ropes, 3) load-relative base positions
        print(f"\n[B] rope-base positions relative to load (bodies={base_names}):")
        print(f"    env 0   : {rel[0].cpu().numpy().round(3).tolist()}")
        print(f"    env {N-1}: {rel[N-1].cpu().numpy().round(3).tolist()}")
        spread = rel.std(dim=0).norm(dim=-1)  # per-rope std across envs
        print(f"[B] per-rope base-position std across envs = {spread.cpu().numpy().round(4).tolist()}")
        print(f"[B] EXPECT per_env -> nonzero (bases physically placed at different anchors)")
    except Exception as e:
        print(f"[B] skipped: {e}")

    # ---------- (C) COM / inertia perturbation correctly modelled ----------
    print("\n[C] COM / inertia perturbation:")
    try:
        robot = u.robot
        load_i = robot.find_bodies("load_link")[0][0]
        view = robot.root_physx_view
        inertias = view.get_inertias().clone()
        coms = view.get_coms().clone()
        print(f"[C] tensor shapes: inertias={tuple(inertias.shape)} coms={tuple(coms.shape)} load_body_idx={load_i}")
        print(f"[C] default load inertia row = {inertias[0, load_i].cpu().numpy().round(5).tolist()}")
        print(f"[C] default load com row     = {coms[0, load_i].cpu().numpy().round(4).tolist()}  "
              f"(expect pos z ~0.06)")

        even = torch.arange(0, N, 2)
        idx = torch.arange(N)
        pert_in = inertias.clone()
        pert_co = coms.clone()
        pert_in[even, load_i] *= 2.0          # double the load inertia on even envs
        pert_co[even, load_i, 0] += 0.15      # shift COM +0.15 m in x on even envs
        view.set_inertias(pert_in, idx)
        view.set_coms(pert_co, idx)

        rb_in = view.get_inertias()
        rb_co = view.get_coms()
        ok_in = torch.allclose(rb_in[even, load_i], pert_in[even, load_i], atol=1e-5)
        ok_co = torch.allclose(rb_co[even, load_i, 0], pert_co[even, load_i, 0], atol=1e-5)
        print(f"[C] set->get round-trip: inertia_match={bool(ok_in)}  com_match={bool(ok_co)}  (both expect True)")

        # dynamics probe: identical zero actions; perturbed load should spin differently
        env.reset()
        # re-apply after reset in case any reset path touches these tensors
        view.set_inertias(pert_in, idx)
        view.set_coms(pert_co, idx)
        act = {a: torch.zeros((N, u.cfg.action_spaces[a]), device=u.device) for a in u.cfg.possible_agents}
        for _ in range(args_cli.steps):
            env.step(act)
        angvel = robot.data.body_com_state_w[:, load_i, 10:13]
        m = torch.norm(angvel, dim=-1)
        odd = torch.arange(1, N, 2)
        print(f"[C] load |angvel| after {args_cli.steps} steps: "
              f"perturbed(even)={m[even].mean():.4f}  nominal(odd)={m[odd].mean():.4f}")
        print(f"[C] EXPECT perturbed != nominal -> COM/inertia is dynamically active in sim")
    except Exception as e:
        import traceback
        print(f"[C] FAILED: {e}")
        traceback.print_exc()

    print("\n===== VALIDATION DONE =====\n")
    env.close()


try:
    main()
finally:
    simulation_app.close()
