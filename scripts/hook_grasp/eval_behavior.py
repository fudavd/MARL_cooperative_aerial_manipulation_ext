# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OBJECTIVE BEHAVIOR TEST for an rsl_rl recurrent hook-grasp policy — is the RIGHT behavior learned?

Beyond catch/hold (eval_rslrl.py), this scores the PHYSICAL CORRECTNESS the user cares about, by decomposing
the hook motion in the assigned wall's orthonormal frame (wall_normal / bar_axis / wall_up) every step:

  TASK success:
    * catch_rate, hold_success_given_caught @100 (identical definitions to eval_rslrl.py)

  PHYSICALITY (the behavior the user wants — a genuine perpendicular pendulum up-swing, not a creep/hang/skim):
    * upsweep_frac        : fraction of catches with v_up > 0.2 m/s at the catch instant (genuine J-hook up-stroke)
    * v_up_catch          : mean hook up-velocity at the catch instant (>0 = sweeping up)
    * perp_ratio          : during unattached approach NEAR the wall, mean|v.wall_normal| /
                            (mean|v.wall_normal| + mean|v.bar_axis|)  -> 1.0 = pure perpendicular swing,
                            0.5 = equal parts sideways skim, <0.5 = mostly skimming ALONG the wall (BAD).
    * deadhang_frac       : fraction of unattached near-wall steps where the hook hangs ~straight below the
                            drone (horiz offset < 0.15 m) AND is barely moving relative to the drone
                            (|v_hook - v_drone| < 0.3 m/s) -> the dead pendulum (BAD).
    * swing_amp           : mean peak-to-peak hook HEIGHT over a ~0.6 s window (a swing arcs; a creep/hang ~0).
    * jerk                : mean drone jerk while attached (control smoothness / real-drone plausibility).

  COMPOSITE physicality_score in [0,1] = mean of [upsweep_frac, perp_ratio, (1-deadhang_frac), swing_amp_norm].

Deterministic (Gaussian mean), LSTM hidden-state carried + reset on done, ALL curricula OFF, radius pinned 0.13.
Emits a single `JSON {...}` line (machine-readable for the weekend autonomous loop) + a human summary.

Example:
    ./.venv/bin/python scripts/hook_grasp/eval_behavior.py --headless --num_envs 256 --steps 8000 \
        --indi_residual --actor_history_len 3 --checkpoint <model.pt> --out /tmp/beh.json
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Objective behavior test for an rsl_rl recurrent hook-grasp policy.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=8000)
parser.add_argument("--hold_steps", type=int, default=100)
parser.add_argument("--capture_radius", type=float, default=0.13)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--indi_residual", action="store_true", default=False)
parser.add_argument("--actor_history_len", type=int, default=0)
parser.add_argument("--near_wall", type=float, default=0.35, help="hook->bar dist (m) under which approach-behavior is scored")
parser.add_argument("--policy_kv", type=str, default="", help="ActorCriticRecurrent overrides — MUST match training arch (e.g. 'rnn_hidden_dim=512,rnn_num_layers=2') or the checkpoint won't load.")
parser.add_argument("--feedforward", action="store_true", default=False,
                    help="Checkpoint was trained with --feedforward (plain ActorCritic MLP, no RNN). Must match training.")
parser.add_argument("--hook_observer", action="store_true", default=False, help="Policy was trained with the hook-observer obs augmentation (+3); loads observer_final.pt/observer_latest.pt next to the checkpoint.")
parser.add_argument("--true_hook", action="store_true", default=False, help="Policy was trained with the TRUE-hook (privileged) obs augmentation (+3); appends ground-truth relative hook, no observer weights needed.")
parser.add_argument("--env_kv", type=str, default="", help="env_cfg overrides applied at eval (e.g. 'attachment.min_chord_angle_deg=10' to test the physical-swing gate). Match the TRUE task.")
parser.add_argument("--out", type=str, default="")
parser.add_argument("--dump_traj", type=str, default="",
                    help="If set, save per-step WORLD-frame trajectories (drone, hook, bar point, basket pose, attached) "
                         "for the first --dump_envs envs to this .npz — for local matplotlib scene animation (no Isaac render).")
parser.add_argument("--dump_envs", type=int, default=4)
parser.add_argument("--dump_steps", type=int, default=1200)
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
import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_rotate_inverse
from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rslrl_adapter import RslRlMarlAdapter, apply_indi_residual_cfg, make_runner_cfg  # noqa: E402

UPSWEEP_THRESH = 0.2      # v_up above this at catch = genuine up-sweep
DEADHANG_HORIZ = 0.15     # m: hook horizontally within this of straight-below-drone
DEADHANG_VREL = 0.3       # m/s: hook-vs-drone rel speed below this = not swinging
SWING_WINDOW = 60         # steps (~0.6 s) for the peak-to-peak height amplitude
SWING_AMP_REF = 0.4       # m: amplitude that maps to swing_amp_norm = 1.0


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    for _attr in ("capture_curriculum", "hold_curriculum", "spawn_curriculum", "attached_spawn_curriculum",
                  "chord_angle_curriculum"):
        if hasattr(env_cfg, _attr):
            setattr(env_cfg, _attr, False)
    if hasattr(env_cfg, "attachment") and args_cli.capture_radius > 0:
        env_cfg.attachment.capture_radius = args_cli.capture_radius
    if hasattr(env_cfg, "hold_success_steps"):
        env_cfg.hold_success_steps = 10_000_000
    if args_cli.indi_residual or args_cli.actor_history_len:
        H = args_cli.actor_history_len or env_cfg.history_len
        apply_indi_residual_cfg(env_cfg, actor_history_len=H, residual=args_cli.indi_residual)
    for tok in args_cli.env_kv.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.partition("=")
        lo = v.strip().lower()
        val = (lo == "true") if lo in ("true", "false") else (
            int(v) if v.strip().lstrip("-").isdigit() else float(v))
        obj, attr = (env_cfg, k.strip())
        if "." in k:
            head, attr = k.strip().split(".", 1)
            obj = getattr(env_cfg, head)
        setattr(obj, attr, val)
        print(f"[eval-behavior] env override: {k.strip()} = {val}")

    # wire the top-level hook mass into the startup mass event (matches train_rslrl; env_kv can set hook_mass_kg)
    if hasattr(env_cfg, "hook_mass_kg") and hasattr(env_cfg, "events"):
        _m = float(env_cfg.hook_mass_kg)
        env_cfg.events.randomize_hook_mass.params["mass_distribution_params"] = (_m, _m)
        print(f"[eval-behavior] hook bob mass = {_m*1000:.0f} g")

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    assert isinstance(base, DirectMARLEnv)
    adapter = RslRlMarlAdapter(base, hook_observer=args_cli.hook_observer, true_hook=args_cli.true_hook)
    if args_cli.true_hook:
        pass  # ground-truth appended live; no weights to load
    elif args_cli.hook_observer:
        adapter.observer_eval = True   # frozen estimator at eval (no online update)
        _ckpt_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
        _obs_wt = os.path.join(_ckpt_dir, "observer_final.pt")
        if not os.path.exists(_obs_wt):
            _obs_wt = os.path.join(_ckpt_dir, "observer_latest.pt")
        assert os.path.exists(_obs_wt), f"hook_observer eval needs observer weights next to the checkpoint; none at {_ckpt_dir}"
        adapter.load_observer(_obs_wt)
        print(f"[eval_behavior] loaded hook observer <- {_obs_wt}")
    # policy_kv MUST match the trained architecture or runner.load() fails on a size mismatch.
    pol = {}
    for tok in args_cli.policy_kv.split(","):
        tok = tok.strip()
        if not tok:
            continue
        k, _, v = tok.partition("=")
        try:
            pol[k.strip()] = int(v)
        except ValueError:
            try:
                pol[k.strip()] = float(v)
            except ValueError:
                pol[k.strip()] = v
    train_cfg = make_runner_cfg(experiment_name="behavior", max_iterations=1, device=device, logger="tensorboard",
                                policy_overrides=pol or None, feedforward=args_cli.feedforward)
    runner = OnPolicyRunner(adapter, train_cfg, log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    runner.eval_mode()
    policy = runner.alg.policy
    obs_normalizer = runner.obs_normalizer

    E, dev = base.num_envs, base.device
    hold_req = args_cli.hold_steps
    obs, _ = adapter.get_observations()
    policy.reset()

    # task accumulators (mirror eval_rslrl)
    ever_caught = torch.zeros(E, dtype=torch.bool, device=dev)
    ever_held = torch.zeros(E, dtype=torch.bool, device=dev)
    consec = torch.zeros(E, dtype=torch.long, device=dev)
    ep_hold_max = torch.zeros(E, dtype=torch.long, device=dev)
    n_ep = n_caught = n_held = 0

    # behavior accumulators (all on-device scalars)
    up_sum = torch.zeros((), device=dev); up_pos = torch.zeros((), device=dev); up_n = torch.zeros((), device=dev)
    perp_sum = torch.zeros((), device=dev); lat_sum = torch.zeros((), device=dev)
    # RELATIVE-frame (drone-pivot) perp/lat — matches what the swing REWARD optimizes (v_swing = hook_vel - drone_vel).
    # World-frame perp_ratio is contaminated by drone translation near the wall; the relative frame is the honest arc-direction metric.
    perp_rel_sum = torch.zeros((), device=dev); lat_rel_sum = torch.zeros((), device=dev)
    # COVARIANCE-based planarity + WHIRL invariant (research-panel fix, 2026-07-23): perp_ratio (an L1 marginal ratio)
    # aliases a cone, a planar-but-misaimed arc, and dead-hang all to 0.50. These are frame-independent discriminators:
    #  - swing_cov = 2x2 covariance of (v_perp_rel, v_lat_rel) -> eigenratio: planarity (rank-1=planar, ~1=isotropic/cone)
    #    and principal-axis angle vs the wall normal = AIM error.
    #  - L_z whirl = vertical component of (rel x v_swing): ~0 for a planar vertical arc, large for a cone/precession.
    cov_pp = torch.zeros(2, 2, device=dev); cov_p = torch.zeros(2, device=dev); cov_n = torch.zeros((), device=dev)
    lz_abs_sum = torch.zeros((), device=dev); whirl_sin_sum = torch.zeros((), device=dev)
    dead_n = torch.zeros((), device=dev); near_n = torch.zeros((), device=dev)
    jerk_sum = torch.zeros((), device=dev); jerk_n = torch.zeros((), device=dev)
    slack_sum = torch.zeros((), device=dev); attspeed_sum = torch.zeros((), device=dev); att_n = torch.zeros((), device=dev)
    overbox_n = torch.zeros((), device=dev); step_n = torch.zeros((), device=dev)  # drone-over-box-footprint fraction
    hook_h_hist = torch.zeros(E, SWING_WINDOW, device=dev)
    amp_sum = torch.zeros((), device=dev); amp_n = torch.zeros((), device=dev)

    # optional world-frame trajectory dump (first K envs) for local scene animation
    Kd = min(args_cli.dump_envs, E)
    dump = bool(args_cli.dump_traj)
    d_drone, d_hook, d_bar, d_att, d_done, d_bpos, d_bquat = [], [], [], [], [], [], []

    for t in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy.act_inference(obs_normalizer(obs))
            obs, _, dones, _ = adapter.step(actions)
            policy.reset(dones)

        i = 0
        hook_pos = base.hook_positions[:, i]
        hook_vel = base.hook_velocities[:, i]
        drone_pos = base.drone_positions[:, i]
        drone_vel = base.drone_linear_velocities[:, i]
        bar_pos, bar_axis, wall_normal = base._assigned_wall_point(hook_pos, i)
        wall_up = base.wall_up_w
        hook_bar = torch.norm(hook_pos - bar_pos, dim=-1)
        attached = base.attached.any(dim=-1)
        newly = base._newly_caught.any(dim=-1)

        if dump and t < args_cli.dump_steps:
            d_drone.append(drone_pos[:Kd].detach().cpu()); d_hook.append(hook_pos[:Kd].detach().cpu())
            d_bar.append(bar_pos[:Kd].detach().cpu()); d_att.append(attached[:Kd].detach().cpu())
            d_done.append(dones[:Kd].detach().cpu())
            d_bpos.append(base.basket_pos_w[:Kd].detach().cpu()); d_bquat.append(base.basket_quat_w[:Kd].detach().cpu())

        # DRONE-OVER-BOX fraction: is the drone horizontally inside the basket footprint (parked ON TOP)?
        drone_bf = quat_rotate_inverse(base.basket_quat_w, drone_pos - base.basket_pos_w)
        over_box = (drone_bf[:, 0].abs() < base.cfg.basket_half_x) & (drone_bf[:, 1].abs() < base.cfg.basket_half_y)
        overbox_n += over_box.float().sum(); step_n += base.num_envs

        v_up = (hook_vel * wall_up).sum(-1)
        v_perp = (hook_vel * wall_normal).sum(-1)
        v_lat = (hook_vel * bar_axis).sum(-1)

        # --- at the catch instant: up-sweep quality ---
        if newly.any():
            vc = v_up[newly]
            up_sum += vc.sum(); up_pos += (vc > UPSWEEP_THRESH).sum(); up_n += newly.sum()

        # --- during unattached approach NEAR the wall: perp/lateral + dead-hang ---
        near = (hook_bar < args_cli.near_wall) & (~attached)
        if near.any():
            perp_sum += v_perp[near].abs().sum()
            lat_sum += v_lat[near].abs().sum()
            # relative-frame (subtract drone/pivot velocity) — the arc direction the reward actually shapes
            v_swing = hook_vel - drone_vel
            vpr = (v_swing * wall_normal).sum(-1)      # perp (into/out of wall), relative
            vlr = (v_swing * bar_axis).sum(-1)         # lateral (along bar), relative
            perp_rel_sum += vpr[near].abs().sum()
            lat_rel_sum += vlr[near].abs().sum()
            rel = hook_pos - drone_pos
            horiz = torch.norm(rel[:, :2], dim=-1)
            vrel = torch.norm(hook_vel - drone_vel, dim=-1)
            dead = near & (horiz < DEADHANG_HORIZ) & (vrel < DEADHANG_VREL)
            dead_n += dead.sum(); near_n += near.sum()
            # covariance + whirl: only over genuinely-SWINGING near-wall samples (kill the 0/0 dead-hang alias)
            sw = near & (vrel > 0.15)
            if sw.any():
                p = torch.stack([vpr[sw], vlr[sw]], dim=-1)     # (M,2) relative swing velocity in the wall frame
                cov_pp += (p.unsqueeze(-1) * p.unsqueeze(-2)).sum(0)
                cov_p += p.sum(0); cov_n += sw.sum()
                # L_z = (rel x v_swing)_z, and normalized whirl = |L_z|/(|rel_h||v_h|) ~ sin(precession angle)
                lz = rel[sw, 0] * v_swing[sw, 1] - rel[sw, 1] * v_swing[sw, 0]
                relh = torch.norm(rel[sw, :2], dim=-1).clamp_min(1e-6)
                vh = torch.norm(v_swing[sw, :2], dim=-1).clamp_min(1e-6)
                lz_abs_sum += lz.abs().sum()
                whirl_sin_sum += (lz.abs() / (relh * vh)).sum()

        # --- swing amplitude: peak-to-peak hook height over a rolling window (near-wall unattached) ---
        hook_h_hist = torch.roll(hook_h_hist, -1, dims=1)
        hook_h_hist[:, -1] = hook_pos[:, 2]
        if t >= SWING_WINDOW and near.any():
            amp = hook_h_hist[near].max(dim=1).values - hook_h_hist[near].min(dim=1).values
            amp_sum += amp.sum(); amp_n += near.sum()

        # --- jerk while attached ---
        if attached.any() and hasattr(base, "_drone_jerk"):
            jm = torch.norm(base._drone_jerk, dim=-1).mean(dim=-1)
            jerk_sum += jm[attached].sum(); jerk_n += attached.sum()
        # --- POST-CATCH ROCKING metrics (2026-08-05, user: "rocking motion makes the wire slack after catch"):
        # slack_frac = fraction of attached steps with the wire SLACK (drone->attach dist < taut threshold);
        # att_speed = mean drone speed while attached (a steady taut hold should hover near-still). ---
        if attached.any():
            taut_thr_m = base._acfg.wire_length - base._acfg.slack_margin
            dist_da_m = torch.norm(drone_pos - base.attach_point_w[:, i], dim=-1)
            slack_sum += ((dist_da_m < taut_thr_m) & attached).float().sum()
            attspeed_sum += torch.norm(drone_vel, dim=-1)[attached].sum()
            att_n += attached.sum()

        # --- task bookkeeping ---
        ever_caught |= newly
        consec = torch.where(attached, consec + 1, torch.zeros_like(consec))
        ep_hold_max = torch.maximum(ep_hold_max, consec)
        ever_held |= consec >= hold_req
        done = dones.bool()
        if done.any():
            n_ep += int(done.sum()); n_caught += int((ever_caught & done).sum()); n_held += int((ever_held & done).sum())
            keep = ~done
            ever_caught &= keep; ever_held &= keep
            consec = torch.where(done, torch.zeros_like(consec), consec)
            ep_hold_max = torch.where(done, torch.zeros_like(ep_hold_max), ep_hold_max)

    catch_rate = n_caught / n_ep if n_ep else 0.0
    hold_gc = n_held / n_caught if n_caught else 0.0
    upsweep_frac = float((up_pos / up_n).item()) if up_n > 0 else 0.0
    v_up_catch = float((up_sum / up_n).item()) if up_n > 0 else 0.0
    denom = (perp_sum + lat_sum)
    perp_ratio = float((perp_sum / denom).item()) if denom > 0 else 0.0
    denom_rel = (perp_rel_sum + lat_rel_sum)
    perp_ratio_rel = float((perp_rel_sum / denom_rel).item()) if denom_rel > 0 else 0.0
    # covariance-based planarity + aim + whirl (frame-independent; only over swinging samples)
    planarity = 0.0; aim_deg = float("nan"); whirl_sin = float("nan"); lz_abs = float("nan")
    if cov_n > 1:
        mean_p = cov_p / cov_n
        cov = cov_pp / cov_n - torch.outer(mean_p, mean_p)
        cov = 0.5 * (cov + cov.t())                                    # symmetrize (numerical)
        evals, evecs = torch.linalg.eigh(cov)                          # ascending: evals[1] >= evals[0]
        l1 = float(evals[1].clamp_min(0).item()); l2 = float(evals[0].clamp_min(0).item())
        planarity = round(1.0 - l2 / l1, 4) if l1 > 1e-9 else 0.0      # 1=planar arc, 0=isotropic/cone
        pv = evecs[:, 1]                                               # principal axis (dominant swing direction)
        aim_deg = round(float(torch.rad2deg(torch.atan2(pv[1].abs(), pv[0].abs())).item()), 2)  # 0=perp(into wall),90=along bar
        whirl_sin = round(float((whirl_sin_sum / cov_n).item()), 4)    # 0=planar vertical arc, ->1=conical whirl
        lz_abs = round(float((lz_abs_sum / cov_n).item()), 4)
    deadhang_frac = float((dead_n / near_n).item()) if near_n > 0 else 0.0
    swing_amp = float((amp_sum / amp_n).item()) if amp_n > 0 else 0.0
    jerk = float((jerk_sum / jerk_n).item()) if jerk_n > 0 else float("nan")

    swing_amp_norm = min(1.0, swing_amp / SWING_AMP_REF)
    # Weight the UNGAMEABLE ARC metrics (perp_ratio, swing_amp) heavily. upsweep_frac and (1-deadhang) are partly
    # MECHANICAL consequences of the catch model (require_upsweep forces v_up>0 on every catch) and the chord-angle
    # gate (forbids near-vertical catches) -> they inflate easily without a real swing (adversarial audit 2026-07-20).
    physicality = round(0.35 * perp_ratio + 0.35 * swing_amp_norm
                        + 0.15 * upsweep_frac + 0.15 * (1.0 - deadhang_frac), 4)

    result = {
        "checkpoint": args_cli.checkpoint,
        "episodes": n_ep, "caught_episodes": n_caught,
        "catch_rate": round(catch_rate, 4),
        "hold_success_given_caught": round(hold_gc, 4),
        "upsweep_frac": round(upsweep_frac, 4),
        "v_up_catch": round(v_up_catch, 4),
        "perp_ratio": round(perp_ratio, 4),
        "perp_ratio_rel": round(perp_ratio_rel, 4),
        "planarity": planarity,        # 1=planar arc, 0=isotropic/conical (covariance eigenratio; de-aliases perp_ratio)
        "aim_deg": aim_deg,            # principal swing axis vs wall normal: 0=perpendicular(good), 90=along the bar
        "whirl_sin": whirl_sin,        # |L_z|/(|r_h||v_h|): 0=planar vertical arc, ->1=conical whirl
        "lz_abs": lz_abs,              # mean |azimuthal angular momentum| (whirl invariant, unnormalized)
        "deadhang_frac": round(deadhang_frac, 4),
        "swing_amp": round(swing_amp, 4),
        "jerk": round(jerk, 3) if jerk == jerk else None,
        "physicality_score": physicality,
        "over_box_frac": round(float((overbox_n / step_n).item()) if step_n > 0 else 0.0, 4),
        "slack_frac": round(float((slack_sum / att_n).item()) if att_n > 0 else 0.0, 4),   # attached steps w/ slack wire (rocking symptom)
        "att_speed": round(float((attspeed_sum / att_n).item()) if att_n > 0 else 0.0, 4), # mean drone speed while attached (want ~0)
        # gates: task goal AND physically-correct behavior
        "GOAL_catch_0.30": catch_rate >= 0.30,
        "GOAL_hold_0.80": hold_gc >= 0.80,
        # Strict gate now REQUIRES a real arc: perp-dominant swing (perp_ratio>=0.60, not the 0.5 skim midpoint)
        # AND meaningful arc amplitude (swing_amp>=0.20 m), on top of the (weaker) upsweep + low-deadhang criteria.
        "GOAL_physical": (upsweep_frac >= 0.6 and perp_ratio >= 0.60
                          and deadhang_frac <= 0.15 and swing_amp >= 0.20),
    }
    result["GOALS_MET_PHYSICAL"] = bool(
        result["GOAL_catch_0.30"] and result["GOAL_hold_0.80"] and result["GOAL_physical"]
    )
    print("JSON " + json.dumps(result))
    print("\n============== BEHAVIOR TEST ==============")
    for k, v in result.items():
        print(f"  {k:28s}: {v}")
    print("==========================================")
    if args_cli.out:
        os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=2); f.flush(); os.fsync(f.fileno())
    if dump and d_drone:
        import numpy as np
        np.savez(args_cli.dump_traj,
                 drone=torch.stack(d_drone, 0).numpy(), hook=torch.stack(d_hook, 0).numpy(),
                 bar=torch.stack(d_bar, 0).numpy(), attached=torch.stack(d_att, 0).numpy(),
                 done=torch.stack(d_done, 0).numpy(),
                 basket_pos=torch.stack(d_bpos, 0).numpy(), basket_quat=torch.stack(d_bquat, 0).numpy(),
                 half_x=float(base.cfg.basket_half_x), half_y=float(base.cfg.basket_half_y),
                 wall_h=float(base.cfg.basket_wall_height),
                 catch_rate=result["catch_rate"], hold=result["hold_success_given_caught"])
        print(f"[eval_behavior] wrote trajectory dump {args_cli.dump_traj}")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
