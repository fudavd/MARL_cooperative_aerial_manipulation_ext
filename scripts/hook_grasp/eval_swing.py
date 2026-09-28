# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verdict metric for the SWING-CONTROLLABILITY test (Isaac-flycrane-swing-test-v0).

Rolls out a trained policy on the clean directed-swing env and measures, in the COMMANDED swing frame
(u_hat = the episode's commanded horizontal axis, w_hat = the off-plane axis), whether the hook swings as a
PLANAR pendulum ALIGNED to the command:

    * planarity = 1 - lam2/lam1 of the 2x2 covariance of (v_swing.u_hat, v_swing.w_hat) over swinging steps
                  -> 1 = a rank-1 planar arc, 0 = isotropic / conical whirl.
    * aim_deg   = acute angle between the principal swing eigenvector and the COMMAND axis u_hat (=(1,0) in
                  this frame) -> 0 = the swing plane is aligned to the command, 90 = orthogonal to it.
    * whirl_sin = mean |L_z| / (|rel_h||v_h|)  (azimuthal angular momentum, normalized) -> 0 = planar
                  vertical arc, ->1 = conical whirl (Miles precession).
    * amp_along = mean peak-to-peak of the in-plane displacement rel.u_hat over a rolling window (arc size).

Reuses the SAME covariance/eigh/L_z math as eval_behavior.py, but in the command frame instead of the wall
frame, and with NO wall/catch gating (the drone free-floats). Compare the three arms:
    blind (no hook obs)  |  --true_hook (direct)  |  --hook_observer (predicted).

VERDICT: if the DIRECT arm reaches high planarity + low aim_deg across random commanded phi, the swing plane
IS controllable given hook state (grasp-task failure was a confound). If it CANNOT even with ground-truth
hook, the conical mode is a genuine dynamics/controllability limit.

Run (ALC, headless):
    ./.venv/bin/python scripts/hook_grasp/eval_swing.py --headless --num_envs 256 --steps 4000 \
        --true_hook --checkpoint logs/rsl_rl/<run>/model_XXXX.pt --out /tmp/swing_direct.json
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Swing-plane controllability metric for a swing-test policy.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-flycrane-swing-test-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=4000)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--swing_vmin", type=float, default=0.15,
                    help="min drone-relative hook speed (m/s) for a sample to count as 'swinging' (kills the dead-hang alias)")
parser.add_argument("--amp_window", type=int, default=60, help="rolling window (steps) for the in-plane peak-to-peak amplitude")
parser.add_argument("--hook_observer", action="store_true", default=False,
                    help="Policy trained with the observer augmentation (+3); loads observer_final.pt/observer_latest.pt next to the checkpoint.")
parser.add_argument("--true_hook", action="store_true", default=False,
                    help="Policy trained with the TRUE-hook (privileged) augmentation (+3); appends ground-truth rel hook.")
parser.add_argument("--policy_kv", type=str, default="",
                    help="ActorCriticRecurrent overrides — MUST match training arch (e.g. 'rnn_hidden_dim=512').")
parser.add_argument("--out", type=str, default="")
parser.add_argument("--dump_traj", type=str, default="",
                    help="If set, save per-step hook/drone trajectories + command for the first --dump_envs envs to this "
                         ".npz (for local matplotlib plotting; no Isaac render needed).")
parser.add_argument("--dump_envs", type=int, default=6, help="How many envs to dump trajectories for.")
parser.add_argument("--dump_steps", type=int, default=600, help="How many steps to dump (<= steps).")
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
from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rslrl_adapter import RslRlMarlAdapter, make_runner_cfg  # noqa: E402


def _parse_policy_kv(s: str) -> dict:
    pol = {}
    for tok in s.split(","):
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
    return pol


def main():
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    assert getattr(env_cfg, "swing_test", False), (
        f"eval_swing expects the swing-test env; task '{args_cli.task}' is not a swing-test cfg (swing_test flag absent)."
    )

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
        print(f"[eval_swing] loaded hook observer <- {_obs_wt}")

    train_cfg = make_runner_cfg(experiment_name="swing_eval", max_iterations=1, device=device, logger="tensorboard",
                                policy_overrides=_parse_policy_kv(args_cli.policy_kv) or None)
    runner = OnPolicyRunner(adapter, train_cfg, log_dir=None, device=device)
    runner.load(args_cli.checkpoint)
    runner.eval_mode()
    policy = runner.alg.policy
    obs_normalizer = runner.obs_normalizer

    E, dev = base.num_envs, base.device
    obs, _ = adapter.get_observations()
    policy.reset()

    # covariance / whirl accumulators in the COMMAND frame (mirrors eval_behavior's wall-frame block)
    cov_pp = torch.zeros(2, 2, device=dev); cov_p = torch.zeros(2, device=dev); cov_n = torch.zeros((), device=dev)
    lz_abs_sum = torch.zeros((), device=dev); whirl_sin_sum = torch.zeros((), device=dev)
    amp_along_sum = torch.zeros((), device=dev)  # mean |rel.u_hat| (in-plane displacement)
    amp_off_sum = torch.zeros((), device=dev)    # mean |rel.w_hat| (off-plane displacement, want ->0)
    disp_n = torch.zeros((), device=dev)
    inplane_hist = torch.zeros(E, args_cli.amp_window, device=dev)  # rolling rel.u_hat for peak-to-peak amplitude
    ptp_sum = torch.zeros((), device=dev); ptp_n = torch.zeros((), device=dev)

    # optional trajectory dump (first K envs) for local plotting -- no Isaac render needed
    K = min(args_cli.dump_envs, E)
    dump = bool(args_cli.dump_traj)
    d_rel, d_phi, d_done = [], [], []   # per step: hook-drone (K,3) | commanded phi (K,) | done (K,)

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
        u, w = base._cmd_u, base._cmd_w              # per-env commanded swing frame (horizontal)

        rel = hook_pos - drone_pos                    # wire vector (pivot -> bob)
        if dump and t < args_cli.dump_steps:
            d_rel.append(rel[:K].detach().cpu())
            d_phi.append(base._cmd_phi[:K].detach().cpu())
            d_done.append(dones[:K].detach().cpu())
        v_swing = hook_vel - drone_vel                # swing velocity about the pivot
        vpr = (v_swing * u).sum(-1)                   # in-plane (command axis) swing velocity
        vlr = (v_swing * w).sum(-1)                   # off-plane swing velocity
        rel_u = (rel * u).sum(-1)                     # in-plane displacement
        rel_w = (rel * w).sum(-1)                     # off-plane displacement

        # in-plane peak-to-peak amplitude over a rolling window
        inplane_hist = torch.roll(inplane_hist, -1, dims=1)
        inplane_hist[:, -1] = rel_u
        if t >= args_cli.amp_window:
            ptp = inplane_hist.max(dim=1).values - inplane_hist.min(dim=1).values
            ptp_sum += ptp.sum(); ptp_n += E

        # displacement means (all steps)
        amp_along_sum += rel_u.abs().sum(); amp_off_sum += rel_w.abs().sum(); disp_n += E

        # covariance + whirl over genuinely-swinging samples (kill the 0/0 dead-hang alias)
        vrel = torch.norm(v_swing, dim=-1)
        sw = vrel > args_cli.swing_vmin
        if sw.any():
            p = torch.stack([vpr[sw], vlr[sw]], dim=-1)          # (M,2) swing velocity in the command frame
            cov_pp += (p.unsqueeze(-1) * p.unsqueeze(-2)).sum(0)
            cov_p += p.sum(0); cov_n += sw.sum()
            lz = rel[sw, 0] * v_swing[sw, 1] - rel[sw, 1] * v_swing[sw, 0]   # L_z = (rel x v_swing)_z
            relh = torch.norm(rel[sw, :2], dim=-1).clamp_min(1e-6)
            vh = torch.norm(v_swing[sw, :2], dim=-1).clamp_min(1e-6)
            lz_abs_sum += lz.abs().sum()
            whirl_sin_sum += (lz.abs() / (relh * vh)).sum()

    # --- reduce ---
    planarity = 0.0; aim_deg = float("nan"); whirl_sin = float("nan"); lz_abs = float("nan")
    if cov_n > 1:
        mean_p = cov_p / cov_n
        cov = cov_pp / cov_n - torch.outer(mean_p, mean_p)
        evals, evecs = torch.linalg.eigh(cov)                    # ascending: evals[1] >= evals[0]
        l1, l2 = float(evals[1].item()), float(evals[0].item())
        planarity = round(1.0 - l2 / l1, 4) if l1 > 1e-9 else 0.0
        pv = evecs[:, 1]                                          # principal swing axis, in (u_hat, w_hat) coords
        # aim = angle of the principal axis from the COMMAND axis u_hat = (1,0): 0 = aligned to command, 90 = orthogonal
        aim_deg = round(float(torch.rad2deg(torch.atan2(pv[1].abs(), pv[0].abs())).item()), 2)
        whirl_sin = round(float((whirl_sin_sum / cov_n).item()), 4)
        lz_abs = round(float((lz_abs_sum / cov_n).item()), 4)

    amp_along = round(float((amp_along_sum / disp_n).item()), 4) if disp_n > 0 else 0.0
    amp_off = round(float((amp_off_sum / disp_n).item()), 4) if disp_n > 0 else 0.0
    swing_amp = round(float((ptp_sum / ptp_n).item()), 4) if ptp_n > 0 else 0.0
    swing_frac = round(float((cov_n / (disp_n + 1e-9)).item()), 4)

    arm = "true_hook(DIRECT)" if args_cli.true_hook else ("observer(PREDICTED)" if args_cli.hook_observer else "hook-blind")
    result = {
        "arm": arm,
        "checkpoint": args_cli.checkpoint,
        "num_envs": args_cli.num_envs, "steps": args_cli.steps,
        "planarity": planarity,        # 1 = planar arc, 0 = isotropic / conical (command-frame covariance eigenratio)
        "aim_deg": aim_deg,            # principal swing axis vs COMMANDED axis: 0 = aligned to command (good), 90 = orthogonal
        "whirl_sin": whirl_sin,        # |L_z|/(|r_h||v_h|): 0 = planar vertical arc, ->1 = conical whirl
        "lz_abs": lz_abs,              # mean |azimuthal angular momentum| (whirl invariant, unnormalized)
        "swing_amp": swing_amp,        # in-plane (command axis) peak-to-peak displacement over the window (arc size, m)
        "amp_along": amp_along,        # mean |rel.u_hat| in-plane displacement (m)
        "amp_off": amp_off,            # mean |rel.w_hat| off-plane displacement (m; want << amp_along)
        "swing_frac": swing_frac,      # fraction of steps genuinely swinging (not dead-hang)
    }
    print("\n==================== SWING-CONTROLLABILITY ====================")
    print(f"  arm            : {arm}")
    print(f"  planarity      : {planarity:>7}   (1=planar arc, 0=conical/isotropic)")
    print(f"  aim_deg        : {aim_deg:>7}   (0=swing plane ALIGNED to command, 90=orthogonal)")
    print(f"  whirl_sin      : {whirl_sin:>7}   (0=planar, ->1=conical whirl / Miles precession)")
    print(f"  swing_amp (ptp): {swing_amp:>7} m (in-plane arc size)")
    print(f"  amp_along/off  : {amp_along:>7} / {amp_off:<7} m (in-plane vs off-plane displacement)")
    print(f"  swing_frac     : {swing_frac:>7}   (fraction of steps actually swinging)")
    print("==============================================================\n")
    if args_cli.out:
        with open(args_cli.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[eval_swing] wrote {args_cli.out}")

    if dump and d_rel:
        import numpy as np
        rel = torch.stack(d_rel, 0).numpy()      # (T, K, 3) hook-drone
        phi = torch.stack(d_phi, 0).numpy()      # (T, K)    commanded azimuth
        done = torch.stack(d_done, 0).numpy()    # (T, K)    episode-boundary flags
        np.savez(args_cli.dump_traj, rel=rel, phi=phi, done=done, arm=arm, **{k: v for k, v in result.items()
                 if isinstance(v, (int, float))})
        print(f"[eval_swing] wrote trajectory dump {args_cli.dump_traj}  rel{rel.shape}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # os._exit in finally would otherwise SWALLOW the traceback (failures look like silent crashes)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    finally:
        os._exit(0)


