# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a video of a trained rsl_rl recurrent (LSTM) HOOK-BLIND actor on the hook-grasp env.

Combines play_mappo_video.py's reliable manual-render pipeline (env.render() -> frames -> imageio ->
os._exit) with eval_rslrl.py's recurrent-policy loading: keeps the env MULTI-AGENT via the adapter,
runs the LSTM actor DETERMINISTICALLY (act_inference = Gaussian mean) on the 135-dim HOOK-FREE obs
with hidden-state carry + reset-on-done. True task (curricula off, radius 0.13).

--scan finds the best-holding env index (no render); then re-run with --env_index to follow it.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Video of a trained rsl_rl recurrent hook-blind controller.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-hook-grasp-v0", help="Task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to an rsl_rl model_*.pt checkpoint.")
parser.add_argument("--steps", type=int, default=900, help="Rollout steps to record.")
parser.add_argument("--frame_every", type=int, default=3, help="Capture a frame every N steps.")
parser.add_argument("--fps", type=int, default=20, help="Output video fps.")
parser.add_argument("--seed", type=int, default=7, help="Seed.")
parser.add_argument("--capture_radius", type=float, default=0.13, help="Catch radius (m); FINAL task=0.13.")
parser.add_argument("--env_index", type=int, default=0, help="Which env the follow-camera tracks.")
parser.add_argument("--scan", action="store_true", default=False, help="No render; report best-holding env index.")
parser.add_argument("--out_dir", type=str, default="scripts/hook_grasp/_render_rslrl", help="Output dir.")
parser.add_argument("--tag", type=str, default="lstm_policy", help="Output file basename.")
parser.add_argument("--indi_residual", action="store_true", default=False,
                    help="Actor was trained with the INDI-residual obs (must match the checkpoint).")
parser.add_argument("--actor_history_len", type=int, default=0,
                    help="ACTOR frame-stack depth H used in training (0 -> env history_len); must match checkpoint.")
parser.add_argument("--policy_kv", type=str, default="",
                    help="ActorCriticRecurrent overrides — MUST match training arch (e.g. 'rnn_hidden_dim=512,rnn_num_layers=2') or the checkpoint won't load.")
parser.add_argument("--feedforward", action="store_true", default=False,
                    help="Checkpoint is a v84 feedforward ActorCritic (plain MLP, no RNN). Must match training.")
parser.add_argument("--true_hook", action="store_true", default=False,
                    help="Actor trained with the TRUE-hook (privileged) obs augmentation (+3); appends ground-truth rel hook (must match checkpoint).")
parser.add_argument("--hook_observer", action="store_true", default=False,
                    help="Actor trained with the learned hook-observer obs augmentation (+3); loads observer_final.pt/observer_latest.pt next to the checkpoint.")
parser.add_argument("--static_cam", action="store_true", default=False,
                    help="Use a fixed overview camera instead of the asset-body follow camera (the follow-cam "
                         "viewport controller silently kills the process on the hybrid-GPU laptop, 2026-08-04).")
parser.add_argument("--env_kv", type=str, default="",
                    help="env_cfg overrides applied at render (e.g. 'attachment.min_chord_angle_deg=25,"
                         "basket_wall_z_hi=0.22') — match the TRUE task the checkpoint was trained/evaluated on.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import MARL_mav_carry_ext.tasks  # noqa: F401
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv
from isaaclab_tasks.utils import parse_env_cfg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rslrl_adapter import RslRlMarlAdapter, apply_indi_residual_cfg, make_runner_cfg  # noqa: E402


def main():
    os.makedirs(args_cli.out_dir, exist_ok=True)
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed

    # honest render at the TRUE task: disable every curriculum / aid (same list as eval_rslrl.py)
    for _attr in ("capture_curriculum", "hold_curriculum", "spawn_curriculum", "attached_spawn_curriculum"):
        if hasattr(env_cfg, _attr):
            setattr(env_cfg, _attr, False)
    if hasattr(env_cfg, "attachment") and args_cli.capture_radius > 0:
        env_cfg.attachment.capture_radius = args_cli.capture_radius
    if hasattr(env_cfg, "hold_success_steps"):
        env_cfg.hold_success_steps = 10_000_000

    # structural actor-obs config (must MATCH the trained checkpoint)
    if args_cli.indi_residual or args_cli.actor_history_len:
        H = args_cli.actor_history_len or env_cfg.history_len
        apply_indi_residual_cfg(env_cfg, actor_history_len=H, residual=args_cli.indi_residual)
    # task-gate overrides (match the TRUE task; same parser as eval_behavior)
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
        print(f"[rslrl-video] env override: {k.strip()} = {val}")
    if hasattr(env_cfg, "chord_angle_curriculum"):
        env_cfg.chord_angle_curriculum = False   # pin the chord gate at its (overridden) target
    # wire the top-level hook mass into the startup mass event (matches train/eval; env_kv can set hook_mass_kg)
    if hasattr(env_cfg, "hook_mass_kg") and hasattr(env_cfg, "events"):
        _m = float(env_cfg.hook_mass_kg)
        env_cfg.events.randomize_hook_mass.params["mass_distribution_params"] = (_m, _m)
        print(f"[rslrl-video] hook bob mass = {_m*1000:.0f} g")

    if args_cli.static_cam:
        # fixed overview framing the spawn/basket region (basket spawns within ~±1 m of the env origin)
        env_cfg.viewer.eye = (3.2, 3.2, 2.4)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.9)
    else:
        # follow-camera on the drone body (same framing as play_mappo_video)
        env_cfg.viewer.origin_type = "asset_body"
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.body_name = "Falcon_base_link_inertia"
        env_cfg.viewer.eye = (2.2, 2.2, 0.6)
        env_cfg.viewer.lookat = (0.0, 0.0, -0.4)
        if hasattr(env_cfg.viewer, "env_index"):
            env_cfg.viewer.env_index = args_cli.env_index

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    base = env.unwrapped
    assert isinstance(base, DirectMARLEnv), "play_rslrl_video expects a DirectMARLEnv (multi-agent) task."
    adapter = RslRlMarlAdapter(base, hook_observer=args_cli.hook_observer, true_hook=args_cli.true_hook)
    if args_cli.hook_observer:
        adapter.observer_eval = True   # frozen estimator at render (no online update)
        import glob as _glob
        _ckpt_dir = os.path.dirname(os.path.abspath(args_cli.checkpoint))
        # accept renamed copies too (best_checkpoints convention prefixes the run name)
        _cands = ([os.path.join(_ckpt_dir, "observer_final.pt"), os.path.join(_ckpt_dir, "observer_latest.pt")]
                  + sorted(_glob.glob(os.path.join(_ckpt_dir, "*observer_final.pt")))
                  + sorted(_glob.glob(os.path.join(_ckpt_dir, "*observer_latest.pt"))))
        _obs_wt = next((p for p in _cands if os.path.exists(p)), None)
        assert _obs_wt, f"hook_observer render needs observer weights next to the checkpoint; none in {_ckpt_dir}"
        adapter.load_observer(_obs_wt)
        print(f"[rslrl-video] loaded hook observer <- {_obs_wt}")

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
    train_cfg = make_runner_cfg(experiment_name="render", max_iterations=1, device=device, logger="tensorboard",
                                policy_overrides=pol or None, feedforward=args_cli.feedforward)
    runner = OnPolicyRunner(adapter, train_cfg, log_dir=None, device=device)
    print(f"[rslrl-video] loading checkpoint: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    runner.eval_mode()
    policy = runner.alg.policy
    obs_normalizer = runner.obs_normalizer

    obs, _ = adapter.get_observations()
    actor_dim = int(obs.shape[-1])
    expected_actor = int(next(iter(base.cfg.observation_spaces.values())))
    if args_cli.true_hook or args_cli.hook_observer:
        expected_actor += 3   # adapter appends the (true/estimated) relative hook position
    print(f"[rslrl-video] actor_obs_dim={actor_dim} (expected {expected_actor})")
    assert actor_dim == expected_actor, f"actor obs {actor_dim} != expected {expected_actor} (indi/H/augment mismatch?)"
    policy.reset()

    E = base.num_envs
    dev = base.device
    ei = args_cli.env_index
    frames, frame_steps = [], []
    caught_total, detached_total, attached_steps = 0, 0, 0
    min_hook_bar = float("inf")
    success_any = False
    consec = torch.zeros(E, dtype=torch.long, device=dev)
    best_hold = torch.zeros(E, dtype=torch.long, device=dev)

    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy.act_inference(obs_normalizer(obs))
            obs, _, dones, _ = adapter.step(actions)
            policy.reset(dones)

        att = base.attached.any(dim=-1)
        consec = torch.where(att, consec + 1, torch.zeros_like(consec))
        best_hold = torch.maximum(best_hold, consec)
        caught_total += int(base._newly_caught.any().item())
        detached_total += int(base._just_detached.any().item())
        attached_steps += int(att[ei].item())
        if hasattr(base, "metrics") and "hook_bar_dist" in base.metrics:
            min_hook_bar = min(min_hook_bar, float(base.metrics["hook_bar_dist"][ei].item()))
        if hasattr(base, "succeeded") and bool(base.succeeded.any().item()):
            success_any = True

        if not args_cli.scan and step % args_cli.frame_every == 0:
            frame = base.render()
            if frame is not None:
                frames.append(np.asarray(frame))
                frame_steps.append(step)

    topk = torch.topk(best_hold, k=min(6, E))
    bh = "best-hold envs (idx:steps): " + " ".join(
        f"{int(i)}:{int(v)}" for v, i in zip(topk.values.tolist(), topk.indices.tolist()))
    print("[rslrl-video] " + bh, flush=True)
    with open(os.path.join(args_cli.out_dir, f"{args_cli.tag}_besthold.txt"), "w") as _f:
        _f.write(bh + "\n")
        _f.flush()

    if frames:
        from PIL import Image
        Image.fromarray(frames[len(frames) // 2]).save(os.path.join(args_cli.out_dir, f"{args_cli.tag}_hero.png"))
        Image.fromarray(frames[-1]).save(os.path.join(args_cli.out_dir, f"{args_cli.tag}_last.png"))
        try:
            import imageio.v2 as imageio
            mp4 = os.path.join(args_cli.out_dir, f"{args_cli.tag}.mp4")
            imageio.mimsave(mp4, frames, fps=args_cli.fps)
            print(f"[rslrl-video] wrote {mp4} ({len(frames)} frames @ {args_cli.fps}fps)")
        except Exception as e:  # noqa: BLE001
            print(f"[rslrl-video] mp4 save skipped: {e}")

    summary = "\n".join([
        "==================== RSL_RL LSTM ROLLOUT ====================",
        f" checkpoint        : {args_cli.checkpoint}",
        f" steps / frames    : {args_cli.steps} / {len(frames)}",
        f" catches / detaches: {caught_total} / {detached_total}",
        f" min hook<->bar (m): {min_hook_bar:.3f}   (capture radius {base._acfg.capture_radius})",
        f" attached steps    : {attached_steps} / {args_cli.steps}",
        f" success (held)    : {success_any}",
        "============================================================",
    ])
    print("\n" + summary)
    with open(os.path.join(args_cli.out_dir, f"{args_cli.tag}_summary.txt"), "w") as f:
        f.write(summary + "\n")
        f.flush()


if __name__ == "__main__":
    # NB: the bare `try: main() finally: os._exit(0)` pattern SWALLOWS every traceback (os._exit fires before
    # the interpreter can print the propagating exception) -> failures look like silent crashes. Print first.
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        os._exit(0)
