# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a presentation clip of one controller on the cooperative transport task.

Drives BOTH controller families through an identical env / camera / rollout loop, so the resulting
clips are visually comparable side by side:

  * ``--policy hebbian`` : an evolved Hebbian plasticity rule (best.pt from scripts/hebbian/train.py)
  * ``--policy mappo``   : the paper's per-agent MAPPO actors (HuggingFace 3_mav.pt)

Examples (see make_videos.sh for the full presentation set):

    python presentation/render_video.py --headless --policy hebbian \
        --checkpoint presentation/checkpoints/hebbian_ABCDW_fixed_fit190.pt \
        --task Isaac-flycrane-payload-decentralized-hovering-v0 --out presentation/videos/x.mp4

    python presentation/render_video.py --headless --policy mappo \
        --checkpoint presentation/checkpoints/mappo_paper_3_mav.pt \
        --task Isaac-flycrane-payload-decentralized-hovering-rand-v0 --attachment_per_env 1 \
        --out presentation/videos/y.mp4
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render a transport-task clip for the presentation.")
parser.add_argument("--policy", choices=["hebbian", "mappo"], required=True, help="Controller family.")
parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint for that controller.")
parser.add_argument("--task", type=str, default="Isaac-flycrane-payload-decentralized-hovering-v0",
                    help="Task id: ...-hovering-v0 (fixed attachment) or ...-hovering-rand-v0 (randomized).")
parser.add_argument("--attachment_per_env", type=int, default=-1,
                    help="Randomized-attachment scope on the rand task: 1 = a fresh cable geometry per env "
                         "(-1 = leave the task default).")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel environments (the camera frames env 0).")
parser.add_argument("--steps", type=int, default=600, help="Control steps to roll out.")
parser.add_argument("--frame_every", type=int, default=2, help="Capture a frame every N steps.")
parser.add_argument("--fps", type=int, default=25, help="Output video fps.")
parser.add_argument("--seed", type=int, default=123, help="Seed (shared across clips = same goal sequence).")
parser.add_argument("--out", type=str, required=True, help="Output .mp4 path.")
parser.add_argument("--cam_eye", type=float, nargs=3, default=(4.2, 4.2, 2.6), help="Camera position.")
parser.add_argument("--cam_lookat", type=float, nargs=3, default=(0.0, 0.0, 1.0), help="Camera target.")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True   # offscreen rendering (needed for env.render() under --headless)
sys.argv = [sys.argv[0]]

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import MARL_mav_carry_ext.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from MARL_mav_carry_ext.algorithms.hebbian.adapter import HebbianPolicy


class PaperMappoPolicy:
    """Deterministic (mean-action) forward pass of the paper's per-agent MAPPO actors.

    Reproduces skrl's GaussianMixin inference: obs -> RunningStandardScaler -> ELU-MLP mean. Same
    reconstruction as scripts/hebbian/compare_paper_vs_hebbian.py, kept here so this script stands alone.
    """

    def __init__(self, uenv, ckpt_path, agents, obs_dim=135, act_dim=5):
        from skrl.resources.preprocessors.torch import RunningStandardScaler

        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.agents = list(agents)
        self.device = uenv.device
        self.nets, self.pre = {}, {}
        for a in self.agents:
            pol = ck[a]["policy"]
            # skrl_mappo_cfg.yaml: layers [1024, 512, 256, 128], ELU activations, linear action output
            net = nn.Sequential(
                nn.Linear(obs_dim, 1024), nn.ELU(),
                nn.Linear(1024, 512), nn.ELU(),
                nn.Linear(512, 256), nn.ELU(),
                nn.Linear(256, 128), nn.ELU(),
                nn.Linear(128, act_dim),
            )
            sd = {k[len("net_container."):]: v for k, v in pol.items() if k.startswith("net_container.")}
            net.load_state_dict(sd)
            net.eval().to(self.device)
            self.nets[a] = net
            sp = RunningStandardScaler(size=obs_dim, device=self.device)
            sp.load_state_dict(ck[a]["state_preprocessor"])
            sp.eval()
            self.pre[a] = sp

    def reset(self, ids=None):
        pass

    @torch.no_grad()
    def act(self, obs):
        return {a: self.nets[a](self.pre[a](obs[a].to(self.device), train=False)) for a in self.agents}


def main():
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    device = args_cli.device if args_cli.device is not None else "cuda:0"
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    env_cfg.partial_obs = True

    # Observation width is set by the controller: the paper MAPPO actors were trained on 3 stacked frames
    # (135), the Hebbian rules on whatever history the checkpoint stores (1 frame -> 45).
    if args_cli.policy == "mappo":
        env_cfg.history_len = 3
        obs_dim = 45 * 3
    else:
        payload = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
        env_cfg.history_len = int(payload.get("history_len", 1))
        obs_dim = 45 * env_cfg.history_len
    env_cfg.observation_spaces = {a: obs_dim for a in env_cfg.possible_agents}

    # Randomized-attachment scope (rand task only): per-env cable geometry needs replicate_physics off.
    # __post_init__ has already run, so both flags are set manually (mirrors scripts/hebbian/train.py).
    if args_cli.attachment_per_env >= 0 and hasattr(env_cfg, "attachment_per_env"):
        env_cfg.attachment_per_env = bool(args_cli.attachment_per_env)
        if args_cli.attachment_per_env == 1:
            env_cfg.scene.replicate_physics = False
        print(f"[render] attachment_per_env={env_cfg.attachment_per_env} "
              f"(replicate_physics={env_cfg.scene.replicate_physics})")

    # static overview camera framing the goal workspace (goals span ~+-1 m in x/y, 0.5-1.5 m in z)
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = tuple(args_cli.cam_eye)
    env_cfg.viewer.lookat = tuple(args_cli.cam_lookat)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    uenv = env.unwrapped
    agents = list(uenv.cfg.possible_agents)

    if args_cli.policy == "mappo":
        policy = PaperMappoPolicy(uenv, args_cli.checkpoint, agents, obs_dim=obs_dim)
        train_best = float("nan")
    else:
        policy = HebbianPolicy(uenv, args_cli.checkpoint)
        train_best = float(torch.load(args_cli.checkpoint, map_location="cpu",
                                      weights_only=False).get("best_fitness", float("nan")))
    print(f"[render] policy={args_cli.policy} obs_dim={obs_dim} train_best={train_best:.1f} "
          f"task={args_cli.task}")

    obs, _ = env.reset()
    policy.reset()
    frames = []
    ep_ret = torch.zeros(uenv.num_envs, device=uenv.device)
    pos_sum = torch.zeros(uenv.num_envs, device=uenv.device)
    steps = torch.zeros(uenv.num_envs, device=uenv.device)
    rets, poss, n_done = [], [], 0
    pos_trace = []

    for step in range(args_cli.steps):
        with torch.inference_mode():
            obs, rewards, terminated, time_outs, _ = env.step(policy.act(obs))
        ep_ret += torch.stack([rewards[a] for a in agents], 0).sum(0)
        pos_sum += uenv.metrics["position_error"]
        pos_trace.append(float(uenv.metrics["position_error"].mean()))
        steps += 1
        done = terminated[agents[0]].bool() | time_outs[agents[0]].bool()
        ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
        for i in ids.tolist():
            n = max(steps[i].item(), 1.0)
            rets.append(ep_ret[i].item())
            poss.append((pos_sum[i] / n).item())
            n_done += 1
        if ids.numel() > 0:
            ep_ret[ids] = 0.0
            pos_sum[ids] = 0.0
            steps[ids] = 0.0
            policy.reset(ids.tolist())
        if step % args_cli.frame_every == 0:
            frame = env.unwrapped.render()
            if frame is not None:
                frames.append(np.asarray(frame))

    summary = {
        "policy": args_cli.policy,
        "checkpoint": os.path.basename(args_cli.checkpoint),
        "task": args_cli.task,
        "attachment_per_env": args_cli.attachment_per_env,
        "obs_dim": obs_dim,
        "train_best_fitness": None if train_best != train_best else round(train_best, 1),
        "steps": args_cli.steps,
        "frames": len(frames),
        "episodes_finished": n_done,
        "mean_episode_return": round(float(np.mean(rets)), 2) if rets else None,
        "mean_position_error_m": round(float(np.mean(poss)), 3) if poss else None,
        # clip-wide tracking error: averaged over every rendered step, so a clip shorter than one
        # episode still reports how well the load was held on its goal
        "clip_mean_position_error_m": round(float(np.mean(pos_trace)), 3) if pos_trace else None,
    }
    print("\n" + json.dumps(summary, indent=2))
    with open(os.path.splitext(args_cli.out)[0] + "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if frames:
        import imageio.v2 as imageio
        imageio.mimsave(args_cli.out, frames, fps=args_cli.fps)
        print(f"[render] wrote {args_cli.out} ({len(frames)} frames @ {args_cli.fps} fps)")
    else:
        print("[render] no frames captured (rendering unavailable on this machine?)")
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
        os._exit(0)   # skip the headless Isaac Sim teardown (can hang and hold the GPU)
