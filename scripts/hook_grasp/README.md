# Hook-grasp task (PPO)

A single Falcon drone carries a flexible 0.55 m wire with a 300 g hook. It must swing the hook up onto a
horizontal bar on the wall of a basket (random pose per episode, observed), latch, and then keep the wire
taut. Task id: `Isaac-flycrane-hook-grasp-v0`.

* **Env**: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/hook_grasp/`
  (`marl_hook_env.py`, `marl_hook_env_cfg.py`, surrogate catch model `attachment.py`).
* **Asset**: `assets/flyhook.py` -> `assets/data/AMR/flyhook/flyhook.usd` (drone + 8-link wire + hook,
  built from `falcon.urdf` by `build_flyhook_asset.py` + `fix_drone_visual.py`; the USD is committed).
* **Learning**: PPO from [rsl_rl](https://github.com/leggedrobotics/rsl_rl) (`pip install rsl-rl-lib`),
  asymmetric actor-critic. Config: `hook_grasp/agents/rsl_rl_ppo_cfg.py`.

## Workflow

```bash
# train (512 envs x 20k iterations = the full recipe; ~7 h on an RTX 4090, ~2x that on an 8 GB laptop GPU)
python scripts/hook_grasp/train.py --headless
# quick smoke test
python scripts/hook_grasp/train.py --headless --num_envs 32 --max_iterations 3

# evaluate a checkpoint on the final task (all curricula off, deterministic policy)
python scripts/hook_grasp/eval.py --headless --checkpoint logs/rsl_rl/hook_grasp/<timestamp>/model_19999.pt

# record a video (follow camera)
python scripts/hook_grasp/play.py --headless --checkpoint logs/rsl_rl/hook_grasp/<timestamp>/model_19999.pt
```

`train.py` arguments: `--num_envs`, `--max_iterations`, `--seed`, `--run_name`, `--resume <ckpt>`, `--wandb`,
`--wandb_project`, plus the usual Isaac Lab launcher flags (`--headless`, `--device`). Everything else (task,
curricula, reward weights, network, PPO hyper-parameters) is defined in the two config files and needs no
command-line overrides. Checkpoints are saved every 100 iterations to `logs/rsl_rl/<run_name>/<timestamp>/`
together with the resolved configs (`params/env.yaml`, `params/agent.yaml`).

## What the policy sees and does

* **Actor observation (153)**: 3 stacked frames of [basket pose, assigned-wall centre, drone pose/velocities,
  goal error] (45) + the 6-dim body-frame INDI disturbance residual (the hook/wire wrench the drone can measure
  onboard). The actor never sees the hook or bar position.
* **Critic observation (180)**: 3 stacked frames of the privileged state (adds hook position/velocity,
  hook-to-bar error, attached flag, wire tension, hold progress).
* **Action (5)**: linear acceleration (3) + body rates (2), tracked by the Geometric -> INDI -> motor cascade.
* **Catch model** (`attachment.py`): within 5 cm of the bar, moving into the wall and sweeping upward, with the
  wire swung out >= 25 deg outward from the wall, a catch is drawn once per approach. A slack wire (drone closer
  than the wire length for more than a short dwell) drops the hook.
* **Success**: hook attached and wire taut for 100 consecutive control steps (1 s); ends the episode.

## Training curricula

All anneal with the number of control steps and reach the final task well before 20k iterations: capture radius
0.13 -> 0.05 m, minimum swing-out angle 0 -> 25 deg, required hold 30 -> 100 steps, spawn spread (distance std
and azimuth) tight -> full, initial swing-assist velocity 2.2 -> 0 m/s, fraction of episodes that start already
attached 0.5 -> 0. `MARLHookGraspEnvCfg.disable_curricula()` pins all of them (used by `eval.py`/`play.py`).

## Reference result

Final-task evaluation of the released recipe (single seed 42, 512 envs x 20k iterations; `eval.py` with
256 envs x 4000 steps):

| metric | value | goal |
|---|---|---|
| catch rate | 0.90 | >= 0.30 |
| hold success given a catch (100 steps) | 0.97 | >= 0.80 |

Catch rate is seed-sensitive; train a few seeds and keep the best if you need the top number.

## Tests

The surrogate catch/slack model has plain-PyTorch tests (no Isaac Sim needed):

```bash
python exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/hook_grasp/tests/test_attachment.py
```
