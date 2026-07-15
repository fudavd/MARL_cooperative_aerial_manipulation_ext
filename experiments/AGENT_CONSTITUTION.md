# Hook-Grasp Improvement Fleet — CONSTITUTION (binding on every agent)

**Mission:** make a single drone learn a *genuine physical pendulum up-swing* to hook a basket
wall-bar, hook-blind, real-drone-transferable. Improve the SETUP (rewards / curricula /
hyperparameters / spawn fields) — never the task, never the observation.

Deadline for the autonomous loop: **2026-07-16 11:00 CEST**. Never idle; when an arm matures,
read its result and launch the next idea.

---

## 0. SUCCESS DEFINITION (immutable)
Judged by DETERMINISTIC eval (mean_actions), all curricula DISABLED:
- `catch_rate ≥ 0.30`  AND  `hold_success_given_caught ≥ 0.80`  at `hold_success_steps = 100`.

## 1. HARD CONSTRAINTS — NEVER CHANGE THESE (a change here = failed task)
1. **Observation space.** Actor obs = `_drone_obs_frame`, 45×3 = **135-dim, HOOK-FREE**.
   - NEVER add hook state / hook→wall distance / tension / attach flags to the ACTOR obs.
   - NEVER change the 135 dim. NEVER change the critic state dim (60×3 = 180).
   - `observation_spaces = {a: 45*3}`, `state_space = 60*3` in marl_hook_env_cfg.py are LOCKED.
2. **The task / physics.** attachment.py FINAL values are LOCKED:
   `capture_radius=0.13`, `require_upsweep=True`, `upsweep_v_min`, detach probabilities,
   `wire_length=0.55`. Basket/wall geometry (`basket_wall_z_lo/hi`, `basket_half_x/y`),
   episode length, decimation, control mode — LOCKED.
3. **Success thresholds** (§0) and `hold_success_steps=100` at eval — LOCKED.
4. **Algorithm = MAPPO / CTDE.** Separate hook-blind actor + privileged critic
   (`use_state_space=True`). Do NOT feed the state to the actor. Do NOT revert to the PPO
   `state_as_observation` path (that was the rejected "hook cheat").

## 2. ALLOWED LEVERS (this is where all improvement happens)
- **Reward shaping** — weights + NEW terms, but every new term MUST be config-gated in
  marl_hook_env_cfg.py and **default OFF / 0.0** so other arms are unaffected. Numerically
  verify sign/scale before launching.
- **Curricula** — schedules AND start values (spawn distance, capture-radius anneal, hold-time
  ramp, attached-spawn reverse curriculum, swing-assist). MUST anneal to the true task values
  in §1 (e.g. capture_radius_start may be 0.25 but must reach 0.13; hold_steps_start 30 → 100).
- **Hyperparameters** — LR, clips, entropy, rollouts, mini_batches, network, `max_log_std`,
  KL/adaptive-LR scheduler, preprocessors. Edit skrl_mappo_cfg.yaml or pass `agent.*` overrides.
- **Spawn velocity/position fields** — swing_assist_v0, spawn_dist_mu, inverse-spawn v0.

## 3. WORKFLOW RULES
- Config-gate every new reward (default off). Smoke-test (`--max_iterations 3`) before a full run.
- GPU cap: keep total training arms ≤ 8 (ALC RTX 4090, 24 GB). Check `nvidia-smi` first;
  each MAPPO arm ≈ 2–3 GB. Retire collapsed/losing arms to free room.
- Every launch: `--algorithm MAPPO --headless --wandb --wandb_project hook-grasp-mappo`
  and a unique `experiment_name`. NO run without `--wandb`.
- Judge by the deterministic EVALUATOR (actor-only, hook-blind), NOT training attached_frac.
- Append every arm + verdict to `experiments/swing_progress.md`. Commit working states.

## 4. ALC LAUNCH TEMPLATE (base = current best config "H2": clip 0.2, entropy 0.004)
```
ssh ALC 'cd ~/AerialMan && nohup ./.venv/bin/python scripts/skrl/train.py \
  --task Isaac-flycrane-hook-grasp-v0 --headless --device cuda:0 --algorithm MAPPO \
  --num_envs 128 --max_iterations 8000 --wandb --wandb_project hook-grasp-mappo \
  env.attachment.capture_radius=0.13 env.capture_radius_start=0.25 env.capture_curriculum_steps=200000 \
  env.hold_success_steps=100 env.hold_success_steps_start=30 env.hold_curriculum_steps=150000 \
  env.alive_weight=1.0 env.proximity_weight=2.0 env.spawn_dist_mu=0.35 env.swing_assist_v0=2.2 \
  env.attachment.upsweep_v_min=0.08 env.impact_toward_weight=6.0 env.anti_sag_weight=6.0 \
  env.slack_penalty_weight=20.0 \
  agent.models.policy.max_log_std=-0.5 agent.agent.ratio_clip=0.2 agent.agent.value_clip=0.2 \
  agent.agent.entropy_loss_scale=0.004 \
  agent.agent.experiment.experiment_name=<NAME> > experiments/<NAME>.log 2>&1 &'
```
Add per-arm overrides after this. Checkpoints: `logs/skrl/mappo_flycrane_hover/*<NAME>*/checkpoints/`.
Read TB scalars for attached_frac/catch/impact_up (source of truth mid-run; eval is final word).

## 5. STATE OF PLAY (2026-07-15 18:45)
Live arms (attached_frac = key mid-run metric):
- v24H2 clip0.2+ent0.004: **af 0.052 @54k, catch 0.35, impact_up +** ← BEST, base config
- v24H3 combo: af 0.034 @53k    · v24H1 hiLR: af 0.019 @54k
- v24MA2 hold-curric: af 0.018 @117k · v24MA1 default HP: af 0.006 @123k (collapsed — conservative)
Verdict forming: wide clip (0.2) + higher entropy (0.004) >> conservative defaults.
Known from PPO era (hook-cheat, invalid for obs but informative on rewards):
catch solvable ~0.4; hold ceiling set by strict DETACH model; taut-margin hold reward fixed
wire-jerk (slack 19%→2%, jerk 49N→22N) but halved catch — a real tradeoff to navigate.
