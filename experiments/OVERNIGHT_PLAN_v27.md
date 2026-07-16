# Overnight autonomous search — 2026-07-16 21:00 → 2026-07-17 11:00

**Mandate (user):** run iterations + improvements until 11:00 AM 2026-07-17. Tune optimizer HP,
actor/critic architecture, rewards, curriculum — anything that does NOT change the task fundamentally.
**HARD CONSTRAINT: do NOT change the observation space.** (Actor obs FROZEN at the residual config
153-dim = 45 base + 6 INDI residual, ×H=3; critic 180. Enforced by `_ENV_KV_BLOCKLIST` in train_rslrl.py.)

## Objective
Raise **CATCH_RATE → 0.30** (the only unmet gate) while keeping **HOLD|caught ≥ 0.80** (SOLVED by the
INDI residual: H3 hold 0.97). Secondary: keep jerk low. Judge by deterministic eval_rslrl (radius 0.13,
curricula off, fixed 50 g tip).

## Base config (winner so far)
`--indi_residual --actor_history_len 3` (obs 153), F3 recipe, fixed 50 g tip. reward_norm: include iff
v27L5(+rn) ≥ v27L3(no rn) at eval.

## Diagnosis → lever priorities
Catch is up-sweep **AIM + TRIGGER-TIMING** limited. The residual carries swing-phase/force info, so the
levers most likely to convert it into catch:
1. **Reward shaping of the up-sweep aim/timing** (`reward_aimed_upsweep`, `reward_pump`, impact_* weights).
2. **Exploration** (entropy_coef, init_noise_std) — catch is a discovery problem.
3. **Capacity** (rnn_hidden_dim, rnn_num_layers) — exploit the residual for timing.
4. **Optimizer** (lr, epochs, mini_batches, desired_kl).
5. **Curriculum** shaping (capture-radius anneal, swing-assist) — no task change.

## OPERATIONAL RULES
- **ALC = 2 arms MAX.** >18 GB GPU → UVM paging cliff → 6× slowdown (2 arms 4.3 it/s each; 3 arms 0.46; 4 arms 0.1).
- Screening horizon **4000 iters** (~15 min/wave at 2 arms). Confirm winners at 6000.
- **Eval on ALC only after the wave's arms EXIT** (local machine = user's busy EA workstation; ALC has no
  spare memory for a concurrent eval). Run-to-completion (auto os._exit) → GPU frees → eval both → next wave.
- Overrides via `--env_kv "k=v,..."` (rewards/curriculum/gates), `--policy_kv`, `--algo_kv`.
- Each wave = 2 NEW hypotheses vs the current best base. Winners carry forward; losers dropped. Adapt from evals.
- Preserve every arm that beats the base to `experiments/best_checkpoints/`. Log each wave verdict in
  `experiments/swing_progress.md` (newest at top).
- **STOP at 11:00 2026-07-17**: final-eval the best, preserve it, write a summary. Each cycle, check the clock.

## Wave queue (adaptive — re-prioritize from results)
- I1: A=`reward_aimed_upsweep=true,upsweep_aim_k=3.0` | B=`reward_pump=true`
- I2: C=`algo entropy_coef=0.01` | E=`policy rnn_hidden_dim=512`
- I3: winner(I1)+winner(I2) combo | D=`policy init_noise_std=1.5` + `algo entropy_coef=0.008`
- I4: G=`algo learning_rate=0.0005,num_learning_epochs=8` | F=`policy rnn_num_layers=2`
- I5+: recombine winners; test curriculum (slower capture-radius anneal for more wide-radius catch practice).

## Result log (append per wave)
- BASE (fixed 50 g, spread-mass-trained read): H3 resid catch 0.12 / hold 0.97 / jerk 139. H1 0.095/0.86.
- **BASE (fixed 50 g, trained fully): v27L3 resid H3 = catch 0.208 / hold 0.944 / jerk 134.** ← THE BASE.
- reward_norm on fixed tip (v27L5): catch 0.117 / hold 0.967 → **reward_norm HURTS catch, DROPPED.**
- RESOURCE: user running 2 Hebbian ES on ALC (~10.8 GB, 12 h). → **1 rsl_rl arm at a time, 256 envs.**
- I1a (base + reward_aimed_upsweep=true,upsweep_aim_k=3.0), 4000 iters — RUNNING.
