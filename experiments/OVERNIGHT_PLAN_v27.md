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
- I1a (base + reward_aimed_upsweep) @4000: catch 0.094 / hold 1.0. ≈ base@4000 (0.109) → **neutral, not a win.**
- **★ KEY: base catch 0.109 @4000 → 0.208 @6000 = STILL CLIMBING STEEPLY (not plateaued).** ⇒ (1) 4000-iter
  screening is TOO SHORT to judge levers (catch undeveloped); (2) **train-LONGER is the top lever.**
  → **v28_baseLong10k: RESUME v27L3 model_5999 → +4000 (to 10k).** Eval 7k/8k/9k/10k trajectory for the
  catch curve toward 0.30. If it plateaus <0.30, THEN test levers (entropy / RNN-512) at a LONG horizon.
- Screening rule REVISED: judge at >=6000 iters (or resume-extend), not 4000.

## ★ RESULT: GOALS MET (2026-07-17 00:40)
- baseLong10k (resume v27L3 6k→10k): catch **0.139→0.175→0.244→0.373** @ 7k/8k/9k/10k, hold 0.89→0.97, jerk→107.
- **catch 0.373 ≥ 0.30 AND hold 0.97 ≥ 0.80 → GOAL. Winning recipe = residual H3 + fixed 50 g tip + F3 + LONG training.**
- Catch still climbing at 10k → resumed to 14k (v28_baseLong14k). Next: eval 11k/12k/13k/14k; if still climbing, extend further; then optionally test if entropy/RNN-512 push the peak even higher.

## Reproducibility + push (2026-07-17 ~03:00)
- seed42 @10k robust: catch 0.323 (CLEARS). seed7 @10k robust: **catch 0.219 (does NOT clear), hold 0.982.**
  → **Goal is SEED-SENSITIVE at catch (0.22-0.32); HOLD robustly solved (0.97-0.98) across seeds.**
- 1 ES run finished → headroom for 2 arms (17.5 GB < cliff). Running:
  - v29_seed123_10k (base recipe) — characterize catch distribution.
  - v29c_seed42_lowkl (desired_kl=0.005 stability lever, 12k) — try to lift/stabilize the peak vs seed42 base 0.323.
