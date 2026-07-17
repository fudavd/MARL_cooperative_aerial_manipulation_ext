# WEEKEND AUTONOMOUS OPERATION — 2026-07-17 → 2026-07-20 11:00

**Mandate (user):** Full compute on ALC. Spawn agents (ultramode) to try different ideas and CODE new
experiments (rewards / scheduling / NN architectures / etc.). Objective-test whether the RIGHT behavior is
learned. When a new good result appears, spawn **two independent, BLIND adversarial agents** that hold it under
scrutiny (review the code, re-analyze the behavior, question the setup/idea) — they do NOT share memory or know
each other's / the proposer's goal; they only inspect the produced policy+code. Keep going until **Mon
2026-07-20 11:00**. **Email the user (fuda.vandiggelen@epfl.ch) on a genuine problem.**

## THE OBJECTIVE (what "right behavior" means)
A hook-blind, real-drone-honest policy that catches RELIABLY via a genuine **perpendicular pendulum up-swing**
(not a creep, a dead hang below the drone, or a sideways skim along the wall), and holds.
Measured by `scripts/hook_grasp/eval_behavior.py` (deterministic, radius 0.13, curricula off):
- **catch_rate ≥ 0.30** (and ideally reproducible across seeds — current champion is seed-sensitive)
- **hold_success_given_caught ≥ 0.80**
- **PHYSICAL:** upsweep_frac ≥ 0.60 · perp_ratio ≥ 0.55 · deadhang_frac ≤ 0.15 (composite `physicality_score`)
- low jerk (real-drone plausibility)
`GOALS_MET_PHYSICAL = catch≥0.30 AND hold≥0.80 AND physical`. Composite ranking key = catch_rate + physicality_score.

## CHAMPION (update as it changes)
- **v28_baseLong_9998** (seed42, resid H3 obs153, fixed 50g tip, F3, 10k iters): catch 0.313 / hold 0.976 / jerk 103.
  **Physicality 0.37 — FAILS GOAL_physical: upsweep 0.22, perp_ratio 0.51, deadhang 0.42, swing_amp 0.06 m.**
  ⇒ champion catches by CREEP/DEAD-HANG, not a genuine perpendicular up-swing. THE weekend target = keep
  catch≥0.30 while driving physicality up (upsweep→0.6, deadhang→<0.15). Composite (catch+phys) = 0.68 to beat.
  v30 swing-reward (1.5/2/2): catch 0.251, physicality 0.35 — swing reward as-weighted did NOT help (investigate).
  Caveat: champion catch seed-sensitive (1/3 seeds clear).
- Recipe scripts: `train_rslrl.py --indi_residual --actor_history_len 3`; per-arm tuning via
  `--env_kv/--policy_kv/--algo_kv` (obs/task fields hard-blocked). Eval: `eval_rslrl.py` + `eval_behavior.py`.

## THE LOOP (one wave ≈ 2–3 h; repeat until Mon 11:00)
1. **DESIGN (Workflow, ~6 agents):** each proposes ONE distinct idea (reward term / curriculum schedule / RNN
   architecture / PPO HP / exploration) with a HYPOTHESIS and an exact launch spec (override flags, or a precise
   code change + the file/lines). Dedupe; pick top ~3 (compute cap).
2. **RUN:** apply any code changes (I, the orchestrator, serially — keep deploys centralized), rsync to ALC,
   launch ≤3 arms (256 envs, ~10k iters). **ALC compute rule: ≤3 arms (GPU pages >~18 GB).** Poller on completion.
3. **EVAL:** each finished arm → `eval_rslrl` (catch/hold) + `eval_behavior` (physicality). Composite score.
4. **ADVERSARIAL SCRUTINY (Workflow, 2 blind agents) — ONLY when an arm beats the champion composite:**
   Give each agent ONLY {checkpoint path, the config/code used, the claimed metrics}. Task each INDEPENDENTLY to
   try to REFUTE the result: (a) audit the code/config for bugs, reward-hacking, eval leakage, curriculum
   cheats; (b) re-run eval_behavior + render a video, inspect the actual motion; (c) question whether the metric
   reflects real behavior. Each returns CONFIRMED / REFUTED + evidence. They share NO context. New champion ONLY
   if BOTH confirm. If either refutes with a valid concern, champion unchanged; log the concern as a new
   constraint for future designs.
5. **UPDATE + LOOP:** promote survivors to champion + `best_checkpoints/`; journal the wave verdict in
   `swing_progress.md`; feed confirmed failure-modes back into the next DESIGN prompt. Check clock; stop at Mon 11:00.

## OPERATIONAL RULES
- ALC (marvin, `ssh ALC`, repo `~/AerialMan`, plain copy — rsync to deploy). Full GPU (24 GB) this weekend.
- ≤3 rsl_rl arms concurrent (UVM paging cliff ~18 GB). Launch detached: `setsid nohup … </dev/null`.
- Training arms LINGER in teardown (procs≠0) — `pkill -9 -f "[x]arm_name"` after the target iter, then eval.
- **ALC headless RENDER segfaults** (Isaac hydra) — render on LOCAL only, and only when local GPU has ~3 GB free.
- `--resume <ckpt>` continues the iter counter (extend-training). Base PEAKS ~10k iters then collapses.
- Levers already tried that did NOT help catch: reward_norm (hurt), aimed_upsweep (neutral), desired_kl=0.005
  (hurt), swing-perp reward @ (1.5/2/2) (catch 0.32→0.24 but check physicality). Untested: RNN 512, more seeds.
- **Email policy:** send ONE email on a genuine blocker (ALC unreachable, every arm crashing, disk full, or a
  confirmed champion improvement worth flagging). Do NOT email routine progress.

## WAVE LOG (append newest at top)
- (pending) Wave 0: behavior baseline of champion + v30; first DESIGN workflow.
