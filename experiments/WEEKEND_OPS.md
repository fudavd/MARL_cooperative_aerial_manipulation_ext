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
- **Email policy:** on a genuine blocker (ALC unreachable, every arm crashing, disk full) OR a scrutiny-CONFIRMED
  champion improvement, create a Gmail DRAFT to fuda.vandiggelen@epfl.ch (mcp__claude_ai_Gmail__create_draft — the
  tool can only DRAFT, not send; user sees it in Drafts). One draft per event; do NOT draft routine progress.

## WAVE LOG (append newest at top)
- **Wave 2 (2026-07-18 ~13:20) LAUNCHED.** Target = PHYSICALITY (wave 1 showed catch & physicality are decoupled).
  - **v32a_bigrnn_pump**: 512x2 LSTM (`--policy_kv rnn_hidden_dim=512,rnn_num_layers=2`) + strong pump — capacity
    hypothesis: does more recurrent memory convert the pump's catch drive into a real timed swing?
  - **v32b_runway_pump**: `spawn_away_v0=1.6, swing_assist_v0=0, spawn_dist_mu=0.6` + strong pump — geometric
    hypothesis: force a fly-IN perpendicular load so creep isn't available.
- **Wave 1 VERDICT (2026-07-18 ~13:10): catch and physicality are DECOUPLED levers.**
  - **v31a_pump: catch 0.31→0.46 (!) hold 0.98 jerk 82 — BIG catch win, but physicality 0.34 (upsweep 0.19,
    deadhang 0.46) — WORSE. Pump buys reliability by intensifying the CREEP, not fixing it.** Preserved as a
    catch-only fallback: best_checkpoints/v31a_pump_9999_catch0.46...CREEP_fallback. NOT a physicality winner →
    no scrutiny fired (creep policy). Composite 0.80 but fails GOAL_physical.
  - v31b_upsweepgate (train-time upsweep_v_min=0.35): catch 0.29, physicality 0.36 — gate didn't force physicality
    (assist provided the v_up in training; reverts to creep at eval). Ruled out.
  - INSIGHT for wave 2+: raising catch ≠ fixing behavior. Must make creep IMPOSSIBLE/unrewarding while keeping the
    pump's catch drive. Levers: capacity (does the creep persist with more memory?), geometry (fly-in runway).
- **Wave 1 (2026-07-18 ~00:30) LAUNCHED.** Design workflow (6 agents) produced 6 pure-flag specs. Key agent
  finding: my swing_perp reward used WORLD-frame perp speed → a dead hook TRANSLATED wallward earns it (pays the
  creep) — explains v30's failure. Launched top-2 orthogonal creep-attacks (2-arm fast regime):
  - **v31a_pump**: `reward_pump=true,pump_windup_decay=false,pump_windup_weight=8,pump_lift_weight=10,swing_still_weight=3`
    (pump uses RELATIVE geometry — can't be farmed by translation; swing_still penalizes the exact deadhang metric).
  - **v31b_upsweepgate**: `attachment.upsweep_v_min=0.35` (creep no longer counts as a catch) `+pump+swing_assist_v0=2.5`.
  QUEUED for wave 2: standoff-runway-pump (spawn_away_v0), deep-wide-lstm (512x2), explore-the-pump (entropy/noise),
  spawn-out-swing-in (dist_mu=1.0). Eval both @peak with eval_rslrl + eval_behavior; adversarial-scrutinize anything
  beating champion composite 0.68 (catch0.31+phys0.37).
- Wave 0: behavior baseline — champion catches by creep/deadhang (upsweep 0.22, deadhang 0.42, phys 0.37); test built.
