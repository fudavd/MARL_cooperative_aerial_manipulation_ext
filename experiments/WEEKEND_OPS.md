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
- **v41 above-box penalty: MEASURED, it WORKS (correcting a frame-based misread).** New metric over_box_frac (fraction
  of steps the drone is horizontally over the box footprint = parked on top). **v37a (no penalty, any-dir gate):
  over_box_frac 0.88 — confirms the user's "always on top" complaint. v41 (above_box_penalty=6 + positive-outward
  10deg gate, 300g): over_box_frac 0.26** — the penalty pushed the drone OFF the top ~70% of the time (hero/last
  frames just caught the residual 26%; "still on top" was a wrong frame read — MEASURE, don't eyeball). BUT catch
  0.37->0.088: **v37a caught 0.37 BECAUSE it sat on top (any-dir gate let it hook from above); v41 is physically
  correct (off-top, positive gate, upsweep 0.89) but the hook-blind actor can't reliably do the outside swing-catch.**
  ⇒ positive gate + above-box penalty BOTH work as designed (force correct behavior); the cost is catch rate — same
  observability wall: enforce WHERE/HOW to catch, but can't learn the big controlled outside-swing to do it well.
- **v37 pendulum-ENERGY reward VERDICT (2026-07-21): NEGATIVE for the arc, but best PRACTICAL policy + likely ROOT
  CAUSE identified.** Rewarding hook PE (g*(z_hook-z_drone+wire_length)) + KE (0.5*|v_hook-v_drone|^2), the most
  physically-principled lever: **v37a (energy+10deg gate): catch 0.47, hold 0.94, deadhang 0.145, jerk 12.5 — best
  practical (reliable, low-creep, smooth) — but swing_amp 0.10, perp_ratio 0.44 (arc STILL unmoved).** v37b (energy+
  15deg perp): deadhang 0.05 (creep gone) but catch 0.11, swing_amp 0.05. **swing_amp & perp_ratio are now IMMOVABLE
  across velocity-, displacement-, AND energy-based rewards x every gate.**
  - **ROOT CAUSE (well-supported): the actor is HOOK-BLIND — no hook position/velocity, only the INDI force residual.
    A large CONTROLLED perpendicular pendulum arc requires timing the swing PHASE, which needs to OBSERVE the hook;
    the residual (force) enables reactive HOLD but not open-loop control of a big arc. PPO therefore converges to a
    small controllable motion regardless of reward. This is an OBSERVABILITY limit, not a reward-tuning gap —
    reward shaping cannot create controllability the obs doesn't provide.** ⇒ a genuine visible swing likely
    REQUIRES giving the actor hook swing-phase obs (relaxing the hook-blind premise) — a task-definition decision.
  - Solid deliverables: catch (v37a 0.47 / v32a 0.60), dead-hang ELIMINATED (v37b 0.05), hold 0.94, jerk 12.5.
    Best practical ckpt: v37a_energy_gate10_...PRACTICALbest.
- **v36 perp-gate VERDICT (2026-07-21): NEGATIVE. Gates cap out — they don't create the arc.** v36 (perpendicular-only
  20deg gate + arc reward, full 10k, RAM stable): **catch 0.022 (collapsed), perp_ratio 0.41, swing_amp 0.058** — the
  arc metrics UNMOVED, catch destroyed. The perp gate only restricts WHERE a catch counts; the drone still approaches
  with a small lateral skim and rarely stumbles into a valid catch (17/771 eps). ⇒ **CONCLUSION across v30–v36: gates
  (chord-angle, perp) reliably ELIMINATE dead-hang (0.42->0.12) but do NOT teach a perpendicular pendulum arc; the
  arc reward's early amplitude (0.34) is untrained flailing that training removes; perp_ratio stuck 0.40–0.51,
  swing_amp stuck 0.06–0.10 across EVERY reward/gate/mass/capacity lever.** The trained policy always converges to a
  small-amplitude precise catch — a big energetic swing is not the efficient solution it finds. Gate/reward shaping is
  exhausted for the arc; a genuine swing likely needs a different mechanism (imitation of a scripted pump, energy-based
  objective, or an action-space/task rethink). Reliable CATCH is solved (v32a 0.60); dead-hang eliminated (v35c 0.12).
- **v35 arc-amplitude fix (2026-07-21 ~00:xx).** Post-scrutiny fix: reward the pendulum ARC DIRECTLY. New reward
  terms `arc_perp_weight` (reward |(hook-drone).wall_normal| = perpendicular swing-out displacement / arc amplitude)
  + `arc_lateral_weight` (penalize along-bar displacement). Behavior test REWEIGHTED: physicality now 0.35*perp_ratio
  + 0.35*swing_amp + 0.15*upsweep + 0.15*(1-deadhang); GOAL_physical now REQUIRES perp_ratio>=0.60 AND swing_amp>=0.20
  (the ungameable arc metrics). Gate raised to 30deg (user). commit 3ae6b4d.
  - **★ KEY SIGNAL: the 30deg gate + arc reward FINALLY produces a real ARC.** v35a_arc30 was OOM-killed at only
    iter 1292/10k, yet its partial ckpt already hit **swing_amp 0.336 m** (vs ~0.08 m for everything before = 4x, a
    genuine ~0.3 m pendulum arc) — because a catch now requires the hook swung out >=30deg, physically forcing the
    arc. catch only 0.03 (undertrained + hard gate). **RELAUNCHED as v35a2_arc30 (single arm, to 10k)** — this is the
    most promising physicality lever found. v35b_arc20 (20deg, full 10k): catch 0.30, swing_amp 0.10, perp_ratio 0.46
    — 20deg too weak to force the arc. ⇒ the GATE ANGLE is what forces swing_amp; 30deg >> 20deg. Watch: does catch
    recover by 10k while swing_amp/perp_ratio stay high? If yes -> the real swing. Then adversarial-scrutinize.
  - **v35a/v35a2 OOM ROOT CAUSE + v35c VERDICT (2026-07-21).** 30deg gate crashed at seed-deterministic iter ~1292:
    early catch ~0.001 -> drone flails -> constant resets -> Isaac USD stage-recomposition memory leak -> OOM SIGTERM.
    FIXED via slow chord anneal (120k steps) + lower weights -> **v35c_arc30_slow** ran full 10k, RAM stable 26G.
    v35c honest eval (30deg gate): catch 0.15, **deadhang 0.42->0.117 (creep ELIMINATED)**, but **swing_amp 0.058,
    perp_ratio 0.38 (WORSE) — NO perpendicular arc.** The undertrained swing_amp 0.34 was FLAILING; trained policy
    converged to a SMALL LATERAL FLICK. ⇒ **the chord-angle gate is DIRECTION-AGNOSTIC (angle from vertical, any
    azimuth) -> the drone swings out SIDEWAYS along the bar to satisfy it, not perpendicular.** NEXT FIX: make the
    gate require swing-out in the WALL-NORMAL plane (perpendicular angle), directly attacking perp_ratio.
- **v34a chord-gate: SCRUTINY REFUTED (both auditors, conf 0.80/0.82) — honest partial progress, NOT a real swing.**
  v34a (chord-angle gate 10deg + 200g hook + relative-frame swing reward, proximity removed): catch 0.41/hold 0.94,
  upsweep 0.22->0.61, deadhang 0.42->0.22, jerk ~100->15, physicality 0.52. TWO BLIND AUDITORS reproduced the numbers
  at fresh seed but REFUTED the "genuine perpendicular swing" claim: (1) the UNGAMEABLE arc metrics didn't move —
  perp_ratio 0.50 (=equal skim), swing_amp 0.08 m (=20% of a real 0.4 m arc); GOAL_physical=False. (2) upsweep 0.61 is
  ~TAUTOLOGICAL (require_upsweep already forbids v_up<=0.12 catches), deadhang drop is partly FORCED by the gate,
  jerk 15 is measured only while ATTACHED and confounded by the 4x heavier hook. (3) baseline is apples-to-oranges
  (old 50g/no-gate task). ⇒ **the chord-gate makes catch HONEST under a real constraint (0.44, verified) and cuts
  dead-hang, but the pendulum ARC is still not there.** Exposed a FLAW in the behavior test (upsweep/deadhang gameable)
  → fixed in v35. The chord-gate impl IS correct + non-bypassable (verified). Best ckpt preserved: v34a_chordgate_...
  Verdict: adversarial mechanism worked — prevented a false "solved" claim. Video: _render_rslrl/v34a_chordgate_1env.mp4.
- **Wave 3 (2026-07-18 ~20:00) LAUNCHED — combine the two wave-2 winning directions.**
  - **v33a_combo**: 512x2 rnn (catch capacity) + fly-in runway (physicality) + pump + swing_still + swing_lateral=3.
  - **v33b_hardrunway**: push physicality harder — spawn_dist_mu=0.8, spawn_away_v0=2.5, swing_still=6, swing_lateral=4,
    pump_lift=12. Goal: drive deadhang 0.26→<0.15 and perp_ratio 0.51→>0.55.
  - Also FIXED a real eval bug: eval_behavior/eval_rslrl ignored --policy_kv → architecture arms silently loaded as
    256x1 (size mismatch). Now pass policy_kv through (commit 5925c18). Re-eval any arch arm WITH its --policy_kv.
- **Wave 2 VERDICT (2026-07-18 ~19:40): two wins on DIFFERENT axes; neither passes physical gate yet.**
  - **v32a_bigrnn_pump (512x2 LSTM): catch 0.605 (!! record) hold 0.97, but physicality 0.39 (deadhang 0.43 — still
    creeps). Capacity is a strong CATCH lever, not a physicality fix.** Preserved: v32a...CATCHrecord.
  - **v32b_runway_pump (spawn_away_v0=1.6, no assist): catch 0.45, physicality 0.44, deadhang 0.42→0.26 — FIRST real
    physicality move.** The fly-in runway geometrically suppresses the dead-hang. Preserved: v32b...PHYSleader.
  - perp_ratio stuck ~0.51 across ALL arms (need 0.55) — no lever moved it yet → added swing_lateral penalty to wave 3.
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
