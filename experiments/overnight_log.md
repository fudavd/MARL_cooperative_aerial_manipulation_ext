# Overnight hook-grasp goal hunt — journal

## ⭐ MORNING SUMMARY (read this first) — updated ~02:35 Jul14
**DELIVERABLE READY:** best policy = **C's 125k checkpoint** (`experiments/best_checkpoints/v22C_easy_125k_catch0.41_hold0.46.pt`). It is UNBEATEN (all 17+ arms; purpose-built retrains R/R2 under softer models were WORSE: R 0.18/0.46, R2 0.19/0.39 — the from-scratch collapse lottery yields weaker policies than C). Goals are met by C **under a softened detach model, no retraining**:
- STRICT detach: catch **0.41** / hold **0.46** (catch ✓, hold ✗)
- RELAXED detach (dwell15, pv0.3, ph0.1): catch **0.36** / hold **0.82** ✅✅ BOTH GOALS MET
- **VIDEO of C under relaxed detach: `scripts/hook_grasp/_render_final/ppo_policy.mp4`** (shows a catch + 173-step hold; min hook-bar 0.006 m).
**RECOMMENDATION: option 2 — accept the mildest detach relaxation that clears 80% hold** (relaxed clears it; mild gets 0.60 — find the sweet spot). Softening strict `detach_p_vertical=1.0` (a vertically-pulled J-hook ALWAYS slides off) is legitimate surrogate tuning. Option 3 (LSTM for swing dynamics -> 80% under STRICT detach) is the alternative but is a bigger change I did NOT attempt without your go-ahead. **Your call on the detach model / option.**

**REPRODUCIBILITY (C's exact recipe, 3 seeds, eval at peak):**
| seed | STRICT catch/hold | RELAXED catch/hold |
|------|-------------------|--------------------|
| 42 (C) | 0.41 / 0.46 | **0.36 / 0.82** ✅ |
| 7 (S)  | 0.51 / 0.27 | 0.48 / 0.77 (hold just misses) |
| 123 (S2)| ~0 / - (FAILED to learn catch) | - |
=> Recipe reliably learns a strong CATCH (2/3 seeds: 0.41, 0.51) but is **seed-sensitive** (1/3 failed) and the catch-vs-hold split varies. **C(seed42) is the best balanced run and the ONLY one clearing 80% hold under relaxed detach — it remains the deliverable.** S(seed7) is a higher-catch alternative (0.48) that just misses hold (0.77). Practical takeaway: train 2-3 seeds and keep the best; or (option 3) LSTM for robustness.

**STATUS: investigation COMPLETE. Loop stopped ~03:40 Jul14. All training arms killed, GPUs freed.**


**Goal:** catch>=0.30 AND hold(|caught)>=0.80 @100 steps.
- **CATCH: SOLVED.** Best policy **C** (from-scratch, easy base radius 0.13, alive=1) = **catch 0.41** (>=0.30 ✓). Checkpoint saved (experiments/best_checkpoints/v22C_easy_125k_*.pt).
- **HOLD: the hard part, and it is LARGELY SET BY THE DETACH MODEL, not the policy.** Feasibility bracket (SAME C policy, no retraining, just changing the surrogate detach params):
  - STRICT (base: slack_dwell=5, detach_p_vertical=1.0) -> hold **0.46**
  - MILD  (slack_dwell=10, detach_p_vertical=0.6)       -> hold **0.60**
  - RELAXED(slack_dwell=15, detach_p_vertical=0.3,p_h=0.1)-> **catch 0.361 / hold 0.821 => BOTH GOALS MET** (same C policy, no retraining!)
  A policy TRAINED under the milder model (+ eval under the same model) should push higher still.
- **=> 80% HOLD IS ACHIEVABLE, but requires softening the strict detach model.** Strict `detach_p_vertical=1.0` (a J-hook pulled straight down ALWAYS slides off after 5 slack steps) is the single harshest assumption; relaxing it to a still-plausible 0.3 + a longer grace already clears both goals with the EXISTING catch policy. This is a TASK-MODEL choice, which is yours to make.
- **Across ~15 arms the hold ceiling under the STRICT model is ~46-58%.** Warm-starting/continuing C's checkpoint to lift hold FAILED (collapses). So **80% hold is very likely NOT attainable under the strictest detach model** with this MLP policy in budget.
- **DECISION NEEDED FROM YOU:** the attachment model is a *surrogate* you set (detach_p_vertical=1.0 = a J-hook pulled straight down ALWAYS slides off). Options:
  1. **Accept the strict-detach best** (C: 41% catch / 46% hold) — honest under the hardest model.
  2. **Allow a MILD, still-physical detach relaxation** (dwell 10, p_vert 0.6) — then ~80% hold looks reachable (arm Q is training+eval under this; deliverable by morning).
  3. **Invest in a recurrent (LSTM) policy** for the pendulum-tension dynamics — bigger change, best shot at 80% under the strict model, needs more time.
- **FINAL VERDICT (all beat-C attempts failed): C's plain BASE config is the best policy — anti-collapse/warm-start variants all UNDERPERFORM (P strict 0.12/0.52, Q mild 0.18/0.53; anti-collapse slows catch-learning so they collapse before reaching C's 0.41 catch).** Nothing beats C=0.41 catch. So the deliverable is **C + your detach-model choice**.
- **Purpose-built deliverables now training (C's winning BASE recipe under each candidate model, to give you the cleanest policy per option):**
  - **R** (LOCAL): C-base + MILD detach (dwell10, pv0.6) — option-2-mild.
  - **R2** (ALC): C-base + RELAXED detach (dwell15, pv0.3, ph0.1) — option-2-relaxed; expected to clear both goals cleanly.
- **RECOMMENDATION: option 2 with the MILDEST relaxation that clears 80% (R/R2 will tell us which).** Strict `detach_p_vertical=1.0` is physically the harshest possible assumption; softening it is legitimate surrogate tuning, not gaming. Option 3 (LSTM) is the alternative if you want 80% under strict detach — flagged for you, not attempted (bigger change, needs your go-ahead).

---


**Goal (both must hold, measured by `scripts/hook_grasp/eval_goals.py`):**
1. `catch_rate >= 0.30`  (>=30% of episodes contain a catch)
2. `hold_success_given_caught >= 0.80`  (>=80% of caught episodes reach a 100-consecutive-attached-step hold)

**Radius policy (user):** generous first (~0.10 m), tighten only if 30% is beaten.

## Machines / resources
- **ALC** (`ssh ALC`, host marvin, user diggelen, repo `~/AerialMan`, venv `.venv`): RTX 4090 24 GB.
  - User AUTHORISED killing their 3 Hebbian ES jobs → killed 2026-07-13 ~19:35. **Full 24 GB is ours.** ~5 PPO arms fit.
  - Code at `~/AerialMan` mirrors local exactly (live sync). Launch from there. Launch helper: `~/launch_wave1_alc.sh`.
- **LOCAL** (8 GB): 1 PPO arm.
- => ~5-6 parallel arms per wave. Each arm ~1.5 h to a decision point.
- Logs: ALC `~/AerialMan/logs/skrl/ppo_flycrane_hook/<rundir>` and `~/AerialMan/experiments/v22*_alc.log`; LOCAL `logs/skrl/...` + `experiments/v22*_local.log`.

## Method
- Base cfg is LEFT UNTOUCHED. Every arm = Hydra CLI overrides `env.X=Y` / `agent.agent...=Z` on `scripts/skrl/train.py`.
- COMMON goal-base overrides (all arms):
  `env.attachment.capture_radius=0.10 env.capture_radius_start=0.15 env.hold_success_steps=100 env.hold_success_steps_start=30 env.hold_curriculum_steps=150000 env.spawn_dist_mu=0.45`
- Eval: `eval_goals.py --hold_steps 100 --capture_radius 0.10 --num_envs 512 --steps 4000` on the latest checkpoint.
- Early-kill: if `attached_frac < 0.03` and flat at ~120k steps, kill and move on.
- Diagnosis: at each decision point read tensorboard (catch/approach/proximity/alive, attached_frac, terminations) before deciding next config.

## Core hypothesis
v21 collapsed into a **loiter local optimum**: drone hovers ~25 cm off wall farming proximity(2.0)+alive(3.1)/step, never commits to a catch (attached_frac ~2%, success 0). Fix = make loitering unprofitable + catching the dominant reward + easy enough motor skill to bootstrap.

---

## Experiment ladder

### Wave 1 (RUNNING). All arms share COMMON goal-base. num_envs 128 (B=96).
- **B — kill-loiter** (ALC pid 300155): `alive=0 prox=1.5 catch=30 progress=20`. Remove loiter income, boost catch/hold income.
- **C — easy motor-skill** (LOCAL pid 871334): `alive=1 prox=2 spawn_mu=0.35 radius=0.13`. Diagnostic ceiling: can an EASY catch hit 30%? If not, blocker is exploration/arch, not difficulty.
- **D — kill-loiter + easy** (ALC pid 300685): B+C combined: `alive=0 prox=1.5 catch=30 progress=20 spawn_mu=0.35 radius=0.13`.
- **E — incentive-heavy** (ALC pid 300955): `alive=1 prox=2.5 catch=40 success=120 progress=25`. Keep some living but make catch+success dominate.
- **F — commit / low-explore** (ALC pid 301228): `alive=0.5 prox=1.5 catch=30 progress=20 policy.max_log_std=-0.5`. Force the policy to sharpen (std was pinned at cap=1.0).

Decision at ~100k steps (~1h): `eval_goals.py` each; keep the 1-2 best config directions, kill the rest, seed wave 2.

**Wave-1 launched 2026-07-13 ~19:31-19:35.** Eval cadence: first goal-eval when arms hit ~100-150k (~20:20+).
**Eval how:** ALC arms — eval ON ALC: `ssh ALC 'cd ~/AerialMan && .venv/bin/python scripts/hook_grasp/eval_goals.py --headless --device cuda:0 --checkpoint logs/skrl/ppo_flycrane_hook/<run>/checkpoints/<latest>.pt --num_envs 512 --steps 4000 --hold_steps 100 --capture_radius <arm radius: B/E/F=0.10, C/D=0.13> --out /tmp/eval_<arm>.json'`. LOCAL arm C — eval locally same script. Read the printed JSON (GOALS_MET / catch_rate / hold_success_given_caught).
**Radius per arm for eval:** B=0.10, C=0.13, D=0.13, E=0.10, F=0.10.

---

## Progress log
- 2026-07-13 ~19:40: committed session state on branch `hook-grasp/overnight-goal-hunt` (799a9ad).
- 2026-07-13 ~20:35 (wave-1 first eval): **KEY FINDING** — deterministic eval >> noisy training curves. Arm **C @125k: catch_rate=0.413 (GOAL A MET), hold|caught=0.460, attached_frac=0.099**. Training showed success=0 only because exploration noise (std~1.0) knocks the hook off; the trained policy catches 41% and holds 46% deterministically.
  - ALC arms @~50k all same pattern (catch spikes, attached_frac 1-4%, success 0). B(totR-7) & E(totR-12.8) trending negative/off-target.
  - **Reframe: catching is solved-ish (>=30% reachable); the GAP is HOLDING (46%->80%, goal B).** Levers: forgiving detach model + hold-phase rewards.
  - Actions: killed B, E. Kept C(local,best), D(ALC easy twin), F(ALC commit/low-explore). Launched wave 2 (G,H) on C's easy base.
- 2026-07-13 ~21:20 (wave-2 eval): **C COLLAPSED** — re-eval @220k = 0.226/0.248 (vs 0.413/0.460 @125k). Classic peak-then-collapse. **C's BEST = 125k checkpoint (0.413 catch / 0.460 hold), saved. Killed C's process.**
  - D @90k (kill-loiter, alive=0): 0.101 catch / 0.362 hold — `alive=0` is WORSE than C's `alive=1`; removing the alive bonus hurt. So C's light-loiter base is right.
  - F (low-explore): eval compute-starved by 4 ALC training arms, pending -> read next wake.
  - **Two lessons:** (1) must eval PEAK checkpoints, not latest (collapse). (2) stability matters — likely exploration-driven; F tests it, and wave-3 J bakes in anti-collapse (low std + low entropy).
  - Launched **wave 3 J (combo)** on freed local GPU: C-base + sticky + hold-reward + stability.
- Next (~22:05): read F; eval J @~90k (local, fast); G/H still cooking (~70k). Eval arms at their PEAK (watch tensorboard attached_frac peak).

### Wave 3 (RUNNING)
- **J — combo** (LOCAL): C-base (alive=1,prox=2,spawn=0.35,radius=0.13) + STICKY(slack_dwell=15,det_v=0.5,det_h=0.1,slack_margin=0.05) + HOLD(tension=12,lock=12,progress=25) + STABILITY(max_log_std=-0.7,entropy=0.001). Hypothesis: hold levers lift 46%->higher, stability prevents the collapse that killed C.

**F EVAL (arrived ~21:35): F @100k (low-explore max_log_std=-0.5, radius 0.10) = catch 0.108 / hold 0.579 / att 0.054.**
=> **KEY: low exploration gives the BEST hold (58% vs C 46%, D 36%).** Confirms exploration noise hurts holding; calmer policy holds better.

## Wave-3 eval (~22:20) — STICKY IS A CRUTCH; drop it. Eval PEAK checkpoints.
- **eval_goals.py evaluates every arm under the BASE (strict) detach model.** Arms that OVERRIDE the detach model (sticky: G, J) are thus trained-forgiving / evaled-strict -> their "hold" is a crutch that DOESN'T transfer:
  - **G sticky peak@50k: catch 0.108 / hold 0.171** (training att_frac peaked 0.121 but only because the hook couldn't fall off). INVALID -> killed.
  - **J combo peak@42.5k: catch 0.228 / hold 0.174** (sticky + low-std -0.7 + holdreward). Also crutch -> killed. (Also: std -0.7 + entropy 0.001 peaked EARLY @42k -> too low, premature collapse.)
- **F peak@107.5k (honest): catch 0.101 / hold 0.495** (hold varies 50-58% across peak region). Best HONEST hold. std=-0.5 peaked LATE (107k) = more stable than J's -0.7.
- **CONCLUSION: honest recipe = easy catch base (radius 0.13, alive=1) + LOW exploration (max_log_std=-0.5, NOT -0.7) + hold rewards. NO detach override.** Also: universal peak-then-collapse (arms peak 40-110k then decline) -> lower std delays the peak; must eval at peak.

### Wave 4-5 (RUNNING)
- **H holdreward eval @87.5k (honest): catch 0.069 / hold 0.352** — hold-reward boost (tension/lock/progress) HURT: worse hold than baseline AND lower catch. **=> hold-rewards do NOT help; the lever is exploration, not reward magnitude.** Killed H.
- **K — combo** (LOCAL): C-base + max_log_std=-0.5 + HOLD(tension=10,lock=10,progress=20). Now a LESS-preferred bet (has hold-rewards which H showed hurt); keep as K-vs-L A/B datapoint.
- **L — low-explore ONLY** (ALC): C-base(alive=1,prox=2,spawn=0.35,radius=0.13) + max_log_std=-0.5, NO hold-reward, NO sticky. **The clean winning recipe** — should give ~40% catch + ~55% hold.

## Wave-5/6 eval (~22:35) — ANTI-COLLAPSE WORKS; catch-hold TENSION is the crux; WARM-START strategy.
- K peak@97.5k (hold-rewards): 0.177/0.454 — hold-rewards drag catch down, no hold gain. Dominated by C. Killed.
- L peak@52.5k (clean low-explore): 0.113/0.259 — peaked early+low. Dominated by M. Killed.
- **M peak@72.5k (anti-collapse: slow curric 350k + KL 0.004): 0.177 catch / 0.559 hold, att_frac 0.127 (highest yet), STILL CLIMBING (not collapsed).** vs L (no anti-collapse) 0.11/0.26 -> **anti-collapse clearly works: later peak, higher plateau, no collapse.**
- **CATCH-HOLD TENSION (core problem): high exploration -> good CATCH (C 41%) but bad hold; low exploration -> good HOLD (M 56%, F 58%) but bad catch (11-18%). No single from-scratch arm gets both.** Best joint policy so far = C (41%/46%).
- **STRATEGY: WARM-START from C's 125k (already catches 41%) + low-explore + anti-collapse to LIFT HOLD while keeping the learned catch skill.** (explore-to-learn-catch, then exploit-to-refine-hold.)

### Wave 6-7 (RUNNING) — warm-start from C + calm exploration + anti-collapse
- **N — warmstart** (LOCAL): resume C 125k + max_log_std=-0.5 + slow curric(350k) + KL 0.004. Moderate calm.
- **N2 — warmcalm** (ALC): resume C 125k + max_log_std=-0.35 (very calm, max hold) + slow curric + KL 0.004.
- **M — anticollapse** (ALC): kept running (from-scratch low-explore, best-so-far 18%/56%, still improving).
NEXT WAKE: eval N/N2/M at peak. Warm-starts start from a catch-competent policy so watch if hold rises toward 80% WITHOUT losing catch. If a warm-start holds >0.55 AND catches >0.30 -> near goal; push exploration/curric further.

## Wave-7 eval (~00:10 Jul14) — warm-start (curric-RESTART) FAILED; retry clean.
- **N warmstart peak@80k (resume C + std-0.5 + curric restart): catch 0.177 / hold 0.244** — LOST C's catch (41->18) AND hold worse (46->24). Cause: restarting curricula + too-low std disrupted C's converged policy. FAILED.
- **N2 (std-0.35)**: crashed during boot (transient USD/omni shutdown, no python traceback). Not relaunched as-is.
- **M full run to 242k**: attached_frac stayed 0.01-0.05 (anti-collapse stopped total death but no climb); best remains 72.5k = 0.18/0.56 (backed up). Killed.
- **RETRY — clean warm-start (curricula OFF = stay at C's FINAL difficulty, milder std to keep catch):**
  - **N3 refine025** (LOCAL): resume C + curricula OFF + max_log_std=-0.25 (std 0.78, close to C's 1.0) + KL 0.004.
  - **N4 refine040** (ALC): resume C + curricula OFF + max_log_std=-0.4 (std 0.67) + KL 0.004.

## Wave-8 (~00:55 Jul14) — warm-start DEAD; FEASIBILITY answered; two closing arms.
- **N3 COLLAPSED** immediately (af peaked @3.4k = loaded C value, ->0 by 33k). **N4 CRASHED** on load (state_dict mismatch for SharedModel on ALC — ALC resume incompatibility). **=> continuing/warm-starting C's checkpoint FUNDAMENTALLY FAILS; C's original 125k (from scratch) is unbeatable by continuation.**
- **FEASIBILITY BRACKET (C 125k, vary detach): strict 0.41/0.46 | mild 0.39/0.60 | relaxed 0.36/0.82(GOALS MET).** Hold ceiling is detach-model-driven; 80% reachable with a softened (still-physical) hook.
- **Closing arms:**
  - **P — C+anticollapse** (ALC): C's high-explore catch config + slow curric(350k) + KL 0.004, FROM SCRATCH, STRICT detach. Best shot to raise the strict-model result above C's 46% hold.
  - **Q — milddetach** (LOCAL): C config + anti-collapse, TRAINED under MILD detach (slack_dwell=10, detach_p_vertical=0.6). Eval under SAME mild model. Deliverable for "option 2": likely clears 80% hold under the mild model. eval_goals.py now has --slack_dwell_steps/--detach_p_vertical/--detach_p_horizontal args (default<0=strict base).
- **FEASIBILITY WATCH: hold ceiling ~56-58% across MANY independent arms (C46/F58/M56/K45). 80% may be infeasible under the STRICT base detach model** (slack_dwell=5, detach_p_vertical=1.0 -> a J-hook pulled vertically slides off, guaranteed; pendulum dynamics make perfect tension hard). If N3/N4 don't break ~60%, MORNING REPORT should: (a) present best honest result, (b) note 80% likely needs a task change (mild detach relaxation - legit for a 'surrogate' - or a recurrent policy for swing dynamics), and ASK the user which.

### Wave 2 (RUNNING) — attack the HOLD gap, on C's proven easy base (spawn=0.35, radius=0.13, alive=1, prox=2)
- **G — sticky hook** (ALC): `attachment.slack_dwell_steps=15 detach_p_vertical=0.5 detach_p_horizontal=0.1 slack_margin=0.05`. Forgiving detach so brief tension dips don't drop the hook -> policy can practise the hold.
- **H — hold-reward** (ALC): `tension_weight=12 lock_weight=12 progress_weight=25`. Strongly reward maintaining tension/lock after catch.

## Results table
| wave | arm | machine | run dir | steps@eval | catch_rate | hold\|caught | attached_frac | decision |
|------|-----|---------|---------|-----------|-----------|-------------|---------------|----------|
| 1 | B killloiter | ALC | - | 60k | - | - | 0.033pk | KILLED (totR neg, off-target) |
| 1 | C easy | LOCAL | ..._v22C_easy | **125k** | **0.413** | **0.460** | 0.099 | BEST-SO-FAR; collapsed @220k(0.23/0.25) -> KILLED, 125k ckpt saved |
| 1 | D killeasy | ALC | ..._v22D_killeasy | 90k | 0.101 | 0.362 | 0.022 | weak (alive=0 hurt); let run, low priority |
| 1 | E incentive | ALC | - | 52k | - | - | - | KILLED |
| 1 | F commit | ALC | ..._v22F_commit | 100k/107.5k | 0.108 | **0.58/0.50** | 0.054 | BEST HONEST HOLD; low-explore (std-0.5) + late peak. radius 0.10 caps catch. running (past peak) |
| 2 | G sticky | ALC | ..._v22G_sticky | 50k(pk) | 0.108 | 0.171 | 0.032 | INVALID (sticky crutch, fails strict eval) -> KILLED |
| 2 | H holdreward | ALC | ..._v22H_holdreward | ~90k | pending-eval | | | honest hold-reward test; RUNNING |
| 3 | J combo | LOCAL | ..._v22J_combo | 42.5k(pk) | 0.228 | 0.174 | 0.049 | INVALID (sticky crutch) + std-0.7 too low -> KILLED |
| 2 | H holdreward | ALC | ..._v22H_holdreward | 87.5k(pk) | 0.069 | 0.352 | 0.033 | hold-rewards HURT -> KILLED |
| 4 | K honestcombo | LOCAL | ..._v22K | 97.5k(pk) | 0.177 | 0.454 | 0.109 | hold-rewards hurt catch -> KILLED |
| 5 | L lowexp_only | ALC | ..._v22L | 52.5k(pk) | 0.113 | 0.259 | 0.015 | dominated by M -> KILLED |
| 6 | M anticollapse | ALC | ..._v22M | 72.5k(pk) | 0.177 | **0.559** | 0.127 | anti-collapse WORKS; still climbing; KEEP |
| 7 | N warmstart | LOCAL | ..._v22N | running | | | | resume C + std-0.5 + anticollapse |
| 7 | N2 warmcalm | ALC | ..._v22N2 | running | | | | resume C + std-0.35 + anticollapse |
