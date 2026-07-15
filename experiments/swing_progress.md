# Physical-Swing Progress Tracker

**Goal (unchanged):** catch_rate ≥ 0.30 AND hold(|caught) ≥ 0.80 @100 steps — AND achieved via a **genuine pendulum up-swing**, not a creep (so it transfers to a real drone).

**Physicality metric** (`diag_catch.py`, at the catch instant): `v_up` = hook upward velocity; **up-sweep%** = fraction of catches with v_up > 0.2 m/s. Real swing → high up-sweep%, v_up ≫ 0. Creep → v_up ≈ 0.

**Eval** = `eval_goals.py` (spawn-fixed, STRICT detach, radius 0.13) for catch/hold + `diag_catch.py` for physicality.

---

## Baseline (old creep policy, for comparison)
| policy | catch | hold | up-sweep% | v_up@catch | verdict |
|---|---|---|---|---|---|
| **C** (old reward, legacy catch) | 0.58 | 0.47 | **10.8%** | ~0.00 | CREEP — unphysical; the thing we're fixing |

## Physical-swing arms (up-sweep-required catch + swing curriculum)
Configs on the C recipe. New physics default: `require_upsweep=True`, `swing_assist_v0`, `impact_up_weight`.

| arm | machine | key config | peak step | catch | hold | up-sweep% | v_up@catch | notes |
|---|---|---|---|---|---|---|---|---|
| P1 | ALC | base (assist1.3, up_w8) | — | — | — | — | — | training |
| P2 | ALC | impact_up_weight=16 | — | — | — | — | — | training |
| P3 | ALC | assist2.2, upsweep_min0.08 | 55k+ | eval running | — | — | — | **WINNER: af 0.125 & climbing** |
| P4 | ALC | up_w16 + assist2.2 (gate0.12) | — | — | — | — | — | collapsing (af→0.017) |
| P5 | local | base (seed7) | — | — | — | — | — | KILLED (af→0) |
| P6 | ALC | assist3.0, upsweep_min0.08 | — | — | — | — | — | KILLED — assist 3.0 too strong (af→0) |
| P7 | ALC | impact_up_weight=24 | — | — | — | — | — | collapsing (af→0.012) |
| Q1 | local | assist2.2+up16+gate0.08 | — | — | — | — | — | FAILED to launch (empty log); GPU reused for P3 eval |
| Q2 | ALC | assist2.6, gate0.08 | — | — | — | — | — | early, af~0.03 (weaker than P3) |
| Q3 | ALC | assist2.2, gate0.08, slow-decay | — | — | — | — | — | early, af~0.06 |
| **R2** | ALC | mod-explore **mls-0.5**, assist2.2/gate0.08 | 57.5k | **0.0197** | 0.39 | **91%** | +0.55 | ✅ WINNER of exploration bracket; catch climbing 0.0072→0.0197; swing tightened to controlled 0.26m arc |
| R1 | ALC | mod-explore mls0.0 | 70k | 0.0074 | 0.38 | 90% | +0.76 | mls0.0 ⇒ ~½ R2 catch (higher noise = less precise); losing branch |
| R3 | ALC | mls0.0 + low-entropy | — | — | — | — | — | training (mls0.0 losing branch) |
| R4 | ALC | mls0.5 | — | — | — | — | — | training (highest noise, expected worst) |
| R5 | ALC | mls0.0 + impact_up16 | — | — | — | — | — | training (mls0.0 losing branch) |
| S1 | ALC | ≈R1 + aimed-upsweep (aim_k3) | — | — | — | — | — | training ~15k (mls0.0; note: missing R2 alive/prox/hold flags) |
| S2 | ALC | ≈R2 + aimed-upsweep (aim_k3) | — | — | — | — | — | training ~15k (mls-0.5; CONFOUNDED: missing R2 alive/prox/hold-curric flags) |
| U1 | ALC | R2 recipe + aimed + WIDE radius (0.25→0.13/200k) | 47.5k (24% anneal) | 0.011 @0.13 / 0.030 @0.22 | 0.52 | 87% | +0.55 | big swing (amp 0.84m); loose 0.22m radius only 2× catch ⇒ AIM scatter, not radius, is the limiter. ⚠️ KILLED by ALC reboot @59k |
| **W1** | ALC | R2 recipe + WIDE radius, **NO aim**, seed7 | 110k (55% anneal) | **0.0273 @0.13** / 0.032 @0.18 | **0.545** | 71% | +0.36 | ✅ **BEST SO FAR: first to beat the 0.02 ceiling AND best hold**, still physical (controlled 0.30m arc). 0.13-vs-0.18 gap narrowing as it anneals. Trajectory ↑ — re-eval at ~180k |
| W2 | ALC | same as W1, seed42 | — | — | — | — | — | training (robustness seed) |
| W3 | ALC | W1 + aimed-upsweep, seed7 | — | — | — | — | — | training (aim thread) |

## Improvement log (newest first)
- **2026-07-15 ~10:55 — 🆕 REVERSE HOLD CURRICULUM implemented (user's idea): spawn a decaying fraction of envs ALREADY ATTACHED at a steady-state hold pose.** Rationale: wide-radius eval confirmed CATCH is ceiling-bound (~0.02-0.03 peak, collapses to ~0.01 at full anneal — W2 0.0099, W3 0.0095 @220k), so HOLD (0.55-0.62 vs 0.80 target) is the more movable gap. New `_attached_spawn_reset`: places drone `wire_length` up-and-out from the assigned wall's bar, sets `attached=True` + pins `attach_point_w` to the bar → policy masters the steady-state HOLD decoupled from catch; fraction decays `frac0 → 0` (basin jitter grows) so it expands "start-at-goal → full catch-then-hold". Config-gated (`attached_spawn_curriculum`, default ON for training; eval sets it False → honest catch-from-scratch). Reuses the existing pin-force + point-blank spawn geometry.
  - **Validated (committed `d89f91b`):** syntax OK; smoke 128/128, 0 errors; **A/B random policy: attached-spawn ON `attached_frac 0.158` vs OFF `0.012` (13×)** ⇒ genuinely injects the hold state.
  - **X-WAVE launched (ALC, alongside finishing W-wave):** X1 = W1 recipe + wide radius + attached_spawn frac0=0.5 · X2 = frac0=0.7 (stronger hold bootstrap) · X3 = narrow radius + frac0=0.5 (hold-focus without the collapsing wide radius, seed42). Decisive metric = **hold-given-caught** in deterministic eval (curriculum OFF) at ~55k+. Target: lift hold 0.55→0.80.


- **2026-07-15 ~09:05 — ✅ WIDE-RADIUS HELPS: W1 beats the ceiling on BOTH catch and hold (mid-anneal).** W1 (wide radius 0.25→0.13, no aim) at 110k / 55% anneal: **catch@0.13 = 0.0273** (> R2 peak 0.0197 and > narrow ceiling ~0.02) with **hold 0.545** (best yet), still genuinely physical (71% up-sweep, controlled 0.30m arc, v_up +0.36). The 0.13-vs-0.18 radius gap has collapsed to 0.027 vs 0.032 → the policy is **generalizing to the tighter radius as it anneals**, so catch@0.13 should keep climbing toward anneal-completion (~180k). This is the first lever to move BOTH metrics up together.
  - Caveat: 0.027 « 0.30 target — even the winning trajectory is an order of magnitude short. Wide-radius is a real improvement but unlikely to close the full gap alone; it does show the constraint is *soft* (curriculum shifts it), not a hard wall. Decisive eval at ~180k (radius≈0.13) pending; W2 (seed42) checks robustness.

- **2026-07-15 ~06:59 — ⚠️ ALC REBOOTED (external), killed all 4 arms @ ~06:18. Relaunched clean wide-radius W-wave.** `uptime` = 41min, logs cut cleanly at U1 step 59519 → not OOM/not my load, an external host reboot. GPU fully free afterwards. Checkpoints survived (every 2.5k). R2/S1/S2 were already characterized (ceiling ~0.02, aimed no-help), so only the wide-radius test was worth restoring — U1 had reached just 24% of its anneal.
  - **W-WAVE (clean, full free GPU, 128 envs each):** W1 = R2 recipe + wide radius (0.25→0.13/200k) NO aim, seed7 [PRIMARY confound-free radius test] · W2 = same, seed42 [robustness] · W3 = wide+aim, seed7 [continues U1's config]. Decisive eval when radius anneals near 0.13 (~150k): does catch@0.13 climb above the ~0.02 narrow-radius ceiling? Given U1's finding (0.22m radius only 2× catch ⇒ aim-scatter-limited), expectation is modest — but this completes the test cleanly.

- **2026-07-15 ~05:15 — ❌ NEGATIVE RESULTS: R2 collapses past peak; aimed-upsweep reward does NOT help. Wide-radius (U1) is the last standing lever.**
  - **R2 peak-then-collapse confirmed:** eval at 103k catch **0.0094** (down from 0.0197@58k) + up-sweep 91%→69% (degrading physics). **Peak ≈55-60k; narrow-radius catch ceiling ≈ 0.02.** Eval sweet spot for these arms is ~55k, not later.
  - **Aimed-upsweep reward = no gain** (both seeds): R1(no-aim,mls0) 0.0074 vs S1(aim,mls0) 0.0057; R2(no-aim,mls-0.5) 0.0197 vs S2(aim,mls-0.5) 0.0032[confounded]. Rewarding v_up·exp(-k·dist) doesn't sharpen the connect — the swing is *already* physical (S1 94% up-sweep, 66 catches); the bottleneck is the fundamental difficulty of timing the up-sweep peak *within the narrow 0.13m radius*, which reward shaping can't fix. Aimed reward stays committed but config-gated OFF (default) — a tested-negative option, not the default path.
  - **Implication:** narrow-radius + genuine-swing + strict-detach caps catch at ~0.02 « 0.30 target. If **U1's wide-radius anneal (0.25→0.13)** doesn't dramatically lift catch, the 0.30 target may be near the physical ceiling under these constraints — key signal for the user's decision. Awaiting U1 @55k.

- **2026-07-15 ~03:10 — R-bracket verdict + WIDE-RADIUS lever identified.** Deterministic eval of the exploration bracket at ~58-70k: **R2 (mls=-0.5) catch 0.0197 = 2.7× R1 (mls=0.0) 0.0074**, both genuinely physical (90-91% up-sweep), R2's swing tighter/more controlled (0.26m vs 0.34m amp). **Lower action-noise ⇒ better catch precision** — confirmed. R2 catch is *climbing* with training (0.0072@27.5k → 0.0197@57.5k).
  - **But best catch 0.0197 @58k vs 0.30 target = 15× gap. Precision is the hard bottleneck** (inherent to the genuine-swing constraint — the up-sweep must peak within 0.13m of the bar with v_up>gate).
  - **KEY FINDING: the capture-radius curriculum was a near-no-op.** Arms trained with `capture_radius_start=0.15` → target 0.13 = only 0.02m of slack, and at 58k the anneal (120k steps) had it at ~0.14. That gives almost no bootstrap help. (Also the cfg *default* `capture_radius_start=0.08` is inverted vs its "loose→tighten" comment — a latent footgun; arms override it so running arms are fine.)
  - **NEW LEVER — real wide-radius anneal:** start radius **0.25 → 0.13 over 200k** so the genuine swing racks up catches early with a forgiving radius, then tightens to true hook geometry. Eval stays honest (`eval_goals` disables capture/hold/spawn curricula → radius pinned at 0.13, swing-assist decays to 0). This is the strongest curriculum lever for a hard-precision task and squarely the "curriculum shaping" the user mandated.
  - **Could not free GPU** (auto-mode classifier blocked killing pre-compaction PIDs — reasonable guard). So launched **ONE** arm in the ~4GB headroom: **U1 = R2 exact recipe + aimed-upsweep + wide radius (0.25→0.13/200k), 96 envs**. R2 (narrow radius) keeps running as the baseline. If U1 catch ≫ R2, wide-radius is the path → scale it up when GPU frees.
  - Next: eval S-wave (aimed test) at ~55k (~45min), re-eval R2 at peak ~100k+, then U1 at ~55k (~1.8h).

- **2026-07-15 ~02:30 — ✅ PHYSICS SOLVED, precision is the remaining gap. Committed aimed-upsweep reward.** R2 (moderate exploration mls=-0.5) DETERMINISTIC eval = **genuine pendulum swing**: v_up +0.70, 85% up-sweep, amplitude 0.79m (≈ real 0.55m-wire arc), hold-given-caught 0.61. This is the physically-correct behaviour the user asked for — the drone up-swings the hook onto the bar, no creep (0% creep-like catches). BUT deterministic catch only 0.0072 → the swing is genuine but doesn't reliably *connect* with the bar (aiming/precision, not physicality).
  - **Reward fix (committed `bac7e6d`):** `reward_aimed_upsweep` — reward the up-stroke weighted by `exp(-k·hook_bar_dist)` so the swing is rewarded most when it peaks ON the bar (not merely inside the coarse impact box). Config-gated (default off; R-wave unaffected). Smoke-validated 160/160, 0 errors.
  - **S-WAVE launched (ALC, alongside R-wave):** S1 = R1 config (mls0.0) + aimed-upsweep, S2 = R2 config (mls-0.5) + aimed-upsweep, both aim_k=3.0, assist2.2/gate0.08. Clean A/B — R1 vs S1 and R2 vs S2 at matched exploration isolate the aimed-reward effect on catch precision.
  - **Eval plan:** let R+S mature to ~55k (peak), deterministic eval_goals + diag_catch on all; pick highest catch that keeps up-sweep% ≫ 80% (must stay physical).

- **2026-07-15 ~01:25 — ⚠️ KEY REFRAME: training attached_frac was EXPLORATION NOISE, not learned skill.** P3@55k DETERMINISTIC eval: **catch 0.0007** (≈0!) vs training af 0.125. diag_catch: only 5 catches, but they ARE up-sweeps (v_up +0.22, 60% up-sweep, 0% creep) with swing amplitude **1.3 m ≫ 0.55 m wire** = wild flailing. Root cause: the ALC regime `max_log_std=2.0` (std→7.4) + entropy 0.005 keeps the policy CHAOTIC — it catches by luck in training, but the mean/deterministic policy hasn't learned a controlled swing. The up-sweep-required catch + assist are working (catches are real up-sweeps), but exploration is too high to converge to a repeatable maneuver.
  - **PIVOT: strong assist (2.2, gate0.08) + MODERATE exploration** (assist provides structured exploration; low action-noise → learnable controlled swing). Killing the chaos-regime arms; launching R-wave bracketing max_log_std ∈ {-0.5, 0.0, 0.5} + low-entropy + strong up-sweep reward, all on assist2.2/gate0.08.
  - **Eval rule going forward: judge by DETERMINISTIC catch (eval_goals), NOT training attached_frac** (the latter is inflated by exploration noise in the high-std regime).
  - **Workflow fixed:** ALC eval SEGFAULTS while training holds the GPU → **ALC = training farm, LOCAL = dedicated eval station** (local yaml adopted ALC's [1024,512,256,128] net so local evals load ALC checkpoints).
  - **R-WAVE (moderate exploration + strong assist2.2/gate0.08), all ALC:** R1 max_log_std=0.0 · R2 =-0.5 · R3 =0.0+entropy0.001(low) · R4 =0.5 · R5 =0.0+impact_up16. Test: does moderate exploration converge to a DETERMINISTIC controlled swing (catch>0 in eval_goals)?

- **2026-07-15 ~00:05** — First read (~35-40k steps, training attached_frac, NOT yet eval'd). Under the up-sweep-required model **every catch is a genuine up-sweep by construction** (creep p=0), so attached_frac here already = physical catches.
  - **WINNER direction: strong swing-assist + EASY up-sweep gate.** P3 (assist2.2, upsweep_min0.08) attached_frac peak **0.099**, sustained ~0.05 & climbing. All weak/hard-gate arms collapse to ~0: P1 base 0.07→0.01, P2 up_w16 0.046→0.006, P4 (up_w16+assist2.2 but DEFAULT gate 0.12) 0.058→0.014, P5 base(local) →0.000.
  - Signal: the **0.08 up-sweep gate** matters as much as assist strength (P4 has assist2.2 but the 0.12 gate → collapses; P3 with 0.08 → sustains). base assist 1.3 cannot bootstrap.
  - **Action:** killed collapsed P1/P2/P5; launched Q wave on the winning direction — Q1(local: assist2.2+up16+gate0.08), Q2(assist2.6+gate0.08), Q3(assist2.2+gate0.08+slow-decay curric). Kept P3(winner), P4, P6(assist3.0), P7(up24).
  - attached_frac 0.05-0.10 is LOWER than the creep-era C (0.15-0.20) — expected: a real up-sweep catch is harder than a creep. Need to grow catch to 0.30 via the assist/gate/reward before worrying about hold.
- **2026-07-14 ~23:20** — Wave P launched (7 arms, both machines). Baseline creep quantified (C: v_up~0, 10.8% up-sweep). Target: reproduce catch≥0.30 with up-sweep%≫baseline (genuine swing), then hold.
