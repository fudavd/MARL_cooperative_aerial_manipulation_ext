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
| P3 | ALC | assist2.2, upsweep_min0.08 | — | — | — | — | — | training |
| P4 | ALC | up_w16 + assist2.2 | — | — | — | — | — | training |
| P5 | local | base (seed7) | — | — | — | — | — | training |
| P6 | ALC | assist3.0, upsweep_min0.08 | — | — | — | — | — | training |
| P7 | ALC | impact_up_weight=24 | — | — | — | — | — | training |

## Improvement log (newest first)
- **2026-07-15 ~00:05** — First read (~35-40k steps, training attached_frac, NOT yet eval'd). Under the up-sweep-required model **every catch is a genuine up-sweep by construction** (creep p=0), so attached_frac here already = physical catches.
  - **WINNER direction: strong swing-assist + EASY up-sweep gate.** P3 (assist2.2, upsweep_min0.08) attached_frac peak **0.099**, sustained ~0.05 & climbing. All weak/hard-gate arms collapse to ~0: P1 base 0.07→0.01, P2 up_w16 0.046→0.006, P4 (up_w16+assist2.2 but DEFAULT gate 0.12) 0.058→0.014, P5 base(local) →0.000.
  - Signal: the **0.08 up-sweep gate** matters as much as assist strength (P4 has assist2.2 but the 0.12 gate → collapses; P3 with 0.08 → sustains). base assist 1.3 cannot bootstrap.
  - **Action:** killed collapsed P1/P2/P5; launched Q wave on the winning direction — Q1(local: assist2.2+up16+gate0.08), Q2(assist2.6+gate0.08), Q3(assist2.2+gate0.08+slow-decay curric). Kept P3(winner), P4, P6(assist3.0), P7(up24).
  - attached_frac 0.05-0.10 is LOWER than the creep-era C (0.15-0.20) — expected: a real up-sweep catch is harder than a creep. Need to grow catch to 0.30 via the assist/gate/reward before worrying about hold.
- **2026-07-14 ~23:20** — Wave P launched (7 arms, both machines). Baseline creep quantified (C: v_up~0, 10.8% up-sweep). Target: reproduce catch≥0.30 with up-sweep%≫baseline (genuine swing), then hold.
