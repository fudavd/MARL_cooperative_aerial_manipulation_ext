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
- **2026-07-14 ~23:20** — Wave P launched (7 arms, both machines). Baseline creep quantified (C: v_up~0, 10.8% up-sweep). Target: reproduce catch≥0.30 with up-sweep%≫baseline (genuine swing), then hold.
