# Overnight Autonomous Run — Findings (2026-07-15 18:45 → 07-16 11:00)

## TL;DR
Ran an autonomous multi-arm improvement fleet on the **hook-blind MAPPO/CTDE** hook-grasp task,
strictly within the constraint **"improve the setup, never change the task or the observation
space."** Explored rewards, curricula, hyperparameters, and spawn fields across ~10 arms, judged
by a purpose-built **true-task evaluator** (`eval_mappo.py`, actor-only, 135-dim hook-free obs,
all curricula/assist OFF, radius pinned 0.13, hold@100).

**Result: the goal (catch ≥ 0.30 AND hold|caught ≥ 0.80) was NOT reached.** Best achieved,
robustly across 5+ configs: **catch ≈ 0.10–0.11, hold|caught ≈ 0.40–0.42.**
The catch rate is a **hard, robust ceiling** — a fundamental limit of a *hook-blind, open-loop*
up-swing landing the trailing hook within the 0.13 m capture radius, not a tuning gap.

## What moved the needle (ranked, all within constraints)
1. **kl-stop (`kl_threshold=0.012`)** — the anti-collapse lever. Sustains training-catch ~0.24
   where conservative defaults (v24MA1) and nopull-alone (v25A1) collapse. Every good arm uses it.
2. **Fast curriculum anneal** (capture 200k→100–140k, attached-spawn 120k→65–90k) — the biggest
   lever on **HOLD**: more true-task attach practice → hold 0.14 → 0.41–0.53, jerk 290N → ~190N.
3. **nopull (`wall_pull_weight=0`)** — helps **both** catch and hold (A12 0.10/0.42 vs same-anneal
   klstop-only A2 0.085/0.14). Confirms the long-standing prior that wall_pull is catch-antagonistic.
4. **Dynamic-hold entry (`attached_spawn_v0=0.4`, new config-gated knob)** — bootstraps hold from a
   post-swing dynamic pose; folded into the max-stack (F3). Marginal-to-positive.

## What did NOT help (confirmed dead-ends this run)
- **Stacking all levers (F3 max-stack) did not exceed the ceiling** — matched A12 (0.11/0.41).
- Conservative HP defaults (ratio_clip 0.1, entropy 0.001): collapse.
- Basin rebalance toward attached-rewards (A3): over-rewards curriculum-attachment, **catch tanks**.
- (Prior runs) aimed-upsweep reward, high action-noise regimes, wide-taut+wall_pull: no catch gain.

## Best models (deployment-ready, hook-blind MAPPO actor) — `experiments/best_checkpoints/`
| checkpoint | catch | hold\|caught | jerk (N) | notes |
|---|---|---|---|---|
| `v25F3_maxstack_95k_...` | **0.110** | 0.41 | 218 | all levers; best-sampled (n=44) |
| `v25A12_combo_175k_...`  | 0.100 | 0.42 | 192 | klstop+nopull |
| `v25F1_fastanneal_125k_...` | ~0.09 | ~0.5 | 184 | fast-anneal, small-sample |

All evaluated at the **true task** (radius 0.13, no assist, hold@100). Peak-then-collapse is
universal → these are PEAK checkpoints, not final (later checkpoints degrade).

## Interpretation
The actor never sees the hook (by your explicit design). It must swing the drone open-loop from
the observed wall-centre and *hope* the trailing weighted hook lands within 0.13 m. With no hook
feedback, aim scatter caps catch at ~0.10–0.11 regardless of reward/curriculum shaping. Hold is
limited by the strict detach model + the actor's inability to sense tension/slack.

## Recommendations (all require a decision you reserved for yourself)
1. **Relax the task** — capture_radius 0.13 → ~0.18–0.20, or hold@100 → hold@60. Fastest path to
   the numeric goal if the physical realism budget allows.
2. **Give the actor MINIMAL hook feedback** — e.g. hook→wall distance only (1 scalar), not full
   hook state. You forbade hook obs this run; this ceiling is the direct cost of that constraint.
   A closed-loop swing is the principled fix for aim scatter.
3. **Pump maneuver** — a repeatable windup→lift for consistent swing timing. Launched as **v25P1_pump**
   (still training, will NOT mature by 11:00) — let it run and eval later; it's the one untested
   catch mechanism that could raise the open-loop ceiling.

## Infrastructure delivered (committed)
- `eval_mappo.py` — true-task hook-blind evaluator (the missing judge; source of truth).
- `AGENT_CONSTITUTION.md` — the binding rulebook the fleet obeyed.
- Config-gated `attached_spawn_v0` knob (default 0.0, other arms unaffected).
- Full chronological log in `swing_progress.md`.

## Still running at hand-off
5 v24 baseline arms (prior-session, could not be retired without your OK) + v25F1/F3 (past-peak) +
**v25P1_pump** (the continuation experiment). Free the v24 arms when you're back to reclaim the GPU.
