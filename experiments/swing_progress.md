# Physical-Swing Progress Tracker

**Goal (unchanged):** catch_rate ≥ 0.30 AND hold(|caught) ≥ 0.80 @100 steps — AND achieved via a **genuine pendulum up-swing**, not a creep (so it transfers to a real drone).

**Physicality metric** (`diag_catch.py`, at the catch instant): `v_up` = hook upward velocity; **up-sweep%** = fraction of catches with v_up > 0.2 m/s. Real swing → high up-sweep%, v_up ≫ 0. Creep → v_up ≈ 0.

**Eval** = `eval_goals.py` (spawn-fixed, STRICT detach, radius 0.13) for catch/hold + `diag_catch.py` for physicality.
**MAPPO eval (NEW, hook-blind, source of truth)** = `scripts/hook_grasp/eval_mappo.py --checkpoint <agent_N.pt>` — loads the CTDE actor only (135-dim hook-free obs, asserted), deterministic, all curricula+assist OFF, radius pinned 0.13, hold measured @100. Judge every MAPPO checkpoint with THIS, not TB attached_frac.

---

# ✅ v63 BREAKTHROUGH → v64 GRASP RECIPE (2026-08-03) — READ FIRST

**The clean swing-test env (Isaac-flycrane-swing-test-v0, no basket/catch/curriculum, commanded random azimuth)
RESOLVED the "observability wall": the swing plane CAN be aimed even hook-blind.** v62 (first reward) collapsed to
dead-hang (any farmable alive stream gets farmed); v63 fixed it with SEED-A-SWING (velocity kick at reset) + alive=0:

| v63 arm | planarity | aim vs command | whirl | swing_amp |
|---|---|---|---|---|
| hook-blind | 0.78 | **0.55°** | 0.52 | 0.63 m |
| observer (+3, deployable) | 0.85 | 1.66° | 0.45 | 0.60 m |
| true hook (+3, privileged) | **0.96** | 2.03° | **0.27** | 0.70 m |

**Decomposed verdict:** AIM is a *control/pivot* problem — solvable blind, given (a) the command axis in obs,
(b) a reward that asks for planar amplitude along it, (c) a swinging start, (d) no hover farm. PLANARITY/whirl
(Miles precession) is where hook info pays: blind 0.78 → observer 0.85 → true 0.96. The old perp_ratio 0.50
"wall" was a degenerate metric + a reward that never asked the right question — NOT an information-theoretic floor.

**✅✅ v64b RESULT (2026-08-03, OBSERVER arm, deterministic n=1053): NEW BEST — catch 0.498 / hold 0.947 @ the
HONEST 0.05 m radius** (upsweep 73% @ v_up +0.37, jerk 20, deadhang 0.12). The +3-dim learned hook estimate beats
blind by +13% relative catch (0.498 vs 0.440) — the info gain shows up as up-sweep TIMING precision, not swing
shape (planarity 0.03 / aim 14° / amp 0.12 — same flick-like arc as blind). Both v64 arms clear catch≥0.30 AND
hold≥0.80 at the true radius; the strict physical-arc gate (amp≥0.20, perp≥0.60) remains open, as does
over_box 0.43. Ckpts + observer weights in experiments/best_checkpoints/ (v64b_planeobs_*BEST.pt). Videos
(matplotlib scene MP4s — Isaac render is dead on BOTH machines: ALC segfaults, local hangs in kit init):
experiments/v64_render/{v64a_blind_catch,v64b_observer_catch,...}.mp4 via eval_behavior --dump_traj +
plot_grasp_traj.py.

**✅ v64a RESULT (2026-08-03, blind arm, deterministic n=1034): BOTH GOALS CLEARED AT THE HONEST 0.05 m RADIUS
FOR THE FIRST TIME — catch 0.440 / hold|caught 0.945** (previous honest best: catch 0.13). At 0.13: catch 0.609 /
hold 0.930. Radius-sensitivity collapsed (−28% from 0.13→0.05 vs v46b's −60%) ⇒ the AIM fix is what lands swings
in the 5 cm window: **aim_deg 5.4°** (was ~72°), upsweep 71% @ v_up +0.36 (genuine up-sweeps, chord gate active),
jerk 18, deadhang 0.11. NOT yet met: the strict physical-arc score — planarity 0.043 / whirl 0.65 (aimed but still
conical) and swing_amp 0.117 < 0.20; also over_box_frac 0.42 (loiters over the box between attempts). Ops notes:
the original v64a died silently @4.2k (PhysX-style, no traceback) — resumed cleanly from model_4200 with a fast
radius re-anneal; the observer arm's MLP DIVERGED on raw obs spikes (mse 0.16→4.0) — fixed with input clamp ±20 +
inference output clamp 0.8 m + lr 3e-4 (mse now 0.012 falling), relaunched as v64b_plane_obs2.

**v64 (launched 2026-08-03): transplant the v63 commanded-plane reward into the grasp task.** u_hat = assigned
wall NORMAL, w_hat = bar axis; +amp/+vel gated near-wall (zone 0.7 m — no far-field swing farm), −off-plane
disp/vel + −|L_z| whirl everywhere unattached (`swing_plane_*` cfg block). On top of the PROVEN stack: F3 recipe +
INDI residual H3 + attached-spawn hold curriculum + swing-assist 2.2/gate 0.08 + above-box/downwash. Old failed
shaping (swing_perp/arc/energy/pump/stop) left OFF. **Honest-radius anneal 0.13 → 0.05** over 150k (judge catch
@0.05 — the real target; 0.13 was inflated). Arms: v64a blind (control) · v64b --hook_observer (v63 says it halves
the whirl gap; deployable). 256 envs × 10k iters (peak window ~10k, don't overtrain), seed 42.

---

# 🎬 VIDEOS + RENDER FIX (2026-08-18): driver 595→580 restored Isaac RTX; both behaviours filmed

Driver saga: the Aug-11 apt auto-upgrade (560.35→595.84-open) broke Isaac 4.5 RTX on the laptop (segfaults;
ALC's 595.71 explains its long-standing render segfaults too). Ubuntu 24.04's 535/550 packages are
transitional shells → 580; user installed **nvidia-driver-580 (580.173.02) — RTX rendering verified working**
(bisect: 560✓ 580✓ 595✗). Real Isaac videos (fps 20, 40 s each):
- **experiments/v84_video/v84r2_hooking.mp4** — r2 at honest gates: up-swing catch + 1364-step continuous
  hold (hold-success termination disabled for filming).
- **experiments/v86_video/v86_carry.mp4** — v86 carry: a FULL 2000-step episode, zero terminations,
  pos_error mean 0.159 m, visibly tracking a tilted commanded pose (3 drones + cables + load).
New: play_transport_video.py (RTX + --dump_traj fallback), plot_transport_traj.py (matplotlib 3D fallback).

# 🎯 v86 RESULT (10000/10000 iters): PPO WEIGHT ADAPTATION WORKS — the attach policy learns to carry

**wandb transport-ppo-handoff/k5xkxdif, finished. The hand-off that had NO channel for evolved head
plasticity (v83/v85: crash in 0.4–0.6 s, flat landscape) trains cleanly under PPO:**
- **Survival: median-61-step frozen r2 → mean episode length ~1330–1445 steps (13–14 s of the 20 s cap) in
  the best rollout windows** (high window-to-window variance is rollout-bucket noise, not regression).
- **Tracking: pos_error 0.82 → ~0.30 m, ori_error 1.77 → ~0.9 rad, per-step crash_term 0.003, episode
  return 0.19 → ~30.**
- Verdict on the user's question: gradient-based ALL-weights adaptation steers the attach behaviour into
  cooperative carrying — the adaptation mechanism was never the blocker, the frozen-innate-behavior channel
  was. NOT yet optimal: most episodes still end in a crash before 20 s and 0.30 m tracking is coarse vs the
  dedicated hover policies (paper MAPPO / July Hebbian ~165-189 return regime).
- Ckpt `best_checkpoints/v86_ppo_handoff_9999_...TRANSPORT.pt`. Open follow-ups: from-scratch PPO control
  (same env, no warm start — does the attach init actually help?), longer training / termination curriculum
  for the last mile, video of the carry behaviour.

# 🎯 v86 PPO HAND-OFF FINETUNE (2026-08-15, launched): gradient weight adaptation from r2 on the transport task

User direction after the v85 null: test whether PPO (all-weights gradient adaptation) can steer the attach
behaviour into carrying, where evolved head plasticity had no channel. New
`scripts/hook_grasp/train_rslrl_transport.py`: `TransportSharedAdapter` maps the 3-drone hover_rand hand-off
env (identical to v83/v85 minus ES-only CRN/slots: 153-dim hook-layout obs + INDI residual, wall directive,
deterministic horizontal spawn, per-env attachment DR, mass pinned) onto rsl_rl as ONE shared policy with
each drone a batch lane (3E; rewards/dones env-shared and tiled; critic = the 84-dim global state tiled —
value is env-level). **WARM START = r2's actor + action std + obs normalizer (verified tensor-equal at
load); critic fresh** (r2's 180-dim hook critic state doesn't exist here). Feedforward ActorCritic [256,256].
512 envs × 10k iters, wandb **transport-ppo-handoff/v86_ppo_handoff**. Watch Mean reward,
Metrics/crash_term (↓ = surviving), Metrics/pos_error (↓ = carrying to the commanded pose).

# 🧬 v85 RESULT (2026-08-14 12:05, 198 gens @ 20 h cap): NULL AGAIN — ABCD-only channel confirmed causally dead

**wandb hebbian-transport-eshead/z6vfstw9: crash_frac 1.000 every generation, pair diffs ~3e-4 (u ~0.014 →
no gradient), best_improvement 0.00035 = noise, theta ≈ 0.** 16× the envs (8192), sigma 0.1, ui 50, and the
better r2 backbone changed nothing. Post-mortem diag: **r2 survives median 61 steps (0.61 s)** on the
transport plant (v74: 38) — crash mix shifted to angle_limit_load 45% / drone_collision 35% — still far
below ui=50, so the plastic head again gets ~1 update before death. **Three-run verdict (v82 hook, v83 v74,
v85 r2+scale): ABCD-only last-layer plasticity has NO causal path to fitness when the frozen innate behavior
dies before plasticity fires. The July-proven channel is the missing one: put the innate weights in the
genome (ΔW block on the head, +1280 → genome 6400) so ES can shape behavior at t=0 — and/or a termination
curriculum so episodes outlive the update interval.**

# 🧬 v85 HAND-OFF RETRY WITH r2 BACKBONE (2026-08-13, launched): ABCD head-plasticity at July ES scale

User direction: v83 rerun with the r2 MLP backbone (v84 winner, catch 0.904/hold 0.970) + the July
working-run OpenAI-ES scale: **pop 1024 × 8 repeats = 8192 envs** (July hover_rand_dr proven), sigma 0.1,
ui 50, hlr 0.05, genome = ABCD of the 256→5 head = **5120** (unchanged — head shape identical to v74's).
v82 honest instrument retained (raw pair diffs u_scale 0.02, SGD lr 0.001, theta anchor, revert, baseline +
center slots, CRN goals + pair-matched attachments). evolve_head_transport.py now auto-detects feedforward
vs LSTM checkpoints. **Smoke signal: r2 crash 0.875→0.50 in 100 steps — NOT the v83 wall-to-wall 1.0; the
MLP attach policy already survives the transport plant longer than v74 did.** ALC experiments/v85_handoff_r2/,
wandb hebbian-transport-eshead, 200 gens / 20 h cap.

# 📉 v84 GRID COMPLETE (2026-08-12 06:39): LSTM unnecessary — MLP-only H3 is the NEW CAMPAIGN BEST; floor found

| run | H | mem | hidden | actor weights | catch | hold\|caught | verdict |
|---|---|---|---|---|---|---|---|
| v74 baseline | 3 | LSTM | 256 | 551k | 0.787 | 0.923 | old best |
| r1_h1_lstm256 | 1 | LSTM | 256 | 447k | 0.750 | 0.856 | ✓ both goals |
| **r2_h3_mlp256** | 3 | — | 256 | **106k** | **0.904** | **0.970** | **✓✓ NEW BEST** (jerk 12.96) |
| r3_h1_mlp256 | 1 | — | 256 | 80k | 0.607 | 0.917 | ✓ (jerk 7.48, calmest ever) |
| r4_h3_mlp128 | 3 | — | 128 | **37k** | 0.616 | 0.934 | ✓ smallest viable (ABCD 146k) |
| r5_h1_mlp128 | 1 | — | 128 | 24k | 0.000 | — | ✗ deadhang collapse (0.66) |
| r6_h1_mlp64 | 1 | — | 64 | 7.7k | 0.005 | — | ✗ deadhang collapse (0.44) |

**Findings (single-seed each — replicate before hard claims):**
1. **The LSTM was never needed and actually hurt**: r2 (H3 frame-stack, plain MLP-256) beats v74 on every
   headline metric — catch 0.904 / hold 0.970 / upsweep 0.996 / jerk 12.96. New campaign best;
   ckpt `best_checkpoints/v84r2_...NEWBEST.pt` (2.6 MB vs v74's 13.6 MB).
2. **The H3 frame-stack is the load-bearing memory**: at every capacity H3 ≫ H1 (0.904→0.607 @256;
   0.616→collapse @128). History cannot be cut to 1 frame without cost; capacity interacts — H1 needs 256.
3. **Capacity floor: 128 hidden with H3** (r4: 0.616/0.934, 37k weights). Below that (H1+128, H1+64) the
   policy collapses to deadhang. Untested: H3+64 — plausible next point if a smaller genome is needed.
4. **For the hand-off genome**: r2 → ABCD 424k (quality-first); r4 → ABCD 146k / ABCD+W 183k (size-first,
   still clears both goals with margin). vs v74's 2.2 M. wandb: hook-grasp-arch-grid (6 runs).

# 📉 v84 ARCHITECTURE-REDUCTION GRID (2026-08-11, launched): how small can a v74-quality attach policy get?

Motivation: all-weights plasticity on v74 needs a 2.2 M ABCD genome (LSTM alone = 76%). If a smaller net
still solves hook-grasp, the hand-off genome becomes tractable. Grid over three reduction axes (history
H3→H1, LSTM→MLP-only via new `--feedforward` flag, hidden 256→128→64), each trained with the EXACT v74
recipe (512 envs × 20k iters, seed 42, INDI residual, 300 g) + honest eval (0.05/25°) after each:
| run | H | memory | hidden | actor-path weights (ABCD genome ×4) |
| v74 baseline | 3 | LSTM256 | 256 | 551,168 (2.20 M) — catch 0.787 / hold 0.923 |
| r1_h1_lstm256 | 1 | LSTM256 | 256 | 446,720 (1.79 M) |
| r2_h3_mlp256 | 3 | — | 256 | 105,984 (424 k) |
| r3_h1_mlp256 | 1 | — | 256 | 79,872 (319 k) |
| r4_h3_mlp128 | 3 | — | 128 | 36,608 (146 k) |
| r5_h1_mlp128 | 1 | — | 128 | 23,552 (94 k) |
| r6_h1_mlp64 | 1 | — | 64 | 7,680 (31 k — July-ES scale) |
**r2_h3_mlp256 RESULT (first grid point, 2026-08-11 19:18): the LSTM is NOT needed — MLP-only BEATS the v74
baseline: catch 0.904 / hold 0.970 honest** (vs 0.787/0.923), upsweep 0.996, jerk 12.96 (best of campaign),
deadhang 0.028, over_box 0.115. The H3 frame-stack alone carries the needed short-term dynamics. Actor path
551k → 106k (ABCD genome 2.2 M → 424 k) with a performance GAIN.

**r1_h1_lstm256 RESULT (19:36): H1 with LSTM still clears both goals but costs performance — catch 0.750 /
hold 0.856** (vs v74 H3-LSTM 0.787/0.923): shrinking history hurts where dropping the LSTM helped.

Framework additions: `make_runner_cfg(feedforward=)` selects plain ActorCritic (rnn keys dropped) +
`hidden_dim` pseudo-key expands to [n,n]; `--feedforward` on train_rslrl + eval_behavior (must match at
load). Chain `run_archgrid.sh` on ALC: 2 lanes × 3 runs (2 concurrent max — 3×512 envs breaks paging),
results append to experiments/archgrid_results.txt, wandb project **hook-grasp-arch-grid**. ~1.5–2 days.

# 🚁 v83 RESULT (2026-08-10 23:44, 300/300 gens): NULL — plasticity never gets to act; v74 dies in 0.4 s

**Completed clean (wandb hebbian-transport-eshead/2zf9sd5d, ~5 h): crash_frac 1.000 every generation,
diff_abs_mean ~7e-5 (pair differences literally zero), improvement never left noise (best 0.001), theta ≈ 0.**
The landscape around the transplanted policy is perfectly flat — σ=0.05 head plasticity does not change when
or how it crashes.

**Post-mortem diagnostic (diag_handoff.py, 32 envs, 939 episodes): pure v74 survives a MEDIAN of 38 control
steps (0.38 s) on the transport plant.** Death causes: cable-angle-vs-drone limit 55%, cable-angle-vs-load
24%, drone collision 14%, cable collision 6%. Mechanism: v74's core competence — pumping the wire to big
swing angles — is precisely what the taut-cable carry task terminates on. And with update_interval=25, the
Hebbian head gets ~ONE update before death: "online adaptation" has no time to exist, so evolution has
nothing to shape. The null is structural, not a tuning miss.

**Levers if pursued (user's call):** (1) termination curriculum — widen cable-angle/collision limits early so
episodes live long enough for plasticity to matter; (2) evolve the initial head weights too (static shift
toward calm, not just ABCD); (3) faster plasticity (ui 25→5) so it can act within 40 steps; (4) drop ES —
PPO-finetune from v74 on the carry task (the dense-reward, big-domain-gap regime is PPO's home turf, cf.
v79-v82: every ES attempt on this stack has lost to PPO).

# 🚁 v83 ATTACH→TRANSPORT HAND-OFF (2026-08-10, launched): v74 policy ×3 drones + evolved head plasticity

User direction: run the v74 attach policy on ALL THREE drones of the transport env (hover_rand, "as if the
drones just did the attachment one by one") and evolve Hebbian ABCD on its last layer (v82 instrument) so
online plasticity adapts attach→carry. New `scripts/hebbian/evolve_head_transport.py` + hover_rand env flags:
- **Obs match by design**: hook env's 45-frame was built to mirror the hover partial layout, and both replace
  the ego one-hot with the wall-centre directive (same `R_load @ c_load` convention). Added
  `actor_indi_residual` cfg flag to hover_rand (appends the 6-dim INDI residual per frame, 45→51, ×H3 = 153 =
  the exact v74 actor input). v74 loaded directly via ActorCriticRecurrent + EmpiricalNormalization
  (ckpt keys: model_state_dict + obs_norm_state_dict), frozen; 3 drones = agent-major batch 3E through one
  shared LSTM (per-drone hidden state + per-drone plastic head w, reset to W0 on episode reset).
- **Conditions matched for the instrument**: goal poses CRN'd per scenario (new `crn` flag, env % n_repeats,
  redrawn per generation); per-env attachment DR condition-slot-matched (`attachment_condition_slots`) so
  antithetic pair members AND the baseline/center slots get identical attachment draws; spawn deterministic
  horizontal (roll/pitch/yaw 0, z offset 1.0); basket mass pinned 1.4 kg (un-CRN'd DR would pollute pair diffs).
- **Rescaled v82 instrument**: hover reward is ~100× smaller than hook (~0.01–0.05/step) → u_scale 0.02,
  revert_margin 0.02 (v82's 0.5 values would zero every gradient).
- **Smoke result**: pipeline clean (impr=0.0000 at theta=0 exactly), but **pure v74 crashes early in 100% of
  envs** — the attach→carry domain gap is the experiment; dense tracking reward gives ES a survive-longer
  gradient. Run: ALC `experiments/v83_handoff/`, pop 256×2 = 512 envs, 300 gens, wandb hebbian-transport-eshead.

# 🧬 v82 ES-HEAD RETRY, HONEST INSTRUMENT (2026-08-10): fix works, verdict = plasticity adds NOTHING (~0.00)

v82 = v81 rerun with the two fixes (user-directed): (1) gradient from RAW baseline-subtracted antithetic pair
differences on a fixed scale (u=clip(Δf/0.5,±3), plain SGD — replaces centered ranks + Adam, which are
offset/scale-invariant and turned no-signal gens into full-size steps), (2) theta anchor 1%/gen toward 0 + trust
region (revert to best theta after 10 straight gens of improvement < −0.5). Two reserved population slots per gen
under identical CRN conditions: genome=0 (pure PPO) and unperturbed theta → `improvement = fit_center − fit_base`
is the honest metric. Run: ALC `experiments/v82_eshead_rel/eshead_..._15-04-41`, wandb
**hook-grasp-es-head/p8orhzyl**, 300/300 gens completed 18:07.

**Results:**
- **The optimizer pathology is gone**: theta_norm stayed ~0.05 the whole run (v81: drifted to 25), 0 reverts,
  center tracked base throughout. The fix did exactly what it was designed to do.
- **But the honest signal is ZERO: improvement = +0.002 ± 0.049 over 300 gens (SEM 0.003, frac>0 = 0.53)** —
  Hebbian head plasticity adds nothing over the plain PPO head. best_improvement 0.186 @gen 16 is a noise draw.
  diff_abs_mean ~0.02/step = pair differences at noise level: the fitness landscape around theta=0 is FLAT at
  sigma 0.05 — there is no direction in which plasticity helps.
- **Retroactive v81 correction**: v82's fit_base (pure PPO) itself fell 4.2 → 0.2/step over generations — that's
  the generation-driven curriculum hardening (radius 0.13→0.05, chord gate annealing, attached-spawn income
  declining), NOT policy damage. So most of v81's "collapse" I attributed to theta drift was actually the task
  getting harder under every genome; v81's real defect (unmeasured drift to norm 25 with no baseline to compare
  against) is exactly what v82's instrument fixes.

**Campaign verdict on the ES line (4 attempts: v79 CMA-ES, v80 July-scheme OpenAI-ES, v81, v82): closed.**
From-scratch ES can't reach the gated catch; plasticity-on-pretrained measures ~0.00 benefit with an honest
baseline-subtracted instrument. PPO v74 (catch 0.787 / hold 0.923 honest, hook-blind) stands as the solution.

# 🧬 v81 ES-HEBBIAN HEAD ON FROZEN PPO BACKBONE (2026-08-10): completed 300/300 gens — NEGATIVE

The third ES attempt (`scripts/hook_grasp/evolve_head.py`, run `experiments/v81_eshead/eshead_p256_s0.05_hlr0.05_ui25_seed42_2026-08-10_11-39-16` on ALC, wandb **hook-grasp-es-head/clil5s2t**, finished cleanly 14:43): freeze the v74 backbone (normalizer+LSTM+MLP hidden), evolve only the ABCD Hebbian plasticity coefficients of the 5×256 action head (genome 5120, theta=0 ⇒ head stays at the PPO weights W0; online rule dw=A·post⊗pre+B·pre+C·post+D, elastic anchor 0.01 to W0, |Δw|≤0.5). OpenAI-ES pop 256×2 repeats, sigma 0.05, Adam lr 0.02, CRN, fitness = per-step mean reward over 2000 steps.

**Result: evolution strictly degraded the pretrained policy and never recovered.**
- Gen 0–2 (theta≈0 = plain PPO ± tiny plasticity): fit_mean 4.2–7.4, attached_frac 0.5–1.0 — the backbone works.
- By gen ~28 the population is permanently broken: fit_mean ~0.15–0.3, attached_frac 0.03–0.2, and **no generation after gen 28 ever beat fit_best 1.1** (gen-275 best: 0.29). fit_best_so_far 7.39 was set at gen 1 and never touched again.
- theta_norm grew monotonically ~+0.08/gen to 25.0 at gen 299 — Adam integrated a noise-dominated rank gradient into a steady drift away from theta=0; the anchor keeps w near W0 *within* an episode but nothing anchors *theta*, so the plasticity rule itself walked into a regime that actively destroys the hold/catch behavior every episode.
- The big gen-to-gen fitness oscillation early on (7.4 / 0.3 / 3.7 / 0.3 …) tracks attached_frac (1.0 / 0.02 / 0.5 …) — it's the CRN spawn condition changing per generation, not genome quality; that bimodal fitness noise is exactly what swamped the ES gradient.

**Verdict: the best genome found is theta≈0, i.e. "don't touch the PPO head."** Hebbian head-plasticity on this task, evolved with per-step-reward fitness, is net harmful — consistent with v79/v80: no ES variant tried (from-scratch CMA-ES, from-scratch July-scheme OpenAI-ES, plasticity-on-pretrained) has added anything over PPO v74 (0.787/0.923). If this direction is ever retried, the fixes suggested by the failure mode are: fitness = *improvement over the theta=0 baseline under identical CRN conditions* (baseline-subtracted, kills the spawn-condition noise), and/or anchor/trust-region on theta itself.

# 🧬 EVOLUTION vs PPO ON THE HONEST TASK (2026-08-08/09): both ES attempts FAIL — catch 0.0

Two evolutionary runs of the FINAL task (honest 0.05 radius annealed, 25° outside chord gate annealed, v74
reward stack, 300 g, hook-blind + INDI residual), honest-evaluated with the patched eval_hebbian (gates pinned,
chord curriculum disabled — no July radius-mismatch):
- **v79 CMA-ES** (my first attempt; pop 48, latent 4, hist 3, fitness=return, 400 gens): fitness 53→94 but ALL
  of it curriculum income (attached-spawn holds + proximity/alive) — honest catch 0.0, and mean episode 122
  steps = can't even sustain from-scratch flight. wandb hebbian-hook-grasp/v9flfmlq.
- **v80 OpenAI-ES = the July es_f3_crn_indi controller scheme** (pop 1024×8=8192 envs, latent 8,
  evolve_init_weights+keep_d, hebbian_lr 0.05 @ ui100, hist 1 + INDI = 51-dim, fitness=mean_reward, CRN) united
  with the current task/reward findings; 512 gens / 20 h: pop mean NEVER left ~0/step, best-so-far flat from
  gen ~70. Honest: catch 0.0 / attached 0.0 — but STABLE FLIGHT (mean episode ~1920/2000). It evolved
  hover/approach, never the swing-catch. wandb hook-grasp-es-f3/woqd6wkf (beside the July baselines).
**Conclusion: the physically-gated catch is out of reach for both ES variants at these budgets** — the July ES
0.11 catch lived off the gateless 0.13-radius task. The sparse 25°-gated catch gives ES no fitness gradient
(PPO's critic + dense advantage does). PPO-LSTM (v74: 0.787/0.923) stands as the solution; ES would need a
staged fitness (flight→approach→swing→catch) or a pretrained backbone to compete.

# 📐 HOOK-MASS CURVE COMPLETE (2026-08-07): 250 g = the sweet spot; goals met from 200 g up

All points = the BLIND v74 recipe (no observer, wall_pull 8) @ 512×20k s42, each trained AND evaluated at its
own mass (hook_mass_kg cfg → startup event), honest 0.05 + 25° outside gate:

| bob mass | 50 g | 100 g | 150 g | 200 g | **250 g** | 300 g |
|---|---|---|---|---|---|---|
| catch (goal .30) | 0.024 | 0.271 | 0.497 | 0.604 ✓ | **0.792 ✓** | 0.787 ✓ |
| hold\|caught (goal .80) | 0.333 | 0.388 | 0.540 | 0.818 ✓ | **0.940 ✓** | 0.923 ✓ |

- **Performance saturates at ~250 g** (250 ≈ 300 within noise; 250 nominally best on both metrics).
- **HOLD has a knee between 150→200 g** (0.54 → 0.82): the bar-seating tension cliff. Catch climbs smoothly.
- Up-sweep% stays ~99-100% at EVERY mass — the swing behavior is mass-independent; conversion + retention are not.
- 50 g straight-wire variant (v77) fixed the wire-whip (planarity 0.75) but NOT the catch (0.001) — light-bob
  failure is momentum/tension, not the whip (whip was a symptom).
- **HARDWARE SPEC: 200 g = minimum viable, 250 g = optimal** (max performance, lightest at the plateau).
Ckpts: v78c_m250 (MASSOPT) + v74 (300 g) in experiments/best_checkpoints/.

# 🎓 FINAL ABLATIONS (2026-08-06): the HOOK-BLIND actor is the true final recipe; light hook FAILS

| honest 0.05 + 25° gate, 512×20k s42 | v73 (observer, 300g) | **v74 (BLIND, 300g)** | v75 (observer, 50g) |
|---|---|---|---|
| catch | 0.776 | **0.787** | 0.024 |
| hold\|caught | 0.936 | 0.923 | 0.333 |
| att_speed / slack | 0.56 / 0.14% | **0.47** / 0.16% | 0.63 / — |
| upsweep / aim | 99.5% / 1.6° | 99.6% / 1.5° | 100% / 2.0° |

**(1) The hook OBSERVER is UNNECESSARY** — the fully hook-blind actor (153-dim, IMU/INDI residual only, the
project's ORIGINAL hard constraint) matches v73 and holds calmest. The July "observability wall" was a
reward-design + training-scale limit, never an information limit. Deployable with onboard sensing only.
**(2) The 50 g hook COLLAPSES catch (0.024)** under the modern gates — its swing is beautifully planar
(planarity 0.78, aim 2°, amp 0.148 — best shape of the campaign!) but the light bob lacks the momentum to
convert swings into 25°-gated catches at 5 cm (and its INDI signature is 6× weaker). The July "50 g made catch
learnable" finding belonged to the gateless 0.13-radius era; with physical gates the HEAVY bob is an asset.
**FINAL RECIPE = v74**: the v73 env_kv @ 512×20k, --indi_residual H3, NO observer, 300 g hook. Ckpt:
best_checkpoints/v74_noobs_blind..._HOOKBLIND_FINAL.pt.

# 🏁 v73 = FINAL RECIPE (2026-08-05): wall_pull restored — best everything, calmest holds

**v73 = v71 + wall_pull_weight 8 (the attached-phase into-wall-velocity penalty F3 had zeroed; user's
"rocking makes the wire slack" fix), 512×20k s42, honest 0.05 + 25° gate:**
catch **0.776** · hold **0.936** · jerk **15.5** (campaign best; v28's creeper was 103) · att_speed 0.74→**0.56**
· slack_frac **0.0014** · upsweep 99.5% · planarity 0.69 · aim 1.6° · over_box 0.164 · swing_amp 0.136.
Every metric ≥ v71; the single restored knob was free performance. Ckpt:
best_checkpoints/v73_wallpull...FINAL.pt (+observer in experiments/v73/). Video: experiments/v73/v73_attempts.mp4.
Residual polish (not blocking): att_speed 0.56 = holds still station-keep dynamically in the lateral pose;
swing_amp 0.136 < the 0.20 strict-arc bar (directed flick-swing, not a large arc).

# 🏆🏆 SEED-CONFIRMED (2026-08-05): v71 recipe meets both goals on BOTH seeds

| honest 0.05 + 25° gate | seed 42 | seed 7 | goal |
|---|---|---|---|
| catch | 0.708 | **0.841** | ≥0.30 ✅✅ |
| hold\|caught | 0.897 | **0.962** | ≥0.80 ✅✅ |
| upsweep | 99.5% | 99.8% | genuine |
| slack_frac (attached) | (rescoring) | 0.0013 | wire ~never slack |
| att_speed | (rescoring) | 0.65 m/s | still moving during holds (the visible "rocking") |

Seed-7 ran into the rare C-level sim death @17.4k, resumed cleanly from ckpt → 20k (curricula pinned). Both
ckpts in experiments/best_checkpoints/ (v71_*GOALSMET.pt, v71s7_*GOALSMET.pt). Remaining polish: att_speed
0.65 (hold is dynamic, not a still taut hover) — v73 (= v71 + wall_pull_weight 8, the into-wall-velocity
penalty F3 had zeroed) targets exactly this; new eval metrics slack_frac/att_speed track it.

# 🏆 v71 (2026-08-05): BOTH GOALS MET AT THE HONEST RADIUS — SCALE WAS THE MISSING LEVER

**v71 = the v70a phase-gated recipe at 512 envs × 20k iters (curricula rescaled 300k/120k), seed 42.
Deterministic eval @ TRUE 0.05 m + 25° outside chord gate + z_hi 0.22, n=1025 episodes:**
| metric | v71 | goal |
|---|---|---|
| **catch** | **0.708** | ≥0.30 ✅ (2.4×) |
| **hold\|caught @100** | **0.897** | ≥0.80 ✅ |
| upsweep / v_up | 99.5% / +0.95 | genuine energetic J-hook up-sweeps |
| perp_ratio_rel / planarity / aim | 0.678 / 0.735 / 1.16° | ≥0.60 ✓ — directed PLANAR swing |
| deadhang / jerk / over_box | 0.016 / 21.8 / 0.223 | clean |
Training af rose to the very END (0.086 @20k, no peak-collapse — unlike the old 10k-peak task) ⇒ possibly still
undertrained. Only strict-arc criteria unmet: swing_amp 0.125 (<0.20) + world-frame perp 0.574 — the catch is a
small DIRECTED planar flick-swing, not a large arc. Caveats: single seed (replicate v71_s7 RUNNING), over_box
0.22 slightly above v70a. Ckpt: experiments/best_checkpoints/v71_scale512x20k_..._GOALSMET.pt (+observer).
v72 (v70a + pull_away transition reward, 256×10k) finishing — candidate to stack for hold margin.
**Lesson: after v63→v70 fixed the BEHAVIOR (aim/planarity/outside-catch), the remaining catch gap was pure
TRAINING SCALE — reward design was already sufficient at v70a.**
- v72 (v70a + pull_away transition reward 6.0, 256×10k s42): catch 0.155 / hold 0.544 — **tested-negative**
  (no hold gain over the v70a band; the cfg term stays available but default-off). Scale, not another reward,
  was the hold fix too. FINAL RECIPE = v70a env config @ 512 envs × 20k ("v71"). Replicate v71_s7 pending.

# 📊 v70a 3-SEED BAND (2026-08-05): the STABLE physically-honest frontier; v71 scale test RUNNING

**v70a (v69c + PHASE-GATED smoothing: penalties OFF in the near-wall unattached swing zone) is the first recipe
CONSISTENT across seeds** — {s42, s7, s123} @ honest 0.05 + strict gates:
catch {0.167, 0.169, 0.119} · hold {0.665, 0.594, 0.573} · aim {0.9°, 0.7°, 0.2°} · planarity ~0.6 ·
upsweep ~100% · over_box ~0.18. ⇒ **Honest reproducible frontier: catch ~0.15, hold ~0.61.** The phase gate
recovered v65b catch while keeping a hold edge; behavior metrics reproduce every seed. NB the v69c s42
hold=0.879 did NOT replicate (s7: 0.53) — that was seed luck; only the v70a band is trustworthy.
**Goals (0.30/0.80) remain ~2x/+0.2 away CONSISTENTLY → reward tuning is exhausted; different lever needed.**
Options: scale (v71 = v70a @ 512 envs x 20k, RUNNING ~5h, curricula rescaled 300k/120k) · task-level help
(wider bar / bar-height curriculum — user decision) · accept & document the frontier.

# ✅ v69c (2026-08-05): the user's SMOOTHING/anti-wobble shaping MET THE HOLD GOAL (single-variable proof)

| honest r=0.05, strict gates | v65b (base) | **v69c = v65b + smoothing only** | v69a = + dyn-hold only |
|---|---|---|---|
| catch | 0.166 | 0.097 | 0.058 |
| hold\|caught | 0.54 | **0.879 ✓ GOAL MET** | 0.63 |
| jerk / over_box | 27.1 / 0.147 | **22.1 / 0.088** | 29.3 / 0.146 |

Smoothing = action_rate/accel 0.15 + drone_perp_osc 2.0. Mechanism: calm control -> taut wire -> no slack-detach
(hold), but the penalties tax the energetic near-wall swing stroke (catch −40%). dyn-hold v0 0.6 = net negative,
DROPPED. v67/v68 bundling mistake documented below (confounded arms, user called it out — single-variable rule).
NaN-std PPO crashes (v64a/v67/v68a) fully fixed: penalty clamps + actor-obs scrub + adapter obs/reward bounding.
**v70 RUNNING: a = v69c + PHASE-GATED smoothing (penalties OFF in the near-wall unattached swing zone — cfg
action_smooth_phase_gate/smooth_free_zone; aims to recover v65b catch while keeping v69c hold) · b = v69c seed-7
replicate (all comparisons so far are single-seed-42; historical seed spread was 0.13-0.32).**

# v65/v66 (2026-08-04, user video feedback on v64: hover-above + top-edge catch + wobble) → v67 RUNNING

**User's v64 video read (correct):** catches were hover-above → slowly lower → hook the TOP EDGE; hold stayed
above the payload (downwash on the load + degraded lift). Root cause: the outward chord gate was NEVER ON in
v64 (attachment defaults 0°/False; the old 10-15° values were per-arm overrides that never entered F3/v64),
and the bar band reached the rim (z_hi=0.30); the attached-spawn taught a ~30°-from-vertical (above-box) hold.

**v65b (observer + outside gates: chord 25° signed-outward annealed, z_hi=0.22, above_box 15, lateral hold pose):
the BEHAVIOR is fixed** — upsweep 97% @ v_up 0.81, over_box 0.43→0.147, deadhang 0.03, planarity 0.52, aim 3.4°,
perp_ratio_rel 0.60 (target hit). **Cost: catch 0.166 / hold|caught 0.54** (strict gate = harder task; lateral
hold harder than the old hover-above hold; still > the old no-gate honest 0.13 catch). Video (real IsaacLab
render, restored pipeline): experiments/v65_66/v65b_isaac*.mp4.

**v66 (v65 + translation-style shaping @ action_rate/accel 0.5 + drone_lat 1.0): COLLAPSED to dead-hang**
(catch 0.0, deadhang 0.68) — penalties outweighed the sparser 25°-gated catch stream (the v62 lesson again).
The translate-not-rotate idea stands; the dose was wrong.

**v67 launched (both observer arms, 10k iters):** v67a = v65 + hold-fix (attached_spawn_frac0 0.7 [X2 evidence],
attached_spawn_v0 0.6 dynamic-hold bootstrap, lock 8, tension 8) · v67b = same + translation shaping at 1/3 dose
(action_rate/accel 0.15, drone_perp_osc 2.0, NO lat penalty).

**Render pipeline restored (local):** (1) hybrid-GPU Vulkan fix — VK_ICD_FILENAMES=nvidia_icd.json + NV PRIME
vars (post-reboot kit picked the Intel iGPU = the "hang"); (2) `try/finally os._exit(0)` was SWALLOWING all
tracebacks (two trivial asserts looked like sim crashes) — all scripts now print tracebacks first. ALC render
still segfaults; render locally.

---

# 🔎 HONEST CORRECTION (2026-07-22, post-adversarial-audit) — READ FIRST

Two blind adversarial auditors corrected two overstatements. Both corrections stand on re-run data:

**(A) CATCH WAS RADIUS-INFLATED.** All evals (incl. the historical v28 "GOALS MET" and the 0.30 threshold itself) used
`capture_radius=0.13 m`. The config's own comment flags this as inflated; the TRUE task radius is 0.05 m (physical hook
0.02–0.03 m). Re-run of the v46b winner:
| radius | catch | hold | note |
|---|---|---|---|
| 0.13 (inflated) | 0.328 | 0.82 | the "GOALS MET" number |
| **0.05 (TRUE)** | **0.130** | 0.85 | honest — does NOT meet 0.30 |
So "catch ≥ 0.30" has only ever been cleared at the inflated radius. **Honest best catch ≈ 0.13.** The 30% goal is
NOT met at the true radius by any policy to date. (hold ≈ 0.85 is radius-robust.)

**(B) THE perp_ratio "OBSERVABILITY WALL" — checked in the RIGHT frame, still ~0.50 (undirected swing CONFIRMED).**
Auditor hypothesis: perp_ratio (world-frame `hook_vel`) was contaminated by drone translation, so the swing might
actually be directed in the drone-RELATIVE frame the reward optimizes. Added `perp_ratio_rel` to eval_behavior.py and
re-ran: v46b perp_ratio_rel = **0.499 @0.13, 0.504 @0.05** — i.e. ALSO ~0.50. The artifact hypothesis is REFUTED; the
swing is genuinely undirected/isotropic (conical) even in the reward's own frame. Note v45b directly penalized
relative-lateral velocity with weight 20 and still got 0.50 → not a simple reward-weight issue. REFRAMED mechanism:
directing a PLANAR perpendicular arc needs phase-locking the pendulum, which needs hook-phase feedback the actor lacks;
a pivot adding blind perpendicular energy yields isotropic hook motion.

**(C) THE LAST LEVER (v48, running):** a planar-DRIVEN pendulum stays planar. Instead of shaping the UNOBSERVED hook
velocity, reward the DRONE's OWN observable velocity perpendicular to the wall (`drone_perp_osc_weight` on
|v_drone·wall_normal|) and penalize along-bar drone motion (`drone_lat_penalty_weight`). Confines the observable pivot
to the wall-normal plane → the hook should follow into a planar perpendicular arc, with clean credit assignment. Arms:
v48a (blend: drone-osc + hook-swing), v48b (pure lever). Success = perp_ratio_rel rises off 0.50 on the honest metric.

---

# 🔑 (earlier, 2026-07-22) — swing_amp UNSTUCK; perp_ratio is the last blocker

The swing rewards (energy pe/ke + arc + swing_perp) **broke swing_amp free**: the previously-immovable ~0.08 m arc
jumped to **0.70–1.17 m** (v44a/b/c), and the hover DIED (`over_box_frac` 0.88 → ~0.03). Two of the three walls fell.
BUT perp_ratio stayed pinned at **0.50** and jerk EXPLODED (12 → 40–106), and catch collapsed to ~0.004.

**Diagnosis (data-backed):** the **hook_ke energy reward = `0.5·|v_hook−v_drone|²` is DIRECTION-AGNOSTIC.** The drone
maximizes it by thrashing the hook in a wild OMNIDIRECTIONAL arc — huge amplitude (swing_amp↑), random direction
(perp_ratio=0.50), violent (jerk↑ → the PhysX hangs), flailing PAST the bar (catch↓). One term, all three failures.

**Two hard walls now understood:**
1. *Observability* — actor is hook-blind (frozen 153-dim obs). Limits fine arc CONTROL.
2. *Physics-stability* — strong/omnidirectional swing rewards yank the wire → tension spike → PhysX blow-up → GPU hang.
   Dose-response CONFIRMED: v41(no swing)=full 10k stable; v44(moderate,no pump)=hang ~3200; v44c(mild pump)=1324;
   v42(strong+pump)=1123; v43b(max+pump)=172. More/omnidirectional swing reward → earlier hang.

**THE FIX (v45, 2026-07-22):** energy OFF; keep ONLY DIRECTIONAL swing terms (`swing_perp`, `arc_perp`, capped) + hard
`swing_lateral` penalty. Arms: v45a(bal,appr2) v45b(hardperp lat=20,appr1) v45c(guidedir,appr4). **All 3 ran the FULL
10k STABLY (no hang) → confirms the energy term was the physics-destabilizer; directional-only defeats the stability wall.**

**v45 RESULT (2026-07-22) — perp_ratio wall is FINAL; achievable frontier improved:**
| arm | catch | hold | perp_ratio | swing_amp | jerk | over_box |
|---|---|---|---|---|---|---|
| v45a | 0.059 | 0.85 | **0.500** | 0.11 | 17.5 | 0.18 |
| v45b (lat=20) | 0.023 | 0.68 | **0.501** | 0.24 | 12.7 | 0.09 |
| v45c (appr4) | **0.236** | 0.84 | **0.487** | 0.13 | 18.7 | 0.17 |

- **perp_ratio = 0.50 is IMMOVABLE** — even a crushing in-plane penalty (v45b lat=20) can't move it. Across ~15 experiments
  (velocity/energy/displacement/directional rewards, every gate) it never leaves 0.50. **CONCLUSION: 0.50 is the
  info-theoretic floor for a HOOK-BLIND actor — you cannot direct an arc you cannot observe. The directed (perpendicular)
  swing is UNREACHABLE with the frozen 153-dim obs.** This is the OBSERVABILITY wall, now proven, not just hypothesized.
- BUT directional-only gives a CLEAN, non-hovering, smooth profile: **jerk 12–18 (vs 103 for v28's creepy champion),
  over_box low (no hover), v45c catch 0.236** — a more PHYSICALLY-PLAUSIBLE catch than v28's hover-creep 0.32.
- **v46 (2026-07-22)** pursues both: v46a(appr6 catch-push), v46b(RNN 512×2 — last capacity test vs perp_ratio wall:
  can a bigger LSTM extract swing phase from the INDI residual?), v46c(seed7 reliability). v45c recipe = current best template.

---

# ⚠️ PHYSICAL-CORRECTNESS STATUS (2026-07-21) — read first

The "catch reliably" goal is met (catch 0.31–0.60 by recipe, hold 0.94–0.98). The **PHYSICALLY-CORRECT swing goal is
NOT yet met** — no policy performs a genuine perpendicular pendulum up-swing; catches are still creep / small-flick /
hook-from-above. Proven by `eval_behavior.py` and confirmed by blind adversarial audits.
- **THE honest metrics: `perp_ratio` (want ≥0.60; stuck ~0.50 = half sideways skim) and `swing_amp` (want ≥0.20 m;
  stuck ~0.08 m = no real arc).** `upsweep_frac`/`deadhang_frac` are PARTLY GAMEABLE (require_upsweep forces v_up>0 on
  every catch; the chord gate mechanically forbids near-vertical catches) → behavior test now weights them low.
- **Chord-angle catch gate** (`attachment.min_chord_angle_deg`, annealed): forbids latching a hook within N° of
  vertical → dead-hang creep catch becomes physically impossible. 10° made catch HONEST (0.44) but audit REFUTED the
  "real swing" claim (no arc). Now testing 30°/20° + a direct ARC-AMPLITUDE reward (v35).
- **Arc reward** (`arc_perp`/`arc_lateral`): rewards perpendicular swing-out DISPLACEMENT, to force a real arc. Hook
  mass 50g→200g (stronger pendulum; also dropped jerk ~100→15). Best less-creepy ckpt: `v34a_chordgate_9999_...`
  (catch 0.44 @10° gate). Best CATCH: `v32a...catch0.60` (creeps). Full trail: `experiments/WEEKEND_OPS.md`.

---

# SOLUTION — CURRENT BEST (2026-07-17) [catch/hold goals; physicality: see status above]

**HOLD gate robustly solved; CATCH gate met on the BEST of 3 seeds (seed-sensitive — see reproducibility below).**
Best checkpoint (seed 42) on the deterministic true-task eval (radius 0.13, fixed 50 g tip, robust n=1078 eps):
**catch 0.323 · hold|caught 0.974 · jerk 103 N → both gates cleared.** This is a genuine goal-meeting policy, but
it is the favorable tail of the seed distribution, NOT a reliable every-seed result.

**REPRODUCIBILITY (catch @~10k iters/61M steps, deterministic true-task):**
| seed / config | catch | hold | clears catch≥0.30 |
|---|---|---|---|
| **seed 42 (base)** | **0.323** (n=1078) | 0.974 | ✅ |
| seed 7 (base) | 0.219 (n=1034) | 0.982 | ❌ |
| seed 123 (base) | 0.131 (n=52) | 0.981 | ❌ |
| seed 42 + desired_kl=0.005 (12k) | 0.148 | 1.000 | ❌ (lever hurt catch) |

⇒ **catch ranges 0.13–0.32 by seed; only 1/3 base seeds clears 0.30.** The residual+fixed-tip+long-training lifts
catch well above the old hook-blind ~0.11 ceiling into 0.13–0.32, and seed 42 clears the gate — but reliably
clearing 0.30 every seed is NOT achieved. **HOLD is robustly solved: 0.97–1.00 across ALL seeds/configs.**

**Best checkpoint:** `experiments/best_checkpoints/v28_baseLong_9998_catch0.37_hold0.97_jerk107_GOALSMET_residH3_fixtip_rslRL.pt`
(filename says 0.37 = first small-n=100 read; the honest large-sample value is **catch 0.323**. Trained on ALC as
`v28_baseLong10k` = v27L3 resumed 6k→10k.)

**The winning recipe (three levers, all real-drone-honest — obs space never widened to hook/wall position):**
1. **INDI disturbance residual** in the actor obs → *solves the HOLD gate* (0.46 → 0.97). 6-dim body-frame
   `[alpha_cmd−omega_dot | Rᵀ(acc_cmd−a_meas)]` = the hook/wire wrench the onboard IMU/INDI measures (force/phase,
   NO position). Actor obs = **153** (45 base + 6 residual, ×H=3); critic stays privileged 180.
2. **Fixed 50 g hook tip** (mass DR off, wire 0.55 m fixed) → *makes CATCH learnable* (0.12 → 0.21): a consistent
   pendulum bob = consistent swing period/momentum → the actor+residual can time the up-sweep.
3. **Long training ~10k rsl_rl ITERATIONS = ~61.4M env-timesteps** (each iter = num_envs 256 × num_steps_per_env
   24 = 6,144 env-steps; so "10k iters" is NOT short — it's ~61M steps, cross-checked vs v26L1's logged "6000
   iters = 36.8M steps"). Catch was undertrained at 6k iters/37M (0.21); it climbs to **0.32 at 10k iters/61M**.
   The base **PEAKS at ~10k then collapses** (11k–13k iters fell to ~0.20) → capture the 10k checkpoint, do NOT overtrain.

**Backend:** rsl_rl `ActorCriticRecurrent` (LSTM h256, actor [256,256], asymmetric 153-actor / 180-critic PPO,
adaptive-KL). F3 env recipe (`apply_f3_env_cfg`). NOT skrl.

**Reproduce:** `scripts/hook_grasp/train_rslrl.py --headless --device cuda:0 --num_envs 256 --max_iterations 10000
--seed 42 --indi_residual --actor_history_len 3` (fixed-50g cfg is now the default). Eval:
`scripts/hook_grasp/eval_rslrl.py --num_envs 256 --steps 8000 --indi_residual --actor_history_len 3 --checkpoint <ckpt>`.

**What did NOT help (tested):** reward normalization (hurt catch 0.21→0.12), aimed-upsweep reward (neutral),
reward-shaping additions generally (F3 rewards already well-tuned). The gate was cleared by obs-signal + task-consistency
+ training length, not by reward/HP tuning.

**Known caveat — catch is SEED-SENSITIVE (see the reproducibility table above):** 3 base seeds → catch
{0.323, 0.219, 0.131}, only seed 42 clears 0.30. Stability lever `desired_kl=0.005` HURT catch (0.15, too
conservative → slow catch development) though it gave the smoothest/best hold (jerk 81, hold 1.0). **Hold robustly
solved across all seeds/configs (0.97–1.00).** Open direction to make catch reliably ≥0.30 every seed (not yet
found): more seeds+keep-best is the pragmatic path; capacity (RNN 512) untested; the catch bottleneck is up-sweep
aim/timing precision, which the residual only partially addresses (it carries force/phase, not hook position).

**Operational notes:** ALC (marvin) GPU pages catastrophically >~18 GB (2 rsl_rl arms max when sharing with the ES
runs; 1 arm when ES heavy). ALC headless offscreen RENDER segfaults (Isaac hydra init) — videos must render on a
GPU with a display (local). rsl_rl `--resume` continues the iteration counter (extend-training works). Per-arm tuning
via `--env_kv/--policy_kv/--algo_kv` (obs/task fields hard-blocked).

---

## AUTONOMOUS IMPROVEMENT FLEET (2026-07-15 18:45 → 2026-07-16 11:00)
Rulebook = `experiments/AGENT_CONSTITUTION.md` (binding: never change task / 135-dim actor obs / 180-dim critic state / catch-detach physics / success thresholds; only rewards·curricula·HP·spawn-fields). wandb project `hook-grasp-mappo`.

- **2026-07-16 14:00 — INDI-RESIDUAL actor obs launched (option b): the FIRST deliberate relaxation of the hook-blind lock, targeting the trigger-timing gap.** User authorized adding the **INDI disturbance residual** to the LSTM actor — the onboard-IMU/INDI-measurable hook+wire wrench (FORCE/phase), which reveals hook DYNAMICS but **no hook or wall POSITION** (real-drone-honest; hook→wall distance stays forbidden). Implemented as a config-gated 6-dim body-frame term appended per actor frame: **[angular: `alpha_cmd − omega_dot`** (the literal INDI increment, indi.py line ~117) **| linear: `Rᵀ(acc_cmd − a_meas)`]** — I included the *linear* channel too (stronger hook channel: wire near CoM makes the angular signature weak). `IndiController.disturbance_residual` exposed; env caches per drone in `_apply_action`, appends in `_drone_obs_frame`. New `actor_history_len` **decouples the ACTOR frame-stack depth H from the critic** (critic keeps history_len=3 → stays 180-dim, clean ablation). `rslrl_adapter.apply_indi_residual_cfg` recomputes `observation_spaces`; train/eval/play take `--indi_residual --actor_history_len`; dim asserts now read the declared space. Smoke-tested locally (H1→51, H3→153, critic 180, no NaN; eval loads a residual ckpt end-to-end). **Two ALC arms live (seed 42, differ ONLY in H):** **v26L2_indiresid_H1** (actor_obs=51: single frame + LSTM memory) · **v26L3_indiresid_H3** (actor_obs=153: 3-frame stack + LSTM). wandb `hook-grasp-rnn`, 6000 iters. GPU 11.6/24.6 GB. **Test:** does the residual let the hook-blind actor TIME the up-sweep velocity gate (v_up>0.08) → catch ≫ 0.11 ceiling, while H ablates how much explicit history the residual needs. Commit `a051d8c`. Eval plan: at peak (~5000 iters, ~1h) run eval_rslrl `--indi_residual --actor_history_len {1,3}` vs the v26L1 baseline (catch 0.10/hold 0.46) and the ~0.11 ceiling.
- **2026-07-17 ~14:00 — REPRODUCIBILITY VERDICT (honest correction): catch is SEED-SENSITIVE, only 1/3 seeds clears 0.30.** Completed 3-base-seed eval @10k: seed42 **0.323** (clears), seed7 **0.219**, seed123 **0.131** — catch spans 0.13–0.32, only seed42 clears the gate. So the earlier unqualified "GOALS MET" is walked back to: **goal met on the BEST seed (42); catch reaches 0.13–0.32 by seed (well above the old hook-blind ~0.11 ceiling) but doesn't reliably clear 0.30.** HOLD robustly solved 0.97–1.00 across ALL seeds. Stability lever **desired_kl=0.005 HURT catch** (seed42+lowkl @12k = 0.148, too conservative → slow catch dev) but gave smoothest/best hold (jerk 81, hold 1.0). Levers that did NOT help catch: reward_norm (0.21→0.12), aimed_upsweep (neutral), desired_kl=0.005 (0.32→0.15). The catch bottleneck (up-sweep aim/timing precision) is only partially addressable by the residual (force/phase, not position). Best deliverable = seed42 model_9998 (catch 0.323/hold 0.974). SOLUTION block at top updated with the reproducibility table.
- **2026-07-17 01:20 — ✅ GOAL ROBUSTLY CONFIRMED (catch 0.323 / hold 0.974, n=1078 eps) + peak-at-10k established.** Big-sample re-eval of model_9998 (256 envs × 8000 steps, 1078 eps / 348 caught): **catch 0.323, hold 0.974, jerk 103, GOALS_MET=True.** (The 0.373 first read was a favorable small n=100 sample; 0.323 is the honest number — still clears 0.30.) 14k trajectory shows the base **PEAKS at ~10k then oscillates/collapses**: 0.373@10k → 0.210@11k → 0.178@12k → 0.224@13k ⇒ training beyond 10k does NOT help (the non-stationarity the user flagged, now visible in catch). **model_9998 (10k) is the deliverable.** Now testing REPRODUCIBILITY: launched v29_seed7_10k (winning recipe, seed 7) — the prior C-recipe was seed-sensitive, so confirming a 2nd seed clears the gate de-risks the claim.
- **2026-07-17 00:40 — 🎯🎯 BOTH GOAL GATES MET. catch 0.373 / hold 0.97 @ 10k iters (deterministic true-task eval).** Resuming the base (v27L3 resid H3, fixed 50 g tip) from 6k → 10k, catch climbed monotonically: **0.139@7k → 0.175@8k → 0.244@9k → 0.373@10k** (hold 0.89→0.97, jerk 143→107, n=100 caught @10k). **catch 0.373 ≥ 0.30 AND hold 0.97 ≥ 0.80 → GOALS_MET=TRUE.** The (7k dip to 0.139 was eval noise, n=36.) ⇒ **The winning recipe = INDI residual (obs 153, H3) + fixed 50 g tip + F3 rewards, trained LONG (10k, was undertrained at 6k).** No reward/HP/arch change needed to clear the gate — the levers that mattered were (1) the residual [hold], (2) fixed tip [catch learnability], (3) enough training [catch still climbing at 6k]. Catch STILL climbing steeply at 10k (0.24→0.37 in last 1k) → **resumed to 14k** to push higher. Checkpoint preserved: best_checkpoints/v28_baseLong_9998_catch0.37_hold0.97_jerk107_GOALSMET.
- **2026-07-16 22:10 — ★ FIXED 50 g TIP boosts CATCH 0.12 → 0.21 (nuisance-mass removed → learnable up-sweep timing). Overnight search begins.** v27L3 (resid H3, trained end-to-end on the fixed 50 g tip, model_5999) fixed-tip eval: **catch 0.208, hold|caught 0.944 (51/54), jerk 134, mean-hold 1313.** vs the spread-mass-trained read (0.12/0.97). The user's fixed-tip request materially HELPS: a consistent pendulum bob (no per-env mass) makes the swing period/momentum consistent → the actor+residual time the up-sweep far better. **Base for the overnight search = v27L3, catch 0.21 / hold 0.94 — 70 % to the 0.30 catch gate, hold already solved.** Remaining lever: push catch to 0.30 via reward/HP/arch tuning (obs FROZEN). **RESOURCE NOTE: user launched 2 Hebbian ES runs on ALC (es_f3_crn, es_f3_crn_indi — adopting the INDI residual + CRN, 8192 envs each, ~10.8 GB, 12 h). To not degrade their ES, my search runs 1 rsl_rl arm at a time (256 envs, ~16.8 GB total, under the 18 GB UVM cliff).** See experiments/OVERNIGHT_PLAN_v27.md.
- **2026-07-16 18:10 — ★★ BREAKTHROUGH: the INDI residual SOLVES the HOLD gate (0.46 → 0.86–0.97). Catch ceiling (~0.11) unmoved.** Deterministic true-task eval of the residual arms at PEAK (model_4900, spread-mass, evaluated at 50 g / radius 0.13, 128 envs / ~300 eps):
  - **v26L3 resid H3: catch 0.120, hold|caught 0.968 (30/31), jerk 139 N, mean-hold 1409 steps.**
  - **v26L2 resid H1: catch 0.095, hold|caught 0.862 (25/29), jerk 216 N, mean-hold 1144 steps.**
  - vs **v26L1 baseline (no residual): catch 0.10, hold 0.46, jerk 223 N.**
  - **BOTH residual arms clear the 0.80 hold gate** (baseline never did). The 6-dim body-frame INDI disturbance residual (`alpha_cmd−omega_dot` + `Rᵀ(acc_cmd−a_meas)`) gives the hook-blind actor the FORCE/tension feedback to keep the wire taut and hold — exactly the "hold via sensed dynamics, not position" hypothesis. **Catch stays ~0.10–0.12**: the residual senses force, not hook POSITION, so the aim/up-sweep-TRIGGER ceiling is unmoved (as diagnosed). **H3 ≻ H1 on every axis** (catch +0.025, hold +0.11, jerk −77 N): the extra explicit history helps the LSTM extract the residual → smoother, better-timed control. **Net: 1 of 2 goal gates (hold≥0.80) now MET; catch≥0.30 remains the open bottleneck (positional aim, needs a different lever than force sensing).** Checkpoints preserved. Fixed-tip (v27) confirmation runs launched.
- **2026-07-16 15:10 — CURRICULUM non-stationarity fix launched (v26L4_rewardnorm): user flagged the value fn chases a moving return scale at each level shift.** Confirmed rsl_rl normalizes OBS only — there is **no value/return normalization on the extrinsic path**, so the critic learns RAW returns while the curriculum shifts reward *reachability* (capture_radius 0.25→0.13, hold_req 30→100, attached_spawn_frac 0.5→0, swing-assist decay): the return SCALE moves under the critic → biased advantages during re-tracking → plausibly the observed **peak-then-collapse** at curriculum thresholds. Fix = **Pathak reward normalization** (rsl_rl ships `EmpiricalDiscountedVariationNormalization` but only wires it into RND): divide per-step reward by a running std of the discounted return so the critic sees a ~stationary scale. Injected opt-in in the adapter (`--reward_norm`, training-only; policy net unaffected). Smoke: reward scale 88→6, no NaN. **v26L4 = v26L1's EXACT config (plain hook-blind LSTM, H=3, obs 135) + reward_norm** — a clean A/B vs v26L1 (catch 0.10/hold 0.46) isolating the curriculum fix (orthogonal to the residual arms L2/L3). Commit `de14182`. **+v26L5 = residual H3 + reward_norm (obs 153, both levers stacked)** added on user request — A/B vs L3 (rn on residual) and L4 (residual on rn). 4 arms live on ALC (GPU 17.6/24.6 GB). Full 2×2-ish design: residual (L1 vs L3), H (L2 vs L3), reward-norm (L1 vs L4), combined (L3/L4 vs L5).
- **2026-07-16 14:35 — LSTM QUALITATIVE finding (video) REFINES the diagnosis: aim solved, catch-TRIGGER timing is the gap.** Rendered v26L1 (play_rslrl_video.py, new). Catch clip (env61): hook to **0.018m**, held **1160/1200 steps** (longer/cleaner holds than MAPPO). Behavior clip (seed7, single env): 0 catches but **min hook-bar 0.034m** — i.e. the LSTM brings the hook WELL INSIDE the 0.13m radius (vs MAPPO's 0.24m in the comparable clip) yet doesn't catch. ⇒ **aim is no longer the bottleneck; the catch fails at the up-sweep VELOCITY gate** (require_upsweep, v_up>0.08 at the catch instant) which the hook-blind actor can't time without seeing hook velocity/phase. Recurrence improved CONTROL (approach precision + hold stability) but can't time the unobservable hook. **Strong implication: a minimal hook-velocity/attached signal (not full hook state) would likely close catch to the goal** — the ceiling is a trigger-timing problem, not aim. Videos: scripts/hook_grasp/_render_rslrl/lstm_catch.mp4, lstm_behavior.mp4. Run COMPLETED 6000 iters (36.8M steps, 1h17m; wandb showed "crashed" only because os._exit skipped wandb.finish() — FIXED). True-task eval: @4900 catch 0.110/hold 0.475; final @5999 **catch 0.105/hold 0.46/jerk 223**. Same ~0.11 catch as skrl MAPPO, hold marginally better (0.46 vs 0.41). ⇒ **catch ceiling is OPEN-LOOP AIM-limited** (hook-blind actor can't place the hook within 0.13m); memory improves phase/hold but not aim. NEXT: video to check the QUALITATIVE question (does the LSTM retry after a miss vs hover?) — a behavioral win possible even at equal catch. Best rsl_rl ckpt preserved. rsl_rl is ~50x faster than skrl MARL (40min vs 13h) — cheap to iterate.
- **2026-07-16 13:30 — BACKEND PIVOT LIVE: rsl_rl LSTM launched on ALC (v26L1_lstm).** Recurrent asymmetric ActorCriticRecurrent (hook-blind 135 actor + privileged 180 critic, LSTM h256), F3 env recipe, adaptive-KL LR. Verified on-machine first (obs routing, LSTM, eval, learning loop; fixed teardown-hang). wandb `hook-grasp-rnn`. Healthy: critic value-loss 646->37, ~7400 steps/s. Hypothesis: LSTM infers latent phase from obs history -> enables retry/hold-switching that the phase-blind MLP couldn't (pump P1 confirmed catch ceiling is hook-blindness, not swing quality). **Incident:** launching the 6th Isaac instance OOM-killed the 5 long-running v24 baselines (ALC 31GB RAM, not GPU, is the concurrency limit ~5 aged instances). No loss — v24 verdicts fully captured. All v25 skrl arms retired (peaks preserved in best_checkpoints/).
- **2026-07-16 12:55 — P1 PUMP eval @100k (just annealed): catch 0.067, hold 0.46 (11/24), jerk 210N.** Learned windup->lift pump lands in the SAME ~0.05-0.11 catch band — does NOT break the ceiling. Confirms the catch limit is hook-BLINDNESS (open-loop aim scatter), not swing quality; a repeatable pump can't aim without hook feedback. Hold 0.46 solid. ⇒ the actor must perceive phase to improve catch — motivates the **rsl_rl LSTM** route (phase-inference from obs history; backend built + verified, awaiting ALC launch).
- **2026-07-16 08:25 — F3 MAX-STACK EVAL: robust catch ceiling confirmed.** F3 (ultra-fast anneal 100k + nopull + dynamic-hold + klstop) @95k (256 envs, n=44 caught — most reliable sample): **catch 0.110, hold|caught 0.41 (18/44), jerk 218N, af 0.049.** Stacking ALL winning levers MATCHES A12 (0.10/0.42), does NOT exceed it. **CONCLUSION: catch ceiling ~0.10-0.11 is ROBUST across 5+ configs** (A2/F1/A12/F3) → it's a fundamental limit of a hook-BLIND open-loop up-swing landing within r=0.13, not a tuning gap. Hold ~0.40-0.42 achievable (fast-anneal + nopull + dynamic-hold). **Goal 0.30/0.80 not reachable with reward/curriculum/HP levers alone** while keeping the actor hook-blind + task fixed. Deliverables preserved in best_checkpoints/ (F3 maxstack, A12 combo, F1). Remaining time: let F3 reach peak, capture best, prepare summary — no futile new arms (nothing matures before 11:00).
- **2026-07-16 04:10 — A12 combo best-so-far + nopull confirmed; launch F3 max-stack.** A12 (slow-anneal, klstop, **nopull**) @175k (256 envs, n=31 caught — best sample): **catch 0.10 (highest), hold 0.42, jerk 192N.** vs A2 (same anneal, wall_pull=8) 0.085/0.14 ⇒ **nopull helps BOTH catch and hold** (aligns with V-wave/research prior: wall_pull is catch-antagonistic). Preserved to best_checkpoints. F2 was missing nopull (wall_pull=8) → retired at 22k, relaunched as **v25F3_maxstack** = ultra-fast anneal (100k) + **nopull** + dynamic-hold (attached_spawn_v0=0.4) + klstop = every winning lever stacked; anneals ~08:40, consolidates to deadline. **Best-known config = F3.** Standings (true-task): A12 catch 0.10/hold 0.42 · F1 0.05/0.31 · A2 0.085/0.14. Catch ceiling ~0.10; hold ~0.4-0.5. Goal 0.30/0.80 still out of reach but F3 is the best shot.
- **2026-07-16 03:40 — F1 fuller eval + catch-ceiling reality.** F1@140k (256 envs/334 eps, fully annealed): **catch 0.048, hold 0.31, jerk 184N** — LOWER than the noisier @125k (128envs) read 0.094/0.53. Lessons: (1) small-sample evals (n≈15-20 caught) are noisy — trust 256-env runs; (2) **peak-then-collapse persists even post-anneal** (F1 degraded 125k→140k) ⇒ must capture PEAK checkpoints not final; (3) **catch is stuck ~0.05-0.09 across the entire klstop family** regardless of anneal speed — a hard ceiling for an open-loop hook-BLIND swing landing within r=0.13. Fast-anneal's real, robust benefit is on HOLD (more true-task attach practice), not catch. Goal 0.30/0.80 unlikely by deadline given the catch ceiling; F2 (ultra-fast+dynamic-hold) is the remaining goal-seeking bet. Best model so far ≈ F1 ~125-140k.
- **2026-07-16 03:00 — DECISIVE FULLY-ANNEALED EVAL: fast-anneal is the biggest lever tonight.**
  - v25A2_klstop @205k (fully annealed, 200k curricula): **catch 0.085, hold|caught 0.14 (2/14), jerk 290N.** Catch trajectory 0.019→0.060→0.085 = decelerating, plateauing ~0.085 ≪ 0.30. Hold bottlenecked by strict detach (86% detach <100 steps).
  - **v25F1_fastanneal @125k (140k curricula, not even fully annealed): catch 0.094, hold|caught 0.53 (10/19), jerk 193N, af 0.047.** ← BEATS A2 on ALL metrics; hold ~4× higher.
  - **VERDICT: more true-task training time (faster curriculum anneal) dramatically improves BOTH catch and hold** (F1 anneals capture@140k/attached-spawn@90k vs A2 200k/120k → more true-task reps → learns to actually catch+hold at radius 0.13 instead of coasting on assist). Still short of 0.30/0.80 goal but the trend is monotonic & steep.
  - **Action:** retire A2 (mature, beaten). Launch **v25F2** = even-faster anneal (capture 100k, attached-spawn 65k, hold 85k) + klstop + **B1 dynamic-hold** (attached_spawn_v0=0.4, targets the hold gap) — goal-seeking stack of the two best levers, ~8h to consolidate before deadline. Keep F1 (leader) + A12 (combo) as controls.
- **2026-07-15 23:25 — A2 trajectory probe @100k (true task).** catch **0.060** (13/217, UP 3× from 0.019@55k), hold|caught 0.23 (3/13), jerk 288N. **Curriculum IS transferring — true-catch rising as radius anneals (~0.22→0.19).** But radius→0.13 still ahead, and hold quality weak (0.23, high jerk) → hold levers (B1/taut) become relevant once catch clears. NOTE: run evals FOREGROUND (background evals die at skrl seed-init; foreground completes <120s at steps=2500).
- **2026-07-15 21:45 — A2 TRUE-TASK EVAL + curriculum-gap insight.** eval_mappo.py on v25A2_klstop @55k (curricula OFF, radius 0.13): **catch 0.0185 (5/271), hold|caught 0.6 (n=5 noisy), jerk 214N** (vs H2 387N). Training-catch 0.24 but true-catch 0.019 ⇒ **huge curriculum gap: at 55k curricula are only ~27% annealed so the policy hasn't faced the true task.** ⟹ true-task eval is PREMATURE until anneal (~200k≈04:00); rank by training-catch until then. **kl-stop (A2) confirmed the anti-collapse winner** (sustains training-catch 0.24@59k while A1 nopull-alone collapsed 0.218→0.080). **Deadline risk:** arms reach true task only ~04:00, leaving ~7h to consolidate. **Action:** retire A1 (nopull-alone verdict in), launch **v25F1_fastanneal** = A2 config + faster curricula (capture 140k, attached-spawn 90k, hold 120k) → hits true task ~01:30 for ~9h true-task training.
- **2026-07-15 21:05 — GEN-25 EARLY VERDICT (matched step ~45k, curricula still active so af is inflated but comparison is fair; `catch` is the clean metric).**
  - v24H2 base: af 0.012 (peak 0.073@34k), **catch 0.080** → fully collapsed (the peak-then-collapse we target).
  - **v25A1 nopull: catch 0.218, af 0.101** · **v25A2 klstop: catch 0.275, af 0.126** ← both decisively beat H2 AND resist the af collapse. Anti-collapse + remove-wall_pull thesis CONFIRMED.
  - v25A3 basin: af 0.133 but **catch 0.058** → basin rebalance (progress16/lock8) backfired: over-rewards the attached state so policy coasts on curriculum-spawned attachment instead of catching. **RETIRED.**
  - **Action:** killed A3 (my slot), launched **v25A12_combo** = A1⊕A2 (wall_pull=0 + kl_threshold=0.012) to stack the two winners (research agent's top trio minus deferred B1). Portfolio 8/8, GPU 23.3/24.5GB.
  - NOTE: real verdict still pending curriculum anneal (~capture 200k, attached-spawn 120k → true task ~01:00-05:00); eval_mappo.py judges then.
- **2026-07-15 18:45 — REALITY CHECK from new eval_mappo.py.** v24H2 @55k on the TRUE task (all curricula+assist OFF): **catch 0.041, attached_frac 0.0065, hold|caught 0.5 (n=8 noisy), jerk 387N.** TB metrics (catch 0.35, af 0.052) are ~8× curriculum/assist-inflated. ⇒ **catch, not hold, is the binding constraint** (0.04 ≪ 0.30). Re-prioritized queue toward catch levers. eval_mappo.py verified (actor 135-dim confirmed, 2 build bugs fixed, skrl patch already applied).
- **HP verdict (mid-run TB):** wide clip(0.2)+entropy(0.004) [H2] ≫ conservative defaults [MA1 collapsed af 0.006@123k]. H2 = new base config.
- **Gen-25 launched (build on H2 base):**
  - **v25A1_nopull** — `wall_pull_weight=0.0` (remove sole catch-antagonistic hold term; V-wave proved it collapses catch, H2 unknowingly ran it at 8.0). Attacks catch+tradeoff.
  - **v25A2_kladapt** — KL-adaptive actor LR (`learning_rate_scheduler_actor=KLAdaptiveLR kl_threshold 0.01`). Anti-collapse throttle.
  - **v25A3_basin** — deepen caught basin / shallow hover basin (`proximity 1.0 alive 0.6 progress 16 lock 8`). Attacks collapse mechanism (dense-shaping dominance).
- **Queued (catch-first):** C2 persist-catch-freq (slower curricula), C3 pump maneuver (MAPPO-fresh), A4 entropy0.006+lowLR. **Hold levers deferred** until catch clears ~0.2: B1 dynamic attached-spawn (needs 3-line code, agent dispatched), A5 detach_penalty, B2 taut-standoff basin.

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
- **2026-07-15 ~16:40 — ✅ HOOK-BLIND MAPPO/CTDE RUNNING (correct architecture) + wandb restored.** Switched training to `--algorithm MAPPO` (skrl CTDE): actor = per-agent hook-free obs (135-dim), centralized critic = privileged state (180-dim, hook — training-only). Required patching the vendored skrl for the SINGLE-DRONE case (2 guards blocked 1-agent MAPPO): (1) `sequential.py` routed 1-agent envs to `single_agent_train` (no shared_states) → now routes MultiAgent/MAPPO to `multi_agent_train`; (2) `base.py` `assert num_agents > 1` in multi_agent_train/eval → relaxed to `>= 1`. Patches saved to `patches/skrl_single_drone_mappo.patch` (thirdparty/skrl is untracked); applied on ALC + local. Smoke passed (512/512, 0 errors).
  - **v24MA1** (CTDE + corrected rewards: up-sweep catch, wide radius, audit fixes impact_toward/anti_sag/slack, swing_assist) · **v24MA2** (+ hold curriculum attached_spawn f0.5). Both mls-0.5, **--wandb project hook-grasp-mappo** (visible in dashboard). err=0. Target ~1.02M timesteps.
  - **PENDING:** eval_goals/diag_catch load checkpoints via the PPO single-agent path — need a MAPPO-compatible evaluator (extract the actor policy) to judge these hook-blind checkpoints. Build while arms mature.


- **2026-07-15 ~16:10 — ⚠️⚠️ MAJOR CORRECTION: the policy was CHEATING (saw the hook). Pivot to MAPPO/CTDE (hook-blind actor).** User caught it: "are you sure we give the hook state...? I explicitly don't want that." **Confirmed:** every script wrapped the env with `multi_agent_to_single_agent(state_as_observation=True)`, feeding the 60-dim privileged STATE (hook−drone, hook vel, **hook→bar distance**, tension) as the POLICY observation, in train AND all evals/play. So ALL prior results (R/W/X/Y/Z waves, catch ceiling ~0.02, every eval) were WITH the policy seeing the hook — invalid for a real drone that lacks hook truth.
  - **User decision = CTDE** (blind actor, privileged critic). skrl **MAPPO already implements this**: actor uses the per-agent obs (`_drone_obs_frame`, 45×3=135-dim, HOOK-FREE: drone+basket+assigned-wall-centre+goal); centralized critic uses the state (60×3=180, with hook, training-only → discarded for deployment). Needs the vendored `thirdparty/skrl` fork (has `shared_states`); confirmed installed editable on ALC. `--state_obs` flag added (default False) for the legacy PPO path.
  - **All env-side work carries over** (up-sweep catch, wide-radius, hold curriculum, pump, reward-audit fixes are in the env cfg). **Stopped all 8 invalid PPO arms; GPU freed.** Relaunching on MAPPO + `--wandb` (my recent launches also lacked --wandb → not visible in wandb; fixed). **This effectively restarts the investigation with the correct, harder, hook-blind actor.**


- **2026-07-15 ~15:50 — 🔍 REWARD AUDIT (user's video concerns: bent wire in hold + sideways swing). Found 2 real reward gaps, fixed (config-gated).**
  - **Concern 3 (wall centre in train/eval) = OK.** Verified `state_as_observation=True` in BOTH train.py & eval_goals.py → policy input is the 60-dim STATE which includes hook−drone, hook vel, **hook→assigned-bar error**, attached, tension. Policy is NOT blind to the hook; the bar target is computed identically train/eval. (The 45-dim actor frame's wall-centre is unused under state_as_observation.)
  - **Concern 2 (sideways swing) = REAL GAP.** `impact_up` rewards v_up (earnable by a sideways-and-up skim); `impact_perp` only PENALIZES excess into-wall speed; NOTHING rewarded a perpendicular approach → sideways skim was a valid optimum. **Fix:** `impact_toward_weight` rewards gentle into-wall speed up to impact_target_speed → impact_perp is now target-seeking (perpendicular strike preferred).
  - **Concern 1 (bent/slack wire in hold) = REAL GAP.** Taut reward (w5) is miscalibrated (ramp needs dist 0.57 but wire caps 0.55 → maxes ~0.6) AND dominated by lock(6)+progress(≤10)+alive(5) paid regardless of tautness; NOTHING penalized sagging/altitude-loss → drone sags then detaches. **Fix:** `anti_sag_weight` (penalize downward drone vel while attached) + `slack_penalty_weight` (penalize drone-anchor dist below taut_thr) = real cost for the crash-down failure.
  - All config-gated (default 0). Numeric-validated. **Z1 launched** (W1 recipe + impact_toward=6 + anti_sag=6 + slack_penalty=20, 96 envs) to test if the fixes give a perpendicular strike + taut hold. Eval at ~55-90k.


- **2026-07-15 ~15:10 — 🆕 PENDULUM PUMP reward (user's maneuver insight) — direct attack on the catch ceiling.** User specified the physically-optimal catch maneuver: approach perpendicular to LOAD the hook behind → plant/reverse to WHIP it forward → LIFT the drone as the hook nears the bar to SEAT it. This is a *repeatable* sequence → consistent swing timing, which is exactly what the aim-scatter ceiling (~0.02) lacks. Implemented `reward_pump` (committed): 2 shaping terms on the drone's OWN controllable motion — (1) windup `drone_toward·hook_behind` far-field (decays w/ curriculum), (2) lift `drone_up·exp(-aim_k·hook_bar)` (taut wire whips hook up at bar). Stop/reverse whip EMERGES as the bridge (not hard-coded → no reward-hacking). Plus `spawn_away_v0` = inverse-spawn runway (away-from-wall seed → drone flies IN, loading the hook behind).
  - **Validated:** smoke 128/128 0 errors; standalone numeric test confirms windup/lift fire ONLY in correct load/lift geometry (0.675 / 2.07) and stay 0 for wrong maneuver + at spawn.
  - **Y-WAVE launched (ALC):** Y1 = W1 recipe + wide radius + pump + spawn_away_v0=1.5 (full runway) · Y2 = same, spawn_away_v0=0 (reward-only, isolates runway). Both swing_assist OFF (pump is the sole swing driver), mls-0.5, seed7. Decisive test: does the pump maneuver lift catch above the ~0.02 ceiling? Eval at ~55-90k.


- **2026-07-15 ~14:20 — Hold curriculum HELPS hold (modestly); catch wall is the real blocker.** Consolidated eval @130k (attached_spawn fully decayed): X2(f0.7) hold **0.643** > X1/X3(f0.5) 0.467 — consistent trend **more attached-spawn ⇒ better hold** (holds at 90k too). X2's 0.643 edges the prior best (0.615). Genuine physical swing throughout (62-74% up-sweep).
  - **BUT catch stays ~0.01 (ceiling-bound)** → GOALS_MET stays False regardless of hold. After testing every catch lever (exploration bracket, aimed reward, wide-radius anneal), catch peaks ~0.02-0.03 and never approaches 0.30. **The catch wall — aim scatter placing the swing peak within 0.13m of the bar — is the binding constraint, and appears near a physical ceiling under strict genuine-swing + strict-detach.**
  - **Following the trend:** launched X4(frac0=0.85) + X5(frac0=1.0, pure hold-start = user's "start fully attached" suggestion) to find the hold ceiling of the curriculum. Expectation: hold climbs further, catch may drop (less catch practice) — completes the frac0 sweep for the deliverable.
  - **Deliverable forming:** creep ELIMINATED / genuine swing SOLVED (user's main concern). Hold movable to ~0.64+ via the reverse-hold curriculum. Catch stuck ~0.02 « 0.30 — meeting BOTH targets needs a task-setup change the user disallowed (bigger hook capture radius = hardware choice, or softer detach). Honest tradeoff for user decision.


- **2026-07-15 ~13:10 — Hold-curriculum first eval INCONCLUSIVE @90k + ⚠️ eval-metric BUG found & fixed.** First X-wave eval returned `hold_given_caught` **>1** (9.0/27.5/24.6) — impossible: `eval_goals`/`diag_catch` disabled only the 3 original curricula by name, so `attached_spawn_curriculum` stayed ON in eval → half the eval envs spawned attached, polluting the ratio. **Fixed** (committed): added `attached_spawn_curriculum` to the disable list in both scripts. (Training was always correct; only the eval readout was wrong.)
  - **Honest re-eval @~90k:** X1(f0.5,wide) catch 0.016 hold 0.375 · X2(f0.7,wide) catch 0.011 hold 0.563 · X3(f0.5,narrow) catch 0.007 hold **0.625** (11 catches). Hold 0.38-0.63 overlaps the prior 0.55-0.62 band → **no clear win yet**, but weak positive trend (more attached-spawn ⇒ higher hold: X2>X1). All still genuinely physical (68-83% up-sweep). Catch remains ceiling-bound so hold-given-caught is noisy (11-36 catches).
  - **Caveat/next:** attached_spawn fraction only fully decays at 120k → the clean read is ~130-150k once the policy consolidates the learned hold into pure catch-then-hold. Also a plausible gap: the curriculum teaches hold from a CALM steady pose, but a real catch leaves the hook in a DYNAMIC just-swung state — the learned hold may not fully transfer. Re-eval at ~130k before judging.


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
