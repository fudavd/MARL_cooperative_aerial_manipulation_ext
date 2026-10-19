# Project status and critical path

Maintained by the project-manager session (-c5). Updated 2026-10-19, 19:45, after status replies from the
hooking session (-6d, flycrane-meta) and the transport / ES session (-0f, flycrane-evo). Companion documents:
`code_quality_control.md` (audit, all defects fixed), `evaluation_protocol.md` (what counts as a result),
`insight_tasks_2026-10-19.md` (identifiability probes, regression checks).

## 1. State in one paragraph

The plant and the observation contract are settled and verified: construction / sequence env of flycrane-meta,
3 flyhook drones, 51-dim per-drone frame x 3 history, 5 actions; regression checks pass on izar at 5166973
(reset 3.9e-5 m, pin 1.05 N, hold risk <= 0.27). Of the three things the goal needs, **hooking** is training on
that plant (izar 3223831, restarted today at iteration 31k with the user's vbox_mu 0.6 / pin_kp 150 / 40 s
episodes) and has not yet produced a 3-drone joint hold from free starts; **transport on that plant does not
exist** (the meta transport family is the evo hover_rand plant re-observed, and the evo ES track runs a 45-dim
drone-only actor that cannot see box heading and whose plastic head has not beaten its warm start); the
**adaptive layer** has offline switching results on imitation bodies (NOTES 2026-10-07/08) but none of the
protocol's static controls, and no plastic-vs-frozen comparison on any plant. The critical path is therefore:
hooking joint hold -> a 51x3 transport teacher on the construction plant -> the static controls -> the plastic
head and the switching test in the sequence env.

## 2. Workstreams

| # | stream | owner | state (evidence) | blocker / risk | next |
|---|---|---|---|---|---|
| A | N-drone hooking + joint hold, construction plant | -6d | izar 3223831 at it 31k on 5166973 + user params. Free starts, it 30k tracker, deterministic, final 5 cm / 25 deg: per-drone catch ~1/3 of 20 s episodes (65 catches / 64 eps x 3 drones), 4 drops; steps with 0/1/2/3 attached 76.6 / 21.0 / 2.35 / 0 %; joint holds 0. Partly hooked starts: 16 % of 75 eps reach a 1 s joint hold. vbox risk while partly attached median 1.11 (old physics). | **Regression after the restart (it 31k -> 32k)**: adaptive-curriculum catch rate single 0.44-0.46 -> 0.22, three-drone ~0.45 -> 0.28; single tracker at 32k 43 catches / 0 drops in 155 eps vs 71 at 30k under the old physics. Candidates: (a) softer pin changes the catch feel, testable by eval (izar 3223928: pin 150 vs 400 x checkpoints it 30k / 32k, single plant, free starts, deterministic); (b) mu 0.8 -> 0.6 raises the slide ratio by 4/3 for the same pull (slide = |F_h| / (mu (m g - F_up))) and with the vbox ramp (~weight 8) taxes the single drone's always-on hold, visible only as a training trend (actor does not observe vbox; reward-only). Pin A/B (3223928) done: pin is NOT the cause (catches 65/165 vs 67/171 at it 30k, 43/155 vs 45/156 at it 32k for pin 150 vs 400; soft pin 9 vs 2 drops at 30k). The policy changed: median height above the box top 0.32 -> 0.41 m, time within 5 cm of the bar 81 % -> 48 %, same outward distance. Reading: under mu 0.6 + the vbox ramp, post-catch holding costs more, so the shared policy avoids the bar; the 3-drone 40 s episodes may add to it. Verdict at 34k vs 32k, with the curriculum easing stated; per-term episode returns 30k vs 32k requested. The 32k 3-drone test was invalid (30 s tracker on 40 s episodes), rerun queued with 8000 steps. Decomposition at it 30k under the new physics (40 s, free starts, 67 eps): median time to 1st / 2nd / 3rd catch 7.6 / 16.1 / 22.8 s, reached by 66 / 18 / 1.5 %; joint hold 1.5 %; drops rare (13 per 102 catches, all slack detaches); vbox risk at 2 attached median 0.78, 46 % of steps > 1. Partly hooked starts: joint hold 23 %. Bottleneck = slow, sequential catching (3-drone curriculum still at 0.75), not drops. Risk split (it 30k): the risk > 1 at 2 attached is sliding (13904 steps slide / 45 tip / 0 lift): both hooked drones pull outward along the v84 hold pose (outward q50 5.0 N per drone, lateral 1.6 N, vertical -0.1 N; horizontal cancellation 0.78, adjacent faces), two 5 N pulls on adjacent faces sum to ~7 N > 0.6 x 9.8 = 5.9 N. Protocol v1.1 joint hold 0 % free / 1.3 % partly hooked (plain all-attached 1 s: 23 %). Fix direction: hold over the bar with a near-vertical wire. The scheduled vbox penalty (full weight at ~it 37.5k) already targets this; PM position: no mid-run reward change, re-check at it 38k, prepare a verticality term for the next restart (decision D5). | decomposition diagnostics (d4b27a1: time to k-th catch, what ends a 2-attached spell, protocol joint hold with pull <= 0.4 x weight) in the tracker at it 32k / 34k. ETA for a usable joint-hold checkpoint: it 40-50k, 1.5-3 days, uncertain. |
| B | Transport, 51x3, construction plant | **nobody** | no policy, no trainer. sequence.py has the plant, target pose, success bookkeeping and attached starts; construction cfg has block_mass_range (0.9, 1.9), com_offset_range (default 0), box_size_dr. | this is the missing skill for the merge. | decision D1 below. Gap estimate (reward port from hover_rand, start distribution, MAPPO recipe transfer, wall time) requested from -0f; -0f will write it only on the user's go-ahead (D1). |
| C | Transport on the development plant (evo hover_rand, 45-dim actor, Hebbian ABCD via ES) | -0f | izar 3223562 / 3223586 (v17, 45 h left). Gens 0-5: paper return 231 -> 300, survival 0.27 -> 0.95, survivor orientation error 78-140 deg (23 deg needed). Unperturbed warm start: 310 / 0.84. Full-obs MAPPO teacher: 854. Fixed-attachment success still 36 % vs MAPPO 95 %. Plastic populations have not exceeded the zero genome. | drone-only obs cannot identify basket heading (known wall). Not on the contract; no number from it enters the merge. **First plastic-on vs frozen number (gen-7 theta, deterministic, curricula off, 1024 fresh scenarios, 2 eval seeds)**: plastic headx4 paper 297 / survival 83 % / protocol success 1.7 % / pos err 0.75 m / ori err 1.8 rad; same theta frozen (= warm-start head) 339 / 86 % / 6.9 % / 0.17 m / 1.6 rad; MAPPO warm-start actor 339 / 86 % / 7.2 % / 0.17 m / 1.6 rad. Plasticity is strictly worse on every protocol metric; the static control holds. Even the frozen head reaches only 7 % protocol success (orientation ~90 deg off). Final eval on the last checkpoints: izar 3223909 (~2026-10-21). Train/eval gap explained: training paper_mean / survival_all are estimated on 4 CRN scenarios per generation (same 4 for every genome), per-scenario survival spans 0.67-0.99, so the curve moves +-0.1 with a near-constant theta; eval.py averages ~1030 scenarios. Rule: compare generations only within a run, quote eval.py numbers for absolute claims. The 4-scenario draw moves the level of a generation, not the ranking: the ES ranks antithetic pair differences on the same scenarios (paired eval), so the draw cancels. The selection noise that does compete with the plasticity effect is chaotic episode-to-episode variation of a fixed genome on a fixed scenario (pair_diff_std 55-80 vs a sigma-0.01 signal of a few points, SNR ~0.02 per genome before pairing, ~5x less noise variance with pairing); that plus the fitness composition (heading weight, relaxed success threshold) is the plausible cause of the v14-v17 plateau. Both go into the closing note. | let v17 finish; deterministic eval of theta vs warm start vs plasticity-frozen, 3 eval seeds, protocol success. First honest plastic-on / plastic-off number (frozen = ABCD zero = the warm start head exactly, since init weights are not evolved; -0f will label it so). No new ES runs on the 45-dim obs (agreed). |
| D | Adaptive layer: static controls | nobody | none of protocol rows 1-3 (static shared head, static GRU, oracle) exist on the construction plant. | without them no adaptivity number is interpretable (H1, H2). | after B exists: train rows 1-3 on the transport task with mass / CoM / size randomised; oracle = hidden variables appended to the frame. |
| E | Adaptive layer: plastic head, within-task | nobody on the contract plant | offline switching heads (IPA / activity) work on imitation bodies; eta_bound now enforced. Probes: mass is the hidden variable (linear R^2 0.2, residual only); attached flag and attachment x/y are frame-visible. | targets must be mass / CoM / inertia, not the switch. | after D: rows 5-6 (frozen vs on) on the train / test splits of the protocol. |
| F | Chain and switching | nobody | sequence env exists (teleport cycles, bookkeeping); no policy has run it. | needs A + B. | method 7 (wrong-start) vs static in the sequence env; time to switch. |
| G | Task distribution | -6d / -c5 | mass randomised (0.9-1.9 kg); box_size_dr ON with the user's narrowed range (half_xy 0.32-0.36, half_z 0.20-0.25); CoM offset off (com_offset_range 0). | changing them mid-run changes the curriculum. | at the next restart of A, or in B from the start: add the CoM offset and widen the size range to the protocol section 2 splits. |
| H | Hygiene | -c5 / user | skrl fork a65eae6 **not pushed** (classifier blocks me). Hook-grasp env has the reset fix (a3666bc) but no reset test. Demos predate the fixes. | - | user: `cd thirdparty/skrl && git push`. -c5: port the reset check to flycrane_hook; re-collect demos after A's next checkpoint. |

## 3. Critical path (ordered; each step gates the next)

1. **Joint hold from free starts** (A). Gate: protocol joint hold v1.1 (all attached, wires taut, vbox risk < 1, 100 steps)
   in >= 10 % of free-start episodes at final curriculum settings, deterministic, 3 evaluation seeds.
2. **Transport teacher on the construction plant, 51x3** (B). Gate: protocol transport success (0.10 m /
   0.35 rad held 50 steps, all attached) >= 80 % from attached starts on the train split of mass. This is the
   step with no owner and no code today; it is the longest pole after A.
3. **Static controls** (D) on the same task and splits. Gate: H1 (oracle beats static on the test split). If
   H1 fails, widen the task distribution before any plastic work.
4. **Plastic head within-task** (E). Gate: H2 and H3 on table A.
5. **Chain + switching** (F) in the sequence env. Gate: H4 on table B; table C with the seed spread.

Steps 3-5 reuse the imitation + ANIL + switching-head code that already exists in flycrane-meta; the new work
is the data (step 2) and the controls (step 3), not the algorithms.

## 4. Decisions needed from the user

- **D1. Who builds the 51x3 transport teacher on the construction plant, and when.** Options: (a) the evo session
  after v17 ends (45 h), using its MAPPO paper recipe; (b) start now on ALC or the free izar slot, in parallel
  with v17; (c) the hooking session after A reaches its gate. My recommendation: (b), owner -0f, since the
  recipe and the person-hours are there and v17 needs only a watcher. The ES / 45-dim track then ends with its
  plastic-on / plastic-off eval and a note.
- **D2. Seeds.** Protocol asks 3 training seeds; a hooking run is ~48 h on one GPU. Proposal: 1 training seed per
  skill until gates 1-2 pass, then 3 seeds for tables A-C only.
- **D3. The between-task claim.** The probes show the switch itself is frame-visible. Proposal: the claim becomes
  "the plastic head keeps the correct load compensation across the switch" (the hidden load each drone carries
  changes at the catch and at lift-off), tested as method 7 vs static. Needs your confirmation before E / F
  are designed.
- **D5. Hold-pose reward for attached drones while others catch.** The v84 hold pose leans out + up, so two hooked drones slide the box. Options: (a) penalise the horizontal component of an attached drone's pull / reward wire verticality while not all are attached; (b) a hold bonus while others catch, with the same verticality condition; (c) wait for the vbox penalty to reach full weight (~it 37.5k, ~10 h) and decide at it 38k. Recommendation: (c) now, (a) prepared in code for the next restart. Done: flycrane-meta f21ac6c adds `w_hold_horizontal` (default 0 = off) and `hold_horizontal_free` (1.0 N): per attached drone, while not all are attached, -w x dt x max(0, |horizontal pin reaction| - free). Suggested starting weight 2.0 (a 5 N lean costs ~8 per second per drone, comparable to the vbox penalty at full weight near risk 1.6). Not synced to izar; switch on only at a restart.
- **D5 / D6 superseded (2026-10-19 late).** Per-term eval (3223932, single plant, it 30k vs 32k, mean return per episode): vbox -0.26 vs -0.22 (negligible), catch 7.1 -> 3.9, hold_success 27.4 -> 15.7, hook_proximity 13.9 -> 15.2, hook_swing 4.8 -> 7.1, hook_above_box -50.2 vs -48.8 (dominates everything). So the stricter vbox is not why the 32k policy avoids the bar; catch and hold both shrank while proximity / swing shaping grew (shaping farmed), and the shared-policy incentive from the 40 s 3-drone episodes is the remaining suspect. User decision (with -6d): simplify the reward to five terms, `reward_mode='simple'` (flycrane-meta b482fea, local): reach = potential shaping on hook-to-bar distance + missing chord angle; catch +C / drop -C; hold = H x taut x g(risk), g = 1 up to 0.8 falling to -1 at 1.2; success = protocol v1.1 for 1 and 3 drones; safety = fail, close, over-box depth, action rate. Legacy reward kept verbatim as `_rewards_legacy`; trainer `--fresh_critic`. Two runs in parallel: fine-tune from the it 31k actor with a fresh critic, and from scratch. Gate before sync: legacy output bit-identical before / after the refactor (in progress), regression_checks pass.
- **D6 (historical). Catch avoidance under the stricter virtual-box check.** After the it 31k restart (mu 0.8 -> 0.6, ramp ~8) the single-drone policy flies 9 cm higher and halves its time near the bar in 1000 iterations; the pin is ruled out by A/B. The vbox penalty can be escaped by not catching, which is cheaper than learning a vertical hold. Options: (a) wait for 34k (~3 h) and see whether the curriculum easing restores catching; (b) restart from it 31k with D5 option (b), a hold bonus for attached drones conditioned on a near-vertical wire, so holding correctly pays more than avoiding; (c) restart from 31k with mu 0.8 on the single plant only. Recommendation: (a) for the trend, then (b) if catching has not recovered, since (b) rewards the behaviour the chain needs and (c) only hides the slide problem. Done: flycrane-meta 9aef9f1 adds `w_hold_vertical` (default 0 = off) and `hold_vertical_cos0` (0.9, ~26 deg): per attached drone with a taut wire, while not all are attached, +w x dt x clamp((cos(wire to vertical) - cos0) / (1 - cos0), 0, 1). izar synced to 9aef9f1 with both optional terms at 0 (behaviour-neutral). Per-term returns eval running (3223932: it 30k vs 32k, training reward cfg, vbox weight fixed at 8.2; 34k to follow).
- **D4. Add the CoM offset and widen the box-size range** to the protocol splits from the next restart of A (G); box size DR is already on with a narrowed range.

## 5. Risks

- A stalls below 10 % joint holds: per-drone catch is 1/3 per 20 s episode from free starts; three simultaneous
  holds compound. Mitigations in order: longer episodes (done: 40 s), partial-hooked-start curriculum kept at
  evaluation off, a hold bonus while others catch, and physics (softer pin, done).
- B is unowned; every week it stays unowned the merge slips a week.
- The ES track consumes a GPU for 45 h for a development-plant number. Acceptable once; not again.
- Compute: one laptop, ALC, two izar slots. Steps 2 and 3 can run in parallel on ALC + izar.

## 6. Oversight cadence

- -6d reports at it 32k and 34k (decomposition, protocol joint hold, catch rate under the softer pin).
- -0f reports when v17's deterministic eval exists and sends the gap estimate for B.
- -c5 keeps this file, the protocol and the memory current; pushes peer commits after review (d4b27a1 pushed).
