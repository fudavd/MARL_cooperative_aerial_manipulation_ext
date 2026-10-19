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
| A | N-drone hooking + joint hold, construction plant | -6d | izar 3223831 at it 31k on 5166973 + user params. Free starts, it 30k tracker, deterministic, final 5 cm / 25 deg: per-drone catch ~1/3 of 20 s episodes (65 catches / 64 eps x 3 drones), 4 drops; steps with 0/1/2/3 attached 76.6 / 21.0 / 2.35 / 0 %; joint holds 0. Partly hooked starts: 16 % of 75 eps reach a 1 s joint hold. vbox risk while partly attached median 1.11 (old physics). | three *simultaneous* catches needed; both catching (1/3 per drone) and keeping are short. Softer pin may change catch dynamics. | decomposition diagnostics (d4b27a1: time to k-th catch, what ends a 2-attached spell, protocol joint hold with pull <= 0.4 x weight) in the tracker at it 32k / 34k. ETA for a usable joint-hold checkpoint: it 40-50k, 1.5-3 days, uncertain. |
| B | Transport, 51x3, construction plant | **nobody** | no policy, no trainer. sequence.py has the plant, target pose, success bookkeeping and attached starts; construction cfg has block_mass_range (0.9, 1.9), com_offset_range (default 0), box_size_dr. | this is the missing skill for the merge. | decision D1 below. Gap estimate (reward port from hover_rand, start distribution, MAPPO recipe transfer, wall time) requested from -0f; -0f will write it only on the user's go-ahead (D1). |
| C | Transport on the development plant (evo hover_rand, 45-dim actor, Hebbian ABCD via ES) | -0f | izar 3223562 / 3223586 (v17, 45 h left). Gens 0-5: paper return 231 -> 300, survival 0.27 -> 0.95, survivor orientation error 78-140 deg (23 deg needed). Unperturbed warm start: 310 / 0.84. Full-obs MAPPO teacher: 854. Fixed-attachment success still 36 % vs MAPPO 95 %. Plastic populations have not exceeded the zero genome. | drone-only obs cannot identify basket heading (known wall). Not on the contract; no number from it enters the merge. | let v17 finish; deterministic eval of theta vs warm start vs plasticity-frozen, 3 eval seeds, protocol success. First honest plastic-on / plastic-off number (frozen = ABCD zero = the warm start head exactly, since init weights are not evolved; -0f will label it so). No new ES runs on the 45-dim obs (agreed). |
| D | Adaptive layer: static controls | nobody | none of protocol rows 1-3 (static shared head, static GRU, oracle) exist on the construction plant. | without them no adaptivity number is interpretable (H1, H2). | after B exists: train rows 1-3 on the transport task with mass / CoM / size randomised; oracle = hidden variables appended to the frame. |
| E | Adaptive layer: plastic head, within-task | nobody on the contract plant | offline switching heads (IPA / activity) work on imitation bodies; eta_bound now enforced. Probes: mass is the hidden variable (linear R^2 0.2, residual only); attached flag and attachment x/y are frame-visible. | targets must be mass / CoM / inertia, not the switch. | after D: rows 5-6 (frozen vs on) on the train / test splits of the protocol. |
| F | Chain and switching | nobody | sequence env exists (teleport cycles, bookkeeping); no policy has run it. | needs A + B. | method 7 (wrong-start) vs static in the sequence env; time to switch. |
| G | Task distribution | -6d / -c5 | mass randomised (0.9-1.9 kg); box_size_dr ON with the user's narrowed range (half_xy 0.32-0.36, half_z 0.20-0.25); CoM offset off (com_offset_range 0). | changing them mid-run changes the curriculum. | at the next restart of A, or in B from the start: add the CoM offset and widen the size range to the protocol section 2 splits. |
| H | Hygiene | -c5 / user | skrl fork a65eae6 **not pushed** (classifier blocks me). Hook-grasp env has the reset fix (a3666bc) but no reset test. Demos predate the fixes. | - | user: `cd thirdparty/skrl && git push`. -c5: port the reset check to flycrane_hook; re-collect demos after A's next checkpoint. |

## 3. Critical path (ordered; each step gates the next)

1. **Joint hold from free starts** (A). Gate: protocol joint hold (all attached, pull <= 0.4 x weight, 100 steps)
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
