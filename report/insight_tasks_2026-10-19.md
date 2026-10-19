# Insight tasks while the cluster runs: identifiability probes, evaluation protocol, simulation regression checks

Date: 2026-10-19 (session marl-cooperative-aerial-manipulation-ext-2e, formerly skrl-01). Companion documents:
`report/code_quality_control.md` (audit), `report/evaluation_protocol.md` (pre-registered methods and tables).
Everything below ran on the laptop (RTX 4070, Isaac Lab 2.0.2) without touching the izar runs.

## 1. Summary

* **Identifiability (section 2).** The hook-blind drone's *attached* flag is readable from a single frame by a
  small non-linear read-out (95 % balanced accuracy), mostly from the drone's own pose relative to the box, and
  from the INDI residual alone once 20 control steps are visible (92 %). It is therefore **not** a hidden switch
  that needs plasticity. The box **mass** is the opposite: a linear read-out recovers R^2 ~ 0.2 from any window
  up to 1 s, the signal lives entirely in the INDI residual, and a non-linear read-out does not transfer across
  held-out attachment geometries. Mass (and by extension CoM offset and inertia) is the variable a plastic layer
  can legitimately target. The attachment point's x / y is visible from one frame (R^2 0.95), its height is not.
* **Evaluation protocol (section 3).** Written as `report/evaluation_protocol.md`: one plant, one observation
  contract, two splits per hidden variable, one success definition per behaviour and for the chain, seven methods
  including the three static controls every plastic claim must beat, four pre-registered hypotheses.
* **Simulation regression checks (section 4).** `projects/flycrane-meta/tests/regression_checks.py`. Reset
  independence passes (trajectory after a violent history matches a fresh process to 4e-5 m). The pin-anchor check
  confirms the audit's "phantom pull": with the old anchor on the collider face the steady pin force is 10 N
  (hook weight 2.9 N, peaks 104 N, box dragged 2.4 m); with the anchor one hook radius outside the face it is 1 N,
  no penetration, all hooks stay attached. The virtual-box check exposed a second spawn inconsistency (attached
  starts placed the hook on the face, not at the anchor), fixed in the same commit.
* **Two corrections to earlier claims.** (a) My audit suggested the attached state as the better switching target
  because the actor is hook-blind; the probe shows it is frame-identifiable, so that argument is withdrawn. The
  label-free switching claim has to rest on something a static read-out cannot do, which section 5 spells out.
  (b) The report's finding P3 (phantom pin force) is now measured, not inferred.

## 2. Identifiability probes

Data: the local demonstration tensors (`projects/flycrane-meta/logs/demos/switch_hook.pt`, v84r2 teacher,
64 task groups x 1400 stored steps x 8 lanes; `variant_transport.pt`, transport teacher `run2_final`, 64 groups x
1400 x 12 lanes). Stored every 2nd control step; each stored frame carries the 3-step actor stack (153). Probes are
fitted on 44 task groups and scored on 20 held-out groups (for transport: unseen attachment geometries), inputs
z-scored, 150k samples per split, linear = logistic / ridge-like regression, MLP = one hidden layer of 128.
"20 steps" = the current frames of the last 10 stored frames inside one episode (510 inputs).

Caveat: the transport demos were collected before the termination and controller-reset fixes; the hook demos
before the controller-reset fix. The INDI residual statistics after the fixes may differ slightly. The
conclusions below are about what is in the observation, which the fixes do not change qualitatively.

### 2.1 Hook: is "my hook is attached" hidden? (balanced accuracy; attached share 9.2 %)

| inputs | linear | MLP-128 |
|---|---|---|
| 1 frame (51) | 0.73 | **0.95** |
| 3 frames (153) | 0.74 | 0.94 |
| 3 frames without the INDI residual | 0.73 | 0.94 |
| 3 frames, INDI residual only (18) | 0.52 | 0.75 |
| 20 control steps (510) | 0.85 | 0.96 |
| 20 control steps, residual only (60) | 0.53 | 0.93 |

Reading: a static non-linear read-out knows the attachment state from one frame. The information is in the
geometry (drone pose relative to the box: an attached drone sits at the hold pose above its bar), not in the
residual; the residual alone needs a 0.2 s window. Part of this is the teacher's own behaviour making the state
visible, but a policy trained on this plant will have the same behaviour. Consequence: hooking vs holding is a
frame-identifiable switch, like hook vs transport (NOTES, 2026-10-07). Neither needs a plastic head to be told
apart.

### 2.2 Transport: is the box mass hidden? (R^2 on held-out attachment groups; mass U(0.9, 1.9) kg, std 0.29)

| inputs | linear | MLP-128 |
|---|---|---|
| 1 frame (51) | 0.21 | -0.48 |
| 3 frames (153) | 0.20 | -0.42 |
| 3 frames without the INDI residual | **0.01** | -0.72 |
| 3 frames, INDI residual only | 0.14 | 0.13 |
| 20 control steps (510) | 0.23 | -0.43 |
| 20 control steps, residual only | 0.14 | 0.10 |
| 50 / 100 control steps (0.5 / 1 s), residual only, linear | 0.14 / 0.15 | - |
| 50 / 100 control steps, residual + own state, linear | 0.12 / 0.10 | - |

Reading: mass is weakly decodable (a fifth of its variance), only through the INDI residual, and a longer window
up to 1 s does not help a linear read-out. The non-linear read-out scores *negative* R^2 on held-out attachment
geometries: it learns an attachment-specific mapping that does not transfer. So mass inference is (i) hard,
(ii) entangled with the attachment geometry, and (iii) not a matter of a short memory. This is the profile of a
variable where a slow, online weight change (a plastic head integrating the residual over the whole episode)
has a job that a 3-frame or 20-step static policy does not do.

### 2.3 Transport: the drone's own attachment point (R^2; std x 0.22, y 0.20, z 0.034 m)

| inputs | linear x / y / z | MLP x / y / z |
|---|---|---|
| 1 frame | 0.11 / 0.07 / -0.03 | **0.95 / 0.94 / 0.12** |
| 3 frames | 0.10 / 0.06 / -0.09 | 0.95 / 0.95 / 0.07 |
| residual only (3 frames) | 0.04 / 0.01 / -0.06 | 0.06 / 0.03 / -0.07 |
| 20 control steps | 0.09 / 0.05 / -0.23 | 0.95 / 0.96 / 0.05 |

Reading: where the cable meets the box in the horizontal plane is visible from the drone's position relative
to the box in one frame (the drone hovers above its anchor). The anchor height is not (its variance is small,
0.02-0.14 m, and it only changes the cable angle slightly). Attachment randomisation is therefore mostly a
*visible* variation for a static policy; what remains hidden is the height, and its effect is small.

### 2.4 What this means for the adaptivity claim

| hidden variable | static read-out from the 3-frame observation | adaptivity target? |
|---|---|---|
| attached / phase | yes (95 %, one frame) | no: a switch the body sees |
| hook vs transport family | yes (NOTES 2026-10-07) | no |
| attachment point x, y | yes (R^2 0.95) | no |
| attachment height | no, but small effect | weak |
| box mass | no (R^2 0.2, residual only, no gain from 1 s windows) | **yes** |
| CoM offset, inertia | not in the demos (never randomised) | expected like mass: add to the task distribution |

The within-task adaptivity claim should be built on mass, CoM offset and (through them) inertia, with the
protocol's static recurrent control as the thing to beat. The between-task claim cannot be "the head notices
the switch" (a static body notices it); it has to be "the head keeps the right behaviour under a hidden
parameter that changes at the switch", for example the load the drone carries once attached, or the mass
split between drones once the box lifts. Section 5 proposes the experiment.

## 3. Evaluation protocol

See `report/evaluation_protocol.md`. The parts the runs should adopt immediately: curricula off and
deterministic actions at evaluation; last checkpoint, 3 seeds; the four success definitions; the task splits for
mass, CoM offset and box size; and the static shared / static recurrent / oracle controls as rows 1-3 of every
table. The identifiability probe of section 2 is the protocol's pre-check (section 7 there).

## 4. Simulation regression checks (`tests/regression_checks.py`)

One Isaac process per check, construction env, 8 envs, laptop. Drones hold position with a PD on the commanded
acceleration (a zero action is a zero *acceleration* command, which lets a loaded drone drift: the first version of
the pin check dragged the box 2.2 m for that reason).

### 4.1 Reset independence (`--check reset`): PASS

Same seed before each reset, 60 steps of zero actions compared between a fresh process, after 150 steps of
random full-scale actions (crashes, flips, hook flailing) and after a benign history.

| | step 1 | step 10 | step 60 |
|---|---|---|---|
| max position difference, violent history vs fresh [m] | 4e-5 | 4e-5 | 4e-5 |
| benign history vs fresh [m] | 4e-5 | 4e-5 | 4e-5 |

The residual 4e-5 m is the reseeded start-state draw itself (float32). The construction plant has no
episode-history leakage. The same check should be run on the hook-grasp env after its reset port (not done:
that env lives in another package; the check is written against the construction env).

### 4.2 Pin anchor vs collider (`--check pin`), attached start, real 1.4 kg block, 200 steps

| anchor offset | steady pin force mean / max [N] | hook centre inside the collider, max [m] | box displacement [m] | hooks still attached |
|---|---|---|---|---|
| 0 (before the fix) | 10.2 / 103.9 | 0.024 | 2.40 | 100 % |
| 0.02 = hook radius (fix) | 1.0 / 4.0 | 0.000 | 0.19 | 100 % |

Hook weight 2.9 N. The old anchor produced a 10 N standing pull per hook, three hooks on three faces of a 1.4 kg
box: the box was dragged across the floor by the hold alone. This also contaminated the virtual-box risk and the
"taut hold needs >= 6.5 N" observation in the git log. The remaining 0.19 m came from the attached *spawn*
placing the hook on the face instead of at the anchor (2 cm stretch at t = 0): fixed, see 4.3.

### 4.3 Virtual-box risk vs real motion (`--check vbox`)

Attached start on a real 1.0 kg block, 300 hold steps, then drone 0 ramps an outward + upward acceleration
command (0 -> 8 m/s^2 in 81 levels of 25 steps) while the other two hold.

| quantity | value |
|---|---|
| virtual-box risk while merely holding (max over envs) | 0.89 (< 1: PASS) |
| box speed after 300 hold steps (max) | 0.22 m/s (creeping: WARN) |
| ramp level at which risk > 1 | 0.1 - 0.7 m/s^2 (one env 2.5) |
| ramp level at which the box tilts > 2 deg, slides > 1 cm or lifts > 1 cm | 0.0 (already moving) |
| hooks still attached at the end of the ramp | 75 % |

Reading: the quasi-static check is consistent in level (risk 0.89 while the box is at the sliding threshold, and
it crosses 1 as the pull grows), but it is not conservative: the real box creeps before the check says so, because
(i) three stiff pins (400 N/m) turn centimetre-level drone position errors into forces of the order of the sliding
threshold of a 1 kg box (0.8 x 9.8 N = 7.8 N), and (ii) once sliding, dynamic friction (0.6) is below the static
0.8 the check uses. Two concrete consequences for the hooking line: the "gentle joint hold" on a real box
demands centimetre hovering precision or a softer pin, and `vbox_mu` should be the dynamic 0.6 for the check to
err on the safe side. The two motion criteria are kept in the test as warnings (physical behaviour to read), the
risk-while-holding criterion as a pass / fail.

Pin check repeated after the spawn fix: steady pin force 1.05 N (max 6 N), zero penetration, all hooks attached,
box displacement 0.22 m during the 2 s hold (the creep above).

## 5. Proposed next experiments (no cluster needed to prepare)

1. **Mass-adaptation table on the transport skill.** Add CoM offset and box size to the construction task
   distribution (knobs exist), train the static shared head and the static GRU on the train split, evaluate on the
   test split with the protocol. If the oracle (mass as input) does not beat the GRU on the test split, mass does
   not matter for the task as rewarded, and the adaptivity claim needs a harder task (heavier range, tighter
   tolerance) before any plastic run.
2. **Plasticity with a job.** Train the plastic head to track the residual-integrated mass estimate: the probe
   says a linear read-out of the residual gives R^2 0.14, which an online rule can accumulate over an episode.
   The frozen-vs-active comparison (protocol rows 5 vs 6) then measures exactly that accumulation.
3. **Switching that a static body cannot do.** In the sequence env, the hidden parameter that changes at the
   switch is the load each drone carries (0.3 kg hook free, then a third of the box). The plastic head's claim
   becomes: after the switch, the head converges to the right load compensation without a label faster than a
   static recurrent policy re-infers it. Measure time-to-steady-tracking after the last catch.
4. **Identifiability on post-fix data.** Re-collect 1000-step demos with the current plant (both fixes) and rerun
   `probe_hidden.py`; the hook demos should be re-collected from session -6d's hooking policy once it holds.
5. **Hook-grasp reset check.** Port 4.1 to `flycrane_hook` (same structure) so the single-drone plant is covered.

## 6. Files

* `report/evaluation_protocol.md` (new).
* `projects/flycrane-meta/tests/regression_checks.py` (new, commit 3476429 + follow-up).
* `projects/flycrane-meta/flycrane_meta/envs/construction.py`: constructor seeding order fix (3476429), attached
  spawn at the pin anchor (follow-up commit).
* Probe scripts and raw outputs: scratchpad of this session
  (`/tmp/claude-1000/-home-fuda-PycharmProjects-MARL-cooperative-aerial-manipulation-ext/329e8f14-e8b2-4967-b92e-0ba44c56f7fb/scratchpad/probe_hidden.py`,
  `probe_mass_long.py`, `probe_results.json`, `regression.jsonl`); copied to `report/data/`.
