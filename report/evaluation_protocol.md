# Evaluation protocol for the adaptive hook-and-transport chain

Status: v1.2, 2026-10-20 (v1.1 draft 2026-10-19, session skrl-01; v1.2 approved by the user 2026-10-20 evening). Scope: every claim the project will make about the N-drone
hooking skill, the cooperative transport skill, the within-task adaptivity of the Hebbian layer and the
hook -> transport switch. Each later run should fill one row of the tables in section 5; a number that is not a
row of these tables is a diagnostic, not a result.

## 1. Plant and observation contract

* Plant: the flycrane-meta construction / sequence env (`Isaac-flycrane-meta-sequence-v0`), 3 flyhook drones
  (0.30 kg hook, 4.8 g wire), one box with bars, controllers Geometric -> INDI -> motor at 300 Hz, control at 100 Hz.
* Observation: the unified per-drone frame, 51 x 3 history = 153 inputs (box pose 12 | assigned wall centre 3 |
  own state 18 | goal error 12 | INDI residual 6). No task label, no phase, no attachment flag, no hidden parameter.
* Action: 5 (linear acceleration 3, body rates 2).
* Both skills and the chain are evaluated on this plant. The evo `hover_rand` plant and the single-drone
  hook-grasp plant are development plants; their numbers are not thesis results.

## 2. Task distribution

Two splits per hidden variable. "Train" is what any learner may see; "test" is held out for every method.

| hidden variable | train | test (held out) | varies per | observable to the actor? |
|---|---|---|---|---|
| box mass [kg] | U(0.9, 1.9) | U(0.7, 0.9) and U(1.9, 2.1) | episode | no (only through the INDI residual) |
| box CoM offset [m, per axis] | U(-0.05, 0.05) | U(-0.08, -0.05) and U(0.05, 0.08) | episode | no |
| box footprint half-width [m] | U(0.22, 0.33) | U(0.20, 0.22) and U(0.33, 0.36) | env (scene build) | yes (box pose given, size not; wall centre given) |
| box height [m] | U(0.6, 1.0) | U(0.5, 0.6) and U(1.0, 1.1) | env | partly (bar height via wall centre) |
| bar band height on the face | random band (`bar_band_random`) | same | env | yes (wall centre) |
| attachment point along the bar | where the hook lands (central 40 %) | same | catch | no |
| box start pose, target pose, assigned faces | per episode | per episode | episode | yes |

Hook mass is fixed at 0.30 kg (user decision). Inertia follows the box mass and size (uniform density); it is
not an independent variable.

Seeds: train seeds {1, 2, 3}; evaluation seed 123 for every method and every table cell; 256 envs x 4 episodes
minimum per cell. Every curriculum off at evaluation (`disable_curricula()` or the construction cfg equivalents:
final capture radius 5 cm, chord gate 25 deg, hold 100 steps, no swing assist, no attached starts, no partial
hooked starts).

## 3. Success definitions (one per behaviour, one for the chain)

| behaviour | success per episode | secondary metrics |
|---|---|---|
| hooking (per drone) | hook attached and wire taut for 100 consecutive control steps (1 s), before time-out, with the box not moved (virtual-box risk < 1 throughout) | catch rate, steps to catch, drops per catch, over-box fraction |
| joint hold (team) | all N drones attached, every wire taut, virtual-box risk < 1 (the box does not slide; dynamic friction) for 100 consecutive steps | time to the 1st / 2nd / 3rd catch, wire tension / box weight quantiles (pin force minus hook weight), collisions |
| transport (team) | box within 0.10 m and 0.35 rad of the target, held 50 steps, all hooks attached | position / orientation error, survival |
| chain (team) | hooking success for all drones, then transport success, within one 120 s episode, no reset in between | time to switch (last catch -> first lift), cycles per episode |

Revision 2026-10-19 (v1.1): the joint-hold row originally capped each pull at 0.4 x box weight. The measured pin
force includes the 0.30 kg hook (2.94 N = 0.30 x the weight of a 1 kg box) and a wire goes slack, and the hook
detaches, below about 4 N, so that cap was unreachable (0 % at it 30k with wire pull / weight q10/50/90 =
0.36 / 0.71 / 0.84). The cap was a proxy for "the box is not dragged"; the virtual-box risk measures that
directly, so it replaces the cap. Tension is reported, not thresholded.

Deterministic actions (policy mean) at evaluation; stochastic only when explicitly marked. Report mean and
standard deviation over the 3 training seeds, never a best-of-N checkpoint. Checkpoint = last, not best.

Revision v1.2 (2026-10-20, APPROVED by the user): the joint-hold row is measured WITH drone-drone collision termination
on (the deployment condition of the sequence env; radius per decision D9 in project_status.md). The stream-A tracker rows up
to it 40k were evaluated with --no_collision; a with-collision row on it_40000 (job 3224525) gave the same 0 protocol holds
and 22 % collision terminations. Reported tables must state the condition until the tracker carries both rows. From v1.2 the
with-collision row is the protocol number; a --no_collision row may be reported next to it as a diagnostic only.

## 4. Methods to compare

Every adaptivity claim is a difference between rows of this list on the same cells. Rows 1-3 are static
controls and must exist before any plastic row is reported.

1. **Static shared head**: the multitask shared body + one linear head, no adaptation.
2. **Static recurrent**: the same body with a GRU (64-128 units) before the head. The strongest non-plastic
   control for hidden-parameter inference; if it matches the plastic rows, plasticity is not needed.
3. **Oracle**: static head given the hidden variables as extra inputs. Upper bound; tells how much the hidden
   variables are worth at all.
4. **Meta head (ANIL)**: k = 0 and k = 1 inner step on the support rollout.
5. **Plastic head, plasticity frozen**: the trained Hebbian head with its learning rate set to 0.
6. **Plastic head, plasticity on**: the actual method.
7. **Plastic head, wrong start**: plasticity on, head initialised at the other task's attractor (switching test).

Rows 5 vs 6 isolate the online update; 6 vs 2 isolates plasticity against memory; 6 vs 3 bounds what remains.

## 5. Result tables

Table A, within-task adaptivity (transport and hooking separately): rows = methods 1-6, columns = train split /
test split for each hidden variable (mass, CoM, size), cells = success rate and the secondary error.

Table B, switching: rows = methods 1, 2, 6, 7; columns = hook success, transport success, chain success, time
to switch; evaluated in the sequence env from free starts.

Table C, robustness: rows = methods 1, 2, 6; columns = seeds 1-3; cells = chain success. Reported with the
spread, this is the number that goes into the abstract.

## 6. Pre-registered hypotheses (what the tables must show)

H1. Hidden parameters matter: oracle (3) beats static (1) by more than the seed spread on the test split.
    If it does not, the task distribution is too narrow for any adaptivity claim; widen it before training.
H2. Memory is not enough: plastic on (6) beats static recurrent (2) on the test split.
H3. The update does the work: plastic on (6) beats plastic frozen (5).
H4. Label-free switching: plastic wrong-start (7) recovers chain success within the time-to-switch budget,
    while static (1) from the wrong head does not.

A result that supports H3 but not H2 is "plasticity works but a GRU is as good"; it must be reported as such.

## 7. Identifiability pre-check (done before training; see `report/insight_tasks_2026-10-19.md`)

Before a plastic row is trained on a hidden variable, the probe in section 7 of that report must show that the
variable is not already decodable from the 3-frame observation with a static read-out. Variables that are
frame-decodable (currently: the attached flag, 95 % balanced accuracy from one frame) are not adaptivity targets;
variables that are not (currently: box mass, linear R^2 0.2 from 3 frames) are.

## 8. Reporting rules

* Every table cell carries the commit hash of flycrane-meta, the env cfg diff from defaults, the seed and the
  number of episodes.
* Numbers measured before the 2026-10-19 fixes (controller reset, terminations, gravity, pin offset) are not
  comparable and are not re-used.
* Curricula on, stochastic actions, best-of-N checkpoints or fewer than 3 seeds make a number a diagnostic.
