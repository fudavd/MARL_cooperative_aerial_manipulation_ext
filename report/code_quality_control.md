# Code quality control: MARL cooperative aerial manipulation workspace

Date: 2026-10-19. Scope: the whole workspace (`exts/MARL_mav_carry_ext`, `projects/flycrane-evo`,
`projects/flycrane-hook-grasp`, `projects/flycrane-meta`, `thirdparty/skrl` fork). Read-only audit; nothing was
changed. Paths below are relative to the workspace root; `EVO/` = `projects/flycrane-evo/flycrane_evo`,
`HOOK/` = `projects/flycrane-hook-grasp/flycrane_hook`, `META/` = `projects/flycrane-meta/flycrane_meta`.

## 0. Verdict in one paragraph

The physics plant, controller cascade and environment code are solid and mutually consistent (same 300 Hz
sim, same Geometric -> INDI -> motor cascade byte-for-byte, correct Isaac Lab APIs for mass / CoM / anchor
randomisation). The workspace does **not yet** test the stated meta-goals, however. (1) N-drone hooking has a
plant (`META/envs/construction.py`, `sequence.py`) but no MAPPO training path and no reported results; the only
hooking result is single-drone rsl_rl PPO. (2) Within-task adaptivity is only genuinely randomised in
flycrane-evo `hover_rand` (mass + attachment); in flycrane-meta the per-task mass mechanism is inert and the hook
mass range is degenerate (0.30-0.30 kg), so nothing there varies mass, inertia or shape. (3) Between-task
switching is never trained in the sequence env and is evaluated against privileged attachment labels; the notes
themselves conclude hook vs transport is identifiable from one frame, so a static head already "switches".
(4) No experiment anywhere credits the plastic/Hebbian component against a frozen static control. Two
reproducibility bugs matter for all past numbers: the hook-grasp env never resets its low-level controller state
between episodes (every v84 number carries episode-to-episode leakage), and the equivalent fix for the transport
env is uncommitted. Details, with file:line, follow.

## 1. What the user's goal requires vs what exists

| Goal component | Where it lives | Status |
|---|---|---|
| N drones individually hook onto an object (MAPPO) | `META/envs/construction.py`, `sequence.py` (3 flyhook drones, real block); hook skill from `HOOK/` (1 drone, rsl_rl PPO) | Plant exists. Training = flycrane-meta's own per-drone PPO (shared actor, centralised critic), **not** skrl MAPPO. No multi-drone hooking results recorded in any README/NOTES. |
| Cooperative transport to a goal pose | upstream `hover` env (paper MAPPO), `EVO/tasks/hover*` (Hebbian ES + MAPPO warm start), `META/envs/transport.py` (re-observed in hook layout) | Solid. Paper MAPPO 96 % success nominal; Hebbian ES far behind (36 % nominal, 1 % under randomisation). All transport numbers in flycrane-meta before 2026-10-12 were on buggy terminations and were withdrawn. |
| Hebbian layer, within-task adaptivity (mass / inertia / shape / attachment) | `EVO/hebbian/*` (ABCD rule, ES); `META/ipa_switch.py`, `hebb_switch.py` (closed-form switching heads) | Only `hover_rand` randomises mass + attachment per episode/env. Shape and inertia are never varied anywhere (block inertia is scaled uniformly with mass, `construction.py:_set_block_mass`). No plastic-vs-frozen comparison exists. |
| Hebbian layer, between-task switching (hook -> transport, no labels) | `META/envs/sequence.py` (one episode: hook -> carry -> teleport), `META/tests/eval_sequence.py` | Switching heads are fitted offline from labelled frames and evaluated against `u.attached` (privileged). Never trained in the sequence env. |
| Chaining the two behaviours end-to-end | `sequence.py` + `ActivitySwitchHead` | Only through `tests/eval_sequence.py`, a print-only script; no result in NOTES. |

Three different adaptation mechanisms coexist (ABCD Hebbian ES in evo; ANIL/MAML + closed-form IPA heads in
meta; nothing plastic in hook) and four success definitions (section 6). Nothing chains them.

## 2. Workspace layout and provenance

* `exts/MARL_mav_carry_ext`: Jack Zeng's paper code (arXiv 2508.01522). Reference only.
* `projects/flycrane-evo` (submodule, branch `dev`, 9 modified + 2 untracked files uncommitted).
* `projects/flycrane-hook-grasp` (submodule, `dev`, clean).
* `projects/flycrane-meta`: **not a submodule, untracked in the root repo, and has no git remote** (`git remote -v`
  is empty). 55 commits of the most recent work (last 2026-10-17) exist only on this laptop's disk. This is the
  highest operational risk in the workspace.
* `thirdparty/skrl` fork: committed local critics / advantage filtering / LR schedulers; uncommitted `MAPPO_RNN`
  and trainer patches. `MAPPO_RNN` is referenced by nothing in `projects/`, `scripts/` or `exts/` (dead code; its
  docstring ties it to the single-drone hook task, which `HOOK/docs/HISTORY.md:24` records as abandoned).
* Compute: this laptop (RTX 4070 8 GB) only runs toy / imitation jobs. Isaac runs were on ALC (RTX 4090) and
  "izar". ALC's copies under `~/repl/` are stale (Sep 28-30) and ALC is currently occupied by another user's job;
  the location of the latest Isaac runs is not discoverable from this machine.

## 3. Physics and simulation layer

### 3.1 Plant
* Drone: Falcon URDF, 0.6 kg + 4 x 4.2 g rotors; thrust applied as body-frame external forces on the rotor
  bodies, reaction yaw moments on the base. Policy action ACCBR (lin acc 3 + body rates 2) -> `GeometricController`
  -> `IndiController` -> first-order motor lag (33 ms) -> thrust 1.5625e-6 w^2, clamp 0-6.25 N.
* Transport cables (evo / meta-transport): 3 rigid 1 m links on 3-DoF revolute chains, load-side anchor =
  `physics:localPos0` edited in USD before play (`EVO/tasks/hover_rand/hover_rand_env.py:300-304`).
* Wire + hook (hook / construction / sequence): 8 rigid links + 8 dummy links, +-90 deg per DoF, 2 cm sphere hook.
  Hook-grasp asset default = 30 g hook on a 21 g wire; the hook env sets 0.30 kg via a startup event
  (`HOOK/tasks/hook_grasp/marl_hook_env_cfg.py:94-102`); construction sets 0.30 kg + 0.5 g links
  (`META/envs/construction_cfg.py:84-85`, `construction.py:58-65`).
* Object: hook-grasp has **no basket body** (analytic pose + surrogate wall spring, k 1500 N/m); construction has
  a real dynamic cuboid 0.55 x 0.55 x 0.32 m, 0.9-1.9 kg, mu 0.8/0.6, optionally kinematic + collider-less
  ("v84 mode"). The transport basket is 0.55 x 0.45 m, top at 0.14 m (`hover_rand_env_cfg.py:46-48`): the object
  shape differs between the hooking and the transport plants.
* Hook-to-bar attachment is a surrogate in every plant: no joint; while `attached` a PD pin (kp 400, kd 20, clamp
  60 N) pulls the hook to the attach point; construction applies the reaction wrench to the block
  (`construction.py:220-226`).

### 3.2 Sim / control parameters

| | hook-grasp | evo hover | meta transport | meta construction / sequence |
|---|---|---|---|---|
| physics dt / decimation | 1/300 s / 3 | same | inherits | same |
| controller rate | 300 Hz (`low_level_decimation=1`) | same | same | same |
| gravity | -9.8066 | -9.8066 | inherits | **-9.81 (Isaac default, not overridden)** |
| episode | 20 s | 20 s | 20 s | 40 s / 120 s |
| replicate_physics | True | True | False (per-env anchors) | True; False iff `box_size_dr` |
| contact sensors | declared, never read | used (illegal contact) | inherits | none |

Controllers are byte-identical across the three copies except the hook `indi.py` adds `disturbance_residual`;
the meta transport env swaps that controller in (`META/envs/transport.py:67-69`), so the residual the actor sees
is computed identically on both tasks. Verified correct.

### 3.3 Physics findings (ranked)

P1 (HIGH, verified). **Hook-grasp env never resets its low-level control state.** `marl_hook_env.py:826-867`
resets only attachment state and observation buffers; the only `.reset(` call is the observation buffer at
`:849`. Motor speeds, INDI/geometric filters, `_forces`, `_moments`, `_hook_force` and the INDI residual all carry
over from the previous episode. The controllers do have `reset(env_ids)` methods (`HOOK/controllers/geometric.py:241`,
`indi.py:188`, `motor_model.py:48`); they are simply never called. The evo working tree fixes the identical bug
and its docstring quantifies the effect: early-crash rate 33 % in a process's first episode, 0.7 % after a
time-out, 67-93 % after an early crash (`EVO/tasks/hover/hover_env.py:901-924`). Every hook number (v84r2 catch
0.90 / hold 0.97, `MetaHookEnv`, all hook demos) was produced with this leakage, and the residual observed on
step 1 is stale.

P2 (HIGH, verified). **The evo reset fix is uncommitted and differs from construction's.** `_reset_low_level`
exists only in the working tree (`git show HEAD:...hover_env.py` has no such function). Construction seeds the
rotor thrusts from the reset motor speed (`construction.py:827-839`), evo zeroes them (`hover_env.py:921`), so on
the first control step the cable observer `a - g - R f/m` sees zero thrust on transport but hover thrust on
construction. The two plants start episodes differently. NOTES say the izar copy was synced at 6a849a8, which
predates this fix.

P3 (MEDIUM-HIGH, unverified in sim). In construction's physical-block mode the pin target is on the collider face
(`construction.py:356-364`, attached spawns at the face centre `:999-1002`) while the hook is a 2 cm sphere pinned by
its centre. Contact keeps the sphere ~2 cm out, so the 400 N/m pin exerts a persistent ~8 N phantom force. The
block sees no net force, but `tension`, the critic feature and `_vbox_risk` read ~8 N for a slack wire, and this
plausibly explains "the taut hold needs >= 6.5 N at the pin" (commit 759d45c) for a 2.9 N hook.

P4 (MEDIUM, verified). With `box_size_dr=True`, `replicate_physics` is turned off (`construction_cfg.py:42-43`) but
`ConstructionEnv._setup_scene` never calls `scene.filter_collisions` (`construction.py:159-174`), unlike
`hover_rand_env.py:225`. Isaac Lab only warns; neighbouring envs can collide physically.

P5 (MEDIUM). Wire re-massed to 0.5 g links without stability re-validation: 600:1 tip/link ratio at 4 solver
iterations in a 17-DoF chain, in a plant that already hung at a 500 g hook (commit d005db7). NOTES record more
drops with the light wire (87 % vs 91 %). Hook-grasp (2.5 g links) and construction (0.5 g) are now different wires.

P6 (MEDIUM-LOW). Contact sensors are activated but never read in hook-grasp (`marl_hook_env_cfg.py:141-147` vs
`_get_dones`); construction has none. Wire/hook dragging on the ground or a drone touching the box is undetected.

P7 (LOW-MEDIUM). Hook-grasp attached spawn writes the hanging joint state while placing the drone a full wire
length up-and-out (`marl_hook_env.py:976-998`): a 0.49 m pin stretch and a 60 N yank at the first step.
`hold_spawn_v84=True` reproduces it in construction.

P8 (LOW). Gravity 9.81 vs 9.8066 (controllers hard-code 9.8066, `geometric.py:43`; `_vbox_risk` uses 9.81): a
constant 0.0034 m/s^2 offset in the INDI linear residual on the construction plant only.

P9 (LOW). `RotorMotor.direction` hard-codes `device="cuda"`; the flyhook USD references the upstream falcon asset
by an absolute `/home/fuda/...` path; controller `falcon_mass=0.6017` vs URDF 0.6168 kg.

### 3.4 Verified correct
Cone cable-angle test and `repeat_interleave` pairing (`hover_env.py:772-797`), rope_2 collision fix applied to env_0
before cloning and copied per env; rotor-thrust reset in construction (commit 3886610) inherited by `SequenceEnv`;
external wrenches rotated to body frame before `set_external_force_and_torque`; mass / inertia / CoM / box scale set
through the correct PhysX-view and USD APIs with CPU ids; per-env attachments correctly force
`replicate_physics=False` + `copy_from_source=True` + ground-only collision filtering; the virtual-box check is a
reasonable quasi-static tip / slide / lift test.

## 4. Environments: randomisation, observation, rewards, terminations

### 4.1 Randomisation inventory

| Quantity | hook-grasp | evo hover_rand | meta transport | meta construction / sequence |
|---|---|---|---|---|
| object mass | hook fixed 0.30 kg | basket U(0.9, 1.9) per episode | same event (per episode) | block U(0.9, 1.9) per episode |
| inertia | - | scaled with mass (uniform density) | same | same |
| shape | - | fixed | fixed | `box_size_dr` (per env at scene build), off by default |
| CoM offset | - | - | - | `com_offset_range`, off by default |
| attachment point | n/a (catch anywhere on the assigned wall) | per env at scene build (`attachment_per_env`, **default False** = one shared draw) | per task group, seeded by `cfg.seed` | fixed in the block frame at the catch point |
| object pose / goal | per episode | per episode | per episode | per episode |
| start state | spawn curricula | optional tilt / velocity / rope noise (new) | inherits | spawn ranges, curricula |

Important: the evo `hover_rand` randomisation, which the README (`EVO/README.md:15`) describes as the randomised
task, only randomises attachment per env when `--attachment_per_env 1` is passed; the default is a single
attachment draw shared by all envs. The meta hook family has `train_mass = test_mass = (0.30, 0.30)`
(`META/envs/hook.py:26-27`) while `META/README.md:14-15` claims 0.15-0.45 / 0.10-0.55 kg.

### 4.2 Terminations
* Evo hover default `cable_angle_mode="euler"` and `fix_rope_collision=False` (`hover_env_cfg.py:186,190`), i.e. the
  inherited buggy terminations (rope paired with another env's basket; rope_2 collider on the ground). The evo
  train / eval / MAPPO scripts never set the fixed mode (grep over `EVO/scripts` finds only the diagnostic
  `check_rope.py`). Only `MetaTransportEnvCfg` (`transport.py:49-50`) opts in. **All flycrane-evo transport
  results, including the README reference table and the MAPPO warm-start checkpoints, were produced on the buggy
  terminations** unless the runs passed overrides not visible in the repo.
* Hook-grasp: fly-low, bounding box, non-finite, runaway, hard-entangle, hold success (`marl_hook_env.py:759-788`).
  No ground / wall contact termination (P6).
* Construction: fly-low, collision (curriculum-gated), out of bounds, block flipped, non-finite, blow-up, optional
  virtual-box failure, joint-hold success (`construction.py:_get_dones`). Reasonable; blow-up zeroes the reward.

### 4.3 Rewards
* Transport (`hover_env.py:602-688`): the paper's six exp-shaped terms plus optional linear / tilt / heading /
  separation / success-bonus / termination-penalty terms, all off by default. A `paper_reward` metric is logged
  with fixed weights for run comparison. Good practice.
* Hook-grasp (`marl_hook_env.py:563-744`): 22 terms, phase-gated. Shared reward even for `_NUM_DRONES > 1`.
* Construction (`construction.py:_get_rewards`): the hook terms ported "term for term", plus hold / gentle-hold /
  joint-hold / lift / carry / place / release / proximity / sector / virtual-box / separation / collision-curriculum
  terms; `per_drone_credit` optional. Roughly 50 weights. The port looked faithful on a spot check (approach,
  proximity, impact, swing, above-box, swing-limit, lock / tension / anti-sag / slack / wall-pull / progress /
  success all match the hook cfg values), but there is no test asserting equivalence between the two
  implementations, and the hook env scales action-rate / accel penalties by a phase gate that the construction
  port replaces with a single `w_action_rate = 0.05`.

### 4.4 Observation contract
All plants expose the unified 51-dim per-drone frame (object pose 12 | assigned-wall centre 3 | own state 18 |
goal error 12 | INDI residual 6), stacked 3x = 153. The hook policy therefore runs unchanged on every drone of
every plant (verified in `META/tests/isaac_checks.py`). Two fragilities were found and documented by the meta
session itself: the wall-centre height slot was a constant in hook-grasp, so the frozen normaliser sees -400 sigma
on any change (`NOTES.md`, 2026-10-13); and the object reference point differs (basket origin vs block base).

## 5. Training setup and algorithms

### 5.1 flycrane-evo
* Hebbian ABCD rule averaged over `update_interval` steps with per-env plastic weights (`hebbian_core.py:143-237`),
  OpenAI-ES with antithetic sampling, centred ranks, Adam (`openai_es.py:66-95`); EGGROLL, PGPE, CMA-ES available.
  MAPPO warm start trains the same bias-free network with PPO + privileged critic; zero genome == PPO policy is
  unit-tested (`tests/test_hebbian_es.py:95-109`). Correct.
* Uncommitted work (coherent, complete, untested): paired antithetic evaluation with common random numbers,
  a privileged frozen-teacher imitation term in the fitness, `actor_extra` oracle inputs, the `_reset_low_level`
  fix, checkpoint = search centre, a MAPPO `best.pt` gate. Two untracked probes: `check_ori.py` and
  `probe_plasticity.py` (ridge / MLP probes of whether the weight drift encodes the hidden task variables: exactly
  the experiment the within-task claim needs; no results in the repo).
* Observation normaliser is updated every step during ES without `init_from` (`adapter.py:585-587`): the fitness
  landscape drifts across generations.

### 5.2 flycrane-hook-grasp
rsl_rl PPO, asymmetric actor-critic, six curricula annealing to the final task; `disable_curricula()` pins them for
eval. Reference result single seed; catch is seed-sensitive (1/3 seeds met the goal at an earlier recipe). P1
applies to all of it.

### 5.3 flycrane-meta
* ANIL / MAML / BOIL / multitask on a shared body with per-family heads and critics; second-order through the head;
  PPO-clipped meta objective; two Isaac processes average gradients over gloo (`algo/meta.py`, `trainer.py`,
  `dist.py`). Inner-gradient and second-order correctness are unit-tested (`tests/test_meta_grad.py`). The notes
  record that the ANIL inner step never helps on Isaac and that the meta head is already as good as each task's
  optimal head.
* Switching heads (`ipa_switch.py`): closed-form D0/D1 so the expected update vanishes at each specialised head,
  anchoring decay, mean-field Jacobian eigenvalues. The stability bound `eta_bound` is computed and stored
  (`:144, :255, :421`) but never enforced or asserted; `step` applies any `eta`.
* `dist.py` desync risk: `perm.chunk(nmb)` yields fewer chunks when `num_tasks % task_minibatches != 0`, so two
  ranks with different task counts issue different numbers of `all_reduce` calls and hang (`meta.py:190-191`).
* Config vs docs drift: `inner_lr=0.5` (README says 0.1), `entropy_coef=0.004`, `learn_inner_lr=False`,
  `sample_minibatches=1` vs the NOTES-stated defaults.
* The per-task mass mechanism is inert for transport: `configure_env_cfg` keeps the per-episode mass event unless
  `continuous=False` (`envs/__init__.py:60-64`); `scripts/eval_adapt.py:61` never passes it, so `--split test`
  masses are overwritten at every reset. Combined with the degenerate hook range, flycrane-meta's "tasks" vary
  nothing physical except the transport attachment geometry (fixed per run).

### 5.4 skrl fork
`MAPPO_RNN` (`skrl/multi_agents/torch/mappo/mappo_rnn.py`): inherits the upstream PPO-RNN hidden-state aliasing
(`:423` with `:350`, sequences replay from h_{t+1}); with >1 agents and separate model instances only the first
agent's parameters enter the optimiser (`:175-190`). Unused, so no effect on results; do not use without a test.

## 6. Tests and evaluation hygiene

Isaac-free suites run and pass here: flycrane-evo 22/22, flycrane-hook-grasp 7/7 (the `__main__` runner of
`test_attachment.py` lists 6 of 7). flycrane-meta: `test_meta_grad.py`, `test_per_family_head.py` are real tests;
`isaac_checks.py` and `construction_checks.py` print PASS/FAIL and always exit 0; every other file under
`META/tests/` is a diagnostic script without assertions.

| Claim | Test? |
|---|---|
| Hebbian trainer bit-equal to pre-vectorisation commit; zero genome == PPO | yes (`test_hebbian_es.py`) |
| Surrogate catch model gates (radius, up-sweep, approach, slack) | yes, but not `chord_angle_perp_only`, `catch_latch`, moving bar (`bar_vel`) as used by construction |
| ANIL inner step == REINFORCE, second order == finite differences | yes |
| First observation after reset is fresh | script (`reset_check.py`) |
| Controller state reset between episodes | **no** (P1, P2) |
| Hook / transport reward port equivalence | no |
| Termination fixes change outcomes | manual script only (`check_rope.py`, `eval_dones.py`) |
| Catch 0.90 / hold 0.97; evo README table; all switching-head numbers; toy gains | no (single-seed logs only) |
| Collision filtering under `box_size_dr` | no |

Evaluation hygiene issues:
* Four incompatible success definitions: evo = episode-mean position error < 0.3 m; hook = catch and 100-step
  taut hold; meta protocol = mean error < 0.3 m; sequence = 0.10 m + 0.35 rad held 50 steps.
* `eval_adapt.py` leaves the hook curricula on (hold requirement 30, not 100 steps), samples stochastic actions
  while every other eval is deterministic, and cannot load `head_norm='ipa'` checkpoints.
* `EVO/scripts/eval.py` claims "same goals for every policy" but does not reseed between policies
  (`common.py:476`).
* Reference checkpoints are best-of-run selections ("selection-biased maximum", acknowledged in the README).
* All withdrawn transport numbers (NOTES, 2026-10-12) still have their checkpoints and demos on disk; nothing
  marks them as invalid at the file level.

## 7. Ranked summary of findings

1. **flycrane-meta has no remote and is not a submodule**: 55 commits of the latest work are unbacked.
2. **Hook-grasp never resets controller / motor state between episodes** (P1): every hook result carries
   episode-history leakage; measured effect on transport was 33 % vs 0.7 % vs 67-93 % early-crash rates.
3. **Evo transport trains on the buggy terminations by default** and the controller-reset fix is uncommitted
   (P2, 4.2): the evo README table and the MAPPO warm-start checkpoints predate both fixes.
4. **Within-task adaptivity is not exercised in flycrane-meta** (degenerate hook mass, inert transport task mass;
   no shape / inertia variation anywhere) and **never credited against a static control** in any project.
5. **Between-task switching is evaluated with privileged labels, never trained in the sequence env**, and is
   frame-identifiable without plasticity by the notes' own analysis.
6. **N-drone hooking has no MAPPO path and no reported results.**
7. Phantom pin force on the real block (P3); missing inter-env collision filtering under box-size DR (P4); wire
   mass change not re-validated (P5).
8. IPA stability bound computed, never enforced; `dist.py` hang risk; config / README drift; `MAPPO_RNN` dead
   code with inherited bugs.
9. Evaluation hygiene: curricula left on, stochastic vs deterministic, four success definitions, no reseeding.

## 8. Recommended direction (project-management view)

1. **Back up flycrane-meta today**: add a remote, push `dev`, register it as a submodule like the other two.
2. **Commit the evo working tree** once `_reset_low_level` and construction's thrust seeding are made identical
   (pick one: construction's init-omega seeding is the self-consistent one). Port the same reset to
   `MARLHookGraspEnv._reset_idx`. Add a cross-plant test: same policy, same start state, same first-step action
   regardless of the env's previous episode.
3. **Switch evo's default to the fixed terminations** (`cable_angle_mode="cone"`, `fix_rope_collision=True`) or set
   them in `scripts/common.py`, then re-measure the README table, the MAPPO warm start and the ES checkpoints.
   Mark everything measured before as "pre-fix" in the READMEs.
4. **Make the adaptivity claim testable before more training**: (a) fix the meta hook mass range and the
   transport task-mass override; (b) add shape (box size) and CoM offset to the task variants (the construction
   cfg already has both knobs); (c) run `probe_plasticity.py` and the `--hebbian_lr 0` frozen control on the same
   seeds, and report plastic vs frozen vs static-MAPPO in one table with one success definition.
5. **Define one success metric per behaviour and one for the chained task**, pin curricula off in every eval,
   use deterministic actions, reseed per policy, and report mean +- std over >= 3 seeds.
6. **Decide the N-drone hooking training path**: either train the construction / sequence plant with the
   skrl MAPPO fork (then the fork's `MAPPO_RNN` needs the fixes in 5.4 or should be deleted) or state that the
   meta trainer's shared-actor PPO is the MAPPO used, and record its first multi-drone hooking numbers.
7. **For the switching claim**, train or at least fine-tune the plastic head inside `sequence.py` and score it
   against a static shared head and against the ANIL head, with the switch inferred from the drone's own
   observations, not from `u.attached`.
8. Smaller: filter inter-env collisions when `box_size_dr` is on; read the contact sensors or drop them; fix the
   `test_attachment.py` runner; make `isaac_checks.py` exit non-zero on failure; enforce `eta <= eta_bound` in
   `IPASwitchHead`; set gravity to 9.8066 in `ConstructionEnvCfg`.

## 9. Open questions for the active sessions

* Where are the latest Isaac runs (izar?) and are those copies at the post-fix commits (evo 6a849a8 + the
  uncommitted `_reset_low_level`; meta 93bf9cb)?
* Which of the three adaptation mechanisms (ABCD Hebbian ES, ANIL, IPA switching head) is the one the thesis
  claims? The repo currently carries all three.
* Is multi-drone hooking meant to be trained with skrl MAPPO (then the fork needs work) or with the meta
  trainer's shared-actor PPO (then say so)?
* Which transport results have been re-measured since the termination fix, and have the hook numbers been
  re-measured since the controller-reset problem was found on the transport side?

## 10. Session status (collected 2026-10-19 after the audit)

### Session marl-cooperative-aerial-manipulation-ext-00 (flycrane-meta, N-drone hooking)
* Objective (set by the user): a 3-drone joint 1 s hold from free starts on the v84-style basket, one shared policy
  on two plants (`seq_hook1`, 1 drone; `seq_hook`, 3 drones), with the virtual-box check enforcing gentle pulls
  until all three are attached.
* Done: v84 port onto the sequence plant, v84 hold pose for hooked starts, collision curriculum, corner-only sector
  penalty, virtual-box slide-ratio fix (the 1e-6 denominator bug), and the run `v84fresh_s1` (wider fixed 1 kg box,
  low-passed box check + taut-pull reward, gradual penalty / termination ramps, adaptive catch curriculum).
* In progress: `v84fresh_s1` at ~28.7k of 60k iterations. Free-start test at the final 5 cm / 25 deg gate
  (iteration 26k): 105 catches single-drone and 105 three-drone, few drops; 2 of 3 drones attached 5.7 % of the
  time; no 3-drone joint hold yet; curriculum stalled at progress 0.80 (single) / 0.65 (three drones).
* Compute: izar, `~/projects/split/flycrane-meta`, rsync of the working tree at commit 93bf9cb (clean).
* Confirms: construction / sequence resets controllers and seeds thrusts (not affected by P1); the original
  v84 / v84r2 hook-grasp numbers are affected. Thesis mechanism = the IPA / activity-driven switching head on a
  task-invariant shared body (ANIL is not the thesis path). N-drone hooking trains with the meta trainer's
  shared-actor PPO, not skrl MAPPO.
* On audit items: fixed 0.30 kg hook mass is deliberate (user request); the README range is stale. `box_size_dr`
  collision filtering noted (8 m spacing mitigates; with `max_dist_from_block` 4 m and `place_xy` 1.5 m the reachable
  ranges of neighbouring envs still overlap by 3 m, so the finding stands as low risk). The transport-mass,
  `eval_adapt`, `eval_sequence` and `eta_bound` items are outside its current scope and are passed to the user.
* Will not add a remote / submodule registration for flycrane-meta without the user's go-ahead (recommendation 1
  remains open and needs a user decision).

### Alignment assessment
The hooking line is on the goal (N drones, individual hooks, gentle joint hold) and its plant is the one the
audit considers correct. Two gaps remain for the goal as stated: the within-task variance of the hooking task is
intentionally zero (fixed hook, fixed 1 kg box), and the switching head is not yet exercised in the sequence env
with this policy. Both are decisions for the user, not defects.

### Session marl-cooperative-aerial-manipulation-ext-09 (flycrane-evo, Hebbian ES transport)
* Objective: an online-adaptive Hebbian ABCD policy for 3-drone transport on `hover_rand` (per-env attachments,
  mass / depth / start randomisation), warm-started from the MAPPO actor; the actor sees drone-only observations
  (one frame, observer cable direction) plus its own start pose, and must infer the basket orientation through the
  plastic layer. Compared on paper return, survival and orientation error.
* Done: per-function audit of the ES path with CPU + Isaac tests; fresh basket orientation in every observation;
  `best.pt` = search centre; equilibrium depth in the common-random-number scenarios; paired antithetic evaluation;
  and today the `_reset_low_level` fix (33 % / 0.7 % / 67-98 % early-crash dependence on the previous episode;
  episodes now bit-reproducible, return correlation 1.000).
* In progress: v17 runs on the fixed code on izar (jobs 3223562 teacher-weight 5, 3223586 heading x4; 4096 x 4
  paired on 8192 envs, 48 h). Open scientific question: whether a drone-only policy can reach the paper's
  orientation tolerance at all (36 % vs MAPPO 95 % on fixed attachments in older runs).
* Compute: izar (SCITAS, account micro-515), `~/projects/evo_v15` = file copy of the local working tree (HEAD
  31b3d06 + the uncommitted edits). The user commits on request only.
* **Correction to section 4.2 / finding 3**: every izar ES / MAPPO run, the teacher checkpoint and the warm-start
  actors pass `env.cable_angle_mode=cone env.fix_rope_collision=true` (plus equilibrium start, 40 s episodes,
  `drone_collision_threshold` 0.5, `termination_penalty` 10) through `~/projects/es_transport/common_env.sh` on
  izar, outside this repo. The repo defaults remain the buggy mode, so anyone running `train.py` / `eval.py`
  without those overrides reproduces the bug, and the README reference table (measured 2026-09-28) predates the
  fix. Recommendation 3 (make the fixed mode the default, or set it in `scripts/common.py`) still applies.
* Thesis mechanism from this line's view: the ABCD Hebbian ES layer on a task-invariant shared body, adaptation
  from neuron activations through the plastic weights, no label or modulator. Agrees that no evolved-static-weights
  control exists yet (the only comparison is zero genome = MAPPO vs evolved ABCD).
* Agrees the hook-grasp env has the same reset bug class and that `_reset_low_level` should port directly.

### Cross-session alignment summary
* Both lines run on izar, from rsync copies of uncommitted working trees. Reproducibility therefore depends on
  this laptop's disk; recommendation 1-2 (push flycrane-meta, commit evo) are the first actions to take.
* The two sessions name different thesis mechanisms (evo: ABCD Hebbian ES; meta: IPA / activity switching head).
  Both describe them as "activity-driven plasticity on a task-invariant shared body", so the intent is the same,
  but the implementations, plants, observation sets (45-dim drone-only, one frame vs 51 x 3 hook layout) and
  success metrics are not shared. This is the main alignment decision for the user: pick the mechanism and the
  observation contract that the chained hook -> transport experiment will use, so that the hooking policy from
  session 00 and the transport policy from session 09 can actually be chained.
* Nobody owns the hook-grasp controller-reset port, the stale READMEs, or the flycrane-meta eval scripts
  (`eval_adapt`, `eval_sequence`, `eta_bound`). These need an owner.

## 2026-10-20 — flycrane-meta trainer and hook-reward port review (gate for the D10 hook ablation)

Scope: `flycrane_meta/trainer.py`, `algo/meta.py`, `rollout.py`, `models.py`, `config.py`, `envs/families.py`,
`envs/construction.py` (`_rewards_legacy`, `_update_attachment`, `_closest_bar_point`), `envs/construction_cfg.py`,
the legacy run's `izar_v84freshd.sbatch`, and the rsl_rl v84 recipe (`flycrane-hook-grasp` `rsl_rl_ppo_cfg.py`,
`marl_hook_env.py`, `marl_hook_env_cfg.py`). Read-only; nothing committed to the submodules by the PM.

**PPO implementation (multitask variant): correct as far as read.** GAE with rsl_rl's time-out bootstrap
(`r += gamma V(s) * trunc`, done masks the next value), global advantage normalisation, PPO clip on the concatenated
support + query batch, adaptive-KL learning rate per minibatch with the same 1.5x rule and 1e-5 / 1e-2 clamps,
entropy on the log-std (same gradient as rsl_rl), gradient clip 1.0 on the shared parameters, log-std clamp, actor
observation normaliser frozen within an iteration and merged from all data afterwards, separate critic on the
privileged state trained after the policy step (MSE, no value clipping). No bug found.

**Legacy hooking arm hyperparameters are not v84's** (sbatch): cosine lr 5e-5 -> 5e-6 (rsl_rl adaptive from 1e-3),
AdamW wd 0.01 (Adam, none), entropy 0 with a scheduled std 0.6 -> 0.2 pinned since it 12k (learned std from 1.0,
entropy 0.004), 16 task minibatches (4), critic 512 x 512 + warm-up 100 (256 x 256, none), 4096 envs x 24 steps
(512 x 24). These came with the two-family std-explosion fixes and were inherited by the hooking arms.

**Reward port** (term-by-term formula diff, subagent, verified in code): all shared hook-phase terms and the
attachment gate compute the same expressions. Deviations to fix in the parity staging: (1) reward target z = band
middle (0.12 m above the base) vs v84's wall centre 0.16 m (`construction.py:172-175, 619` vs
`marl_hook_env.py:163-167, 586`); (2) `_update_attachment()` runs in `_get_observations` while Isaac steps
dones -> rewards -> reset -> observations and `_get_dones` refreshes kinematics, so rewards see post-step kinematics
with one-step-stale attachment flags (`construction.py:510, 844`; `direct_marl_env.py:381-396`); (3)
`bar_lateral_frac` default 0.4 vs v84's full face (the legacy single plant already used 1.0). Residuals accepted:
above_box gated by unattached, repeatable hold milestone, one curriculum clock, critic form, spawn distribution.
The v84 regulariser port by -6d (uncommitted at review time) matches `marl_hook_env.py:700-754` for one drone.

## 2026-10-22 PM review of the D15/D16 trainer commits on flycrane-meta dev (local, unpushed; user asked for a check of -6d's code)

| commit | scope | verdict |
|---|---|---|
| da89fc6 | `--single_rollout`, `--flat_minibatches` (T2) | no defect: same minibatch count / size as the task blocks, critic sees the same samples once, normaliser moments counted once |
| dcdf545 | `--critic_shared_lr`, `--value_clip` (T3) | no defect: rsl_rl's clipped value loss exactly, collection-time values detached |
| a6392af | `scripts/train_rsl_rl.py` three drones, `--init_checkpoint`, `--final` (D16) | no defect: drone-major rows consistent for obs / actions / rewards / dones / critic; verified empirically later by -6d's probe_align.py (exact) |
| d644b49 | `--critic_warmup`, `--max_lr` CappedAdam (D16b) | no defect: KL 0 while frozen keeps the adaptive rule idle, cap clamps the rate per minibatch |
| 85f02c6 | `--gamma` + reward_rates phase / stochastic / event split | no defect; LIVE in D16s 3226754 (fm_d16c md5 900edfcc verified against the blob) |
| 0467f8e | `--resume_full`, `--init_critic`, `--init_actor_optim` (d16ctl2) | no defect: Adam state copied by name with a 13 = 13 guard, rsl_rl load restores the iteration counter |
| 3628049 | ego critic layout + one-drone critic lift + `--critic_clip` (D16d) | no defect: column arithmetic checked against `_state` (frame 40 + 36 D), one-hot drop exact, constant columns neutralised; probe |V3 - V_R0| 4.9e-4 |
| aea6d09 | `--warmup_lr`, `--warmup_vloss` gate, EV log (D16e, never run) | logic correct; `os._exit(3)` skips env.close, acceptable |
| 56b43d4 | reward_rates `_get_dones` spy (episode ends by drone) | fine |
| tracker | d16s rows every 2k from 4k | fine |
| 2633497 | construction: `sector_ramp_begin/end` (control-step ramp 0 -> w_sector, same shape as vbox_ramp) + `sector_attached_only` (gate 1 - na, na = unattached, so attached drones pay) (D16 sector redesign, 2026-10-25) | no defect: defaults reproduce the old term exactly (tested on R0 5k: default sector -39.247 unchanged, attached-only 0.0, ramp 300k-900k at step 0 0.0; catches / positive return identical); ramp keyed on `_ctrl_step` so it needs `curriculum_steps`, not `adaptive_curriculum`; with attached-only keep `sector_inward_margin` 10 or the hold pose is taxed. Local only, unpushed, not synced to izar; regression_checks before any sync |
| 895be5e | diag_seq_hook: per-spell share of steps with vbox risk < 1 (`spell_frac_risk_lt1`, sorted) and count of spells whose median risk is < 1 (`spells_median_risk_lt1`) (2026-10-26) | no defect: `p2_lt1` accumulates on the same boundaries as `p2_len` (carried when the class is unchanged, restarted from the current step's flag on a new class, zeroed outside exactly-two); the spell is appended with the pre-update counters, so its share is over exactly its own steps; the end-of-rollout flush matches; a NaN risk (no `rraw`) counts as not < 1; share > 0.5 is equivalent to median risk < 1. `cur`/`lt1` moved after `rk`, no change in semantics. Diagnostic only, +13/-5, no env change. Local, unpushed |
| 466095c | diag_seq_hook: exactly-two spells by face pair (`spells`, distinct `envs`, `spell_s_q25_50_75` added to `two_attached_by_face_pair`) (2026-10-26) | no defect: per-env class `cur` (-1 outside exactly-two) is compared with the previous step's class; a spell ends on class change, on leaving exactly-two or on episode end (the `live` mask drops the env, so `cur` = -1), length resets to 1 on a new class and running spells are flushed after the rollout; step sums equal spell-length sums (smoke test at dt 0.01). `cur` stays -1 when `_hook_force` is None, which only ends spells on the first step. Diagnostic only, +17/-1, no env change. Local, unpushed |
| 49e3f64 | diag_seq_hook: exactly-two-attached steps by face pair (`two_attached_by_face_pair`: adjacent / opposite / same_face, steps + q10/50/90 of net horizontal force, cancellation, risk) (2026-10-25) | no defect: class = same_face if faces equal, opposite if `face // 2` equal (faces 0/1 = +-x, 2/3 = +-y, matching `_face_frame` / the `face < 2` half-extent choice in construction.py), else adjacent; diagnostic only, +18 lines, no env change. Local, unpushed |

D16s cfg vs R0's process (the live run): curricula match R0's; `per_drone_credit` was already in R0's cfg; the eye(D) critic tag generalises R0's constant one-hot; `w_sector` 30 is new approach shaping (all_s1 reward); `episode_length_s` 40 with `end_on_hold_success=false` is right because `construction.py:880` ends the episode as a success when the joint hold reaches `joint_hold_steps` (100), the three-drone analogue of R0's end-on-hold. Outcome of the whole D16 line (not code defects): a converged actor collapses within ~1k iterations against any critic not calibrated to the current env's returns; the user closed the fine-tune line on 2026-10-22 (retrain from scratch).
