# Hook-Based Grasping — Environment Design Workplan

**Task (one-liner).** A *single* quadrotor carries a flexible **wire** with a **small weight** and a
**4-pronged releasable hook** at the bottom. It must fly up to a **basket** (mesh wall with horizontal
bars), **swing** the weighted wire, and **hook onto a horizontal bar** anywhere on the basket wall, then
**keep pulling** so the wire stays taut and the hook does not fall off.

This document is the design/plan. **No environment code is written until the open questions in §7 are
resolved with the user.**

---

## 1. What we are reusing from the existing codebase

| Piece | Source | Reuse for hook task |
|---|---|---|
| Env base class | `DirectMARLEnv` (`hover/marl_hover_env.py`) | → single-agent `DirectRLEnv` (or 1-agent MARL, see §7 Q2) |
| Drone asset | `FALCON_CFG` (`assets/falcon.py`), mass **0.6017 kg** | the single drone |
| **Flexible cable model** | `flycrane_rod.usd`: each cable = **multi-link chain of spherical joints** (`rope_*_sphere_joint_*_joint_x/y`, damping 0.005) | the weighted swinging wire — matches FLARE's rigid-chain approach that captures taut/slack hybrid dynamics |
| Control cascade | `ACCBR` mode: policy → `[lin_accel(3), body_rates(2)]` → Geometric → INDI → motor model, ~100 Hz (decimation 3, sim dt 1/300) | drone actuation |
| USD anchor editing | `hover_rand._randomize_attachments` edits `physics:localPos0` pre-sim | pattern for randomizing wire length / bar positions |
| Contact sensors | `ContactSensorCfg` on all bodies | wire/hook–wall contact detection |
| Obs history | `CircularBuffer`, `history_len` | partial-observability memory |
| Reward/termination bookkeeping | `_episode_sums`, `extras["log"]`, per-term logging | keep the same logging structure |
| Debug vis | `VisualizationMarkers` (goal/body frames) | visualize target bar + hook + attach state |

## 2. Reference-paper takeaways (see agent extractions)

- **FLARE** (RA-L 2026) — the RL blueprint: cable = serial rigid-link chain (hybrid taut/slack); action =
  normalized **CTBR** through a Betaflight-like inner loop @100 Hz; payload state encoded as **two cable
  deviation angles (φ, θ)** in body frame + previous action; PPO, tiny MLP (2×128); reward = progress
  (λ₂=10) + swing-limit safety penalty (r_excess=3) + smoothness (λ₁=1e-4) + workspace-crash termination
  (r_bound=10); DR over payload swing angle (±10%), mass & cable length (±50%). Nominal cable **0.5 m**,
  payload **70 g**, drone 305 g, ω_max=[15,15,5] rad/s.
- **Hook from moving platform** (ICRA 2025) — the catch model: success = **position + yaw + relative-velocity
  match** simultaneously within the **hook radius (~2.3 cm)**: `f_gr = ‖r_L−r_p‖ + ‖ψ−ψ_r‖ + ‖ṙ_L−v_r‖ = 0`.
  Relative velocity is an explicit tunable knob (gain ζ: "more aggressive grasping"). Rigid 2-DoF rod,
  viscous joint friction b=0.29. Drone 0.605 kg, rod **0.40 m**, hook 0.01 kg, hook radius ~0.02–0.023 m.
- **Autonomous hook grasping** (TCST 2025) — 1-DoF rigid pole, success = `‖r_b^L‖ ≤ 0.5·d_h` (hook radius);
  5-segment traj-opt + geometric SE(3) + LQR swing damping. Rod 0.40 m, hook Ø0.04 m.
- **Model-based meta-RL** (RA-L 2021) — least relevant (learned dynamics, 4 Hz, external camera); only
  contributes realistic small payload (10–15 g) / short cable (18–30 cm) ranges.

**Novelty vs papers:** all three hook papers use a **rigid rod** and **never model hook slip / cable
slack**. Our flexible weighted wire + probabilistic catch + slack-detach (rules 1–4) is a genuine
extension. The catch-success + relative-velocity structure from ICRA-2025 is what we translate into a
**probabilistic** surrogate.

## 3. The four physical requirements → how each is modeled

1. **Small weight at wire bottom** → an end-effector rigid body (hook+weight lumped) with mass `m_e`
   (~0.02–0.05 kg) at the tip of the multi-link wire chain. Gives pendulum momentum for swing-up.
2. **Hook not guaranteed to attach on contact** → attachment is **stochastic**: on a valid contact we
   sample `Bernoulli(p_catch)`.
3. **Relative velocity governs `p_catch`** → `p_catch` is highest for hook motion **parallel-up** along the
   wall, lowest **parallel-down**, scaled by contact quality:
   `p_catch = p0 · g_dir(v_rel) · g_prox(d) · g_normal(v_rel)`, where `g_dir` maps the angle between hook
   relative-velocity and the wall's up-tangent from **down → p_min** to **up → p_max** (e.g.
   `g_dir = clip((cosθ_up+1)/2, p_min, p_max)` form), `g_prox` gates on hook–bar distance < capture radius,
   `g_normal` requires some into-wall approach. (Exact form tuned; unit-tested in M3.)
4. **Slack → hook falls off** → while attached, track wire **tension**; if the wire goes slack (drone
   drifts closer than wire length so tension ≤ 0) for longer than a small dwell, force **detach**. This is
   the signal that makes the policy learn to "keep pulling."

## 4. Attachment mechanism — proposed implementation (**surrogate**, see §7 Q1)

Per-env state: `attached∈{0,1}`, `attach_point_w` (world pos on the bar), `attach_bar_id`, `slack_timer`.

- **Detect (while not attached):** nearest bar segment to hook tip; compute hook relative velocity vs bar,
  decomposed into wall-up-tangent / along-bar / wall-normal. If within capture radius → evaluate `p_catch`
  (§3.3) → Bernoulli sample. On success: `attached=1`, snap `attach_point_w` to the bar contact point.
- **Hold (while attached):** pin the hook tip to `attach_point_w` with a stiff PD spring-damper via
  `set_external_force_and_torque` on the end-effector body. The existing wire chain then transmits tension
  to the drone naturally (no runtime joint creation needed → fully GPU-parallel).
- **Release:** tension = magnitude of the pin's pull along the wire; when the wire–anchor distance < natural
  wire length (slack) → tension≈0 → increment `slack_timer`; if it exceeds dwell → `attached=0`, hook drops.

**Alternatives (rejected unless user prefers):** (B) real rigid hook geometry catching on real bars via
PhysX contact — most realistic but hard to make probabilistic and slow/unstable at scale; (C) runtime
dynamic PhysX D6 joint on catch — awkward per-env at scale in Isaac Lab.

## 5. Env interface (proposed)

- **Action** — reuse `ACCBR`: `[lin_accel(3), body_rates(2)]` (yaw held), ~100 Hz. (FLARE-style CTBR is an
  option; ACCBR keeps parity with the existing controller stack.)
- **Observation** (compact, FLARE-inspired, + history buffer):
  drone lin-vel(3), rotation matrix(9), body rates(3); wire swing angles φ,θ in body frame + hook velocity;
  attachment flag + tension/slack indicator; **target bar** relative position + wall up-direction; previous
  action.
- **Reward** (staged; weights ∝ step_dt, same logging pattern as hover):
  - approach: progress toward a pre-catch pose in front of the target wall (FLARE progress term).
  - swing: reward hook kinetic energy / up-parallel velocity as it nears the bar.
  - **catch bonus**: large reward on `attached` 0→1 transition.
  - **hold**: while attached, reward maintained tension; penalize slack; big penalty on detach.
  - regularizers: action smoothness, body-rate, effort, swing-limit safety (avoid wire-in-props).
- **Terminations:** crash (drone too low / hits basket / out of bounds), wire–prop entanglement (swing
  limit), timeout, success (attached & held ≥ T_hold). Detach-after-catch behavior: see §7 Q4.

## 6. Milestones & testable goals

- **M0 — Asset & physics sanity.** New `hook_grasp` task dir + USD (drone + weighted wire chain + static
  basket with bars). *Test:* run N envs with hover action; wire hangs & swings like a pendulum, no NaNs,
  contact sensors report wall hits.
- **M1 — Approach.** Stage-A reward only. *Test:* policy reaches a pre-catch pose in front of a target bar
  within tolerance.
- **M2 — Swing-up.** Stage-B reward. *Test:* measurable rate of hook reaching bar height with up-parallel
  relative velocity.
- **M3 — Attachment model (unit-tested in isolation).** *Test:* scripted hook velocities (up / down / normal
  / tangential) → measured catch rates reproduce the intended monotonic ordering (up > sideways > down) and
  slack → detach fires correctly.
- **M4 — Full task.** Approach→swing→catch→hold end-to-end. *Test:* success rate (attached & held ≥ T) over
  N envs on the existing skrl/PPO (or MARL) stack.
- **M5 — Robustness / DR.** Randomize wire length, end mass, bar height, basket pose. *Test:* success rate
  under DR.

## 7. The larger vision (user's ultimate goal)

The single-drone hook-grasp is a **building block** of a multi-agent, multi-phase, meta-optimized pipeline:

> 3 *loose* drones **approach** a basket → **attach** one-by-one at *random* locations on the basket wall →
> cooperatively **transport** the basket to a target pose → **release** the hooks → move to the **next
> basket** → repeat. The attach skill and the transport skill are **co-optimized in tandem** via a MAML-like
> scheme or an overarching evolution.

Mapping to what exists / what is new:

| Phase | Status |
|---|---|
| Approach | part of the new attach env |
| **Attach** (swing hook onto wall bar, hold tension) | **this new env** — the hard, novel skill |
| Transport (carry basket to target pose+ori) | **already exists** = the `hover` transport MARL task |
| Release (slide hook off) | small extension of the attach env (reverse of catch) |
| Sequencing + co-adaptation | a **meta-layer** over the shared policy/genome |

### Architecture recommendation (resolves old Q2)

- Build the new env as **`DirectMARLEnv` parameterized by `num_drones`**, homogeneous agents, sharing the
  observation/action/reward idioms of the existing transport task. **Develop & validate with `num_drones=1`
  first** (isolated swing→catch→hold + surrogate attachment), then flip to 3 with **sequential per-agent
  attachment** — no architecture rewrite. This plugs straight into the Hebbian/CMA-ES shared-genome stack.
- **Meta-layer:** start with **evolution/CMA-ES over the shared Hebbian genome** (already in the stack;
  handles the non-differentiable catch + phase sequencing); keep **MAML** as a later comparison. (Not needed
  to build the attach env.)
- Consequence for §5: observation carries the **basket target pose** (random per episode, observed — like
  the transport task's `pose_command_w`) and, later, a per-agent "assigned wall region / attach slot".

## 8. Resolved design decisions

1. **Attachment mechanism** → **surrogate probabilistic model** (§4). ✓ Extended: `p_max=0.7` +
   per-attempt `catch_latch` (retry-after-failure); angle-dependent detach fused with the slack dwell.
2. **Env base** → **`DirectMARLEnv`, `num_drones`-parameterized, single-drone first** (§7). ✓ Confirmed & built.
3. **Basket** → **random pose per episode, provided in the observation**, static within an episode. ✓
4. **Detach after catch** → **continue + penalty, allow re-catch**; success = attached & held ≥ T_hold. ✓
5. **Success metric** → **eval SUCCESS RATE at the true 0.05 m radius (curriculum-OFF)**, NOT `attached_frac`
   (structurally capped by success-termination). Curricula (capture-radius, hold-time) are training-only. ✓
6. **Target assignment** → two modes, same env: **any-wall** (`_nearest_wall_point`, hook may catch on any
   wall) and **assigned-wall** (per-episode random wall; obs carries its centre relative to basket CoM;
   reward+catch restricted to it) — the assigned-wall mode is the primitive for the 3-drone attach phase. ✓
7. **Actor observation** → currently the **hover-parity 45-dim frame with NO hook state** (H=1, no history);
   only the CTDE critic sees the hook. **Known bottleneck** — adding a hook→wall error to the actor is the
   top open lever (not yet done). ⚠

## 10. M0 implementation manifest (concrete)

New package `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/hook_grasp/`:

- `attachment.py` — **surrogate catch/slack model** (pure torch, no Isaac dep → unit-testable now). Novel
  core. Contains `HookAttachmentCfg` + `compute_catch_probability(...)` + `update_attachment_state(...)`.
- `marl_hook_env_cfg.py` — `MARLHookGraspEnvCfg(DirectMARLEnvCfg)`; `num_drones` param (default 1); basket
  random-pose command + abstract bar target params; ACCBR action; obs/reward weights; attachment cfg.
- `marl_hook_env.py` — `MARLHookGraspEnv(DirectMARLEnv)`; `_setup_scene` builds drone + **procedural wire
  chain** (spherical-joint links + weight+hook tip body, pxr in `_setup_scene`, no prebuilt USD) + static
  basket with abstract bar lines; obs/reward/dones/reset; integrates `attachment.py`.
- `__init__.py` — gym-register `Isaac-flycrane-hook-grasp-v0`.
- `agents/` — copy skrl PPO/MAPPO + `hebbian_cfg.yaml` from `hover_rand`.

**Asset strategy:** build the wire procedurally in `_setup_scene` (chain of small rigid links + spherical
joints below the drone, ending in a lumped hook+weight body), so wire length / #segments / end-mass are
config-driven and no binary-USD authoring is needed. The drone (`FALCON_CFG`) + wire form one articulation
so `robot.data.body_*_w` exposes the hook state. Basket = static rigid body at the observed random pose;
bars = parametric horizontal line segments on the 4 walls (known to `attachment.py`), plus a light visual
mesh. **Requires an Isaac Sim session to build/validate** (pxr absent from the plain venv).

**M3 down-payment (doable now, no Isaac):** unit-test `attachment.py` in the venv — scripted hook velocities
must reproduce catch-rate ordering up > sideways > down, proximity gate, and slack→detach.

## 9. Status log

- 2026-07-07: Parsed existing hover/hover_rand env, flycrane_rod asset, ACCBR control cascade; parsed 4
  reference papers. Drafted plan; got user answers (surrogate catch; random observed basket pose; continue-on-
  detach; revealed full multi-agent pipeline vision). Updated plan with the larger vision + architecture
  recommendation.
- 2026-07-07: User confirmed build path (DirectMARLEnv, num_drones-param, 1-drone first) and abstract bar
  targets. Implemented `hook_grasp/attachment.py` (surrogate catch/slack model) + `tests/test_attachment.py`.
  **All 6 unit tests pass on GPU** (up>side>down ordering p=0.90/0.46/0.02, proximity gate, approach gate,
  Monte-Carlo catch rates, slack->detach dwell, taut-stays-attached). M3 core down-payment done.
  M3 core down-payment done.
- 2026-07-07: Built the full M0 env (code complete, `py_compile` clean): `assets/flyhook.py` (`FLYHOOK_CFG`
  + body-name contract), `hook_grasp/marl_hook_env_cfg.py`, `hook_grasp/marl_hook_env.py`, `__init__.py`,
  registered in `directMARL/__init__.py` + `assets/__init__.py`. Basket is analytic (visual marker only, no
  static collider); attachment surrogate wired in; hook held by a PD-pin force. Minimalist `checklist.md`
  added. **Next blocker: author `data/AMR/flyhook/flyhook.usd` via `scripts/hook_grasp/build_flyhook_asset.py`
  (needs an Isaac Sim session), then run M0 sanity.** See checklist "Open verification risks".
- 2026-07-07: Wrote `scripts/hook_grasp/build_flyhook_asset.py` — extends the real `falcon.urdf` (drone
  mass/inertia/rotors unchanged) with an 8-link wire (revolute-x/y spherical decomposition, `flycrane_rod`
  pattern) + lumped `hook_link` sphere tip. **Generated & XML-validated `flyhook.urdf`** (23 links/22 joints)
  in the plain venv. USD conversion pending Isaac: `build_flyhook_asset.py --convert`. Hook mesh deemed
  cosmetic-only (surrogate catch) — not blocking M0.
- 2026-07-07: **M0 PASSES.** Converted URDF→USD in Isaac; wrote `scripts/hook_grasp/play_sanity.py`; debugged
  3 issues (configclass comprehension scope, missing `ang_acc` in control state, Isaac teardown hang →
  file-write + `os._exit`). Sanity (200 steps, 1 env): 0 NaNs, hook hangs at −0.572 m (taut 0.55 m), wire
  swings, attachment plumbing runs, clean exit. **Rendered** the env (drone + hanging wire + hook + bar
  marker) → `scripts/hook_grasp/_render/hook_grasp_env.png`. Next: M1 (approach reward) — train a policy so
  the drone flies the hook to the target bar.
- 2026-07-07: Render/physics validated per user review. Added `--maneuver` (oscillating lateral accel +
  P altitude-hold) + attitude logging to `play_sanity.py`. Findings: (1) drone is **UPRIGHT** (roll/pitch
  <4°) — not flipped; (2) wire hangs & swings correctly (zero-action=hover is why it first looked static);
  (3) the flycrane controller's `acc_load` (carry-task) term cancels scripted lateral accel (5 m/s² cmd →
  3.9° tilt), so open-loop maneuvers barely swing — a trained policy is the right demo; (4) minor reset
  transient collapses the wire ~1 step (fix in `_reset_idx`). Hebbian `_overall_best` checkpoint is
  hover-task (obs 45, hist 1) — incompatible with hook obs (126); can't drive this env. See checklist
  "Findings". Proceed to M1.
- 2026-07-07: **★ Found & fixed a CRITICAL control bug — the M0 drone was UNPOWERED.** The env issued **3
  separate** `set_external_force_and_torque` calls; Isaac (`articulation.py:821`) recomputes
  `has_external_wrench = forces.any() or torques.any()` **per call and the last wins**, so the hook pin-force
  call (all-zeros while unattached) reset the flag to False and wiped the rotor-thrust wrench — the drone
  never actually tilted. A `_dbg_ctrl` trace proved the controller commanded correct tilt (qcmd 37.8° +
  differential thrusts) but no wrench was applied. **FIX:** merge all wrenches into ONE call over
  `self._ext_body_ids` (`combined_forces`/`combined_torques`). After fix, `play_sanity.py --maneuver
  --impulse --accel 6.0`: pitch **+26.6°**, body up [−0.002,+0.002,+1.000], **wire swing 74.6°** (was 22°),
  0 NaNs, reset clean (hook rel-z −0.60) → **RESULT: PASS**, montage confirms banked quad + swinging taut
  wire. Also resolved the user's two questions: **acc_load** — added `load_comp_gain` to `geometric.py`
  (default 1.0, hover unchanged) gated to `attached` so the drone perceives **zero load until it hooks on**
  (correct design, though NOT the maneuver bottleneck — the wrench bug was); **reset** — confirmed FINE
  (clean −0.60 hang, all terminations timeouts). The earlier "M0 PASS" only checked wire-hang + NaNs on the
  unpowered drone. **M0 now genuinely PASSES with a controllable drone. Proceed to M1 (approach reward +
  train a PPO policy).**
- 2026-07-08/09: **M1–M4 reached — full approach→catch→hold trained end-to-end via skrl-PPO.** Reward fully
  **redesigned** (see `_get_rewards`): approach = **hook→nearest-wall** telescoping progress (gated to the
  unattached phase); proximity `exp(-2·d)`; **impact_perp** penalizes only the *excess* into-wall speed over a
  soft-contact target (squared); **impact_up** rewards up-the-wall velocity; **lock** + **tension** while
  attached; one-off **catch** (40) and terminal **success** (50) bonuses; **detach** penalty (5). Target is the
  **whole basket** (any of the 4 walls), computed each step from the observed basket pose.
- 2026-07-08: **★ Key metric insight — judge by SUCCESS RATE, not `attached_frac`.** `attached_frac` is
  structurally capped low (~0.15–0.20) because success **terminates** the episode. Built
  `scripts/hook_grasp/eval_success_rate.py` (batched, curriculum-OFF, true 0.05 m radius, per-episode
  success/catch/retry stats) — the honest metric. A "plateau" at `attached_frac≈0.2` was actually ~0.98
  success under the training curriculum.
- 2026-07-08/09: **Attachment refinements.** (1) `p_max` 0.9→**0.7** + a **per-attempt `catch_latch`** (one
  Bernoulli draw per approach, reset on leaving the capture zone) → forces **retry-after-failure** instead of
  the per-step draw compounding to a near-certain catch. (2) **Capture-radius curriculum** (anneal 0.15→0.05
  over 6000 control steps) to bootstrap a dense catch signal. (3) **Angle-dependent detach** on tension release
  (**vertical→100%**, horizontal→20%, interp by |cos∠-from-vertical|) **in conjunction with the slack dwell**
  (brief slack tolerated `slack_dwell_steps`, then the angle roll fires once) — models a J-hook that slides off
  vertically but can rest on the bar at an angle.
- 2026-07-09: **Wall-collision surrogate + observability finding.** Walls are visual-only (no collider), so a
  spring-damper `_wall_collision_force` keeps the hook out of the footprint (top open so it can be swung over
  the rim). Measured: a *soft* force **leaks** (max poke-through ~230–270 mm in ~0.1% of steps) — a fundamental
  limit; kept the stiff surrogate (`check_wall_penetration.py` quantifies it). **Observation is the real
  bottleneck:** the actor obs is the hover-parity 45-dim frame with **no hook state** (only the CTDE critic
  sees the hook) → catching is hard to learn. Flagged as the top lever (add hook→wall error to the actor).
- 2026-07-10: **Result baseline (any-wall, blind obs).** Best checkpoint (~746k) eval at the true 0.05 m
  radius: **≈37% success / 60% catch** (2021 eps). Two failure modes: ~40% never catch (approach precision),
  and of catches ~38% slip before the hold. Training **peaked ~92%** (curriculum-inflated) around 746k then
  **collapsed to ~5% by 2M** — classic PPO **entropy/std collapse** (entropy loss drifted toward 0). Fix
  direction: stop early and/or regularize entropy.
- 2026-07-11: **★ Assigned-wall task variant (divide-the-walls).** Each drone is assigned a **random wall per
  episode**. The ego **one-hot id in the obs is replaced by the assigned-wall centre** expressed **relative to
  the basket CoM** (`_assigned_wall_center`), and **both the approach reward and the catch are restricted to
  that one wall** (`_assigned_wall_point`; `_nearest_wall_point` kept for diagnostics only). This is the direct
  primitive for the 3-drone attach pipeline (one wall each). Runtime-verified (obs dim unchanged at 45).
- 2026-07-11: **Training-config overhaul** (anti-collapse + faster) and switch to **Weights & Biases**:
  `_HISTORY_LEN 3→1` (**no history**, obs 135→45); `mini_batches 1→32`; **cosine LR** 1e-3→1e-5
  (`CosineAnnealingLR`, `train.py` auto-sizes `T_max` = updates×epochs from the run length);
  `entropy_loss_scale 0.005→0.01`; `min_log_std −20→−1.5` (std floor ~0.22 to block the collapse); a
  **hold-time curriculum** (required hold ramps 10→50 steps over 8000 control steps, logged as
  `Metrics/hold_req`, disabled for eval); `experiment.wandb: True` (project `hook-grasp-flycrane`). Eval/play
  scripts also pin `hold_curriculum=False`. New **750k-timestep-capped** run live (stops before the ~746k
  collapse zone).
