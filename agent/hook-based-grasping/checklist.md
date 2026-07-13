# Hook-Grasp — Implementation Checklist

Minimalist tracker. See `workplan.md` for rationale. Paths relative to repo root.
Package root: `exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/hook_grasp/`

## M3 core — surrogate attachment (DONE ✅)
- [x] `hook_grasp/attachment.py` — `HookAttachmentCfg`, `compute_catch_probability`, `update_attachment_state`, `is_slack`
- [x] `hook_grasp/tests/test_attachment.py` — 6 GPU tests pass (`.venv/bin/python .../tests/test_attachment.py`)
  - up>side>down `p=0.90/0.46/0.02` · proximity gate · approach gate · MC rates · slack→detach · taut-stays

## M0 — env scaffolding (code DONE ✅, needs Isaac to run)
- [x] `hook_grasp/marl_hook_env_cfg.py` — `MARLHookGraspEnvCfg` (num_drones=1, ACCBR=5, **obs 45 (H=1, no history)**, **state 59** privileged, basket/bar params, reward weights, embeds `HookAttachmentCfg`; capture + hold curricula)
- [x] `hook_grasp/marl_hook_env.py` — `MARLHookGraspEnv` (all methods: control cascade reuse, hook PD-pin `_compute_hook_pin_force`, `_update_attachment`, obs/state, staged rewards, dones, reset+`_reset_target_pose`, debug vis)
- [x] `hook_grasp/__init__.py` — gym-register `Isaac-flycrane-hook-grasp-v0`
- [x] `hook_grasp/agents/` — copied skrl_ppo / skrl_mappo / hebbian cfg
- [x] `tasks/directMARL/__init__.py:5` — `from .hook_grasp import *`
- [x] all files `py_compile` clean

## Asset
- [x] `assets/flyhook.py` — `FLYHOOK_CFG` + body-name constants (contract for the builder)
- [x] register in `assets/__init__.py:4`
- [x] `scripts/hook_grasp/build_flyhook_asset.py` — extends real `falcon.urdf` (drone mass/inertia/rotors kept) + N-link wire (revolute-x/y spherical decomposition, `flycrane_rod` pattern) + lumped `hook_link` tip
- [x] **URDF generated & XML-validated** → `data/AMR/falcon_data/flyhook.urdf` (23 links / 22 joints, 8 wire links, 16 DoFs, hook_link, drone bodies preserved) — ran in plain venv
- [x] **convert URDF→USD (Isaac):** `python scripts/hook_grasp/build_flyhook_asset.py --convert`
  (wraps `convert_urdf.py`, no `--merge-joints`) → `data/AMR/flyhook/flyhook.usd`
- [x] **graft working drone body visual (Isaac):** `python scripts/hook_grasp/fix_drone_visual.py --headless`
  — the current importer can't convert `falcon_without_rotors.dae` (dropped nodes → broken/dark body,
  *looks* flipped though physics up-axis=[0,0,+1]). Reuses the committed `falcon.usd` body visual (older
  importer, renders fine): deactivates the broken visual, adds an override referencing falcon.usd's
  `visuals` scope. **0 unresolved warnings after**; drone body now renders as a proper upright quad.
  Pipeline is 2 steps: `--convert` then `fix_drone_visual.py`.

## Note
- 4-prong hook mesh is **cosmetic only** (catch is surrogate); `hook_link` is a sphere collider now, swap a visual mesh later if desired. Do NOT block M0 on hook CAD.

## Open verification risks (flag on first Isaac run)
- hook PD-pin force frame: `_compute_hook_pin_force` sends force in hook body frame (mirrors rotor-thrust convention) — verify sign/magnitude holds the hook without blowing up
- ~~`set_external_force_and_torque` called 3× (falcon moments, rotor forces, hook force) — confirm they coexist~~ **RESOLVED — this WAS the bug (see Findings).** Now ONE combined call.
- obs dim (**45**, H=1) / state dim (**59**) must match `_drone_obs_frame` / `_get_states` exactly

## PPO run (M1, reward-shaped) — 2026-07-07
- **state_space bug (FIXED):** `_STATE_PER_DRONE` was 37 but `_get_states()` emits **35** (it omits the
  φ,θ swing angles that are in the obs frame). Decentralized Hebbian never used the state; PPO's centralized
  critic does → `RuntimeError: shape '[-1,37]' invalid for size 16*35`. Fixed constant to 35.
- **reward shaping (NEW):** added a dense **progress** term = `progress_weight*(prev_dist-cur_dist)` per step
  (clamped ±0.1 m, tames reset spikes; NOT dt-scaled — telescoping delta). This is the fix for the Hebbian
  plateau (flat `exp(-1.5d)` gave no gradient at >1 m). Also widened approach decay to `exp(-1.0d)`, bumped
  swing 0.5→1.0 / catch 50→80 / hold 2→3, swing active under `swing_near_range=0.3` m. See `_get_rewards`.
- **skrl PPO:** `--algorithm PPO` (uppercase; lowercased internally) → single-agent wrap of the DirectMARLEnv.
  cfg `skrl_ppo_cfg.yaml` (experiment dir → `ppo_flycrane_hook`). Smoke (16 envs, 1920 ts) PASSES clean.
- **video after training:** patched `scripts/skrl/play.py` to make `DirectMARLPlotter` optional (it hard-needs
  a `load_odometry_sensor_link` body + `pose_command_w` → crashes on hook env at instantiation). Wrapper
  `scripts/hook_grasp/run_ppo_then_video.sh` = timeout-capped train → newest ckpt → `play.py --video` → copy
  to `_render/ppo_policy.mp4`. Params: `TRAIN_SECONDS` (default 39600=11h), `NUM_ENVS` (default 128).

## Milestones (tests)
- [x] **M0 sanity — PASS ✅ (now with a CONTROLLABLE drone)** `scripts/hook_grasp/play_sanity.py`. After the
  critical wrench-bug fix: `--maneuver --impulse --accel 6.0` → drone pitch **+26.6°**, body up-axis
  [−0.002,+0.002,+1.000] (upright), **wire swing 74.6°**, 0 NaNs, reset probe clean (hook rel-z −0.60),
  RESULT: PASS. Montage `scripts/hook_grasp/_render/montage.png` shows the quad banking + wire swinging &
  hanging taut. (The prior M0 PASS was on an unpowered drone — see ★ finding.)
- [x] **M1 approach · M2 swing-up · M4 full task — PASS ✅** (trained end-to-end via PPO; hook→wall approach,
  catch, and hold-to-success all learned). Honest eval ≈**37% success / 60% catch** at true 0.05 m radius.
- [ ] **M5 DR** — hook-mass DR is on (`randomize_hook_mass`); wire-length / bar-height / basket-pose DR TBD.

## Findings from render/physics validation
- **★ CRITICAL BUG (FIXED) — drone was uncontrollable, thrust silently cancelled.** The env made **3 separate**
  `set_external_force_and_torque` calls (falcon moments, rotor forces, hook pin-force). Isaac's
  `articulation.py:821` sets `has_external_wrench = forces.any() or torques.any()` **per call**, and the LAST
  call wins. When the hook was unattached its pin-force was all-zeros → `has_external_wrench=False` → the whole
  external-wrench buffer (incl. rotor thrusts) was cleared. Net: the drone never actually tilted/flew even
  though the controller commanded correct tilt+differential-thrust. **FIX:** combine all wrenches into ONE call
  (`combined_forces`/`combined_torques` over `self._ext_body_ids`). After fix: pitch 26.6°, x accelerates,
  **wire swing 74.6°** (was 22°), 0 NaNs → `play_sanity.py --maneuver --impulse` returns PASS. The earlier
  M0 "PASS" was on an unpowered drone — it only checked wire-hang + NaNs, not real control.
- **Drone body "flip" = broken visual mesh, NOT dynamics.** Proven: body up-axis=[0,0,+1], roll/pitch 0°.
  Root cause: current Isaac URDF importer drops nodes of `falcon_without_rotors.dae`. FIXED by grafting the
  committed `falcon.usd` body visual (`fix_drone_visual.py`) — drone now renders as a proper upright quad.
- **acc_load — question resolved.** The geometric controller's `acc_load` (carry-task load feed-forward) was
  suspected of resisting maneuvers, but a debug trace showed it was NOT the bottleneck (the wrench bug was).
  Still, gating it is the correct design for the free-pendulum phase: added `load_comp_gain` to
  `geometric.py` (default 1.0 → hover unchanged) and set it per-env to `attached[:, i]` in the env, so the
  drone perceives **zero load until the hook attaches**. Answers the user's "set perceived load 0 until
  attachment?" — yes, done.
- **Reset — question resolved, reset itself is FINE.** Post-reset hook rel-z = −0.60 (clean taut hang),
  terminations are all timeouts, no spurious entanglement on reset. The transient `hook rel-z ≈ −0.09` seen
  earlier is an in-*flight* swing artifact, not a reset bug. (Optional polish: explicitly reset wire
  link/joint states in `_reset_idx` — noted, non-blocking.)

## Debug log (M0 bring-up)
- URDF→USD via `--convert` OK (SIGABRT is Isaac's teardown, *after* files written; harmless).
- Fixed: dict-comprehension in configclass body can't see class attrs → hoisted dims to module-level `_*`.
- Fixed: control cascade needs `ang_acc` in the drone-state dict (was omitted).
- Isaac app teardown hangs → `play_sanity.py` writes `summary.txt` + `os._exit()` to capture results/render.
- Cosmetic only: `falcon_without_rotors.dae` visual ref unresolved after convert (drone body renders partially;
  rotors/wire/hook fine). Not blocking; revisit if a clean drone mesh is wanted.
- ★ Fixed the `has_external_wrench` bug (3 calls → 1 combined). Traced via `_dbg_ctrl` capture: controller
  commanded qcmd_tilt 37.8° + differential thrusts, but wrench never applied until the calls were merged.

## Reward redesign (2026-07-09) — DONE ✅
- Full `_get_rewards` rewrite. Terms (dt-scaled unless noted): **approach** = hook→nearest-wall telescoping
  progress (×`not_att`, clamp ±0.1 m, not dt-scaled); **proximity** `3·exp(-2·d)·not_att`; **impact_perp**
  `-5·clamp(into_speed−0.3, 0)²·in_zone` (penalize only *excess* into-wall speed → soft contact optimal);
  **impact_up** `+8·v_up·in_zone`; **lock** `6·attached`; **tension** `5·clamp(tension/30,0,1)·attached`;
  one-off **catch** `40`, terminal **success** `50·succeeded`, **detach** `-5`; + control regularizers.
- Target = the **whole basket** (any of 4 walls) in any-wall mode; the **assigned wall** in assigned mode.

## Attachment refinements — DONE ✅
- `p_max` 0.9→**0.7** + **per-attempt `catch_latch`** (one draw per approach; reset on leaving the zone) →
  **retry-after-failure**. Unit-tested.
- **Capture-radius curriculum** (0.15→0.05 over 6000 ctrl steps; `capture_curriculum`, eval pins it off).
- **Angle-dependent detach fused with the slack dwell**: brief slack tolerated `slack_dwell_steps`, then one
  angle-weighted roll (**vertical→1.0**, horizontal→0.2, interp |cos∠-from-vertical|). `test_attachment.py`
  updated → **6/6 pass** (grace = no detach, then vertical 1.000 / horizontal ~0.20).

## Assigned-wall variant (2026-07-11) — DONE ✅
- Per-episode **random wall assignment** (`_assigned_wall`, sampled in `_reset_target_pose`).
- Obs **one-hot id → assigned-wall centre relative to basket CoM** (`_assigned_wall_center`); dims unchanged (45).
- **Approach reward + catch restricted to the assigned wall** (`_assigned_wall_point`, `_wall_point_by_index`;
  `_nearest_wall_point` kept for diagnostics/metrics). Headless smoke-verified (all 4 walls sampled; catch
  restricted). Primitive for the 3-drone attach phase.

## Training-config overhaul (2026-07-11) — DONE ✅
- Env cfg: `_HISTORY_LEN 3→1` (**no history**, obs 135→45); **hold-time curriculum** (10→50 over 8000 ctrl
  steps, `Metrics/hold_req`, eval pins it off).
- `skrl_ppo_cfg.yaml`: `mini_batches 1→32`; `learning_rate_scheduler: CosineAnnealingLR` (1e-3→1e-5);
  `entropy_loss_scale 0.005→0.01`; `min_log_std −20→−1.5` (anti-collapse std floor); `experiment.wandb: True`
  (project `hook-grasp-flycrane`).
- `scripts/skrl/train.py`: resolves the cosine class + **auto-sizes `T_max`** = (timesteps/rollouts)×epochs.
- Motivation: the 2M any-wall run **collapsed after ~746k** (entropy/std collapse: 92%→5%). Now **capped at
  750k** and regularized. Eval/play scripts pin `capture_curriculum=False` **and** `hold_curriculum=False`.

## Tooling added
- `scripts/hook_grasp/eval_success_rate.py` — batched curriculum-OFF success/catch/retry eval (the honest metric).
- `scripts/hook_grasp/check_wall_penetration.py` — max/settled hook penetration into the wall footprint.
- `scripts/hook_grasp/play_ppo_video.py` — reliable manual-render video of a trained PPO policy.

## Top open levers (not yet done)
- **Actor observability** — obs has NO hook state (only the CTDE critic does); adding a hook→wall error vector
  to the actor is the highest-expected-gain change for the ~40% no-catch failures.
- **Multi-drone** — flip `num_drones` to 3 with per-drone assigned walls + a cooperative term.
- **Physics validation** — validate the final policy with real hook/wall colliders (surrogate leaks).
