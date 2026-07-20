# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the hook-grasping task: a single drone swings a weighted wire + hook onto a basket wall bar.

Modeled on ``hover_rand/marl_hover_env_cfg.py`` (same control cascade, sim, logging idioms). Differences:
    * single drone (``num_drones=1``, but the class is written to generalize),
    * a flexible weighted **wire + hook** instead of the pre-attached rope (asset ``FLYHOOK_CFG``),
    * a **basket** at a random-but-observed pose with abstract horizontal **bar targets** on its walls,
    * a **surrogate probabilistic attachment** model (``HookAttachmentCfg``) with slack-detach.
"""

from __future__ import annotations

import math

from MARL_mav_carry_ext.assets.flyhook import (
    FLYHOOK_CFG,
    FLYHOOK_DRONE_BODY,
    FLYHOOK_HOOK_BODY,
    FLYHOOK_ROTOR_BODY,
    FLYHOOK_WIRE_BODY,
)

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .attachment import HookAttachmentCfg

# Module-level constants: referenced inside the class-body dict comprehensions below. (A comprehension in
# a class body runs in its own scope that sees module globals but NOT sibling class attributes, so these
# cannot live on the class itself.)
_NUM_DRONES = 1
_ACTION_DIM = 5                      # ACCBR: [lin_acc(3), body_rates(2)]
_HISTORY_LEN = 3                     # observation history frames (back to 3 to test whether H=1 hurt)
# Observation is IDENTICAL to the hover/transport partial obs (basket = "load", pose_command = "goal"), so a
# shared genome transfers across the attach + transport phases. Per frame (see _drone_obs_frame):
#   load(basket) pos (3) + rot matrix (9)                                          = 12
#   ego one-hot (3) + drone pos (3) + rot (9) + lin vel (3) + ang vel (3)          = 21
#   goal_pos_error (3) + difference_matrix (9)                                     = 12
_OBS_PER_FRAME = 12 + 21 + 12             # = 45 (matches the hover partial observation)
# Centralized (PRIVILEGED) critic state (see _get_states): load(basket) full state (18) + per-drone
# state (18) + per-drone privileged hook block (12: hook rel/vel/->bar-error/attached/tension + hold_progress,
# the [0,1] fraction of the way to the success milestone so the critic can predict the ramping progress reward)
# + goal (12).
_STATE_DIM = 18 + (18 + 12) * _NUM_DRONES + 12   # = 60 for one drone
_POSSIBLE_AGENTS = [f"falcon{i + 1}" for i in range(_NUM_DRONES)]


@configclass
class EventCfg:
    """Reset / disturbance events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Start the drone at a random pose in a volume in front of / away from the basket. The wire+hook
    # hang below and start swinging from the reset transient.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.5, 1.5),
                "y": (-1.5, 1.5),
                "z": (1.0, 2.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")},
        },
    )

    # Hook+weight tip mass PINNED at 50 g -- deliberately NOT randomized (degenerate range = every env
    # identical). The hook is the pendulum bob, so its mass sets the swing period and the momentum delivered at
    # the bar; the old (0.02, 0.05) DR handed every env a different bob and, because an EA evaluates genome g
    # only on envs [g*n_repeats, (g+1)*n_repeats), each genome was permanently stuck with its own mass block --
    # a nuisance variable no number of repeats averages out. The wire length is likewise fixed (0.55 m, baked
    # into the flyhook USD; see attachment.wire_length), so the pendulum is now a fixed physical system.
    randomize_hook_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[FLYHOOK_HOOK_BODY]),
            "mass_distribution_params": (0.20, 0.20),   # kg: fixed 200 g (heavier bob -> stronger pendulum, resists creep/hang)
            "operation": "abs",
        },
    )


@configclass
class MARLHookGraspEnvCfg(DirectMARLEnvCfg):
    # --- control -------------------------------------------------------------------------
    control_mode = "ACCBR"  # ACCBR (accel + body rates) or geometric; reuses the flycrane cascade
    decimation = 3
    episode_length_s = 20

    # --- agents / spaces -----------------------------------------------------------------
    num_drones = _NUM_DRONES
    possible_agents = _POSSIBLE_AGENTS

    partial_obs = True
    history_len = _HISTORY_LEN

    # --- optional INDI disturbance-residual actor observation (real-drone-honest hook signal) ---------------
    # When ``actor_indi_residual`` is on, each ACTOR frame is extended by the 6-dim body-frame INDI disturbance
    # residual (angular INDI increment alpha_cmd-omega_dot + linear commanded-vs-measured specific force; see
    # IndiController.disturbance_residual). This is what the onboard IMU/INDI measures of the hook+wire wrench --
    # it carries the hook DYNAMICS (force/phase) but NO hook or wall POSITION, so the actor stays position-blind.
    # ``actor_history_len`` decouples the ACTOR frame-stack depth H from the critic's history_len (which stays
    # _HISTORY_LEN=3, keeping the privileged critic 180-dim): 0 -> fall back to history_len; else H frames for the
    # actor only. Set BOTH via rslrl_adapter.apply_indi_residual_cfg (it also recomputes observation_spaces).
    actor_indi_residual: bool = False
    actor_history_len: int = 0            # 0 -> use history_len; else the ACTOR-only frame-stack depth H

    # per-frame observation layout = the hover/transport partial obs (see _drone_obs_frame). basket = "load",
    # pose_command = "goal":
    #   load(basket) pos (3) + rot matrix (9)                                   = 12
    #   ego one-hot (3) + drone pos (3) + rot (9) + lin vel (3) + ang vel (3)   = 21
    #   goal_pos_error (3) + difference_matrix (9)                              = 12
    obs_dim_per_frame = _OBS_PER_FRAME
    action_dim = _ACTION_DIM

    action_spaces = {a: _ACTION_DIM for a in _POSSIBLE_AGENTS}
    observation_spaces = {a: _OBS_PER_FRAME * _HISTORY_LEN for a in _POSSIBLE_AGENTS}
    # centralized critic state (see _get_states): load(basket) full state + per-drone state + goal terms,
    # FRAME-STACKED over _HISTORY_LEN steps. state_as_observation feeds this to the policy+value nets, so the
    # stacking gives them memory of the approach->swing dynamics (a single frame can't see the swing phase).
    state_space = _STATE_DIM * _HISTORY_LEN

    # --- simulation ----------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=0.0033333333333333335,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.8066),
    )

    # --- robot ---------------------------------------------------------------------------
    robot_cfg = FLYHOOK_CFG.replace(prim_path="/World/envs/env_.*/flyhook")
    robot_cfg.spawn.activate_contact_sensors = True

    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/flyhook/.*", update_period=0.0, history_length=3, debug_vis=False
    )
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*")
    contact_sensor_threshold = 0.1

    # body-name regexes (contract with FLYHOOK_CFG)
    falcon_names = FLYHOOK_DRONE_BODY
    falcon_rotor_names = FLYHOOK_ROTOR_BODY
    wire_names = FLYHOOK_WIRE_BODY
    hook_name = FLYHOOK_HOOK_BODY

    # --- low-level control ---------------------------------------------------------------
    low_level_decimation: int = 1
    max_thrust_pp = 6.25  # N

    # --- basket + bar targets (analytic; basket is visualization-only, no static collider) ----------
    # Basket geometry in its own frame: vertical walls, half-extents (hx, hy), wall height range.
    # Box footprint matches the hover/transport payload (load_length_x = load_length_y = 0.275 -> a
    # 0.55 x 0.55 square), so the attach-phase basket is dimensionally the same box the transport task carries.
    basket_half_x: float = 0.275
    basket_half_y: float = 0.275
    basket_wall_z_lo: float = 0.02   # bars can sit from just above the base ...
    basket_wall_z_hi: float = 0.30   # ... up to the top of the wall (m, basket frame)
    # Random basket pose per episode (world). Placed a bit away so the drone must approach.
    basket_pose_range = {
        "pos_x": (-1.0, 1.0),
        "pos_y": (-1.0, 1.0),
        "pos_z": (0.6, 1.4),
        "yaw": (-math.pi, math.pi),   # basket only yaws (walls stay vertical)
    }
    # Half-length of a target bar segment along its wall (m). A wide bar = easier target.
    bar_half_len: float = 0.18
    reward_center_target: bool = True   # approach/proximity reward aims at each wall's CENTRE (designated,
    # MAPPO-safe, load-balanced) point; the CATCH still fires anywhere the hook contacts the bar (False = reward
    # follows the closest point, i.e. "aim anywhere on the wall").

    # --- transport goal pose (pose_command) ----------------------------------------------
    # Where the hooked basket should be carried to (observed like the hover task's pose_command_w). The full
    # task = approach basket -> hook onto a wall -> transport the basket to this goal. For the construction
    # demo the transport is SHORT, so the goal = the basket's start pose + a small random offset (keeps the
    # whole scene near the initial drone/basket region). Sampled per episode in _reset_target_pose.
    goal_offset_range = {
        "x": (-0.75, 0.75),
        "y": (-0.75, 0.75),
        "z": (-0.3, 0.5),
        "yaw": (-math.pi / 4, math.pi / 4),
    }
    goal_z_floor: float = 0.4   # never place the transport goal below this height (m)

    # --- surrogate attachment model ------------------------------------------------------
    attachment: HookAttachmentCfg = HookAttachmentCfg()

    # --- capture-radius curriculum -------------------------------------------------------
    # The true capture radius (attachment.capture_radius = 0.05 m) makes a catch razor-thin: with the
    # into-wall + up-sweep + Bernoulli gates, catches are rare and PPO plateaus at attached_frac ~0.20
    # (approach solved, but the catch/hold signal is too sparse to push past). Anneal the radius from a
    # LOOSE start to the true value over the first part of training: early on catches are frequent so the
    # policy learns approach->catch->hold densely, then the target tightens back to the real 0.05 m.
    # capture_radius_end is taken from attachment.capture_radius (the true target). EVAL-SAFE: set
    # capture_curriculum=False (play scripts do) so the radius is pinned at the true value -> no inflated eval.
    # NOTE: superseded by the SPAWN curriculum below (widening the catch target biases learning toward APPROACH
    # first and then makes the catch progressively HARDER; instead we spawn the drone AT the wall and back it off).
    # Left here (default OFF) as a toggle.
    capture_curriculum: bool = True
    capture_radius_start: float = 0.08        # loose bootstrap radius (m); anneals down to attachment.capture_radius
    capture_curriculum_steps: int = 120000    # control steps to anneal start -> end (~1:1 with trainer timesteps)

    # --- CHORD-ANGLE curriculum (physical-swing gate; see attachment.min_chord_angle_deg) ------------------
    # Anneal the min wire-from-vertical angle required for a catch from ``chord_angle_start_deg`` -> the target
    # ``attachment.min_chord_angle_deg`` over ``chord_angle_steps`` control steps. Start at 0 (any angle catches,
    # so the drone learns to reach the wall) then tighten so a near-vertical dangling hook can NO LONGER latch ->
    # forces a genuine swung-out pendulum up-swing. EVAL-SAFE: chord_angle_curriculum=False -> the gate is pinned
    # at attachment.min_chord_angle_deg (eval scripts disable it, so eval uses the full gate). Default OFF.
    chord_angle_curriculum: bool = False
    chord_angle_start_deg: float = 0.0
    chord_angle_steps: int = 60000

    # --- spawn curriculum: FIXED-MEAN GAUSSIAN distance, GROWING std (reverse curriculum) ------------------
    # The drone spawns in front of its assigned wall at a radial distance drawn from a GAUSSIAN whose MEAN is
    # FIXED at the catchable point-blank distance (spawn_dist_mu = 0.6 m) and whose STD grows slowly with the
    # curriculum (spawn_std_start -> spawn_std_end). Because the mode never moves off 0.6 m, there is ALWAYS a
    # strong fraction of easy point-blank spawns (the geometry a hook-blind policy can catch) -> no catastrophic
    # forgetting, while the widening tail gradually introduces harder far/near spawns. This fixes the hard-shell
    # failure where the ramping mean marched the drone out to 2.2 m and the catch collapsed. Distance is clamped
    # to [spawn_dist_min, spawn_dist_max] so nothing lands inside the basket footprint or absurdly far. The
    # azimuth still widens (0 -> +/-pi) with the curriculum. Progress is generation-driven for EA
    # (self._spawn_generation) and control-step-driven for PPO. EVAL-SAFE: spawn_curriculum=False -> full spread
    # (frac=1); eval scripts disable it.
    spawn_curriculum: bool = True
    # --- COMMON RANDOM NUMBERS (CRN) -- variance reduction for the EA/ES path only -------------------
    # An EA ranks genomes by a single scalar and has NO critic/baseline, so any per-episode randomness lands
    # straight in the ranking. The attached-spawn curriculum is the worst offender: it flips an independent coin
    # per episode (frac0=0.5), and an attached episode banks the success bonus almost for free -> a genome's
    # fitness largely reflects how many free spawns it drew, not how good it is. (MAPPO is immune: its critic
    # sees `attached` in the privileged state and the advantage baselines the windfall away.)
    # With crn=True every genome in a generation faces an IDENTICAL set of episode conditions: env e draws its
    # randomness from scenario (e % n_repeats) -- the trainer sets _crn_repeats -- so the luck becomes a shared
    # constant that CANCELS in the rank transform, leaving pure policy differences. The table is redrawn each
    # generation (common WITHIN, independent ACROSS), so the curriculum and the task distribution are untouched.
    # Requires num_envs == popsize * n_repeats with genome = env//n_repeats (the Hebbian trainer's layout).
    crn: bool = False
    crn_seed: int = 0                  # per-generation table seed: crn_seed*1000003 + generation
    spawn_dist_mu: float = 0.6         # m: FIXED mean radial distance from the wall (the catchable zone; mode
                                       # stays here forever so easy spawns are always present)
    # STRIPPED DIAGNOSTIC: std pinned tiny (fixed ~0.6 m point-blank), azimuth pinned tiny (always in-front).
    # (full-task values were 0.1 -> 0.6 std, azimuth up to pi.)
    spawn_std_start: float = 0.1       # m: initial (tight) std of the distance Gaussian
    spawn_std_end: float = 0.6         # m: final (wide) std -> reaches out to ~mu +/- 1.2 m at 2 sigma
    spawn_azimuth_max: float = math.pi # rad: cap on azimuth spread; full task = up to pi (all around the box)
    spawn_dist_min: float = 0.5        # m: clamp floor -> stays outside the basket footprint (nothing inside box)
    spawn_dist_max: float = 2.2        # m: clamp ceiling on the far Gaussian tail
    spawn_height: float = 0.7          # m above basket CoM so the dangling hook reaches the bar band
    spawn_pos_noise: float = 0.15      # m small extra xyz jitter on the spawn position
    spawn_curriculum_steps: int = 200000     # PPO: control steps over which std + azimuth grow (start -> end)
    spawn_curriculum_generations: int = 175  # EA: generations over which they grow (trainer sets _spawn_generation)
    # STRIPPED DIAGNOSTIC: fixed target wall (0 = +x). Set -1 to restore the random 4-wall assignment.
    fixed_wall: int = -1

    # --- reverse SWING-PHASE curriculum (physical up-sweep bootstrap) --------------------------------------
    # With the up-sweep-required catch (attachment.require_upsweep), the drone must PUMP the pendulum so the
    # hook sweeps UP onto the bar -- a hard-to-discover, energetic maneuver. To bootstrap it we give the drone
    # an initial velocity TOWARD its assigned wall at spawn: the drone carries momentum at the wall, then as it
    # arrests the hook (lagging on the wire) swings forward and UP onto the bar -- an ASSISTED up-sweep that
    # produces catches early so the policy gets the "up-sweep -> catch" signal. The assist DECAYS with the same
    # spawn-curriculum fraction (strong at frac=0 start -> 0 at frac=1 end) so by the end the drone must
    # generate the whole approach+pump itself. EVAL-SAFE: spawn_curriculum=False -> frac=1 -> zero assist.
    swing_assist_curriculum: bool = True
    swing_assist_v0: float = 1.3       # m/s: peak initial toward-wall drone speed at curriculum start (frac=0)
    swing_assist_up: float = 0.3       # fraction of v0 added as an initial UP velocity (helps seed the up-sweep)

    # reverse HOLD curriculum (spawn-attached): spawn a FRACTION of envs ALREADY ATTACHED at a near-steady-state
    # hold pose (drone hovering wire_length up-and-out from the bar, hook pinned to the bar), so the policy masters
    # the steady-state HOLD first -- the harder unmet target (hold ~0.55 -> 0.80 needed) -- decoupled from the
    # ceiling-bound catch. The attached fraction DECAYS with the curriculum (frac0 -> 0 over the steps below), so
    # training expands "start at the goal -> full catch-then-hold". A small pose jitter GROWS with the curriculum
    # so the hold is learned from an ever-wider basin. EVAL-SAFE: attached_spawn_curriculum=False -> 0 attached
    # spawns (honest catch-from-scratch).
    attached_spawn_curriculum: bool = True
    attached_spawn_frac0: float = 0.5    # fraction of envs spawned attached at curriculum start (step 0)
    attached_spawn_steps: int = 120000   # control steps over which the attached fraction decays frac0 -> 0
    attached_spawn_out: float = 0.5      # steady pose dir = normalize(out*n_wall + up*z); horizontal (out) share
    attached_spawn_up: float = 0.87      # ... vertical (up) share -> ~30deg from vertical, wire taut, tension holds hook on bar
    attached_spawn_jitter: float = 0.15  # m: max pose jitter at curriculum END (grows 0 -> this with frac) to widen the hold basin
    attached_spawn_v0: float = 0.0       # residual up-and-out speed (m/s) imparted to the drone at attached-spawn reset, so hold is bootstrapped from a dynamic post-swing state instead of a static pose; 0.0 = static (default, other arms unaffected)

    # --- rewards -------------------------------------------------------------------------
    # Redesigned reward (2026-07-09). Structure:
    #   * APPROACH on the DRONE->BASKET distance (coarse driver: get the drone to the target).
    #   * IMPACT term: at the wall, decompose the hook velocity (relative to the static bar) into a
    #     component PERPENDICULAR to the wall and a component IN the wall plane, and shape each (below).
    #   * LOCKING reward for catching + staying locked, and a TENSION reward while locked (keep the wire taut).
    #   * The hover/transport CONTROL regularizers (smoothness, body-rate, effort, downwash) reused verbatim.
    # All dense per-step terms are scaled by step_dt EXCEPT the approach progress delta (already a per-step
    # delta) and the one-off catch/detach bonuses.

    # approach: HOOK -> NEAREST WALL. Coarse driver = telescoping progress (reward for closing the
    # hook->nearest-wall distance; dense gradient at any range). Targets the HOOK at the nearest wall (not the
    # drone at the basket centre, which would dangle the hook in the middle away from every wall) so approach,
    # proximity and impact all pull the same direction.
    approach_weight = 15.0        # reward per metre of hook->nearest-wall distance closed (delta clamped per step)
    approach_delta_clip = 0.1     # max |hook->wall distance change| credited per step (m); tames reset spikes
    # proximity: fine HOOK -> BAR proximity, exp(-proximity_k * ||hook - bar||_3D). Keyed to the HOOK being
    # near the wall bar in full xyz (not the drone near the basket), and concentrated near the wall so it
    # only meaningfully rewards a genuinely close hook. This supplies the gradient that lifts the dangling
    # hook up to the bar once the drone has reached the basket (unattached phase only). See _get_rewards.
    proximity_weight = 6.0        # RESTORED (removing it made living net-negative -> the agent chose to CRASH to
                                  # escape the negative reward stream). Being near the bar must PAY so the agent
                                  # wants to stay alive there; the grasp rewards (catch/lock/success) then dominate.
    proximity_k = 2.0             # exp(-k*d): ~0.67 @0.2 m, ~0.41 @0.45 m, ~0.14 @1 m -> concentrated at the wall

    # impact: decompose the hook velocity at the wall (relative to the static bar), active only inside the
    # impact zone (hook within impact_range of the target bar and still unattached):
    #   * PERPENDICULAR to the wall (wall_normal component v_normal): PUNISHED SQUARED -> penalize hard
    #     slamming into the wall. The catch model still needs a gentle into-wall touch to engage, so the
    #     optimum is a soft contact, not zero and not a slam.
    #   * IN-PLANE of the wall: the wall_up component v_up (= |v_in_plane| * sin(angle-from-horizontal), the
    #     requested "sinus of v_wall"): moving UP the wall is rewarded, purely horizontal (along the bar) is
    #     0, moving DOWN is penalized -- all captured by the signed v_up.
    impact_range = 0.2            # hook->bar distance (m) under which the impact velocity shaping is active
    # impact_perp now penalizes only the EXCESS into-wall speed over a soft-contact target (see _get_rewards):
    # -(w)*clamp(into_speed - impact_target_speed, 0)^2. A gentle touch (<= target) and moving AWAY are free,
    # so it no longer fights the catch model's required into-wall engagement -- only true slamming is punished.
    impact_perp_weight = 5.0      # penalty weight on the SQUARED excess into-wall speed above impact_target_speed
    impact_target_speed = 0.3     # m/s into-wall contact speed treated as "soft"; only faster is penalized
    impact_up_weight = 8.0        # reward weight on the signed in-plane up component v_up
    # TOWARD-WALL approach (fixes the "sideways swing" gap): reward the GENTLE perpendicular into-wall hook speed
    # UP TO impact_target_speed, so a PERPENDICULAR pendulum strike is preferred over a sideways skim (which earns
    # v_up just as well). Complements impact_perp (which only PENALIZES excess into-wall speed). Zone+unattached.
    impact_toward_weight: float = 0.0   # reward on min(into_wall_speed, impact_target_speed); off by default
    # ANTI-SAG / straight-wire hold (fixes concern 1): while ATTACHED, penalize the drone LOSING ALTITUDE (down
    # velocity) and penalize the drone-anchor distance dropping BELOW taut_thr (wire visibly bending). Gives a
    # real cost for the "crash down -> wire loose -> detach" failure, beyond merely losing the small taut bonus.
    anti_sag_weight: float = 0.0        # penalty on downward drone speed while attached
    slack_penalty_weight: float = 0.0   # penalty on slack depth clamp(taut_thr - dist_da, 0) while attached
    # AIMED up-sweep (raises catch PRECISION): reward the up-stroke weighted by nearness to the bar exp(-k*dist)
    # instead of the binary impact-box gate, so the swing is rewarded most when it peaks ON the bar. Opt-in.
    reward_aimed_upsweep: bool = False
    upsweep_aim_k: float = 3.0    # exp(-k*hook_bar_dist): ~0.55@0.2m, ~0.26@0.45m -> concentrates the up-reward at the bar

    # --- PERPENDICULAR-SWING shaping (2026-07-17): force a genuine pendulum swinging PERPENDICULAR to the target
    # wall, instead of the two failure modes seen in the videos: (a) the hook hanging DEAD straight below the drone,
    # (b) the hook skimming SIDEWAYS along the wall face. Decomposes the hook velocity in the assigned wall's
    # orthonormal frame (wall_normal / bar_axis / wall_up) and shapes each component separately:
    #   * PERPENDICULAR |v.wall_normal| -> REWARDED in BOTH directions (in AND out). Swinging back and forth
    #     through the wall normal is exactly how pendulum amplitude is pumped; rewarding only the inbound stroke
    #     would invite a one-way charge (and impact_perp already punishes slamming). Capped by swing_perp_cap so
    #     it cannot be farmed by simply flying fast.
    #   * LATERAL |v.bar_axis| -> PENALIZED (the sideways skim along the wall face; it earns v_up just as well but
    #     never seats the hook on the bar).
    #   * VERTICAL v_up is deliberately NOT penalized even though it lies IN the wall plane: the catch REQUIRES
    #     v_up > attachment.upsweep_v_min to seat on a horizontal bar, and a true perpendicular pendulum arc is
    #     precisely perpendicular+vertical with ZERO lateral. Penalizing v_up would fight the catch gate.
    #   * STANDSTILL BELOW THE DRONE -> PENALIZED: peaks when the hook hangs ~straight under the drone AND is
    #     barely moving RELATIVE TO THE DRONE (the dead pendulum). Relative velocity is used on purpose so that
    #     carrying a dead-hanging hook around at speed still counts as a standstill (the pendulum is not swinging).
    # All terms are UNATTACHED-phase only (once locked we want a taut hold, not a swing). Opt-in: weights default
    # 0.0, so every existing arm/recipe is unaffected.
    swing_perp_weight: float = 0.0      # reward on clamp(|v_hook . wall_normal|, max=swing_perp_cap), unattached
    swing_perp_cap: float = 1.5         # m/s: cap on the rewarded perpendicular speed (anti reward-farming)
    swing_lateral_weight: float = 0.0   # penalty on |v_hook . bar_axis| (sideways skim along the wall face)
    swing_still_weight: float = 0.0     # penalty on the dead hang (hook straight below the drone AND not swinging)
    swing_still_radius: float = 0.15    # m: horizontal hook-under-drone offset scale; exp(-(horiz/r)^2) -> 1 = dead below
    swing_still_speed: float = 0.3      # m/s: hook-vs-drone relative speed at/above which it is NOT a standstill
    # --- ARC-AMPLITUDE shaping (2026-07-20): reward the pendulum ARC directly, not just its velocity. The velocity
    # terms (swing_perp) can be satisfied by a small fast wobble; the auditors showed swing_amp stayed ~0.08 m (no
    # real arc) and perp_ratio ~0.50 (half sideways skim). These reward the hook's DISPLACEMENT from straight-below
    # the drone, decomposed in the wall frame: PERPENDICULAR displacement (|(hook-drone).wall_normal|) is REWARDED
    # (a big swing-out in the plane perpendicular to the wall = real arc amplitude), LATERAL displacement
    # (|(hook-drone).bar_axis|) is PENALIZED (keeps the arc in the perpendicular plane, not skimming along the bar).
    # Paired with swing_perp (velocity) this rewards being displaced AND moving -> a genuine oscillating arc, and
    # with the chord-angle gate + require_upsweep a static hold cannot catch. Unattached-phase only. Opt-in (0 = off).
    arc_perp_weight: float = 0.0        # reward on clamp(|(hook-drone).wall_normal|, max=arc_disp_cap) -- arc amplitude
    arc_lateral_weight: float = 0.0     # penalty on |(hook-drone).bar_axis| -- sideways (along-bar) displacement
    arc_disp_cap: float = 0.5          # m: cap on rewarded perpendicular displacement (~wire_length; 0.55*sin(65deg))

    # PENDULUM PUMP shaping (raises catch via a REPEATABLE pump-release-lift maneuver -> consistent swing timing,
    # vs scattershot up-sweeps). Two shaping terms on the drone's OWN (controllable) motion; the "stop/reverse to
    # whip the hook forward" phase EMERGES as the bridge between them (not hard-coded -> avoids reward-hacking):
    #   (1) LOAD: charge toward the wall with the hook trailing BEHIND (rel*wall_normal>0) -> stores pendulum energy
    #   (2) LIFT: drone rises (v_drone*wall_up>0) concentrated at the bar -> the taut wire whips the hook UP onto it
    # Gated to the unattached phase. windup DECAYS with the spawn curriculum (bootstrap the strategy, then refine).
    reward_pump: bool = False
    pump_windup_weight: float = 1.5   # (1) LOAD: drone_toward[m/s] * hook_behind[m], far field (hook_bar>impact_range)
    pump_lift_weight: float = 2.0     # (2) LIFT: drone_up[m/s] * exp(-upsweep_aim_k*hook_bar), the seating timing
    pump_windup_decay: bool = True    # scale windup by (1-spawn_frac): strong at curriculum start, ~0 at eval/frac=1
    # inverse-spawn RUNWAY: seed an initial AWAY-from-wall drone velocity so it must fly IN, naturally loading the
    # hook behind (the "approach with perpendicular velocity" seed). Decays with the curriculum like swing assist.
    spawn_away_v0: float = 0.0        # m/s away-from-wall initial drone speed at curriculum start (frac=0); pairs w/ pump

    # Attaching / holding = the PRIMARY objective. Rewards:
    #   * catch_bonus  : +bonus on EVERY attach event (not one-off) -> lure to actually engage the hook,
    #   * detach_penalty: -penalty on EVERY detach event, BALANCED (== catch_bonus) so an attach->detach cycle
    #     nets ~0 -> the drone only profits by STAYING hooked (keeps the catch bonus, avoids the detach penalty),
    #   * lock         : positive EVERY step the hook is ATTACHED = continuous "reward for hooking",
    #   * tension      : positive EVERY step, scaled by how TAUT the wire is pulled (~0 when slack) = "taut",
    #   * success_bonus: one-off + when the hold MILESTONE is first reached (restored to match run 15-28-03,
    #     whose catch rate climbed to ~0.15 WITH this bonus active; removing it flattened catching at ~0.03).
    catch_bonus = 20.0           # + on each attach (_newly_caught 0->1); lure to engage the hook
    # detach_penalty < catch_bonus so a catch->detach still nets POSITIVE (+35): this is the exploration lure
    # that drove the good run 15-28-03 (a blind drone can't hold yet, so most early catches end in a detach; a
    # balanced -40 zeroed the net signal and catching decayed). "Holding > cycling" is instead ensured by the
    # continuous lock+tension stream (a long hold out-earns rapid cycling), not by balancing the event bonuses.
    detach_penalty = 3.0         # - on each detach (_just_detached); matches 15-28-03
    success_bonus = 80.0        # large one-off + when the hold milestone is reached; success now TERMINATES the
                                 # episode (see _get_dones), so this is a big terminal payoff for a completed grasp
    progress_weight = 10.0       # per-step shaped reward = weight * (hold_counter / hold_req) in [0,1]: ramps UP
                                 # as the hold accumulates toward success and drops to 0 the instant the hook is
                                 # lost (hold_counter resets). Dense guidance up the last stretch to the milestone.
    lock_weight = 6.0            # per-step reward while attached ("locked on" = hooked)
    # tension: while LOCKED, reward wire tension (the drone actively pulling the wire taut). Proxy = the pin
    # force magnitude the wire transmits to the hook (see _compute_hook_pin_force), normalized by tension_ref.
    tension_weight = 5.0         # NOW weights the TAUT-MARGIN reward (see marl_hook_env _get_rewards): reward for
                                 # keeping drone->attach distance a safe margin ABOVE the slack threshold, i.e.
                                 # holding the wire firmly taut / pulling AWAY from the wall (not the saturated
                                 # raw pin-force it used to be, which gave no gradient and let the drone loiter slack).
    tension_ref = 30.0           # N; (legacy) pin-force magnitude -- no longer used by the taut-margin reward
    taut_margin_target = 0.05    # m; taut reward ramps 0->1 over [taut_thr, taut_thr+this] (taut_thr=wire-slack_margin
                                 # =0.52). Peak ~0.57m = comfortably taut (wire 0.55), 0 at the 0.52 slack edge.
    wall_pull_weight = 8.0       # penalty on the drone's INTO-wall velocity while attached: discourages pulling
                                 # toward the wall (goes slack + unrealistic jerk through the stiff pin).

    # --- survival economics (fixes the "suicidal agent": net-negative living + free death -> crash on purpose) ---
    alive_weight = 5.0           # + EVERY step the episode continues (alive_weight * dt) -> baseline living is
                                 # POSITIVE (so the agent stays alive), but kept modest so hovering-to-timeout does
                                 # NOT out-earn a catch+success (success ends the episode, forfeiting the stream).
    death_penalty = 40.0        # - one-off on any FAILURE termination (fly_low / hard_entangle / out-of-bounds /
                                 # runaway / nonfinite) -- NOT on success or timeout. Makes crashing clearly worse
                                 # than living, so early exploration can't collapse into "die fast".

    # control regularizers (reused from the hover/transport task)
    action_smoothness_weight = 0.5
    body_rate_penalty_weight = 0.5
    force_penalty_weight = 0.5
    downwash_rew_weight = 0.5     # reward keeping the hook out of the prop-wash line (hover downwash analog)
    downwash_scale = 3.0
    swing_limit_weight = 2.0      # penalize the wire swinging up near the props; now the SOLE entanglement
                                  # deterrent (entanglement is no longer a hard termination), so strengthened 4x

    # success = attached & held (taut) for this many consecutive control steps. Reaching it grants the one-off
    # success_bonus (above). The hold-time curriculum ramps the required hold from hold_success_steps_start ->
    # hold_success_steps over hold_curriculum_steps, so the milestone is reachable early (10 steps) and tightens
    # as the policy improves (matches run 15-28-03, where hold_req ramped 10 -> 13 over the first 14k steps).
    hold_success_steps: int = 400             # final (hard) required hold -> higher time, reached late in training
    hold_curriculum: bool = True
    hold_success_steps_start: int = 10        # initial (easy) required hold, in control steps (only if curriculum on)
    hold_curriculum_steps: int = 250000       # control steps over which to ramp start -> hold_success_steps
    hold_curriculum_power: float = 1.5        # ramp exponent (hfrac**power): >1 keeps the required hold LOW for
                                              # LONGER (easy, frequent successes early), then rises steeply to the
                                              # higher final time -> "low for a long time, then increase".

    # --- surrogate wall collision --------------------------------------------------------
    # The basket walls are visualization-only (no physics collider), so the real hook body would otherwise fly
    # THROUGH them. This analytic surrogate keeps the hook out of the wall shell: when the hook penetrates the
    # footprint below the rim, apply an outward spring-damper force along the nearest lateral wall normal (the
    # top is left open so the hook can still be swung over the rim). margin < capture_radius so a genuine catch
    # (triggered outside the wall) still fires before the repulsion. See _wall_collision_force.
    wall_collision_k: float = 1500.0      # N/m outward stiffness on penetration depth (stiff -> shallow poke-through)
    wall_collision_damp: float = 60.0     # N per (m/s) damping on the into-wall velocity component (kills inward motion)
    wall_collision_fmax: float = 250.0    # N clamp on the repulsion magnitude
    wall_collision_margin: float = 0.02   # m: expand the footprint by ~hook radius (barrier sits at the hook surface)

    # --- terminations --------------------------------------------------------------------
    min_drone_height = 0.1
    entangle_margin = 0.1   # SOFT: hook within this of the drone -> swing_limit penalty + diagnostic (not a kill)
    hard_entangle_margin = 0.6   # HARD reset only on a TRUE over-the-top wrap (loosened 0.25->0.6 so a normal
                                 # catch up-swing near the bar does NOT trip it); also dropped from death_penalty
    runaway_speed = 30.0    # HARD reset: drone speed (m/s) above this = physics divergence; reset before it NaNs
    bounding_box_threshold = 6.0

    # --- debug vis -----------------------------------------------------------------------
    # The basket + target bar are analytic (no collider) -> render them as visualization-only markers so
    # the scene actually SHOWS where the basket is and which bar to hook. One visualizer with two prototypes:
    #   * "wall" (index 0): a unit cuboid, drawn as 4 translucent brown panels (scaled/placed per wall).
    #   * "bar"  (index 1): a unit cuboid, drawn as one bright-red bar along the target bar segment.
    # See _debug_vis_callback, which builds the per-env panel/bar transforms from basket_pos_w/basket_quat_w.
    basket_wall_height: float = 0.32   # visual wall height (m); covers the bar z-range (0.02-0.30)
    debug_vis: bool = True
    if debug_vis:
        marker_cfg_basket = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/basket",
            markers={
                "wall": sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.35, 0.15), opacity=0.28),
                ),
                "bar": sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.9, 0.1, 0.1), emissive_color=(0.4, 0.0, 0.0)
                    ),
                ),
            },
        )

    # --- scene ---------------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=8.0, replicate_physics=True)

    events = EventCfg()
