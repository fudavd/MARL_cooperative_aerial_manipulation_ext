# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the hook-grasp task: a single drone swings a weighted wire + hook onto a basket wall bar.

Same control cascade, simulation and logging idioms as the hover task. Differences:
    * single drone (``num_drones = 1``; the env is written to generalize),
    * a flexible weighted **wire + hook** instead of the pre-attached rope (asset ``FLYHOOK_CFG``),
    * an analytic **basket** at a random-but-observed pose with horizontal **bar** targets on its walls,
    * a **surrogate probabilistic attachment** model (``HookAttachmentCfg``) with slack-detach.

The defaults below ARE the final task and the training recipe that solves it with PPO (see
``scripts/hook_grasp/README.md``): a hook-blind actor (no hook or wall position in its observation, only the
onboard INDI disturbance residual), a privileged critic, and curricula that anneal to the final task during
training. Evaluation calls :meth:`MARLHookGraspEnvCfg.disable_curricula` so every gate sits at its final value.
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

# Module-level constants (referenced by the class-body dict comprehensions below, which cannot see sibling
# class attributes).
_NUM_DRONES = 1
_ACTION_DIM = 5                      # ACCBR: [lin_acc(3), body_rates(2)]
_HISTORY_LEN = 3                     # critic frame-stack depth
_ACTOR_HISTORY_LEN = 3               # actor frame-stack depth
# Per-frame ACTOR observation = the hover/transport partial observation layout (basket = "load",
# pose_command = "goal"), see MARLHookGraspEnv._drone_obs_frame:
#   basket pos (3) + basket rot matrix (9)                                             = 12
#   assigned-wall centre (3) + drone pos (3) + rot (9) + lin vel (3) + ang vel (3)     = 21
#   goal_pos_error (3) + difference_matrix (9)                                         = 12
# plus the 6-dim body-frame INDI disturbance residual (the hook/wire wrench the drone senses onboard).
_OBS_PER_FRAME = 12 + 21 + 12             # 45
_INDI_RESIDUAL_DIM = 6
_ACTOR_OBS_PER_FRAME = _OBS_PER_FRAME + _INDI_RESIDUAL_DIM   # 51
# Privileged critic state (see _get_states): basket full state (18) + per-drone state (18) + per-drone hook
# block (12: hook rel/vel, hook->bar error, attached, tension, hold_progress) + goal (12).
_STATE_DIM = 18 + (18 + 12) * _NUM_DRONES + 12   # 60 for one drone
_POSSIBLE_AGENTS = [f"falcon{i + 1}" for i in range(_NUM_DRONES)]

HOOK_MASS_KG = 0.30
"""Mass of the lumped hook + weight tip body (kg). The hook is the pendulum bob: its mass sets the swing period
and the momentum delivered at the bar. 300 g trains reliably; below ~200 g the catch collapses."""


@configclass
class EventCfg:
    """Reset / startup events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Default drone reset pose (overridden per episode by the spawn logic in MARLHookGraspEnv._reset_idx).
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

    # Fixed hook mass (degenerate range = identical in every env).
    randomize_hook_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[FLYHOOK_HOOK_BODY]),
            "mass_distribution_params": (HOOK_MASS_KG, HOOK_MASS_KG),
            "operation": "abs",
        },
    )


@configclass
class MARLHookGraspEnvCfg(DirectMARLEnvCfg):
    # --- control -------------------------------------------------------------------------
    control_mode = "ACCBR"  # accel + body rates; reuses the flycrane Geometric -> INDI -> motor cascade
    decimation = 3
    episode_length_s = 20

    # --- agents / spaces -----------------------------------------------------------------
    num_drones = _NUM_DRONES
    possible_agents = _POSSIBLE_AGENTS

    partial_obs = True
    history_len = _HISTORY_LEN
    # The actor observation is hook-BLIND: no hook or wall position, only what the drone can measure onboard.
    # ``actor_indi_residual`` appends the 6-dim body-frame INDI disturbance residual (angular INDI increment
    # alpha_cmd - omega_dot, and R^T(acc_cmd - a_meas)) to every actor frame; it carries the hook/wire wrench
    # (force and phase) without any position information. ``actor_history_len`` is the actor frame-stack depth
    # (0 -> ``history_len``); the critic keeps ``history_len`` frames regardless.
    actor_indi_residual: bool = True
    actor_history_len: int = _ACTOR_HISTORY_LEN

    obs_dim_per_frame = _OBS_PER_FRAME
    action_dim = _ACTION_DIM
    action_spaces = {a: _ACTION_DIM for a in _POSSIBLE_AGENTS}
    observation_spaces = {a: _ACTOR_OBS_PER_FRAME * _ACTOR_HISTORY_LEN for a in _POSSIBLE_AGENTS}   # 153
    state_space = _STATE_DIM * _HISTORY_LEN                                                          # 180

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

    # --- basket + bar targets (analytic; the basket is visualization-only, no collider) ----------------
    # Footprint matches the hover/transport payload (0.55 x 0.55 m box).
    basket_half_x: float = 0.275
    basket_half_y: float = 0.275
    basket_wall_z_lo: float = 0.02   # bars sit from just above the base ...
    basket_wall_z_hi: float = 0.22   # ... up to this height (m, basket frame)
    # Random basket pose per episode (world). Walls stay vertical (yaw only).
    basket_pose_range = {
        "pos_x": (-1.0, 1.0),
        "pos_y": (-1.0, 1.0),
        "pos_z": (0.6, 1.4),
        "yaw": (-math.pi, math.pi),
    }
    bar_half_len: float = 0.18            # half-length of the drawn target bar (m), visual only
    reward_center_target: bool = True     # approach/proximity rewards aim at the assigned wall's CENTRE; the
    # catch itself still fires anywhere along that wall.

    # --- transport goal pose (observed like the hover task's pose_command; the basket start pose + offset) --
    goal_offset_range = {
        "x": (-0.75, 0.75),
        "y": (-0.75, 0.75),
        "z": (-0.3, 0.5),
        "yaw": (-math.pi / 4, math.pi / 4),
    }
    goal_z_floor: float = 0.4   # never place the goal below this height (m)

    # --- surrogate attachment model (final task values) ----------------------------------
    attachment: HookAttachmentCfg = HookAttachmentCfg(
        capture_radius=0.05,          # m: the hook must pass within 5 cm of the bar
        upsweep_v_min=0.08,           # m/s: a genuine up-stroke is required to seat the hook
        min_chord_angle_deg=25.0,     # deg: the wire must be swung out >= 25 deg ...
        chord_angle_perp_only=True,   # ... OUTWARD from the wall (drone outside, hook swung up from outside)
    )

    # --- training curricula (all anneal to the final task; ``disable_curricula`` pins them for eval) ------
    # capture radius: loose start -> attachment.capture_radius
    capture_curriculum: bool = True
    capture_radius_start: float = 0.13
    capture_curriculum_steps: int = 300000        # control steps
    # chord-angle gate: 0 deg -> attachment.min_chord_angle_deg
    chord_angle_curriculum: bool = True
    chord_angle_start_deg: float = 0.0
    chord_angle_steps: int = 120000
    # hold requirement: hold_success_steps_start -> hold_success_steps (power ramp: easy for long, then rises)
    hold_success_steps: int = 100                 # final: consecutive taut-attached control steps = success
    hold_curriculum: bool = True
    hold_success_steps_start: int = 30
    hold_curriculum_steps: int = 85000
    hold_curriculum_power: float = 1.5
    # drone spawn: in front of the assigned wall at a Gaussian radial distance with FIXED mean (the catchable
    # point-blank distance) and a std that grows start -> end; the azimuth widens 0 -> +/-pi.
    spawn_curriculum: bool = True
    spawn_dist_mu: float = 0.35        # m: mean radial distance from the wall
    spawn_std_start: float = 0.1       # m
    spawn_std_end: float = 0.6         # m
    spawn_azimuth_max: float = math.pi
    spawn_dist_min: float = 0.5        # m: clamp floor (outside the basket footprint)
    spawn_dist_max: float = 2.2        # m: clamp ceiling
    spawn_height: float = 0.7          # m above the basket CoM
    spawn_pos_noise: float = 0.15      # m: extra height jitter
    spawn_curriculum_steps: int = 200000
    # swing assist: initial drone velocity toward the wall (+ a small up kick) that decays to 0 with the spawn
    # curriculum, so assisted up-sweep catches happen early in training.
    swing_assist_curriculum: bool = True
    swing_assist_v0: float = 2.2       # m/s at curriculum start
    swing_assist_up: float = 0.3       # fraction of v0 added as an initial UP velocity
    # attached spawn: a decaying fraction of envs starts ALREADY hooked at a taut hold pose (learn the hold first)
    attached_spawn_curriculum: bool = True
    attached_spawn_frac0: float = 0.5
    attached_spawn_steps: int = 65000
    attached_spawn_out: float = 0.8    # steady pose dir = normalize(out * wall_normal + up * z)
    attached_spawn_up: float = 0.6
    attached_spawn_jitter: float = 0.15  # m: pose jitter at curriculum end (grows 0 -> this)
    attached_spawn_v0: float = 0.4     # m/s: residual up-and-out speed at the attached spawn

    # --- rewards (per control step; dense terms are scaled by step_dt in the env) -------------------------
    # approach: telescoping progress of the hook toward its target on the assigned wall (unattached phase)
    approach_weight = 15.0
    # proximity: exp(-k * ||hook - target||) (unattached phase)
    proximity_weight = 2.0
    proximity_k = 2.0
    # impact shaping inside the impact zone (hook within impact_range of the bar, unattached):
    #   * excess into-wall speed above impact_target_speed is penalized (no slamming),
    #   * gentle into-wall speed up to impact_target_speed is rewarded (perpendicular strike, not a skim),
    #   * the signed up-the-wall hook velocity is rewarded (the J-hook up-sweep).
    impact_range = 0.2
    impact_perp_weight = 5.0
    impact_target_speed = 0.3
    impact_toward_weight: float = 6.0
    impact_up_weight = 8.0
    # planar swing shaping (unattached phase): reward pendulum amplitude/speed IN the wall-normal plane near the
    # wall, penalize off-plane (along-bar) displacement/speed and azimuthal whirl everywhere, and reward the
    # drone's own velocity perpendicular to the wall (a planar-driven pendulum stays planar).
    swing_plane_amp_weight: float = 8.0
    swing_plane_amp_cap: float = 0.5      # m
    swing_plane_vel_weight: float = 4.0
    swing_plane_vel_cap: float = 1.5      # m/s
    swing_plane_off_weight: float = 5.0
    swing_plane_vlat_weight: float = 3.0
    swing_plane_whirl_weight: float = 4.0
    swing_plane_zone: float = 0.7         # m: hook->target distance under which the positive terms are active
    drone_perp_osc_weight: float = 2.0
    drone_perp_osc_cap: float = 1.5       # m/s
    # attach / hold
    catch_bonus = 20.0           # + on every attach event
    detach_penalty = 3.0         # - on every detach event
    success_bonus = 80.0         # one-off when the hold milestone is reached (terminates the episode)
    progress_weight = 10.0       # per-step hold_counter / hold_req in [0, 1]
    lock_weight = 6.0            # per-step while attached
    tension_weight = 5.0         # per-step taut-margin reward while attached
    tension_ref = 30.0           # N: pin-force normalization for the critic state
    taut_margin_target = 0.05    # m: taut reward ramps 0 -> 1 over [taut_thr, taut_thr + this]
    anti_sag_weight: float = 6.0        # penalty on downward drone speed while attached
    slack_penalty_weight: float = 20.0  # penalty on slack depth while attached
    wall_pull_weight = 8.0       # penalty on the drone's into-wall velocity while attached
    # survival
    alive_weight = 1.0           # + every living step
    death_penalty = 40.0         # - on any failure termination (not on success / timeout)
    # control regularizers
    action_smoothness_weight = 1.0
    body_rate_penalty_weight = 1.0
    action_rate_penalty_weight: float = 0.15    # ||a_t - a_{t-1}||^2
    action_accel_penalty_weight: float = 0.15   # ||a_t - 2 a_{t-1} + a_{t-2}||^2
    action_smooth_phase_gate: bool = True       # the two penalties above are OFF in the near-wall swing zone
    smooth_free_zone: float = 0.7               # m: hook->target distance of that zone (unattached)
    force_penalty_weight = 0.5
    downwash_rew_weight = 3.0    # keep the hook off the prop-wash axis
    downwash_scale = 3.0
    swing_limit_weight = 2.0     # penalize the hook rising toward the props
    above_box_penalty_weight = 15.0   # penalize the drone hovering horizontally over the box footprint
    above_box_margin = 0.15

    # --- surrogate wall collision (keeps the hook out of the collider-less walls) --------
    wall_collision_k: float = 1500.0
    wall_collision_damp: float = 60.0
    wall_collision_fmax: float = 250.0
    wall_collision_margin: float = 0.02

    # --- terminations --------------------------------------------------------------------
    min_drone_height = 0.1
    entangle_margin = 0.1          # soft: hook within this of the drone -> penalty + diagnostic only
    hard_entangle_margin = 0.6     # hard reset only on a true over-the-top wrap
    runaway_speed = 30.0           # hard reset: drone speed above this = physics divergence
    bounding_box_threshold = 6.0

    # --- debug vis: the analytic basket walls + target bar are drawn as markers -------------------------
    basket_wall_height: float = 0.32
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

    def disable_curricula(self) -> None:
        """Pin every training aid at its FINAL value (honest evaluation of the true task)."""
        self.capture_curriculum = False        # radius = attachment.capture_radius
        self.chord_angle_curriculum = False    # gate = attachment.min_chord_angle_deg
        self.hold_curriculum = False           # hold requirement = hold_success_steps
        self.spawn_curriculum = False          # full spawn spread, zero swing assist
        self.swing_assist_curriculum = False
        self.attached_spawn_curriculum = False  # every episode starts unattached
