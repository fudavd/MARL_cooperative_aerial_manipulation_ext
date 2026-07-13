# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import math

from MARL_mav_carry_ext.assets import FLYCRANE_CFG

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass


@configclass
class EventCfg:
    """Events for the hovering task.

    Resetting states on resets, disturbances, etc.
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.5, 1.5),
                "roll": (-0, 0),
                "pitch": (-0, 0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    # domain randomization: basket (load) weight, nominal 1.4 kg +/- 0.5 kg.
    # reset mode -> each env draws a fresh mass every episode (no fixed env->mass mapping to overfit;
    # attachment geometry is still frozen per env since a joint anchor can't be moved at runtime).
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["load_link"]),
            "mass_distribution_params": (0.9, 1.9),  # nominal 1.4 kg +/- 0.5 kg
            "operation": "abs",
        },
    )

    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {
    #             "x": (0.0, 0.0),
    #             "y": (0.0, 0.0),
    #             "z": (1.5, 1.5),
    #             "roll": (-0, 0),
    #             "pitch": (-0, 0),
    #             "yaw": (0.0, 0.0),
    #         },
    #         "velocity_range": {
    #             "x": (-0.0, 0.0),
    #             "y": (-0.0, 0.0),
    #             "z": (-0.0, 0.0),
    #             "roll": (-0.0, 0.0),
    #             "pitch": (-0.0, 0.0),
    #             "yaw": (-0.0, 0.0),
    #         },
    #     },
    # )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )


@configclass
class MARLHoverRandEnvCfg(DirectMARLEnvCfg):
    # control mode
    control_mode = "ACCBR"  # ACCBR or geometric
    # env
    decimation = 3
    episode_length_s = 20

    # history of observations
    partial_obs = True  # if only local observations are used
    history_len = 3

    possible_agents = ["falcon1", "falcon2", "falcon3"]
    num_drones = len(possible_agents)
    if control_mode == "geometric":
        action_dim_geo = 12
        action_spaces = {"falcon1": action_dim_geo, "falcon2": action_dim_geo, "falcon3": action_dim_geo}
        obs_dim_geo = 87
        observation_spaces = {"falcon1": obs_dim_geo, "falcon2": obs_dim_geo, "falcon3": obs_dim_geo}
        state_space = 84
    elif control_mode == "ACCBR":
        action_dim_accbr = 5
        action_spaces = {"falcon1": action_dim_accbr, "falcon2": action_dim_accbr, "falcon3": action_dim_accbr}
        if partial_obs:
            obs_dim_accbr = 45 * history_len
        else:
            obs_dim_accbr = 87
        observation_spaces = {"falcon1": obs_dim_accbr, "falcon2": obs_dim_accbr, "falcon3": obs_dim_accbr}
        state_space = 84

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.0033333333333333335,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.8066),
    )
    # robot
    robot_cfg: ArticulationCfg = FLYCRANE_CFG.replace(prim_path="/World/envs/env_.*/flycrane")
    robot_cfg.spawn.activate_contact_sensors = True

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="/World/envs/env_.*/flycrane/.*", update_period=0.0, history_length=3, debug_vis=False
    )
    sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*")
    contact_sensor_threshold = 0.1

    # falcon CoM names
    falcon_names = "Falcon.*_base_link_inertia"
    # rotor names
    falcon_rotor_names = "Falcon.*_rotor_.*"
    # payload name
    payload_name = "load_odometry_sensor_link"
    # rope name and termination terms
    rope_name = "rope_.*_link"
    cable_angle_limits_drone = 0.0  # cos(angle) limits
    cable_angle_limits_payload = -math.sqrt(2) / 2  # cos(angle) limits
    cable_collision_threshold = 0.2
    cable_collision_num_points = 10
    drone_collision_threshold = 0.6
    bounding_box_threshold = 5.0
    goal_achieved_range = 0.3
    goal_achieved_ori_range = 0.4

    # low level control
    low_level_decimation: int = 1
    max_thrust_pp = 6.25  # N

    # rewards
    pos_track_weight = 1.5
    ori_track_weight = 1.5
    action_smoothness_weight = 0.5
    body_rate_penalty_weight = 0.5
    force_penalty_weight = 0.5
    downwash_rew_weight = 0.5

    # goal terms
    goal_range = {
        "pos_x": (-1.0, 1.0),
        "pos_y": (-1.0, 1.0),
        "pos_z": (0.5, 1.5),
        "roll": (-math.pi / 4, math.pi / 4),
        "pitch": (-math.pi / 4, math.pi / 4),
        "yaw": (-math.pi, math.pi),
    }
    range_curriculum_steps = 7500
    # reward/task scheduling: when True, widen goal_range to the harder ranges after
    # range_curriculum_steps control steps (see _reset_idx in marl_hover_env.py). Off by default.
    range_curriculum: bool = False
    # harder goal ranges activated once the curriculum kicks in
    goal_range_hard = {
        "pos_x": (-2.0, 2.0),
        "pos_y": (-2.0, 2.0),
        "pos_z": (0.5, 2.5),
        "roll": (-math.pi / 4, math.pi / 4),
        "pitch": (-math.pi / 4, math.pi / 4),
        "yaw": (-math.pi, math.pi),
    }

    make_quat_unique_command = False
    # goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/goal_marker",
    #     markers={
    #         "goal": sim_utils.SphereCfg(
    #             radius=0.0335,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 1.0)),
    #         ),
    #     },
    # )

    # debug visualization
    debug_vis: bool = True
    if debug_vis:
        marker_cfg_goal = FRAME_MARKER_CFG.copy()
        marker_cfg_goal.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_goal.prim_path = "/Visuals/Command/goal_pose"

        marker_cfg_body = FRAME_MARKER_CFG.copy()
        marker_cfg_body.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg_body.prim_path = "/Visuals/Command/body_pose"

    # --- cable attachment-point domain randomization ---------------------------------------
    # Perturbs where the 3 cables attach to the load (physics:localPos0 of the
    # rope_*_sphere_joint_0_joint_x joints, parent = load_link). Only the load-side anchor is
    # moved; the rope segments below it are untouched, so the cable length stays the same.
    # Implemented in MARLHoverRandEnv._randomize_attachments (edits the USD before sim start).
    randomize_attachment: bool = True
    # scope:
    #   per_env=True  -> every parallel env gets its OWN attachment triangle (wide coverage),
    #                    but forces replicate_physics=False (slower / more VRAM at high N).
    #   per_env=False -> ONE attachment sampled per process, shared by all envs (full speed).
    attachment_per_env: bool = False
    # basket geometry (metres, load_link frame): half-extents + top-rim z, from the load's
    # inertia/collision box (0.275 x 0.225 x 0.075 half-extents, COM z=0.06, top rim z~0.14).
    attachment_basket_half_x: float = 0.275
    attachment_basket_half_y: float = 0.225
    attachment_basket_top_z: float = 0.14
    # Walls-only attachment, parametrized by arc length s around the basket PERIMETER (the rim
    # outline), so each cable's anchor slides across corners onto multiple walls -> genuine x AND y
    # variation and coverage of all four side walls (NO top-face attachments). s runs CCW from the
    # (+x,-y) corner: [0,2hy)=+x wall, [2hy,2hy+2hx)=+y wall, ...; perimeter length P=4(hx+hy)=2.0 m
    # for the 0.275x0.225 basket (see _perimeter_point). Each rope gets its own non-overlapping arc
    # with >=0.25 m gaps, so anchors stay ordered around the rim and the cables never cross/overlap.
    # Tuple = (s_lo, s_hi, z_lo, z_hi); s_hi may exceed P (wraps). z runs DOWN the wall from the rim.
    # per-env: wide arcs tiling ~71% of the rim (the max with a hard non-crossing guarantee -- the
    # three unavoidable ~0.20 m gaps, centered on the +x/+y/-y wall middles, keep anchors above the
    # cable-collision threshold). rope_1/rope_2 span the full +x wall onto the +y/-y walls (x in
    # [-0.005,0.275]); rope_3 spans +y->-x->-y. All four walls covered.
    attachment_arcs_per_env = {
        "rope_1": (0.33, 0.73, 0.02, 0.14),   # +x(full) -> +y(right half)
        "rope_2": (1.72, 2.12, 0.02, 0.14),   # -y(right half) -> +x(full)  [wraps past P=2.0]
        "rope_3": (0.93, 1.52, 0.02, 0.14),   # +y(left) -> -x(full) -> -y(left)
    }
    # per-run: narrow arcs near the nominal corners -> realistic shared attachment.
    attachment_arcs_per_run = {
        "rope_1": (0.40, 0.49, 0.09, 0.14),
        "rope_2": (1.955, 2.045, 0.09, 0.14),
        "rope_3": (1.15, 1.30, 0.09, 0.14),
    }

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=8.0, replicate_physics=True)

    events = EventCfg()

    def __post_init__(self):
        # per-env attachment randomization needs independently-parsed physics per env, which is
        # incompatible with the fast replicate_physics path. Switch it off automatically so the
        # user only flips one flag (attachment_per_env). See perf notes for the cost at scale.
        parent_post = getattr(super(), "__post_init__", None)
        if callable(parent_post):
            parent_post()
        if self.randomize_attachment and self.attachment_per_env:
            self.scene.replicate_physics = False
