# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hook-grasp DirectMARL env: a drone swings a weighted wire + hook onto a basket wall bar and holds it taut.

Reuses the flycrane control cascade (Geometric -> INDI -> motor model) from the hover task. The novel pieces
are: a flexible weighted wire + hook (asset ``FLYHOOK_CFG``), an analytic basket with horizontal **bar** targets
at a random-but-observed pose, and the **surrogate probabilistic attachment** (``attachment.py``) with
slack-detach. Written for ``num_drones`` agents (used with 1).
"""

from __future__ import annotations

import math

import torch
from collections.abc import Sequence

from MARL_mav_carry_ext.controllers import GeometricController, IndiController
from MARL_mav_carry_ext.controllers.motor_model import RotorMotor

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import CircularBuffer
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    matrix_from_quat,
    normalize,
    quat_from_euler_xyz,
    quat_from_matrix,
    quat_rotate,
    quat_rotate_inverse,
)

from . import attachment as att
from .marl_hook_env_cfg import MARLHookGraspEnvCfg


class MARLHookGraspEnv(DirectMARLEnv):
    cfg: MARLHookGraspEnvCfg

    def __init__(self, cfg: MARLHookGraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # --- body indices -----------------------------------------------------------------
        self._falcon_idx = self.robot.find_bodies(cfg.falcon_names)[0]
        self._falcon_rotor_idx = self.robot.find_bodies(cfg.falcon_rotor_names)[0]
        self._hook_idx = self.robot.find_bodies(cfg.hook_name)[0]
        # Bodies that receive external wrenches, in ONE combined set_external_force_and_torque call:
        # [falcon (body torques), rotors (thrusts), hook (PD-pin force)]. A single call is required:
        # set_external_force_and_torque sets has_external_wrench=False when the passed forces are all zero, so a
        # separate zero-valued hook call (unattached) would cancel the rotor thrusts.
        self._ext_body_ids = list(self._falcon_idx) + list(self._falcon_rotor_idx) + list(self._hook_idx)

        self._num_drones = len(self._falcon_idx)
        self._control_mode = cfg.control_mode
        self._acfg = cfg.attachment

        # --- curricula (control-step driven; see cfg) -------------------------------------
        self._ctrl_step = 0
        self._capture_end = float(cfg.attachment.capture_radius)
        self._capture_start = float(cfg.capture_radius_start)
        if cfg.capture_curriculum:
            self._acfg.capture_radius = self._capture_start
        self._chord_angle_end = float(cfg.attachment.min_chord_angle_deg)
        if cfg.chord_angle_curriculum:
            self._acfg.min_chord_angle_deg = float(cfg.chord_angle_start_deg)
        self._current_hold_req = (
            int(cfg.hold_success_steps_start) if cfg.hold_curriculum else int(cfg.hold_success_steps)
        )

        # --- observation history ----------------------------------------------------------
        # actor frame-stack depth (actor_history_len; 0 -> history_len). The critic keeps cfg.history_len frames.
        self._actor_hist = int(cfg.actor_history_len) or int(cfg.history_len)
        self._observation_buffers = {
            agent: CircularBuffer(self._actor_hist, self.num_envs, device=self.device)
            for agent in self.cfg.possible_agents
        }

        # --- action / wrench buffers ------------------------------------------------------
        self._forces = torch.zeros(self.num_envs, len(self._falcon_rotor_idx), 3, device=self.device)
        self._moments = torch.zeros(self.num_envs, len(self._falcon_idx), 3, device=self.device)
        self._hook_force = torch.zeros(self.num_envs, len(self._hook_idx), 3, device=self.device)
        self._setpoints = {a: {} for a in self.cfg.possible_agents}
        self.prev_actions = {
            a: torch.zeros(self.num_envs, cfg.action_dim, device=self.device) for a in self.cfg.possible_agents
        }
        # one step further back, for the 2nd-difference (action-accel) penalty
        self.prev_prev_actions = {
            a: torch.zeros(self.num_envs, cfg.action_dim, device=self.device) for a in self.cfg.possible_agents
        }

        # --- drone state buffers ----------------------------------------------------------
        self.drone_positions = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_orientations = torch.zeros(self.num_envs, self._num_drones, 4, device=self.device)
        self.drone_orientations[..., 0] = 1.0
        self.drone_rot_matrices = torch.zeros(self.num_envs, self._num_drones, 3, 3, device=self.device)
        self.drone_linear_velocities = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_angular_velocities = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_linear_accelerations = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.drone_angular_accelerations = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._drone_jerk = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._drone_prev_acc = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        # latest INDI disturbance residual per drone (body frame, 6): [alpha_cmd - omega_dot | R^T(acc_cmd - a_meas)],
        # read from each drone's INDI controller after _apply_action; fed to the actor obs iff cfg.actor_indi_residual.
        self._indi_residual = torch.zeros(self.num_envs, self._num_drones, 6, device=self.device)

        # --- hook state buffers (one hook per drone) --------------------------------------
        self.hook_positions = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.hook_velocities = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.hook_orientations = torch.zeros(self.num_envs, self._num_drones, 4, device=self.device)
        self.hook_orientations[..., 0] = 1.0

        # --- controllers / motor models (per drone), identical cascade to the hover task ----
        self.geo_controllers = {i: GeometricController(self.num_envs, self._control_mode) for i in range(self._num_drones)}
        self._indi_controllers = {i: IndiController(self.num_envs) for i in range(self._num_drones)}
        self._ll_counter = 0
        self._constant_yaw = torch.zeros([self.num_envs, 1], device=self.device)
        initial_rpm = torch.tensor([[1400.0, 1350.0, 1350.0, 1400.0]], device=self.device).repeat(self.num_envs, 1)
        self.motor_models = {i: RotorMotor(self.num_envs, initial_rpm.clone()) for i in range(self._num_drones)}
        self.sampling_time = self.sim.get_physics_dt() * self.cfg.low_level_decimation

        # --- attachment state (per drone) -------------------------------------------------
        self.attached = torch.zeros(self.num_envs, self._num_drones, dtype=torch.bool, device=self.device)
        self.slack_timer = torch.zeros(self.num_envs, self._num_drones, dtype=torch.long, device=self.device)
        # catch_latch: True once the single per-dwell catch attempt has been drawn while the hook is inside the
        # capture radius; reset when the hook leaves the zone (one draw per approach, not per step).
        self.catch_latch = torch.zeros(self.num_envs, self._num_drones, dtype=torch.bool, device=self.device)
        self.attach_point_w = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self.hold_counter = torch.zeros(self.num_envs, self._num_drones, dtype=torch.long, device=self.device)
        self._newly_caught = torch.zeros(self.num_envs, self._num_drones, dtype=torch.bool, device=self.device)
        self._just_detached = torch.zeros(self.num_envs, self._num_drones, dtype=torch.bool, device=self.device)
        # previous-step hook->target distance for the approach progress reward (per drone)
        self._prev_hook_wall = torch.zeros(self.num_envs, self._num_drones, device=self.device)
        # envs reset since the last reward pass: seed _prev_hook_wall with the post-spawn distance on the next
        # _get_rewards so the first-step approach delta is 0 (no cross-episode jump)
        self._reset_approach_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # envs reset since the last _get_states: refill their frame-stack with the fresh state
        self._state_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # per-episode TARGET WALL assignment (index 0..3 into the 4 basket walls), sampled at reset. The drone is
        # told (via its observation) the centre of ITS assigned wall; approach reward and catch use that wall only.
        self._assigned_wall = torch.zeros(self.num_envs, self._num_drones, dtype=torch.long, device=self.device)
        # wire pin-force magnitude transmitted to the hook ~ tension (for the critic state)
        self._hook_tension = torch.zeros(self.num_envs, self._num_drones, device=self.device)

        # --- basket + target-bar buffers (world frame, per episode) ----------------------
        self.basket_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.basket_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.basket_quat_w[:, 0] = 1.0
        self.bar_center_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.bar_axis_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.bar_axis_w[:, 1] = 1.0
        self.wall_normal_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.wall_normal_w[:, 0] = 1.0
        self.wall_up_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.wall_up_w[:, 2] = 1.0  # vertical walls -> up is world +z
        # centres of the 4 walls in the basket frame (+x, -x, +y, -y), panel centre at z = H/2
        _hx, _hy = self.cfg.basket_half_x, self.cfg.basket_half_y
        _H = self.cfg.basket_wall_height
        self._wall_centers_const = torch.tensor(
            [[_hx, 0.0, _H / 2], [-_hx, 0.0, _H / 2], [0.0, _hy, _H / 2], [0.0, -_hy, _H / 2]], device=self.device
        )  # (4, 3)

        # --- transport GOAL pose (observed like the hover task's pose_command) ------------
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:, 3] = 1.0  # identity quat

        # --- reward logging ---------------------------------------------------------------
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "approach", "proximity", "impact_perp", "impact_up",
                "catch", "success", "alive", "death", "lock", "tension", "wall_pull", "progress", "detach",
                "action_smoothness", "body_rate_penalty", "force_penalty", "downwash", "swing_limit",
                "swing_shape", "above_box", "action_rate", "action_accel",
            ]
        }
        self.metrics = {
            "hook_bar_dist": torch.zeros(self.num_envs, device=self.device),
            "attached_frac": torch.zeros(self.num_envs, device=self.device),
            "capture_radius": torch.zeros(self.num_envs, device=self.device),
            "hold_req": torch.zeros(self.num_envs, device=self.device),
            "spawn_std": torch.zeros(self.num_envs, device=self.device),
            "wall_penetration": torch.zeros(self.num_envs, device=self.device),
        }

        # --- termination buffers ----------------------------------------------------------
        self.drone_fly_low = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.entangled = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.body_pos_outside = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.nonfinite = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.runaway = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.hard_entangle = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # hold milestone: succeeded = the hold requirement is met THIS step; _succeeded_once = met at any point
        # this episode (per-episode metric); _newly_succeeded = the first step it was met (one-off bonus).
        self.succeeded = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._succeeded_once = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._newly_succeeded = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.set_debug_vis(cfg.debug_vis)

    # =====================================================================================
    # scene
    # =====================================================================================
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        contact_sensors = ContactSensor(self.cfg.contact_forces)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # Weld a coordinate-frame visual to the hook body IN THE USD (child of hook_link) so it is driven by the
        # physics->render pipeline like the hook mesh (no lag). Spawned under the source env before cloning.
        if self.cfg.debug_vis:
            frame_cfg = sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd", scale=(0.08, 0.08, 0.08)
            )
            frame_cfg.func(
                "/World/envs/env_0/flyhook/hook_link/hook_frame", frame_cfg, translation=(0.0, 0.0, -0.02)
            )
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["contact_forces"] = contact_sensors
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # =====================================================================================
    # actions
    # =====================================================================================
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        for agent in self.cfg.possible_agents:
            self.prev_prev_actions[agent][:] = self.prev_actions[agent]   # a_{t-2}
            self.prev_actions[agent][:] = self.actions[agent]             # a_{t-1}
            self.actions[agent][:] = actions[agent]                       # a_t

        for i, agent in enumerate(self.cfg.possible_agents):
            action = actions[agent]
            if self._control_mode == "ACCBR":
                self._setpoints[agent]["lin_acc"] = action[:, :3]
                self._setpoints[agent]["body_rates"] = torch.cat((action[:, 3:5], self._constant_yaw), dim=-1)
            self._setpoints[agent]["yaw"] = self._constant_yaw
            self._setpoints[agent]["yaw_rate"] = self._constant_yaw
            self._setpoints[agent]["yaw_acc"] = self._constant_yaw

    def _apply_action(self) -> None:
        if self._ll_counter % self.cfg.low_level_decimation == 0:
            all_thrusts, all_moments = [], []
            drone_pos = self.robot.data.body_com_state_w[:, self._falcon_idx, :3] - self.scene.env_origins.unsqueeze(1)
            drone_quat = self.robot.data.body_com_state_w[:, self._falcon_idx, 3:7]
            drone_lin_vel = self.robot.data.body_com_state_w[:, self._falcon_idx, 7:10]
            drone_ang_vel = self.robot.data.body_com_state_w[:, self._falcon_idx, 10:13]
            drone_lin_acc = self.robot.data.body_acc_w[:, self._falcon_idx, :3]
            drone_ang_acc = self.robot.data.body_acc_w[:, self._falcon_idx, 3:6]

            for i in range(self._num_drones):
                agent = self.cfg.possible_agents[i]
                ds = {
                    "pos": drone_pos[:, i], "quat": drone_quat[:, i],
                    "lin_vel": drone_lin_vel[:, i], "ang_vel": drone_ang_vel[:, i],
                    "lin_acc": drone_lin_acc[:, i], "ang_acc": drone_ang_acc[:, i],
                }
                self._drone_jerk[:, i] = (ds["lin_acc"] - self._drone_prev_acc[:, i]) / self.step_dt
                ds["jerk"] = self._drone_jerk[:, i]
                self._drone_prev_acc[:, i] = ds["lin_acc"]

                alpha_cmd, acc_load, acc_cmd, q_cmd = self.geo_controllers[i].getCommand(
                    ds, self._forces[:, i * 4 : i * 4 + 4], self._setpoints[agent]
                )
                target_rpm = self._indi_controllers[i].getCommand(
                    ds, self._forces[:, i * 4 : i * 4 + 4], alpha_cmd, acc_cmd, acc_load
                )
                # cache the INDI disturbance residual (hook/wire wrench sensed onboard) for the actor obs
                self._indi_residual[:, i] = self._indi_controllers[i].disturbance_residual
                thrusts, moments = self.motor_models[i].get_motor_thrusts_moments(target_rpm, self.sampling_time)
                all_thrusts.append(thrusts)
                all_moments.append(moments)

            forces = torch.cat(all_thrusts, dim=-1)
            torques = torch.cat(all_moments, dim=-1)
            self._forces[..., 2] = forces
            self._moments[..., 2] = torques.view(self.num_envs, self._num_drones, 4).sum(-1)
            self._compute_hook_pin_force()
            self._ll_counter = 0
        self._ll_counter += 1

        # Apply ALL external wrenches in ONE call (see _ext_body_ids): body torques on the falcon, thrusts on the
        # rotors, PD-pin force on the hook.
        combined_forces = torch.cat((torch.zeros_like(self._moments), self._forces, self._hook_force), dim=1)
        combined_torques = torch.cat(
            (self._moments, torch.zeros_like(self._forces), torch.zeros_like(self._hook_force)), dim=1
        )
        self.robot.set_external_force_and_torque(combined_forces, combined_torques, self._ext_body_ids)

    def _compute_hook_pin_force(self):
        """While attached, pin the hook tip to ``attach_point_w`` with a stiff PD spring-damper; the wire chain
        then transmits tension to the drone. Force is expressed in the hook body frame (like the rotor thrusts)."""
        hook_pos = self.robot.data.body_com_state_w[:, self._hook_idx, :3] - self.scene.env_origins.unsqueeze(1)
        hook_vel = self.robot.data.body_com_state_w[:, self._hook_idx, 7:10]
        hook_quat = self.robot.data.body_com_state_w[:, self._hook_idx, 3:7]
        kp, kd, fmax = 400.0, 20.0, 60.0
        err = self.attach_point_w - hook_pos                       # (E, D, 3) world
        pin_w = torch.clamp(kp * err - kd * hook_vel, -fmax, fmax)
        pin_w = pin_w * self.attached.unsqueeze(-1)                # pin only when attached
        self._hook_tension[:] = torch.norm(pin_w, dim=-1)         # ~ wire tension (world frame)
        # surrogate wall collision: keep the hook out of the (collider-less) wall shell; does NOT count as tension
        wall_w = self._wall_collision_force(hook_pos, hook_vel)
        force_w = pin_w + wall_w
        # world -> hook body frame
        force_b = quat_rotate_inverse(hook_quat.reshape(-1, 4), force_w.reshape(-1, 3)).view_as(force_w)
        self._hook_force[:] = force_b

    def _wall_collision_force(self, hook_pos_w: torch.Tensor, hook_vel_w: torch.Tensor) -> torch.Tensor:
        """Outward spring-damper that stops the hook penetrating the (visualization-only) basket walls.

        The 4 vertical walls form a footprint box of half-extents (hx, hy) spanning z in [0, wall_height]; the TOP
        is open (the hook is swung over the rim). Inside that shell, push out along the nearest LATERAL wall normal
        with force = clamp(k*depth - c*v_into_wall, 0, fmax). Returns a world-frame force (E, D, 3).
        """
        E, D = hook_pos_w.shape[0], hook_pos_w.shape[1]
        q = self.basket_quat_w.unsqueeze(1).expand(E, D, 4).reshape(-1, 4)
        rel = (hook_pos_w - self.basket_pos_w.unsqueeze(1)).reshape(-1, 3)
        hook_bf = quat_rotate_inverse(q, rel)
        vel_bf = quat_rotate_inverse(q, hook_vel_w.reshape(-1, 3))
        hx = self.cfg.basket_half_x + self.cfg.wall_collision_margin
        hy = self.cfg.basket_half_y + self.cfg.wall_collision_margin
        H = self.cfg.basket_wall_height
        x, y, z = hook_bf[:, 0], hook_bf[:, 1], hook_bf[:, 2]
        inside = (x.abs() < hx) & (y.abs() < hy) & (z > 0.0) & (z < H)   # in the wall shell (top open)
        dx, dy = hx - x.abs(), hy - y.abs()                             # lateral penetration into x-/y-walls
        use_x = dx < dy                                                 # nearest lateral face
        depth = torch.where(use_x, dx, dy)
        zc = torch.zeros_like(x)
        nrm = torch.stack(
            [torch.where(use_x, torch.sign(x), zc), torch.where(~use_x, torch.sign(y), zc), zc], dim=-1
        )                                                              # outward lateral normal (basket frame)
        v_out = (vel_bf * nrm).sum(-1)                                  # +out, -into the wall
        mag = (self.cfg.wall_collision_k * depth - self.cfg.wall_collision_damp * v_out).clamp(min=0.0)
        mag = (mag * inside.float()).clamp(max=self.cfg.wall_collision_fmax)
        force_bf = nrm * mag.unsqueeze(-1)
        return quat_rotate(q, force_bf).reshape(E, D, 3)

    # =====================================================================================
    # state refresh + attachment update
    # =====================================================================================
    def _refresh_states(self):
        self.drone_positions[:] = self.robot.data.body_com_state_w[:, self._falcon_idx, :3] - self.scene.env_origins.unsqueeze(1)
        self.drone_orientations[:] = self.robot.data.body_com_state_w[:, self._falcon_idx, 3:7]
        self.drone_rot_matrices[:] = matrix_from_quat(self.drone_orientations)
        self.drone_linear_velocities[:] = self.robot.data.body_com_state_w[:, self._falcon_idx, 7:10]
        self.drone_angular_velocities[:] = self.robot.data.body_com_state_w[:, self._falcon_idx, 10:13]
        self.hook_positions[:] = self.robot.data.body_com_state_w[:, self._hook_idx, :3] - self.scene.env_origins.unsqueeze(1)
        self.hook_velocities[:] = self.robot.data.body_com_state_w[:, self._hook_idx, 7:10]
        self.hook_orientations[:] = self.robot.data.body_com_state_w[:, self._hook_idx, 3:7]

    def _wall_candidates_bf(self, hook_pos: torch.Tensor) -> torch.Tensor:
        """Closest point on each of the 4 wall faces to the hook, in the BASKET frame. (E, 4, 3).

        Each wall face spans its full length over the bar z-band [basket_wall_z_lo, basket_wall_z_hi]; the
        candidate is the hook projected onto that face and clamped into the face's extent. Face order is
        (+x, -x, +y, -y). ``hook_pos`` is world (env-local) (E, 3).
        """
        q = self.basket_quat_w
        hook_bf = quat_rotate_inverse(q, hook_pos - self.basket_pos_w)          # hook in basket frame (E, 3)
        hx, hy = self.cfg.basket_half_x, self.cfg.basket_half_y
        x, y, z = hook_bf[:, 0], hook_bf[:, 1], hook_bf[:, 2]
        zc = z.clamp(self.cfg.basket_wall_z_lo, self.cfg.basket_wall_z_hi)  # closest point in the catch z-band
        cy = y.clamp(-hy, hy)                                               # along an x-wall (axis = y)
        cx = x.clamp(-hx, hx)                                               # along a y-wall (axis = x)
        hxf, hyf = torch.full_like(x, hx), torch.full_like(x, hy)
        return torch.stack(
            [
                torch.stack([hxf, cy, zc], -1),
                torch.stack([-hxf, cy, zc], -1),
                torch.stack([cx, hyf, zc], -1),
                torch.stack([cx, -hyf, zc], -1),
            ],
            dim=1,
        )                                                                      # (E, 4, 3)

    def _wall_point_by_index(self, hook_pos: torch.Tensor, idx: torch.Tensor):
        """Closest point on the SPECIFIED wall (``idx`` (E,) in 0..3) to the hook, plus that wall's frame (world).

        Returns ``(point_w, bar_axis_w, wall_normal_w)``, each (E, 3).
        """
        q = self.basket_quat_w
        pts = self._wall_candidates_bf(hook_pos)                               # (E, 4, 3) basket frame
        ar = torch.arange(hook_pos.shape[0], device=self.device)
        chosen_bf = pts[ar, idx]                                               # (E, 3)
        normals = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]], device=self.device)[idx]
        axes = torch.tensor([[0.0, 1, 0], [0.0, 1, 0], [1.0, 0, 0], [1.0, 0, 0]], device=self.device)[idx]
        point_w = self.basket_pos_w + quat_rotate(q, chosen_bf)
        normal_w = normalize(quat_rotate(q, normals))
        axis_w = normalize(quat_rotate(q, axes))
        return point_w, axis_w, normal_w

    def _assigned_wall_point(self, hook_pos: torch.Tensor, i: int):
        """Closest point on drone ``i``'s ASSIGNED wall to the hook + that wall's frame (world)."""
        return self._wall_point_by_index(hook_pos, self._assigned_wall[:, i])

    def _assigned_wall_center(self, i: int) -> torch.Tensor:
        """Offset (x, y, z) of the CENTRE of drone ``i``'s assigned wall RELATIVE TO THE BASKET CoM, in world
        axes. (E, 3). Replaces the ego one-hot id in the observation: it identifies the drone's role and points
        it at the side of the basket to approach and latch onto."""
        c_bf = self._wall_centers_const[self._assigned_wall[:, i]]             # (E, 3) basket frame
        return quat_rotate(self.basket_quat_w, c_bf)                           # (E, 3) relative to basket CoM

    def _target_bar_points(self, hook_pos: torch.Tensor, i: int) -> torch.Tensor:
        """Closest point on drone ``i``'s assigned wall to the hook (world). ``hook_pos`` (E, 3)."""
        return self._assigned_wall_point(hook_pos, i)[0]

    def _update_attachment(self):
        """Advance the curricula and the surrogate attachment state for every drone."""
        self._ctrl_step += 1
        # capture-radius curriculum: shrink linearly start -> end over capture_curriculum_steps, then hold
        if self.cfg.capture_curriculum:
            frac = min(1.0, self._ctrl_step / max(1, self.cfg.capture_curriculum_steps))
            self._acfg.capture_radius = self._capture_start + (self._capture_end - self._capture_start) * frac
        # chord-angle curriculum: tighten the min swing-out angle start -> target over chord_angle_steps
        if self.cfg.chord_angle_curriculum:
            cfrac = min(1.0, self._ctrl_step / max(1, self.cfg.chord_angle_steps))
            self._acfg.min_chord_angle_deg = self.cfg.chord_angle_start_deg + (self._chord_angle_end - self.cfg.chord_angle_start_deg) * cfrac
        # hold-time curriculum: power ramp (stays near the easy start for long, then rises to the final hold)
        if self.cfg.hold_curriculum:
            hfrac = min(1.0, self._ctrl_step / max(1, self.cfg.hold_curriculum_steps)) ** self.cfg.hold_curriculum_power
            self._current_hold_req = int(round(
                self.cfg.hold_success_steps_start
                + (self.cfg.hold_success_steps - self.cfg.hold_success_steps_start) * hfrac
            ))
        self._newly_caught[:] = False
        self._just_detached[:] = False
        for i in range(self._num_drones):
            hook_pos = self.hook_positions[:, i]
            hook_vel = self.hook_velocities[:, i]
            anchor = self.drone_positions[:, i]               # wire drone-side anchor ~ drone CoM
            # closest point on this drone's ASSIGNED wall + that wall's frame (catch ONLY on the assigned side)
            bar_pos, bar_axis, wall_normal = self._assigned_wall_point(hook_pos, i)
            self.bar_center_w = bar_pos
            self.bar_axis_w = bar_axis
            self.wall_normal_w = wall_normal
            attached, slack_timer, attach_pt, newly, detached, latch = att.update_attachment_state(
                self.attached[:, i], self.slack_timer[:, i], self.attach_point_w[:, i],
                hook_pos=hook_pos, hook_vel=hook_vel, anchor_pos=anchor,
                bar_pos=bar_pos, bar_axis=bar_axis, wall_normal=wall_normal, wall_up=self.wall_up_w,
                cfg=self._acfg, catch_latch=self.catch_latch[:, i],
            )
            self.attached[:, i] = attached
            self.slack_timer[:, i] = slack_timer
            self.attach_point_w[:, i] = attach_pt
            self._newly_caught[:, i] = newly
            self._just_detached[:, i] = detached
            self.catch_latch[:, i] = latch
            # held = attached AND taut (not slack); count consecutive held steps for success
            taut = attached & (~att.is_slack(anchor, attach_pt, self._acfg))
            self.hold_counter[:, i] = torch.where(taut, self.hold_counter[:, i] + 1, torch.zeros_like(self.hold_counter[:, i]))

    # =====================================================================================
    # observations
    # =====================================================================================
    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._refresh_states()
        self._update_attachment()

        observations = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            frame = self._drone_obs_frame(i, agent)
            self._observation_buffers[agent].append(frame)
            observations[agent] = self._observation_buffers[agent].buffer.reshape(self.num_envs, -1)
        return observations

    def _drone_obs_frame(self, i: int, agent: str) -> torch.Tensor:
        # Hook-BLIND actor frame with the hover/transport partial-observation layout (45): the BASKET plays the
        # role of the hover "load" and pose_command_w the transport "goal":
        #   basket pos (3) + basket rot matrix (9)                                                = 12
        #   assigned-wall centre rel. basket (3) + drone pos (3) + rot (9) + lin vel (3) + ang vel (3) = 21
        #   goal_pos_error = goal - basket (3) + difference_matrix (goal vs basket orient) (9)     = 12
        # plus, iff cfg.actor_indi_residual, the 6-dim INDI disturbance residual (hook/wire wrench sensed
        # onboard: force and phase, NO position).
        basket_matrix = matrix_from_quat(self.basket_quat_w)                         # (E, 3, 3)
        goal_matrix = matrix_from_quat(self.pose_command_w[:, 3:7])                  # (E, 3, 3)
        goal_pos_error = self.pose_command_w[:, :3] - self.basket_pos_w              # goal - basket position
        difference_matrix = torch.matmul(goal_matrix, basket_matrix.transpose(-1, -2))
        assigned_wall_center = self._assigned_wall_center(i)                         # (E, 3)
        parts = [
            self.basket_pos_w,                                       # 3
            basket_matrix.reshape(self.num_envs, 9),                 # 9
            assigned_wall_center,                                    # 3
            self.drone_positions[:, i],                              # 3
            self.drone_rot_matrices[:, i].reshape(self.num_envs, 9), # 9
            self.drone_linear_velocities[:, i],                     # 3
            self.drone_angular_velocities[:, i],                    # 3
            goal_pos_error,                                          # 3
            difference_matrix.reshape(self.num_envs, 9),            # 9
        ]                                                            # = 45
        if self.cfg.actor_indi_residual:
            parts.append(self._indi_residual[:, i])                 # 6
        # scrub non-finite values: a wire-tension transient can spike the residual on the very step the env is
        # reset by _get_dones; that frame still enters the rollout buffer.
        return torch.nan_to_num(torch.cat(parts, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)

    def _get_states(self) -> torch.Tensor:
        # PRIVILEGED critic state: the actor never sees the hook; the critic additionally gets the full hook state
        # and the hook->bar error. Layout (60 per frame):
        #   basket: pos(3) + rot(9) + lin vel(3, =0 static) + ang vel(3, =0)                   = 18
        #   per drone: pos(3) + rot(9) + lin vel(3) + ang vel(3)                                = 18
        #   per drone hook: hook_rel_drone(3) + hook vel(3) + hook->bar err(3)
        #                   + attached(1) + tension(1) + hold_progress(1)                       = 12
        #   goal: goal_pos_error(3) + difference_matrix(9)                                      = 12
        basket_matrix = matrix_from_quat(self.basket_quat_w)
        goal_matrix = matrix_from_quat(self.pose_command_w[:, 3:7])
        goal_pos_error = self.pose_command_w[:, :3] - self.basket_pos_w
        difference_matrix = torch.matmul(goal_matrix, basket_matrix.transpose(-1, -2))
        zeros3 = torch.zeros(self.num_envs, 3, device=self.device)
        parts = [
            self.basket_pos_w,                             # 3
            basket_matrix.reshape(self.num_envs, 9),       # 9
            zeros3,                                        # 3  basket lin vel (static)
            zeros3,                                        # 3  basket ang vel
        ]
        for i in range(self._num_drones):
            parts += [
                self.drone_positions[:, i],                                # 3
                self.drone_rot_matrices[:, i].reshape(self.num_envs, 9),   # 9
                self.drone_linear_velocities[:, i],                       # 3
                self.drone_angular_velocities[:, i],                      # 3
            ]
        for i in range(self._num_drones):
            hook_pos = self.hook_positions[:, i]
            bar_pos = self._target_bar_points(hook_pos, i)
            tension = (self._hook_tension[:, i] / self.cfg.tension_ref).clamp(0.0, 1.0).unsqueeze(-1)
            # fraction of the required hold already accumulated (resets to 0 the instant the hook is lost): lets
            # the critic predict the ramping per-step progress reward.
            hold_progress = (self.hold_counter[:, i].float() / max(1, self._current_hold_req)).clamp(0.0, 1.0)
            parts += [
                hook_pos - self.drone_positions[:, i],     # 3  hook relative to drone
                self.hook_velocities[:, i],                # 3  hook velocity
                bar_pos - hook_pos,                        # 3  hook -> bar error vector
                self.attached[:, i].float().unsqueeze(-1), # 1  attached flag
                tension,                                   # 1  wire tension (normalized)
                hold_progress.unsqueeze(-1),               # 1  progress toward the hold milestone [0, 1]
            ]
        parts += [goal_pos_error, difference_matrix.reshape(self.num_envs, 9)]  # 3 + 9
        single = torch.nan_to_num(torch.cat(parts, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)   # (E, 60)

        # frame-stack the last history_len states
        H = self.cfg.history_len
        if not hasattr(self, "_state_hist") or self._state_hist.shape[1] != H:
            self._state_hist = single.unsqueeze(1).repeat(1, H, 1)          # first call: fill all frames
        else:
            self._state_hist = torch.cat([self._state_hist[:, 1:], single.unsqueeze(1)], dim=1)
            rm = self._state_reset_mask
            if rm.any():                                                    # just-reset envs: refill so no
                self._state_hist[rm] = single[rm].unsqueeze(1).repeat(1, H, 1)  # stale cross-episode frames
            self._state_reset_mask[:] = False
        return self._state_hist.reshape(self.num_envs, H * single.shape[-1])

    # =====================================================================================
    # rewards
    # =====================================================================================
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        dt = self.step_dt
        approach = torch.zeros(self.num_envs, device=self.device)
        proximity = torch.zeros(self.num_envs, device=self.device)
        impact_perp = torch.zeros(self.num_envs, device=self.device)
        impact_up = torch.zeros(self.num_envs, device=self.device)
        lock = torch.zeros(self.num_envs, device=self.device)
        tension = torch.zeros(self.num_envs, device=self.device)
        wall_pull = torch.zeros(self.num_envs, device=self.device)
        progress = torch.zeros(self.num_envs, device=self.device)
        swing_limit = torch.zeros(self.num_envs, device=self.device)
        swing_shape = torch.zeros(self.num_envs, device=self.device)
        above_box = torch.zeros(self.num_envs, device=self.device)
        # envs where a drone is in the near-wall UNATTACHED swing zone: action-variability penalties OFF there
        smooth_free = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for i in range(self._num_drones):
            drone_pos = self.drone_positions[:, i]
            hook_pos = self.hook_positions[:, i]
            hook_vel = self.hook_velocities[:, i]
            attached = self.attached[:, i]
            not_att = (~attached).float()

            # --- hook -> ASSIGNED wall: closest point (where the catch can fire) and the shaping target -------
            bar_pos, bar_axis, wall_normal = self._assigned_wall_point(hook_pos, i)
            hook_bar = torch.norm(hook_pos - bar_pos, dim=-1)
            if self.cfg.reward_center_target:
                center_w = self.basket_pos_w + self._assigned_wall_center(i)     # world centre of the assigned wall
                hook_tgt = torch.norm(hook_pos - center_w, dim=-1)
            else:
                hook_tgt = hook_bar

            # freshly-reset envs: seed prev with the post-spawn distance so the first delta is 0
            rm = self._reset_approach_mask
            self._prev_hook_wall[rm, i] = hook_tgt[rm]

            # --- approach: potential-based progress toward the target (unattached phase) -------------------
            delta = self._prev_hook_wall[:, i] - hook_tgt
            approach += self.cfg.approach_weight * delta * not_att
            self._prev_hook_wall[:, i] = hook_tgt

            # --- proximity: exp(-k * dist) concentrated at the target (unattached phase) --------------------
            proximity += self.cfg.proximity_weight * torch.exp(-self.cfg.proximity_k * hook_tgt) * not_att

            if self.cfg.action_smooth_phase_gate:
                smooth_free |= (hook_tgt < self.cfg.smooth_free_zone) & (~attached)

            # --- impact shaping inside the impact zone (hook near the bar, unattached) ---------------------
            in_zone = ((hook_bar < self.cfg.impact_range) & (~attached)).float()
            v_normal = (hook_vel * wall_normal).sum(-1)          # perpendicular to wall (>0 out, <0 into)
            v_up = (hook_vel * self.wall_up_w).sum(-1)           # in-plane, along the wall's up axis
            # penalize only the EXCESS into-wall speed above the soft-contact target (no slamming) ...
            into_speed = torch.clamp(-v_normal, min=0.0)
            excess = torch.clamp(into_speed - self.cfg.impact_target_speed, min=0.0)
            impact_perp += -self.cfg.impact_perp_weight * excess.square() * in_zone
            # ... and reward a gentle into-wall speed up to that target (a perpendicular strike beats a skim)
            if self.cfg.impact_toward_weight > 0.0:
                impact_perp += self.cfg.impact_toward_weight * torch.clamp(into_speed, max=self.cfg.impact_target_speed) * in_zone
            # signed up-the-wall velocity: up rewarded, horizontal 0, down penalized
            impact_up += self.cfg.impact_up_weight * v_up * in_zone

            # --- planar swing shaping (unattached phase) --------------------------------------------------
            # pendulum swing velocity = hook velocity RELATIVE to the drone (a dead hook carried by the drone
            # has zero swing velocity)
            v_swing = hook_vel - self.drone_linear_velocities[:, i]
            v_lat_rel = (v_swing * bar_axis).sum(-1)         # sideways swing along the wall face
            # the drone's OWN velocity perpendicular to the wall (an in-plane pump the actor fully observes):
            # a planar-driven pendulum stays planar
            if self.cfg.drone_perp_osc_weight > 0.0:
                v_dr_perp = (self.drone_linear_velocities[:, i] * wall_normal).sum(-1)
                swing_shape += self.cfg.drone_perp_osc_weight * torch.clamp(v_dr_perp.abs(), max=self.cfg.drone_perp_osc_cap) * not_att
            # commanded-plane terms: u_hat = wall_normal (the perpendicular catch plane), w_hat = bar_axis (off-plane).
            # Positive terms (amp, vel) are gated NEAR the wall so a far-field "swing in place" cannot be farmed;
            # the penalties (off-plane displacement/speed, azimuthal whirl) run over the whole unattached phase.
            if self.cfg.swing_plane_amp_weight > 0.0 or self.cfg.swing_plane_whirl_weight > 0.0:
                rel_sp = hook_pos - drone_pos                      # wire vector (pivot -> bob)
                relh = rel_sp.clone(); relh[:, 2] = 0.0            # horizontal part
                vh = v_swing.clone(); vh[:, 2] = 0.0
                amp_along = (relh * wall_normal).sum(-1).abs()     # in-plane arc amplitude
                amp_off = (relh * bar_axis).sum(-1).abs()          # off-plane displacement (along the bar)
                vel_along = (v_swing * wall_normal).sum(-1).abs()  # in-plane swing speed
                l_z = relh[:, 0] * vh[:, 1] - relh[:, 1] * vh[:, 0]  # azimuthal angular momentum (whirl)
                in_zone_sp = (hook_tgt < self.cfg.swing_plane_zone).float() * not_att
                # penalties are clamped too: a PhysX wire-tension transient can spike the hook velocity
                swing_shape += self.cfg.swing_plane_amp_weight * amp_along.clamp(max=self.cfg.swing_plane_amp_cap) * in_zone_sp
                swing_shape += self.cfg.swing_plane_vel_weight * vel_along.clamp(max=self.cfg.swing_plane_vel_cap) * in_zone_sp
                swing_shape += -self.cfg.swing_plane_off_weight * amp_off.clamp(max=1.0) * not_att
                swing_shape += -self.cfg.swing_plane_vlat_weight * v_lat_rel.abs().clamp(max=3.0) * not_att
                swing_shape += -self.cfg.swing_plane_whirl_weight * l_z.abs().clamp(max=2.0) * not_att

            # --- above-box penalty: drone horizontally OVER the box footprint (props over the load) ----------
            if self.cfg.above_box_penalty_weight > 0.0:
                drone_bf = quat_rotate_inverse(self.basket_quat_w, drone_pos - self.basket_pos_w)
                inside_x = torch.clamp(self.cfg.basket_half_x + self.cfg.above_box_margin - drone_bf[:, 0].abs(), min=0.0)
                inside_y = torch.clamp(self.cfg.basket_half_y + self.cfg.above_box_margin - drone_bf[:, 1].abs(), min=0.0)
                over = (inside_x > 0) & (inside_y > 0)
                above_box += self.cfg.above_box_penalty_weight * torch.minimum(inside_x, inside_y) * over.float()

            # --- attached phase: lock, taut hold, anti-sag, anti-pull-to-wall ------------------------------
            lock += self.cfg.lock_weight * attached.float()
            # taut-margin reward: drone->attach-point distance a comfortable margin ABOVE the slack threshold
            # (ramps 0 -> 1 over [taut_thr, taut_thr + taut_margin_target])
            taut_thr = self._acfg.wire_length - self._acfg.slack_margin
            dist_da = torch.norm(self.drone_positions[:, i] - self.attach_point_w[:, i], dim=-1)
            tension += (
                self.cfg.tension_weight
                * ((dist_da - taut_thr) / self.cfg.taut_margin_target).clamp(0.0, 1.0)
                * attached.float()
            )
            if self.cfg.anti_sag_weight > 0.0:
                v_drone_up = (self.drone_linear_velocities[:, i] * self.wall_up_w).sum(-1)   # + = rising
                tension += -self.cfg.anti_sag_weight * torch.clamp(-v_drone_up, min=0.0) * attached.float()
            if self.cfg.slack_penalty_weight > 0.0:
                tension += -self.cfg.slack_penalty_weight * torch.clamp(taut_thr - dist_da, min=0.0) * attached.float()
            # penalize the drone moving INTO the wall while attached (goes slack and yanks the stiff pin)
            v_drone_normal = (self.drone_linear_velocities[:, i] * wall_normal).sum(-1)   # + = away, - = into wall
            wall_pull += self.cfg.wall_pull_weight * torch.clamp(-v_drone_normal, min=0.0) * attached.float()

            # --- progress: hold_counter / hold_req in [0, 1]; drops to 0 the instant the hook is lost ---------
            progress += (self.hold_counter[:, i].float() / max(1, self._current_hold_req)).clamp(0.0, 1.0)

            # --- swing-limit soft penalty: hook rising toward the drone (prop entanglement) ----------------
            rise = hook_pos[:, 2] - (drone_pos[:, 2] - 0.15)
            swing_limit += self.cfg.swing_limit_weight * torch.clamp(rise, min=0.0)

        self._reset_approach_mask[:] = False      # consumed

        r_approach = approach                     # already a per-step delta; not scaled by dt
        r_proximity = proximity * dt
        r_impact_perp = impact_perp * dt
        r_impact_up = impact_up * dt
        r_lock = lock * dt
        r_tension = tension * dt
        r_wall_pull = -wall_pull * dt
        r_progress = self.cfg.progress_weight * progress * dt
        r_swing_limit = -swing_limit * dt
        r_swing_shape = swing_shape * dt
        r_above_box = -above_box * dt

        # event bonuses on EVERY attach / detach, one-off success bonus on first reaching the hold milestone
        r_catch = self.cfg.catch_bonus * self._newly_caught.any(dim=-1).float()
        r_detach = -self.cfg.detach_penalty * self._just_detached.any(dim=-1).float()
        r_success = self.cfg.success_bonus * self._newly_succeeded.float()

        # survival: + every living step, - on a failure termination (not on success / timeout / entanglement)
        failure = (self.drone_fly_low | self.body_pos_outside | self.runaway | self.nonfinite)
        r_alive = self.cfg.alive_weight * dt * (~failure).float()
        r_death = -self.cfg.death_penalty * failure.float()

        # control regularizers (as in the hover task) + action-variability penalties
        cur = torch.cat([self.actions[a] for a in self.cfg.possible_agents], dim=-1)
        prev = torch.cat([self.prev_actions[a] for a in self.cfg.possible_agents], dim=-1)
        diff = ((cur - prev).abs()) / self._num_drones
        r_smooth = self.cfg.action_smoothness_weight * torch.exp(-torch.norm(diff, dim=-1).square()) * dt
        prev_prev = torch.cat([self.prev_prev_actions[a] for a in self.cfg.possible_agents], dim=-1)
        a_rate = (cur - prev) / self._num_drones                       # 1st diff: abrupt command changes
        a_accel = (cur - 2.0 * prev + prev_prev) / self._num_drones    # 2nd diff: command shake
        smooth_gate = (~smooth_free).float() if self.cfg.action_smooth_phase_gate else 1.0
        r_action_rate = -self.cfg.action_rate_penalty_weight * (a_rate * a_rate).sum(-1) * dt * smooth_gate
        r_action_accel = -self.cfg.action_accel_penalty_weight * (a_accel * a_accel).sum(-1) * dt * smooth_gate
        body_rates = torch.cat([self.actions[a][:, 3:5] for a in self.cfg.possible_agents], dim=-1)
        r_body_rate = self.cfg.body_rate_penalty_weight * torch.exp(-torch.norm(body_rates / self._num_drones, dim=-1)) * dt
        eff = torch.max(self._forces[..., 2] / self.cfg.max_thrust_pp, dim=-1)[0]
        r_effort = self.cfg.force_penalty_weight * torch.exp(-eff) * dt
        r_downwash = self.cfg.downwash_rew_weight * self._downwash_reward() * dt

        reward_terms = {
            "approach": r_approach, "proximity": r_proximity,
            "impact_perp": r_impact_perp, "impact_up": r_impact_up,
            "catch": r_catch, "detach": r_detach, "success": r_success,
            "alive": r_alive, "death": r_death,
            "lock": r_lock, "tension": r_tension, "wall_pull": r_wall_pull, "progress": r_progress,
            "action_smoothness": r_smooth, "body_rate_penalty": r_body_rate, "force_penalty": r_effort,
            "action_rate": r_action_rate, "action_accel": r_action_accel,
            "downwash": r_downwash, "swing_limit": r_swing_limit,
            "swing_shape": r_swing_shape, "above_box": r_above_box,
        }
        for k, v in reward_terms.items():
            self._episode_sums[k] += v
        shared = sum(reward_terms.values())
        # a divergent step (reset by _get_dones this same step) must not emit a NaN/Inf reward into the batch
        shared = torch.nan_to_num(shared, nan=0.0, posinf=0.0, neginf=0.0)
        return {agent: shared for agent in self.cfg.possible_agents}

    def _downwash_reward(self) -> torch.Tensor:
        """Reward keeping the hook OUT of the drone's prop-wash line (thrust axis): ``1 - exp(-scale * d)`` with
        ``d`` the perpendicular distance from the hook to the thrust axis. Min over drones."""
        zb = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs * self._num_drones, 1)
        thrust_dir = quat_rotate(self.drone_orientations.reshape(-1, 4), zb).view(self.num_envs, self._num_drones, 3)
        rel = self.hook_positions - self.drone_positions                      # (E, D, 3) drone -> hook
        along = (rel * thrust_dir).sum(-1, keepdim=True) * thrust_dir         # component along thrust axis
        line_dist = torch.norm(rel - along, dim=-1)                           # (E, D) perpendicular distance
        return 1.0 - torch.exp(-self.cfg.downwash_scale * line_dist.min(dim=-1).values)

    # =====================================================================================
    # terminations
    # =====================================================================================
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.drone_fly_low = (self.drone_positions[:, :, 2] < self.cfg.min_drone_height).any(dim=-1)
        # entanglement: hook rises to near the drone -- SOFT (penalty) diagnostic only
        self.entangled = (self.hook_positions[:, :, 2] > self.drone_positions[:, :, 2] - self.cfg.entangle_margin).any(dim=-1)
        self.body_pos_outside = (self.drone_positions.abs() > self.cfg.bounding_box_threshold).any(dim=-1).any(dim=-1)
        # robustness guards: a diverging env MUST reset, else its NaN/Inf state poisons the whole batch
        speed = torch.norm(self.drone_linear_velocities, dim=-1)                      # (E, D)
        finite = (torch.isfinite(self.drone_positions).all(-1)
                  & torch.isfinite(self.drone_linear_velocities).all(-1)
                  & torch.isfinite(self.hook_positions).all(-1))                      # (E, D)
        self.nonfinite = ~finite.all(dim=-1)
        self.runaway = (speed > self.cfg.runaway_speed).any(dim=-1) | (~torch.isfinite(speed).all(dim=-1))
        # catastrophic wrap: hook risen WELL ABOVE the drone (a real over-the-top wire wrap)
        self.hard_entangle = (self.hook_positions[:, :, 2]
                              > self.drone_positions[:, :, 2] + self.cfg.hard_entangle_margin).any(dim=-1)
        # hold milestone: every drone has held (attached & taut) for the required window -> one-off success
        # bonus AND the episode ends (a completed grasp)
        self.succeeded = (self.hold_counter >= self._current_hold_req).all(dim=-1)
        self._newly_succeeded = self.succeeded & (~self._succeeded_once)
        self._succeeded_once |= self.succeeded

        self._update_metrics()

        terminations = (self.drone_fly_low | self.body_pos_outside | self.succeeded
                        | self.nonfinite | self.runaway | self.hard_entangle)
        self.time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = {a: terminations for a in self.cfg.possible_agents}
        time_outs = {a: self.time_out for a in self.cfg.possible_agents}
        return terminated, time_outs

    def _update_metrics(self):
        with torch.no_grad():
            hook_bar = torch.zeros(self.num_envs, device=self.device)
            for i in range(self._num_drones):
                bar_pos = self._target_bar_points(self.hook_positions[:, i], i)
                hook_bar += torch.norm(self.hook_positions[:, i] - bar_pos, dim=-1)
            self.metrics["hook_bar_dist"] = hook_bar / self._num_drones
            self.metrics["attached_frac"] = self.attached.float().mean(dim=-1)
            self.metrics["capture_radius"] = torch.full(
                (self.num_envs,), float(self._acfg.capture_radius), device=self.device
            )
            self.metrics["hold_req"] = torch.full(
                (self.num_envs,), float(self._current_hold_req), device=self.device
            )
            _spawn_std = self.cfg.spawn_std_start + (
                self.cfg.spawn_std_end - self.cfg.spawn_std_start
            ) * self._spawn_frac()
            self.metrics["spawn_std"] = torch.full((self.num_envs,), float(_spawn_std), device=self.device)
            # wall penetration: hook depth INSIDE the wall footprint below the rim (should stay ~0)
            self.metrics["wall_penetration"] = self._hook_wall_penetration().amax(dim=-1)

    def _hook_wall_penetration(self) -> torch.Tensor:
        """Depth (m) each drone's hook is inside the basket wall footprint below the rim; 0 outside/above. (E, D)."""
        E, D = self.num_envs, self._num_drones
        q = self.basket_quat_w.unsqueeze(1).expand(E, D, 4).reshape(-1, 4)
        rel = (self.hook_positions - self.basket_pos_w.unsqueeze(1)).reshape(-1, 3)
        hb = quat_rotate_inverse(q, rel)
        x, y, z = hb[:, 0], hb[:, 1], hb[:, 2]
        hx, hy, H = self.cfg.basket_half_x, self.cfg.basket_half_y, self.cfg.basket_wall_height
        inside = (x.abs() < hx) & (y.abs() < hy) & (z > 0.0) & (z < H)
        depth = torch.minimum(hx - x.abs(), hy - y.abs()).clamp(min=0.0) * inside.float()
        return depth.view(E, D)

    # =====================================================================================
    # reset
    # =====================================================================================
    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        self._reset_target_pose(env_ids)
        # spawn the drone in front of its assigned wall (after the reset events + basket/assigned wall are set)
        self._spawn_curriculum_reset(env_ids)

        # reset attachment + histories
        self.attached[env_ids] = False
        self.slack_timer[env_ids] = 0
        self.catch_latch[env_ids] = False
        self.hold_counter[env_ids] = 0
        self.attach_point_w[env_ids] = 0.0
        self._succeeded_once[env_ids] = False
        self._newly_succeeded[env_ids] = False
        # attached-spawn curriculum: a decaying fraction of these envs starts ALREADY attached (must run after the
        # attach buffers are zeroed and the spawn pose is set, since it overrides both for the selected subset)
        self._attached_spawn_reset(env_ids)
        self._reset_approach_mask[env_ids] = True
        self._state_reset_mask[env_ids] = True
        for agent in self.cfg.possible_agents:
            self._observation_buffers[agent].reset(env_ids)
            self.prev_actions[agent][env_ids] = 0.0
            self.prev_prev_actions[agent][env_ids] = 0.0
            self.actions[agent][env_ids] = 0.0

        if "log" not in self.extras:
            self.extras["log"] = dict()
        for name, buf in [
            ("fly_low", self.drone_fly_low), ("entangled", self.entangled),
            ("bounding_box", self.body_pos_outside), ("success", self._succeeded_once),
            ("time_out", self.time_out), ("nonfinite", self.nonfinite),
            ("runaway", self.runaway), ("hard_entangle", self.hard_entangle),
        ]:
            self.extras["log"][f"Episode_Termination/{name}"] = torch.count_nonzero(buf[env_ids]).item()
        for key in self._episode_sums:
            self.extras["log"][f"Episode_Reward/{key}"] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        for name, val in self.metrics.items():
            self.extras["log"][f"Metrics/{name}"] = val.mean()

    def _reset_target_pose(self, env_ids):
        n = len(env_ids)
        rng = self.cfg.basket_pose_range

        def _uni(lohi):
            return torch.empty(n, device=self.device).uniform_(*lohi)

        self.basket_pos_w[env_ids, 0] = _uni(rng["pos_x"])
        self.basket_pos_w[env_ids, 1] = _uni(rng["pos_y"])
        self.basket_pos_w[env_ids, 2] = _uni(rng["pos_z"])
        yaw = _uni(rng["yaw"])
        zeros = torch.zeros(n, device=self.device)
        self.basket_quat_w[env_ids] = quat_from_euler_xyz(zeros, zeros, yaw)

        # per-episode TARGET-WALL assignment: a random one of the 4 walls per drone
        self._assigned_wall[env_ids] = torch.randint(
            0, 4, (n, self._num_drones), device=self.device, dtype=torch.long
        )

        # transport GOAL pose = the basket's start pose + a small random offset (z clamped above a floor)
        go = self.cfg.goal_offset_range
        self.pose_command_w[env_ids, 0] = self.basket_pos_w[env_ids, 0] + _uni(go["x"])
        self.pose_command_w[env_ids, 1] = self.basket_pos_w[env_ids, 1] + _uni(go["y"])
        self.pose_command_w[env_ids, 2] = (self.basket_pos_w[env_ids, 2] + _uni(go["z"])).clamp_min(self.cfg.goal_z_floor)
        goal_yaw = yaw + _uni(go["yaw"])
        self.pose_command_w[env_ids, 3:7] = quat_from_euler_xyz(zeros, zeros, goal_yaw)

    def _spawn_frac(self) -> float:
        """Spawn-curriculum progress in [0, 1] (0 = tight/in-front, 1 = full range). 1.0 when the curriculum is
        off (eval -> full range, zero swing assist)."""
        if not self.cfg.spawn_curriculum:
            return 1.0
        return min(1.0, self._ctrl_step / max(1, self.cfg.spawn_curriculum_steps))

    def _spawn_curriculum_reset(self, env_ids):
        """Spawn the drone in front of its assigned wall: radial distance ~ N(spawn_dist_mu, std) with a std that
        grows with the curriculum, azimuth widening 0 -> +/-pi, height spawn_height above the basket. Optionally
        seeds an initial velocity toward the wall (swing assist) that decays with the curriculum."""
        M = len(env_ids)
        frac = self._spawn_frac()
        q = self.basket_quat_w[env_ids]                                        # (M, 4)
        idx = self._assigned_wall[env_ids, 0]                                  # (M,) drone-0 assigned wall
        normals_bf = torch.tensor(
            [[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]], device=self.device
        )[idx]                                                                 # (M, 3) outward normal (basket frame)
        normal_w = quat_rotate(q, normals_bf)                                  # (M, 3) world axes
        base_ang = torch.atan2(normal_w[:, 1], normal_w[:, 0])                 # (M,)
        az = min(frac * math.pi, self.cfg.spawn_azimuth_max)
        jitter = (torch.rand(M, device=self.device) * 2 - 1) * az
        ang = base_ang + jitter
        std = self.cfg.spawn_std_start + (self.cfg.spawn_std_end - self.cfg.spawn_std_start) * frac
        r = self.cfg.spawn_dist_mu + torch.randn(M, device=self.device) * std
        r = r.clamp(self.cfg.spawn_dist_min, self.cfg.spawn_dist_max)
        basket_c = self.basket_pos_w[env_ids]                                  # (M, 3) env-local
        # only the height is jittered: with r >= spawn_dist_min (> the basket half-diagonal) the spawn is
        # guaranteed outside the box footprint in every azimuth direction
        z_noise = (torch.rand(M, device=self.device) * 2 - 1) * self.cfg.spawn_pos_noise
        spawn_local = torch.empty(M, 3, device=self.device)
        spawn_local[:, 0] = basket_c[:, 0] + torch.cos(ang) * r
        spawn_local[:, 1] = basket_c[:, 1] + torch.sin(ang) * r
        spawn_local[:, 2] = basket_c[:, 2] + self.cfg.spawn_height + z_noise
        spawn_local[:, 2] = spawn_local[:, 2].clamp_min(self.cfg.min_drone_height + 0.2)
        yaw = (torch.rand(M, device=self.device) * 2 - 1) * math.pi
        zeros = torch.zeros(M, device=self.device)
        quat = quat_from_euler_xyz(zeros, zeros, yaw)                          # (M, 4)
        world_pos = spawn_local + self.scene.env_origins[env_ids]             # absolute (incl env origin)
        self.robot.write_root_pose_to_sim(torch.cat([world_pos, quat], dim=-1), env_ids)
        # swing assist: initial velocity TOWARD the wall (+ a little UP), assist = v0 * (1 - frac)
        vel = torch.zeros(M, 6, device=self.device)
        if self.cfg.swing_assist_curriculum:
            assist = self.cfg.swing_assist_v0 * (1.0 - frac)
            vel[:, 0] = -normal_w[:, 0] * assist                              # horizontal, toward the wall
            vel[:, 1] = -normal_w[:, 1] * assist
            vel[:, 2] = assist * self.cfg.swing_assist_up                     # small UP kick
        self.robot.write_root_velocity_to_sim(vel, env_ids)

    def _attached_spawn_reset(self, env_ids):
        """Attached-spawn curriculum: a DECAYING fraction of envs starts ALREADY attached at a near-steady-state
        hold pose (drone ``wire_length`` up-and-out from the bar, hook pinned to the bar), so the hold is learned
        first. The fraction decays ``frac0 -> 0`` and the pose jitter grows ``0 -> jitter`` over the curriculum."""
        if not self.cfg.attached_spawn_curriculum:
            return
        M = len(env_ids)
        if M == 0:
            return
        prog = min(1.0, self._ctrl_step / max(1, self.cfg.attached_spawn_steps))
        frac = self.cfg.attached_spawn_frac0 * (1.0 - prog)
        if frac <= 0.0:
            return
        sel = torch.rand(M, device=self.device) < frac
        if not bool(sel.any()):
            return
        sub = env_ids[sel]                                                     # (n,) envs to spawn attached
        n = len(sub)
        i = 0                                                                  # single-drone task
        q = self.basket_quat_w[sub]                                           # (n, 4)
        idx = self._assigned_wall[sub, i]                                     # (n,) assigned wall 0..3
        # bar point (basket frame): wall face, horizontal centre, middle of the catch z-band
        hx, hy = self.cfg.basket_half_x, self.cfg.basket_half_y
        z_mid = 0.5 * (self.cfg.basket_wall_z_lo + self.cfg.basket_wall_z_hi)
        faces = torch.tensor([[hx, 0, 0], [-hx, 0, 0], [0, hy, 0], [0, -hy, 0]], device=self.device)[idx]
        bar_bf = faces.clone()
        bar_bf[:, 2] = z_mid
        bar_w = self.basket_pos_w[sub] + quat_rotate(q, bar_bf)               # (n, 3) env-local world
        normals_bf = torch.tensor([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]], device=self.device)[idx]
        n_out = normalize(quat_rotate(q, normals_bf))                         # (n, 3) outward wall normal
        up = torch.zeros_like(n_out); up[:, 2] = 1.0
        steady_dir = normalize(self.cfg.attached_spawn_out * n_out + self.cfg.attached_spawn_up * up)
        wl = float(self._acfg.wire_length)
        drone_w = bar_w + wl * steady_dir                                     # (n, 3)
        jit = self.cfg.attached_spawn_jitter * prog                          # basin widens with the curriculum
        if jit > 0.0:
            drone_w = drone_w + (torch.rand(n, 3, device=self.device) * 2 - 1) * jit
        drone_w[:, 2] = drone_w[:, 2].clamp_min(self.cfg.min_drone_height + 0.2)
        yaw = torch.atan2(-n_out[:, 1], -n_out[:, 0])                        # face the wall
        zeros = torch.zeros(n, device=self.device)
        quat = quat_from_euler_xyz(zeros, zeros, yaw)
        world_pos = drone_w + self.scene.env_origins[sub]
        self.robot.write_root_pose_to_sim(torch.cat([world_pos, quat], dim=-1), sub)
        # residual up-and-out speed so the hold starts from a dynamic post-swing state (0 = static pose)
        vel = torch.zeros(n, 6, device=self.device)
        if self.cfg.attached_spawn_v0 > 0.0:
            vel[:, :3] = self.cfg.attached_spawn_v0 * steady_dir
        self.robot.write_root_velocity_to_sim(vel, sub)
        # attachment state: pinned to the bar, latched; hold counter starts at 0 (policy must sustain it)
        self.attached[sub, i] = True
        self.attach_point_w[sub, i] = bar_w
        self.catch_latch[sub, i] = True
        self.hold_counter[sub, i] = 0
        self.slack_timer[sub, i] = 0

    # =====================================================================================
    # debug vis
    # =====================================================================================
    def _set_debug_vis_impl(self, debug_vis: bool):
        # The hook coordinate frame is welded to the hook body in the USD (see _setup_scene). The basket walls
        # and target bar (analytic, no collider) are drawn as markers so the scene shows them.
        if debug_vis:
            if not hasattr(self, "basket_visualizer"):
                self.basket_visualizer = VisualizationMarkers(self.cfg.marker_cfg_basket)
                self._init_basket_marker_geom()
            self.basket_visualizer.set_visibility(True)
        elif hasattr(self, "basket_visualizer"):
            self.basket_visualizer.set_visibility(False)

    def _init_basket_marker_geom(self):
        """Constant (cfg-only) geometry for the basket markers: the 4 wall panels' centres/sizes in the basket
        frame and the target-bar size."""
        hx, hy = self.cfg.basket_half_x, self.cfg.basket_half_y
        H = self.cfg.basket_wall_height
        t = 0.02  # panel thickness (m)
        self._wall_centers_bf = torch.tensor(
            [[hx, 0.0, H / 2], [-hx, 0.0, H / 2], [0.0, hy, H / 2], [0.0, -hy, H / 2]], device=self.device
        )  # (4, 3)
        self._wall_sizes = torch.tensor(
            [[t, 2 * hy, H], [t, 2 * hy, H], [2 * hx, t, H], [2 * hx, t, H]], device=self.device
        )  # (4, 3)
        self._bar_size = torch.tensor(
            [2 * self.cfg.bar_half_len, 1.5 * t, 1.5 * t], device=self.device
        )  # (3,)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        E = self.num_envs
        origins = self.scene.env_origins                                   # (E, 3)
        q = self.basket_quat_w                                             # (E, 4)
        base = self.basket_pos_w + origins                                 # (E, 3) world basket origin
        # 4 wall panels
        q_rep = q.unsqueeze(1).expand(E, 4, 4).reshape(-1, 4)
        c_rep = self._wall_centers_bf.unsqueeze(0).expand(E, 4, 3).reshape(-1, 3)
        wall_pos = base.unsqueeze(1) + quat_rotate(q_rep, c_rep).reshape(E, 4, 3)   # (E, 4, 3)
        wall_quat = q.unsqueeze(1).expand(E, 4, 4)                                  # (E, 4, 4)
        wall_scale = self._wall_sizes.unsqueeze(0).expand(E, 4, 3)                  # (E, 4, 3)
        # target bar: unit cuboid with local x = bar_axis, z = wall_up
        xb = self.bar_axis_w                                              # (E, 3)
        zb = self.wall_up_w                                               # (E, 3)
        yb = torch.cross(zb, xb, dim=-1)
        bar_quat = quat_from_matrix(torch.stack([xb, yb, zb], dim=-1))    # (E, 4)
        bar_pos = self.bar_center_w + origins                            # (E, 3)
        bar_scale = self._bar_size.unsqueeze(0).expand(E, 3)             # (E, 3)
        translations = torch.cat([wall_pos.reshape(-1, 3), bar_pos], dim=0)
        orientations = torch.cat([wall_quat.reshape(-1, 4), bar_quat], dim=0)
        scales = torch.cat([wall_scale.reshape(-1, 3), bar_scale], dim=0)
        marker_indices = torch.cat([
            torch.zeros(E * 4, dtype=torch.long, device=self.device),
            torch.ones(E, dtype=torch.long, device=self.device),
        ])
        self.basket_visualizer.visualize(
            translations=translations, orientations=orientations,
            marker_indices=marker_indices, scales=scales,
        )
