# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Swing-controllability test env: a clean, confound-free directed-pendulum task on the hook PLANT.

Subclasses :class:`MARLHookGraspEnv` to inherit the ENTIRE plant + control cascade (asset spawn, geometric
-> INDI -> motor model, INDI disturbance-residual read, state refresh) and overrides ONLY the task surface
(observation / privileged state / reward / reset / dones). There is no basket, no catch, no wall, no
curriculum: each episode commands a random horizontal swing-plane azimuth ``phi`` and the sole objective is a
PLANAR pendulum swing in the vertical plane containing ``phi``. See ``marl_swing_env_cfg.py`` for the full
rationale and the observation/state/reward layout, and ``Research/Combined/master_report.md`` for the
controllability question this isolates.
"""

from __future__ import annotations

import math

import torch
from collections.abc import Sequence

from .marl_hook_env import MARLHookGraspEnv
from .marl_swing_env_cfg import MARLSwingTestEnvCfg

# reward-term keys (also the per-episode logging groups)
_SWING_REWARD_KEYS = [
    "swing_amp", "swing_offplane", "swing_vel", "swing_vlat", "swing_whirl",
    "station", "alt", "bodyrate", "alive", "death",
]


class MARLSwingTestEnv(MARLHookGraspEnv):
    cfg: MARLSwingTestEnvCfg

    def __init__(self, cfg: MARLSwingTestEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # per-episode commanded swing azimuth phi and its horizontal frame (u_hat = in-plane axis, w_hat = the
        # off-plane / whirl axis to suppress). Sampled in _reset_idx.
        self._cmd_phi = torch.zeros(self.num_envs, device=self.device)
        self._cmd_cos = torch.ones(self.num_envs, device=self.device)
        self._cmd_sin = torch.zeros(self.num_envs, device=self.device)
        self._cmd_u = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_u[:, 0] = 1.0
        self._cmd_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_w[:, 1] = 1.0
        # station-keeping reference (per drone): drone spawn position, captured on the first reward after reset.
        self._spawn_pos = torch.zeros(self.num_envs, self._num_drones, 3, device=self.device)
        self._swing_reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # per-episode reward sums for logging (mirrors the base _episode_sums pattern, swing terms only)
        self._swing_sums = {
            k: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for k in _SWING_REWARD_KEYS
        }

    # =====================================================================================
    # reset: free-float spawn (base EventCfg) + a fresh commanded azimuth. NO basket / curriculum.
    # =====================================================================================
    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # Skip the WHOLE MARLHookGraspEnv._reset_idx (basket pose, spawn/hold/attached curricula, task logging)
        # and call its grandparent DirectMARLEnv._reset_idx directly -> applies the EventCfg reset (free-float
        # drone spawn in the reset volume + 300 g bob randomize) and resets episode_length_buf.
        super(MARLHookGraspEnv, self)._reset_idx(env_ids)

        n = len(env_ids)
        phi = torch.empty(n, device=self.device).uniform_(0.0, math.pi)   # swing PLANE is a line -> [0, pi) covers all
        cos, sin, zero = torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)
        self._cmd_phi[env_ids] = phi
        self._cmd_cos[env_ids] = cos
        self._cmd_sin[env_ids] = sin
        self._cmd_u[env_ids] = torch.stack([cos, sin, zero], dim=-1)      # in-plane horizontal axis
        self._cmd_w[env_ids] = torch.stack([-sin, cos, zero], dim=-1)     # off-plane / whirl axis (to suppress)

        # v63 SEED-A-SWING: kick the drone root with a random-horizontal-direction velocity so the compliant wire
        # makes the hook lag into a pendulum swing (a random plane, INDEPENDENT of the command -> the policy must
        # STEER it to the commanded plane). Runs AFTER the reset event (which zeroed the root velocity). Mirrors
        # the grasp task's swing_assist (write_root_velocity_to_sim). Breaks the dead-hang: the episode starts
        # with swing kinetic energy, so a static hang is a costly choice to actively damp it out, not a free start.
        if getattr(self.cfg, "swing_seed_v0", 0.0) > 0.0:
            kick_ang = torch.empty(n, device=self.device).uniform_(-math.pi, math.pi)   # random, != command
            vel = torch.zeros(n, 6, device=self.device)
            vel[:, 0] = torch.cos(kick_ang) * self.cfg.swing_seed_v0
            vel[:, 1] = torch.sin(kick_ang) * self.cfg.swing_seed_v0
            self.robot.write_root_velocity_to_sim(vel, env_ids)

        self._swing_reset_mask[env_ids] = True   # capture spawn pos on the next reward pass
        self._state_reset_mask[env_ids] = True   # refill the critic frame-stack (no stale history)
        for agent in self.cfg.possible_agents:
            self._observation_buffers[agent].reset(env_ids)
            self.prev_actions[agent][env_ids] = 0.0
            self.prev_prev_actions[agent][env_ids] = 0.0
            self.actions[agent][env_ids] = 0.0

        # per-episode logging (reward sums + termination counts)
        if "log" not in self.extras:
            self.extras["log"] = dict()
        for k in self._swing_sums:
            self.extras["log"][f"Episode_Reward/{k}"] = torch.mean(self._swing_sums[k][env_ids]) / self.max_episode_length_s
            self._swing_sums[k][env_ids] = 0.0
        for name, buf in [
            ("fly_low", self.drone_fly_low), ("bounding_box", self.body_pos_outside),
            ("runaway", self.runaway), ("nonfinite", self.nonfinite), ("time_out", self.time_out),
        ]:
            self.extras["log"][f"Episode_Termination/{name}"] = torch.count_nonzero(buf[env_ids]).item()

    # =====================================================================================
    # observations: drone kinematics + command + INDI residual (base actor stays position-blind).
    # =====================================================================================
    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._refresh_states()   # NO _update_attachment (no catch model in the swing test)
        observations = {}
        for i, agent in enumerate(self.cfg.possible_agents):
            frame = self._swing_obs_frame(i)
            self._observation_buffers[agent].append(frame)
            observations[agent] = self._observation_buffers[agent].buffer.reshape(self.num_envs, -1)
        return observations

    def _swing_obs_frame(self, i: int) -> torch.Tensor:
        # 26 = drone pos(3) + rot(9) + lin vel(3) + ang vel(3) + command cos/sin(2) + INDI residual(6)
        parts = [
            self.drone_positions[:, i],                                   # 3
            self.drone_rot_matrices[:, i].reshape(self.num_envs, 9),      # 9
            self.drone_linear_velocities[:, i],                          # 3
            self.drone_angular_velocities[:, i],                         # 3
            self._cmd_cos.unsqueeze(-1),                                  # 1  commanded swing azimuth (cos)
            self._cmd_sin.unsqueeze(-1),                                  # 1  ...                     (sin)
            self._indi_residual[:, i],                                    # 6  onboard hook/wire wrench (dynamics, no pos)
        ]
        return torch.cat(parts, dim=-1)

    # =====================================================================================
    # privileged critic state: drone kinematics + TRUE hook rel/vel + command (26 per frame, stacked H).
    # =====================================================================================
    def _get_states(self) -> torch.Tensor:
        parts = []
        for i in range(self._num_drones):
            parts += [
                self.drone_positions[:, i],                                  # 3
                self.drone_rot_matrices[:, i].reshape(self.num_envs, 9),     # 9
                self.drone_linear_velocities[:, i],                         # 3
                self.drone_angular_velocities[:, i],                        # 3
                self.hook_positions[:, i] - self.drone_positions[:, i],     # 3  hook rel drone (privileged)
                self.hook_velocities[:, i],                                 # 3  hook velocity  (privileged)
            ]
        parts += [self._cmd_cos.unsqueeze(-1), self._cmd_sin.unsqueeze(-1)]  # 2  command
        single = torch.nan_to_num(torch.cat(parts, dim=-1), nan=0.0, posinf=0.0, neginf=0.0)

        H = self.cfg.history_len
        if not hasattr(self, "_swing_state_hist") or self._swing_state_hist.shape[1] != H:
            self._swing_state_hist = single.unsqueeze(1).repeat(1, H, 1)
        else:
            self._swing_state_hist = torch.cat([self._swing_state_hist[:, 1:], single.unsqueeze(1)], dim=1)
            rm = self._state_reset_mask
            if rm.any():
                self._swing_state_hist[rm] = single[rm].unsqueeze(1).repeat(1, H, 1)
            self._state_reset_mask[:] = False
        return self._swing_state_hist.reshape(self.num_envs, H * single.shape[-1])

    # =====================================================================================
    # reward: directed PLANAR swing in the commanded plane. Pendulum quantities are drone-RELATIVE.
    # =====================================================================================
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        dt = self.step_dt
        sums = {k: torch.zeros(self.num_envs, device=self.device) for k in _SWING_REWARD_KEYS}
        c = self.cfg
        for i in range(self._num_drones):
            drone_pos = self.drone_positions[:, i]
            drone_vel = self.drone_linear_velocities[:, i]
            hook_pos = self.hook_positions[:, i]
            hook_vel = self.hook_velocities[:, i]

            # capture the spawn position for station-keeping on the first reward after a reset
            rm = self._swing_reset_mask
            if rm.any():
                self._spawn_pos[rm, i] = drone_pos[rm]

            u, w = self._cmd_u, self._cmd_w                      # command frame (horizontal in-plane / off-plane)
            rel = hook_pos - drone_pos                            # wire vector (pivot -> bob)
            relh = rel.clone(); relh[:, 2] = 0.0                 # horizontal displacement
            v_swing = hook_vel - drone_vel                        # swing velocity about the pivot
            vh = v_swing.clone(); vh[:, 2] = 0.0

            amp_along = (relh * u).sum(-1).abs()                 # in-plane arc amplitude (along command)
            amp_off = (relh * w).sum(-1).abs()                   # off-plane displacement (to suppress)
            vel_along = (v_swing * u).sum(-1).abs()              # in-plane swing speed (anti dead-hang)
            vel_off = (v_swing * w).sum(-1).abs()                # off-plane swing speed
            l_z = relh[:, 0] * vh[:, 1] - relh[:, 1] * vh[:, 0]  # azimuthal angular momentum (whirl)

            sums["swing_amp"] += c.swing_amp_weight * amp_along.clamp(max=c.swing_amp_cap)
            sums["swing_offplane"] += -c.swing_offplane_weight * amp_off
            sums["swing_vel"] += c.swing_vel_weight * vel_along.clamp(max=c.swing_vel_cap)
            sums["swing_vlat"] += -c.swing_vlat_weight * vel_off
            sums["swing_whirl"] += -c.swing_whirl_weight * l_z.abs()

            # station-keeping: pump IN PLACE, don't translate one-way (which would fake directed motion by dragging)
            dxy = (drone_pos[:, :2] - self._spawn_pos[:, i, :2]).norm(dim=-1)
            dz = (drone_pos[:, 2] - self._spawn_pos[:, i, 2]).abs()
            sums["station"] += -c.swing_station_weight * dxy
            sums["alt"] += -c.swing_alt_weight * dz
            sums["bodyrate"] += -c.swing_bodyrate_weight * self.drone_angular_velocities[:, i].square().sum(-1)

        self._swing_reset_mask[:] = False
        for k in sums:                                            # scale the dense per-step terms by dt
            sums[k] = sums[k] * dt

        # survival economics: +alive per living step, -death on any FAILURE termination (flags from _get_dones)
        failure = (self.drone_fly_low | self.body_pos_outside | self.runaway | self.nonfinite)
        sums["alive"] = c.alive_weight * dt * (~failure).float()
        sums["death"] = -c.death_penalty * failure.float()

        for k in sums:
            self._swing_sums[k] += sums[k]
        shared = torch.nan_to_num(sum(sums.values()), nan=0.0, posinf=0.0, neginf=0.0)
        return {agent: shared for agent in self.cfg.possible_agents}

    # =====================================================================================
    # dones: divergence / floor / bounds / timeout only (a big up-swing above the drone is DESIRED here).
    # =====================================================================================
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self.drone_fly_low = (self.drone_positions[:, :, 2] < self.cfg.min_drone_height).any(dim=-1)
        self.body_pos_outside = (self.drone_positions.abs() > self.cfg.bounding_box_threshold).any(dim=-1).any(dim=-1)
        speed = torch.norm(self.drone_linear_velocities, dim=-1)
        finite = (torch.isfinite(self.drone_positions).all(-1)
                  & torch.isfinite(self.drone_linear_velocities).all(-1)
                  & torch.isfinite(self.hook_positions).all(-1))
        self.nonfinite = ~finite.all(dim=-1)
        self.runaway = (speed > self.cfg.runaway_speed).any(dim=-1) | (~torch.isfinite(speed).all(dim=-1))
        self.time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminations = self.drone_fly_low | self.body_pos_outside | self.nonfinite | self.runaway
        terminated = {a: terminations for a in self.cfg.possible_agents}
        time_outs = {a: self.time_out for a in self.cfg.possible_agents}
        return terminated, time_outs
