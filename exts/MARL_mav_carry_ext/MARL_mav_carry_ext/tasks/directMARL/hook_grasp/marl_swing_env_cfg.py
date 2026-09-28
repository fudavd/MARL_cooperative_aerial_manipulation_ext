# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the SWING-CONTROLLABILITY test env: a clean, confound-free directed-pendulum task.

This is NOT the hook-grasp task. It reuses the exact same PLANT (quadrotor + flexible weighted wire +
hook bob = a spherical pendulum on a moving pivot) and the same control cascade, but strips away
EVERYTHING that muddied every prior reading of the "can the swing plane be steered?" question:

    * NO basket, NO wall, NO surrogate catch, NO wall-repulsion, NO approach/proximity/lock/hold rewards.
    * NO curricula (capture radius / chord angle / spawn / swing-assist / attached-spawn / hold).
    * The drone free-floats; each episode samples a COMMANDED horizontal swing-plane azimuth phi and the
      ONLY objective is to make the hook swing as a PLANAR pendulum in the vertical plane containing phi
      (large in-plane amplitude, zero off-plane displacement, zero azimuthal whirl L_z).

The observation-vs-control question is answered by the SAME adapter arms used on the hook task:
    * hook-blind  (base obs = drone kinematics + command + INDI residual)  -> the control/reference,
    * --true_hook (append ground-truth relative hook position, +3)         -> the DIRECT-obs upper bound,
    * --hook_observer (append the learned estimate, +3)                    -> the PREDICTED-obs arm.

Verdict: if the DIRECT-obs arm CAN command the swing plane (high planarity + aim~0 to phi across random
phi) then the plane IS controllable and the hook-task failure was a task/reward confound; if it CANNOT
even with perfect hook info, the conical/whirl mode is a genuine dynamics/controllability limit
(Miles precession of a forced spherical pendulum). See Research/Combined/master_report.md.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .marl_hook_env_cfg import MARLHookGraspEnvCfg, _NUM_DRONES, _POSSIBLE_AGENTS

# --- swing-test observation / state layout -------------------------------------------------------------
# ACTOR frame (base, before the adapter appends the +3 true/predicted hook): the drone's OWN observable
# kinematics + the commanded swing azimuth + the onboard INDI disturbance residual (the hook/wire wrench
# the IMU/INDI actually measures -- carries hook DYNAMICS/phase but NO hook position, so the base actor
# stays position-blind exactly like the hook task).
#   drone pos (3) + drone rot matrix (9) + drone lin vel (3) + drone ang vel (3)  = 18
#   command (cos phi, sin phi)                                                    =  2
#   INDI disturbance residual (6)                                                =  6
_SWING_OBS_PER_FRAME = 18 + 2 + 6            # = 26
# PRIVILEGED critic frame: drone kinematics (18) + hook rel (3) + hook vel (3) + command (2) = 26.
_SWING_STATE_PER_FRAME = 18 + 3 + 3 + 2      # = 26


@configclass
class MARLSwingTestEnvCfg(MARLHookGraspEnvCfg):
    # marker that flips the env into the clean swing-control task (the env class overrides obs/reward/reset/
    # dones on this flag; the train script skips the F3 recipe + INDI-cfg when it is set).
    swing_test: bool = True

    # --- spaces (self-contained; NOT the hover-matching 45 layout) -----------------------
    # H stays 3 (frame-stack gives the LSTM the swing phase a single frame can't see).
    observation_spaces = {a: _SWING_OBS_PER_FRAME * 3 for a in _POSSIBLE_AGENTS}
    state_space = _SWING_STATE_PER_FRAME * 3
    # actor_indi_residual is irrelevant here (the swing obs builds the residual in unconditionally); leave the
    # base value. history_len stays 3.

    # no basket markers to draw -> disable debug vis (base __init__ would otherwise build basket geometry).
    debug_vis: bool = False

    # shorter episodes than the 20 s grasp task: the pendulum period is ~1.5 s, so ~12 s gives several full
    # swings per episode while resetting the commanded azimuth often (more command variety per PPO iteration).
    episode_length_s = 12

    # --- turn EVERY curriculum OFF (the swing env overrides _reset_idx and never calls them, but keep the
    #     flags honest so nothing anneals a stale knob) -----------------------------------
    capture_curriculum: bool = False
    chord_angle_curriculum: bool = False
    spawn_curriculum: bool = False
    swing_assist_curriculum: bool = False
    attached_spawn_curriculum: bool = False
    hold_curriculum: bool = False

    # =====================================================================================
    # directed-planar-swing reward (the whole objective). Command frame per env:
    #   u_hat = (cos phi, sin phi, 0)  = the commanded horizontal swing axis (the plane is span{u_hat, z}),
    #   w_hat = (-sin phi, cos phi, 0) = the horizontal axis to SUPPRESS (off-plane / whirl direction).
    # Pendulum quantities use the drone-RELATIVE hook state (the true swing about the wire pivot):
    #   rel      = hook_pos - drone_pos ,  rel_h = horizontal part
    #   v_swing  = hook_vel - drone_vel ,  v_h   = horizontal part
    #   L_z      = rel_h x v_h  (azimuthal angular momentum about the vertical through the pivot; >0 = whirl)
    # =====================================================================================
    swing_amp_weight: float = 12.0      # + reward clamp(|rel . u_hat|, max=cap): in-plane arc AMPLITUDE along command
    swing_amp_cap: float = 0.5          # m: cap on rewarded in-plane displacement (~wire length horizontal)
    swing_offplane_weight: float = 8.0  # - penalty |rel . w_hat|: displacement OUT of the commanded plane
    swing_vel_weight: float = 6.0       # + reward clamp(|v_swing . u_hat|, max=cap): in-plane swing SPEED (anti dead-hang)
    swing_vel_cap: float = 1.5          # m/s: cap on rewarded in-plane swing speed
    swing_vlat_weight: float = 4.0      # - penalty |v_swing . w_hat|: off-plane swing speed
    swing_whirl_weight: float = 6.0     # - penalty |L_z|: azimuthal angular momentum (collapses the cone into a plane)

    # v63 SEED-A-SWING (fixes the v62 dead-hang collapse): the v62 policies hovered still and farmed `alive`
    # instead of paying to pump a swing. Fix = (1) start each episode with the pendulum ALREADY swinging by
    # kicking the drone root with a random-horizontal-direction velocity at reset (the compliant wire makes the
    # hook lag into a swing, random plane INDEPENDENT of the command -> the policy must STEER it to the command);
    # (2) drop the hover farm: alive_weight 5->0 (stillness now earns nothing) and cut station-keeping 2->0.5
    # (it fought the very motion needed). Dead-hang is now a costly departure from a moving start, not a free
    # positive baseline. Mirrors the grasp task's swing_assist velocity write (write_root_velocity_to_sim).
    swing_seed_v0: float = 1.8          # m/s: random-horizontal drone kick at reset to excite the pendulum swing
    alive_weight = 0.0                  # (was base 5.0) NO hover farm -> a static dead-hang earns nothing
    # station-keeping: the drone must PUMP IN PLACE, not translate one-way along u_hat (which would fake a
    # directed hook motion by dragging the bob). Kept SMALL (v63) just to bound drift, not to suppress the swing.
    swing_station_weight: float = 0.5   # - penalty ||drone_xy - spawn_xy||: keep the pivot loosely near its spawn
    swing_alt_weight: float = 2.0       # - penalty |drone_z - spawn_z|:      hold altitude
    swing_bodyrate_weight: float = 0.5  # - penalty ||drone_ang_vel||^2:      light attitude damping (stability)
    # death_penalty (base 40.0) is reused for divergence economics.