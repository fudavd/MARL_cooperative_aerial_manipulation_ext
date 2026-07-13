# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset config for the hook-grasping robot: one Falcon drone + a flexible weighted wire + a hook tip.

The wire is a serial chain of small rigid links joined by spherical joints (2-DoF revolute x/y per node,
damping ~0.005), following the ``flycrane_rod`` cable pattern (see ``flycrane_rod.py``); the tip link lumps
the small weight and the 4-pronged hook into one body. The whole thing is ONE articulation rooted at the
drone, so ``robot.data.body_*_w`` exposes the hook tip state directly.

The USD (``data/AMR/flyhook/flyhook.usd``) is authored once by ``scripts/hook_grasp/build_flyhook_asset.py``
(needs an Isaac Sim session). Body-name conventions below are the contract between that builder, this config,
and the env (``marl_hook_env.py``).
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

script_dir = Path(__file__).parent
usd_path = script_dir / "data/AMR/flyhook/flyhook.usd"

# --- body/joint name conventions (regex) --------------------------------------------------
FLYHOOK_DRONE_BODY = "Falcon_base_link_inertia"   # drone CoM body
FLYHOOK_ROTOR_BODY = "Falcon_rotor_.*"            # 4 rotor bodies (thrust application points)
FLYHOOK_WIRE_BODY = "wire_link_.*"                # N wire chain links
FLYHOOK_HOOK_BODY = "hook_link"                   # lumped weight + hook tip body
FLYHOOK_WIRE_JOINT = "wire_.*_joint_.*"           # spherical-joint DoFs along the wire

# number of wire links the builder emits (kept here so cfg/env can reason about it)
FLYHOOK_NUM_WIRE_LINKS = 8

##
# Configuration
##

FLYHOOK_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/flyhook",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(usd_path.resolve()),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        # rotors are driven by external forces (like the other flycrane assets) -> passive joints
        "rotor_joints": ImplicitActuatorCfg(
            joint_names_expr=["Falcon_rotor_.*_joint"],
            stiffness=None,
            damping=None,
        ),
        # wire spherical DoFs: passive with light damping so the pendulum settles realistically
        "wire_joints": ImplicitActuatorCfg(
            joint_names_expr=["wire_.*_joint_.*"],
            stiffness=0.0,
            damping=0.005,
        ),
    },
)
"""One Falcon drone with a flexible weighted wire + hook tip (single articulation)."""
