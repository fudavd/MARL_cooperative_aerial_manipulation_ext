#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate the ``flyhook`` URDF (drone + flexible weighted wire + hook tip) and convert it to USD.

Why a URDF: every robot asset in this repo was authored as a URDF and converted with IsaacLab's
``UrdfConverter`` (see the ``config.yaml`` next to each ``*.usd``). We keep that pipeline: this script
**extends the real ``falcon.urdf``** (so the drone's mass / inertia / rotor layout stay identical and the
Geometric+INDI+motor controller keeps working) by appending:

  * an N-link wire, each node a spherical joint modelled as two revolute DoFs in series
    (``wire_link_i_joint_x`` about X, ``wire_link_i_joint_y`` about Y) through a tiny massless dummy link —
    the exact pattern ``flycrane_rod`` uses for its ropes;
  * a lumped **hook + weight** tip body ``hook_link`` (a simple sphere collider — the 4-prong hook geometry
    is purely cosmetic and can be swapped onto this link later; the catch is a *surrogate*, so the tip
    physics only needs the right mass and a rough collider).

Body / joint names match the contract in ``assets/flyhook.py``:
    drone  : Falcon_base_link_inertia, Falcon_rotor_.*        (from falcon.urdf, "/" -> "_" on convert)
    wire   : wire_link_.*  with joints  wire_.*_joint_.*
    hook   : hook_link

Usage
-----
Generate the URDF only (no Isaac needed -- run anywhere, incl. the plain venv):
    python scripts/hook_grasp/build_flyhook_asset.py

Generate + convert to USD (needs an Isaac Sim session; shells out to IsaacLab's converter):
    python scripts/hook_grasp/build_flyhook_asset.py --convert

Or convert manually with the stock tool:
    python thirdparty/IsaacLab/scripts/tools/convert_urdf.py \
        exts/.../assets/data/AMR/falcon_data/flyhook.urdf \
        exts/.../assets/data/AMR/flyhook/flyhook.usd \
        --joint-stiffness 0.0 --joint-damping 0.0 --joint-target-type none --headless
(Do NOT pass --merge-joints: the fixed base/hook joints must stay so ``Falcon_base_link_inertia`` and
``hook_link`` survive as distinct bodies.)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# --- geometry / mass parameters (keep in sync with assets/flyhook.py + attachment.HookAttachmentCfg) ----
NUM_WIRE_LINKS = 8          # FLYHOOK_NUM_WIRE_LINKS
WIRE_LENGTH = 0.55          # m, total taut length (== HookAttachmentCfg.wire_length)
WIRE_RADIUS = 0.004         # m, collision/visual cylinder radius of a wire segment
WIRE_TOTAL_MASS = 0.02      # kg, spread over the links
HOOK_MASS = 0.03            # kg, lumped weight+hook (DR overrides at runtime: 0.02-0.05)
HOOK_RADIUS = 0.02          # m, tip sphere collider (~paper hook radius)
ATTACH_Z = 0.0              # m, wire attaches to Falcon/base_link at this z (drone origin)
JOINT_LIMIT = 1.5708        # rad, +/- per revolute DoF (90 deg -> wire may reach horizontal, not invert)
JOINT_DAMPING = 0.005       # matches the wire actuator damping in FLYHOOK_CFG

REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets/data/AMR"
FALCON_URDF = ASSETS / "falcon_data/falcon.urdf"
OUT_URDF = ASSETS / "falcon_data/flyhook.urdf"          # next to falcon meshes so .dae refs resolve
OUT_USD = ASSETS / "flyhook/flyhook.usd"
CONVERT_TOOL = REPO / "thirdparty/IsaacLab/scripts/tools/convert_urdf.py"


def _inertia_box(m, sx, sy, sz):
    """Solid-box inertia (kg m^2)."""
    return (m * (sy**2 + sz**2) / 12.0, m * (sx**2 + sz**2) / 12.0, m * (sx**2 + sy**2) / 12.0)


def _wire_and_hook_urdf() -> str:
    seg = WIRE_LENGTH / NUM_WIRE_LINKS
    m_link = WIRE_TOTAL_MASS / NUM_WIRE_LINKS
    ixx, iyy, izz = _inertia_box(m_link, WIRE_RADIUS, WIRE_RADIUS, seg)
    out: list[str] = []

    for i in range(NUM_WIRE_LINKS):
        parent = "Falcon/base_link" if i == 0 else f"wire_link_{i - 1}"
        joint_z = ATTACH_Z if i == 0 else -seg  # offset from parent origin down to this node's joint
        dummy = f"wire_dummy_{i}"
        link = f"wire_link_{i}"
        out.append(f"""
  <!-- wire node {i}: spherical joint = revolute-X then revolute-Y through a massless dummy -->
  <joint name="wire_link_{i}_joint_x" type="revolute">
    <origin rpy="0 0 0" xyz="0 0 {joint_z:.6f}"/>
    <axis xyz="1 0 0"/>
    <limit effort="0.0" lower="{-JOINT_LIMIT}" upper="{JOINT_LIMIT}" velocity="100.0"/>
    <dynamics damping="{JOINT_DAMPING}"/>
    <parent link="{parent}"/>
    <child link="{dummy}"/>
  </joint>
  <link name="{dummy}">
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="1e-4"/>
      <inertia ixx="1e-7" ixy="0.0" ixz="0.0" iyy="1e-7" iyz="0.0" izz="1e-7"/>
    </inertial>
  </link>
  <joint name="wire_link_{i}_joint_y" type="revolute">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit effort="0.0" lower="{-JOINT_LIMIT}" upper="{JOINT_LIMIT}" velocity="100.0"/>
    <dynamics damping="{JOINT_DAMPING}"/>
    <parent link="{dummy}"/>
    <child link="{link}"/>
  </joint>
  <link name="{link}">
    <inertial>
      <origin xyz="0 0 {-seg / 2:.6f}"/>
      <mass value="{m_link:.6f}"/>
      <inertia ixx="{ixx:.3e}" ixy="0.0" ixz="0.0" iyy="{iyy:.3e}" iyz="0.0" izz="{izz:.3e}"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 {-seg / 2:.6f}"/>
      <geometry><cylinder length="{seg:.6f}" radius="{WIRE_RADIUS}"/></geometry>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 {-seg / 2:.6f}"/>
      <geometry><cylinder length="{seg:.6f}" radius="{WIRE_RADIUS}"/></geometry>
    </collision>
  </link>""")

    # hook + weight tip: rigidly fixed to the last wire link (the wire chain already provides the DoF)
    hix, hiy, hiz = (2.0 / 5.0 * HOOK_MASS * HOOK_RADIUS**2,) * 3  # solid sphere
    out.append(f"""
  <!-- lumped hook + weight tip -->
  <joint name="hook_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 {-WIRE_LENGTH / NUM_WIRE_LINKS:.6f}"/>
    <parent link="wire_link_{NUM_WIRE_LINKS - 1}"/>
    <child link="hook_link"/>
  </joint>
  <link name="hook_link">
    <inertial>
      <origin xyz="0 0 {-HOOK_RADIUS:.6f}"/>
      <mass value="{HOOK_MASS:.6f}"/>
      <inertia ixx="{hix:.3e}" ixy="0.0" ixz="0.0" iyy="{hiy:.3e}" iyz="0.0" izz="{hiz:.3e}"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 {-HOOK_RADIUS:.6f}"/>
      <geometry><sphere radius="{HOOK_RADIUS}"/></geometry>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 {-HOOK_RADIUS:.6f}"/>
      <geometry><sphere radius="{HOOK_RADIUS}"/></geometry>
    </collision>
  </link>
""")
    return "".join(out)


def generate_urdf() -> Path:
    falcon = FALCON_URDF.read_text()
    if "</robot>" not in falcon:
        raise RuntimeError(f"unexpected falcon URDF (no </robot>): {FALCON_URDF}")
    head = falcon.rsplit("</robot>", 1)[0].rstrip()
    head = head.replace('<robot name="Falcon">', '<robot name="Flyhook">', 1)
    # NOTE: the falcon body mesh (falcon_without_rotors.dae) does not survive this Isaac version's URDF
    # importer (dropped Collada nodes -> body renders broken; physics frame is still upright). We keep the
    # drone definition verbatim here and fix the *visual only* afterwards by grafting the working body
    # visual from the committed falcon.usd -> run scripts/hook_grasp/fix_drone_visual.py after --convert.
    urdf = head + "\n" + _wire_and_hook_urdf() + "\n</robot>\n"
    OUT_URDF.write_text(urdf)
    # sanity: XML well-formedness
    import xml.dom.minidom as minidom

    minidom.parseString(urdf)
    print(f"[flyhook] wrote {OUT_URDF}  ({NUM_WIRE_LINKS} wire links, L={WIRE_LENGTH} m, "
          f"hook {HOOK_MASS} kg) — XML OK")
    return OUT_URDF


def convert_to_usd():
    OUT_USD.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(CONVERT_TOOL), str(OUT_URDF), str(OUT_USD),
        "--joint-stiffness", "0.0", "--joint-damping", "0.0", "--joint-target-type", "none", "--headless",
    ]
    print("[flyhook] converting URDF -> USD:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[flyhook] wrote {OUT_USD}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the flyhook URDF (+ optional USD conversion).")
    p.add_argument("--convert", action="store_true", help="also convert to USD (needs Isaac Sim).")
    args = p.parse_args()
    generate_urdf()
    if args.convert:
        convert_to_usd()
    else:
        print("[flyhook] URDF only. Convert inside Isaac with:  "
              "python scripts/hook_grasp/build_flyhook_asset.py --convert")
