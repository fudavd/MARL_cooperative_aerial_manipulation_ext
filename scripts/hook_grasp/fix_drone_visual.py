# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Graft the working drone-body visual from the committed ``falcon.usd`` onto ``flyhook.usd``.

Why: the falcon body mesh (``falcon_without_rotors.dae``) does not survive a fresh URDF->USD conversion on
this Isaac version (dropped Collada nodes -> the body renders broken/dark, which *looks* flipped although
the physics frame is upright: body up-axis = [0,0,+1]). The committed ``falcon.usd`` was converted by an
older importer that handled the mesh, so its drone body renders correctly. Rather than re-convert, we reuse
that working visual: deactivate the broken visual scope under ``Falcon_base_link_inertia`` in flyhook.usd and
add an override that *references* the good visual subtree from ``falcon.usd``. Physics / masses / names are
untouched (this only edits the visual).

Run inside Isaac:
    python scripts/hook_grasp/fix_drone_visual.py --headless
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
sys.argv = [sys.argv[0]]
app = AppLauncher(args_cli).app

from pxr import Usd, UsdGeom  # noqa: E402

REPO = "/home/fuda/PycharmProjects/MARL_cooperative_aerial_manipulation_ext"
ASSETS = f"{REPO}/exts/MARL_mav_carry_ext/MARL_mav_carry_ext/assets/data/AMR"
# candidate source drones with a correctly-converted body mesh (older importer); first non-empty wins
SRC_CANDIDATES = [f"{ASSETS}/falcon/falcon.usd", f"{ASSETS}/flycrane_rod/flycrane_rod.usd"]
DST = f"{ASSETS}/flyhook/flyhook.usd"         # our asset (root layer, composes configuration/*.usd)


def _count_meshes(stage, root_path):
    """Count Mesh prims under root_path, descending INTO instances (prototype geometry)."""
    root = stage.GetPrimAtPath(root_path)
    n = 0
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if p.IsA(UsdGeom.Mesh):
            pts = p.GetAttribute("points")
            if pts and pts.Get():
                n += 1
    return n


def _find_body_visual_scope(stage):
    """Return (scope_path, n_meshes) for the 'visuals' scope under a *base_link_inertia* body.

    These assets are instanceable, so we locate the visual SCOPE by name (its mesh children live in a
    prototype and are invisible to a plain Traverse) and count meshes via an instance-proxy range.
    """
    for p in stage.Traverse():
        path = str(p.GetPath())
        if "base_link_inertia" in path and p.GetName() in ("visuals", "visual"):
            return p.GetPath(), _count_meshes(stage, p.GetPath())
    return None, None


def main():
    src_path, src_scope = None, None
    for cand in SRC_CANDIDATES:
        if not os.path.exists(cand):
            continue
        scope, n_meshes = _find_body_visual_scope(Usd.Stage.Open(cand))
        print(f"[fix] candidate {os.path.basename(cand)}: scope={scope}  meshes={n_meshes}")
        if scope is not None and n_meshes and n_meshes > 0:
            src_path, src_scope = cand, scope
            break
    if src_path is None:
        raise SystemExit("[fix] no source USD has a non-empty body visual scope")
    print(f"[fix] using SRC={os.path.basename(src_path)}  scope={src_scope}")
    SRC = src_path

    dst = Usd.Stage.Open(DST)
    body = None
    for p in dst.Traverse():
        if p.GetName() == "Falcon_base_link_inertia":
            body = p
            break
    if body is None:
        raise SystemExit("[fix] Falcon_base_link_inertia not found in flyhook.usd")
    print(f"[fix] DST body prim: {body.GetPath()}")

    # deactivate the broken visual scope(s) under the body
    for child in body.GetChildren():
        if "visual" in child.GetName().lower():
            print(f"[fix]   deactivating broken visual: {child.GetPath()}")
            child.SetActive(False)

    # add a fresh visual as an override that references the working scope from falcon.usd
    reused = dst.DefinePrim(body.GetPath().AppendChild("visual_reused"), "Xform")
    reused.GetReferences().AddReference(SRC, str(src_scope))
    print(f"[fix]   added {reused.GetPath()} -> reference {SRC}{src_scope}")

    dst.GetRootLayer().Save()
    print(f"[fix] saved {DST}")


if __name__ == "__main__":
    main()
    os._exit(0)
