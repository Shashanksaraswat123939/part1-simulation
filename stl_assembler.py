"""
stl_assembler.py --- Assemble full car and right-half STL from component meshes.

Mirrors right sidepod to left, concatenates all components, and slices
the full car at y=0 to produce the right-half STL for CFD.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def _mirror_right_to_left(right_mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """
    Mirror right sidepod to create left sidepod.

    1. Copy mesh
    2. Flip y coordinate (y *= -1)
    3. Invert face winding (normals reversed by mirror)
    4. Fix normals for consistent outward orientation
    """
    import trimesh

    left = right_mesh.copy()
    left.vertices[:, 1] *= -1.0
    left.invert()
    trimesh.repair.fix_normals(left)
    return left


def assemble_stl(
    meshes: dict[str, "trimesh.Trimesh"],
    candidate_id: str,
    out_dir: str,
) -> tuple[str, str]:
    """
    Returns (full_stl_path, half_stl_path).

    Full car:
    1. Mirror right sidepod -> left sidepod
    2. Concatenate all components
    3. Fix winding and normals
    4. Check watertight
    5. Export full STL

    Right-half STL (for Part 2 CFD — CONFIRMED half-car simulation):
    Slicing the assembled full car at y=0 fails for complex mesh topologies
    (open edges at the symmetry plane where multiple components meet).
    Instead, we slice each COMPONENT individually at y=0, cap the cross-section,
    then concatenate the per-component halves. Each component's cross-section is
    topologically simple (a single closed polygon), making capping reliable.
    The sidepod is already right-half only and is included directly.
    """
    import trimesh

    # Validate all components are present
    for name in ("nose", "sidepod", "rearpod", "main_body"):
        if name not in meshes:
            raise ValueError(f"Missing '{name}' mesh.")
    if "sidepod" not in meshes:
        raise ValueError("Missing 'sidepod' mesh for mirroring.")

    # ── Full-car assembly ──────────────────────────────────────────────────
    left_sidepod = _mirror_right_to_left(meshes["sidepod"])

    parts_full = [meshes[n] for n in ("nose", "sidepod", "rearpod", "main_body")]
    parts_full.append(left_sidepod)
    full_car = trimesh.util.concatenate(parts_full)
    trimesh.repair.fix_winding(full_car)
    trimesh.repair.fix_normals(full_car)

    if not full_car.is_watertight:
        trimesh.repair.fill_holes(full_car)
        if not full_car.is_watertight:
            from surface_extraction import MeshQualityFailure
            raise MeshQualityFailure("Full car assembly is not watertight.")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    full_path = out_path / f"car_{candidate_id}_full.stl"
    full_car.export(str(full_path))

    # ── Right-half STL (per-component slicing) ─────────────────────────────
    # Slice each symmetric component individually, then add the sidepod directly.
    # Per-component slicing is reliable because each component's cross-section at
    # y=0 is a single simple polygon (not the complex multi-hole polygon that
    # results from slicing the entire assembled car).
    half_parts = []

    for name in ("nose", "sidepod", "rearpod", "main_body"):
        mesh = meshes[name]
        try:
            sliced = trimesh.intersections.slice_mesh_plane(
                mesh,
                plane_normal=(0, 1, 0),
                plane_origin=(0, 0, 0),
                cap=True,
            )
            trimesh.repair.fill_holes(sliced)
            trimesh.repair.fix_normals(sliced)
            half_parts.append(sliced)
        except Exception:
            # Fallback: take faces with centroid y >= 0
            centres = mesh.triangles_center
            right_face_idx = np.where(centres[:, 1] >= -1e-6)[0]
            if len(right_face_idx) > 0:
                sub = mesh.submesh([right_face_idx], append=True)
                half_parts.append(sub)

    right_half = trimesh.util.concatenate(half_parts)
    trimesh.repair.fix_winding(right_half)
    trimesh.repair.fix_normals(right_half)

    if not right_half.is_watertight:
        trimesh.repair.fill_holes(right_half)
        trimesh.repair.fix_normals(right_half)
        if not right_half.is_watertight:
            from surface_extraction import MeshQualityFailure
            raise MeshQualityFailure(
                f"Right-half STL for candidate '{candidate_id}' is not watertight "
                f"after per-component slicing and repair. Check component meshes for "
                f"complex topology at the y=0 symmetry plane."
            )

    # Verify no y < 0 vertices (numerical tolerance)
    if np.any(right_half.vertices[:, 1] < -1e-6):
        from surface_extraction import MeshQualityFailure
        raise MeshQualityFailure("Right half has vertices with y < 0 after slicing.")

    half_path = out_path / f"car_{candidate_id}_half.stl"
    right_half.export(str(half_path))

    return (str(full_path.resolve()), str(half_path.resolve()))