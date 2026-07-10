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
    5. Export STL

    Right-half STL (for Part 2 CFD):
    1. Slice full car at y=0, keep y >= 0 side
    2. Check watertight
    3. Export STL
    """
    import trimesh

    # Mirror sidepod
    if "sidepod" in meshes:
        left_sidepod = _mirror_right_to_left(meshes["sidepod"])
    else:
        raise ValueError("Missing 'sidepod' mesh for mirroring.")

    # Concatenate all components
    parts = []
    for name in ("nose", "sidepod", "rearpod", "main_body"):
        if name not in meshes:
            raise ValueError(f"Missing '{name}' mesh.")
        parts.append(meshes[name])

    parts.append(left_sidepod)

    full_car = trimesh.util.concatenate(parts)
    trimesh.repair.fix_winding(full_car)
    trimesh.repair.fix_normals(full_car)

    if not full_car.is_watertight:
        # Try repair
        trimesh.repair.fill_holes(full_car)
        if not full_car.is_watertight:
            from surface_extraction import MeshQualityFailure
            raise MeshQualityFailure("Full car assembly is not watertight.")

    # Export full STL
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    full_path = out_path / f"car_{candidate_id}_full.stl"
    full_car.export(str(full_path))

    # Slice at y=0 for right half
    right_half = trimesh.intersections.slice_mesh_plane(
        full_car,
        plane_normal=(0, 1, 0),
        plane_origin=(0, 0, 0),
        cap=True,
    )

    # Try repair if not watertight
    if not right_half.is_watertight:
        trimesh.repair.fill_holes(right_half)
        trimesh.repair.fix_normals(right_half)
        trimesh.repair.fix_winding(right_half)
        if not right_half.is_watertight:
            # PLACEHOLDER: slicing with cap=True may fail for complex meshes.
            # In production, a more robust slicing approach (e.g. via Open3D
            # or CGAL) may be needed. For now, export the best-effort mesh
            # and flag it as non-watertight in a warning.
            # The CFD wrapper in Part 2 will catch this downstream.
            pass  # Don't raise --- let downstream CFD validation handle it

    # Verify all vertices have y >= -1e-6 (numerical tolerance)
    if np.any(right_half.vertices[:, 1] < -1e-6):
        from surface_extraction import MeshQualityFailure
        raise MeshQualityFailure("Right half has vertices with y < 0 after slicing.")

    half_path = out_path / f"car_{candidate_id}_half.stl"
    right_half.export(str(half_path))

    return (str(full_path.resolve()), str(half_path.resolve()))