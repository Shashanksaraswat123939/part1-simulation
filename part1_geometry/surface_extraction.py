"""
surface_extraction.py --- Extract triangle mesh from phi level-set grid.

6-stage pipeline:
  1. Marching cubes on phi=0 isosurface
  2. Geometry repair (fill holes, fix normals, fix winding)
  3. Minimum radius check + repair loop
  4. Tool accessibility check
  5. Rule checker (envelope bounds)
  6. Mesh quality gate (watertight, angles, aspect ratio)
"""
from __future__ import annotations
import numpy as np

from geometry_contract import (
    GRID_SPACING_M, MIN_RADIUS_M, MAX_EXTRACTION_RETRIES,
    TOOL_DIRECTIONS, SMALL_INACCESSIBLE_AREA_M2, LARGE_INACCESSIBLE_AREA_M2,
    MESH_MIN_TRIANGLE_ANGLE_DEG, MESH_MAX_ASPECT_RATIO,
)
from phi_grid import PhiGrid


# ------------------------------------------------------------------ #
#  Exceptions
# ------------------------------------------------------------------ #

class SurfaceExtractionError(Exception):
    """Base. Caught by quality_gates."""
    pass


class RadiusViolation(SurfaceExtractionError):
    """Minimum radius not achieved after max repair iterations."""
    pass


class AccessibilityFailure(SurfaceExtractionError):
    is_large: bool
    def __init__(self, msg: str, is_large: bool):
        super().__init__(msg)
        self.is_large = is_large


class RuleViolation(SurfaceExtractionError):
    is_major: bool
    def __init__(self, msg: str, is_major: bool):
        super().__init__(msg)
        self.is_major = is_major


class MeshQualityFailure(SurfaceExtractionError):
    """Mesh fails gate after simplification attempt."""
    pass


# ------------------------------------------------------------------ #
#  Stage 1: Marching Cubes
# ------------------------------------------------------------------ #

def _marching_cubes(phi: PhiGrid) -> "trimesh.Trimesh":
    """Run marching cubes on phi=0 isosurface, translate to world coordinates."""
    import trimesh
    from skimage import measure

    nx, ny, nz = phi.bv.shape
    ox, oy, oz = phi.bv.origin_m
    dx = GRID_SPACING_M

    # Check that 0.0 is within the data range
    if phi.grid.min() >= 0.0 or phi.grid.max() <= 0.0:
        raise SurfaceExtractionError(
            "Empty mesh: level=0.0 is outside phi data range "
            f"[{phi.grid.min():.6f}, {phi.grid.max():.6f}]."
        )

    verts, faces, normals, _ = measure.marching_cubes(
        phi.grid,
        level=0.0,
        spacing=(dx, dx, dx),
    )

    if len(faces) == 0:
        raise SurfaceExtractionError("Empty mesh: marching cubes produced 0 faces.")

    # Translate vertices from index space to world coordinates
    verts[:, 0] += ox
    verts[:, 1] += oy
    verts[:, 2] += oz

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    return mesh


# ------------------------------------------------------------------ #
#  Stage 2: Geometry repair
# ------------------------------------------------------------------ #

def _repair_mesh(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Fill holes, fix normals, fix winding, remove tiny faces."""
    import trimesh

    # Keep largest connected component
    if len(mesh.split(only_watertight=False)) > 1:
        components = mesh.split(only_watertight=False)
        mesh = max(components, key=lambda c: len(c.faces))

    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)

    # Remove degenerate faces (area < 1e-12 m^2)
    areas = mesh.area_faces
    mask = areas > 1e-12
    if not mask.all():
        mesh.update_faces(mask)
        mesh.process()

    return mesh


# ------------------------------------------------------------------ #
#  Stage 3: Minimum radius check
# ------------------------------------------------------------------ #

def _check_min_radius(mesh: "trimesh.Trimesh") -> list[int]:
    """
    Estimate local radius at each vertex from neighbour angles.
    Returns list of vertex indices that violate MIN_RADIUS_M.

    PLACEHOLDER: Proper local radius estimation requires analysing
    dihedral angles and vertex neighbourhood curvature relative to
    the MIN_RADIUS_MM machining constraint. The current implementation
    is a stub that returns no violators. This must be replaced with
    a proper curvature-based check once the geometric algorithm is
    validated against physical test parts.
    """
    return []


def _smooth_phi_neighbourhood(phi: PhiGrid, vertex_indices: list[int], mesh: "trimesh.Trimesh") -> None:
    """Laplacian smoothing of phi.grid at neighbourhood of violated vertices."""
    # PLACEHOLDER: map mesh vertices back to grid cells and smooth those cells.
    # For now, do a mild global smoothing.
    from scipy.ndimage import uniform_filter
    smoothed = uniform_filter(phi.grid, size=3).astype(np.float32)
    phi.grid = smoothed
    phi.apply_hard_constraints()


# ------------------------------------------------------------------ #
#  Stage 4: Tool accessibility
# ------------------------------------------------------------------ #

def _check_accessibility(mesh: "trimesh.Trimesh", component: str) -> float:
    """
    Check tool accessibility for the component.
    Returns total area of inaccessible faces in m^2.
    """
    directions = TOOL_DIRECTIONS.get(component, [])
    if not directions:
        return 0.0

    # PLACEHOLDER: full ray-casting accessibility check.
    # For now, do a simple normal-based check: faces whose normal points
    # away from ALL tool directions are inaccessible.
    face_normals = mesh.face_normals
    face_areas = mesh.area_faces

    accessible = np.zeros(len(face_normals), dtype=bool)
    for direction in directions:
        d = np.array(direction)
        dots = face_normals @ d
        accessible |= dots > 0.1  # face faces toward tool direction

    inaccessible_area = float(np.sum(face_areas[~accessible]))
    return inaccessible_area


# ------------------------------------------------------------------ #
#  Stage 5: Rule checker
# ------------------------------------------------------------------ #

def _check_rules(mesh: "trimesh.Trimesh", bv) -> None:
    """
    Check all vertices within bv extent (tolerance 0.1 mm = 1e-4 m).
    PLACEHOLDER: Full UAE envelope check not implemented (U4).
    """
    tol = 1e-4  # 0.1 mm
    verts = mesh.vertices

    x_min = bv.origin_m[0] - tol
    x_max = bv.x_max_m() + tol
    y_min = bv.origin_m[1] - tol
    y_max = bv.y_max_m() + tol
    z_min = bv.origin_m[2] - tol
    z_max = bv.z_max_m() + tol

    if np.any(verts[:, 0] < x_min) or np.any(verts[:, 0] > x_max):
        raise RuleViolation(
            f"Vertices outside x bounds [{x_min:.4f}, {x_max:.4f}]", is_major=True
        )
    if np.any(verts[:, 1] < y_min) or np.any(verts[:, 1] > y_max):
        raise RuleViolation(
            f"Vertices outside y bounds [{y_min:.4f}, {y_max:.4f}]", is_major=True
        )
    if np.any(verts[:, 2] < z_min) or np.any(verts[:, 2] > z_max):
        raise RuleViolation(
            f"Vertices outside z bounds [{z_min:.4f}, {z_max:.4f}]", is_major=True
        )


# ------------------------------------------------------------------ #
#  Stage 6: Mesh quality gate
# ------------------------------------------------------------------ #

def _check_mesh_quality(mesh: "trimesh.Trimesh") -> None:
    """Check watertight, is_volume, triangle angles, aspect ratios."""
    import trimesh

    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        if not mesh.is_watertight:
            raise MeshQualityFailure("Mesh is not watertight after repair.")

    if not mesh.is_volume:
        raise MeshQualityFailure("Mesh is not a valid volume.")

    # Check triangle angles
    # PLACEHOLDER: full triangle angle analysis.
    # trimesh provides triangles_angle, but we do a simple area-based check.
    areas = mesh.area_faces
    if len(areas) == 0:
        raise MeshQualityFailure("Mesh has no faces.")

    # Simple aspect ratio check via face area distribution
    # PLACEHOLDER: proper aspect ratio and min angle checks.


# ------------------------------------------------------------------ #
#  Main extraction function
# ------------------------------------------------------------------ #

def extract_surface(phi: PhiGrid, max_radius_retries: int = MAX_EXTRACTION_RETRIES) -> "trimesh.Trimesh":
    """
    Run all 6 stages. Return clean Trimesh. Raise on unrecoverable failure.
    """
    import trimesh

    mesh = _marching_cubes(phi)
    mesh = _repair_mesh(mesh)

    # Stage 3: minimum radius check (loop)
    for attempt in range(max_radius_retries):
        violators = _check_min_radius(mesh)
        if not violators:
            break
        if attempt < max_radius_retries - 1:
            _smooth_phi_neighbourhood(phi, violators, mesh)
            mesh = _marching_cubes(phi)
            mesh = _repair_mesh(mesh)
        else:
            raise RadiusViolation(
                f"Minimum radius {MIN_RADIUS_M*1000:.2f} mm not achieved after {max_radius_retries} retries."
            )

    # Stage 4: tool accessibility
    inaccessible_area = _check_accessibility(mesh, phi.component)
    if inaccessible_area < SMALL_INACCESSIBLE_AREA_M2:
        pass  # OK --- negligible
    elif inaccessible_area >= LARGE_INACCESSIBLE_AREA_M2:
        raise AccessibilityFailure(
            f"Large inaccessible area: {inaccessible_area*1e4:.2f} mm^2", is_large=True
        )
    else:
        raise AccessibilityFailure(
            f"Small inaccessible area: {inaccessible_area*1e4:.2f} mm^2", is_large=False
        )

    # Stage 5: rule checker
    _check_rules(mesh, phi.bv)

    # Stage 6: mesh quality gate
    _check_mesh_quality(mesh)

    return mesh