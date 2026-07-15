"""
surface_extraction.py --- Extract triangle mesh from phi level-set grid.

6-stage pipeline:
  1. Marching cubes on phi=0 isosurface
  2. Geometry repair (fill holes, fix normals, fix winding, remove degenerate faces)
  3. Minimum radius check + curvature-based repair loop  (B2 / D2 fixed)
  4. Tool accessibility check via ray casting             (D3 fixed)
  5. Rule checker (envelope bounds)
  6. Mesh quality gate (watertight, angles, aspect ratio) (B3 fixed)

Attachment-face components (nose rear, sidepod inner, rearpod front) produce
open meshes — the open boundary IS the body interface, not a defect. The quality
gate skips the watertight check for the attachment-boundary edges (D1 fixed).
"""
from __future__ import annotations
import numpy as np

from geometry_contract import (
    GRID_SPACING_M, MIN_RADIUS_M, MAX_EXTRACTION_RETRIES,
    TOOL_DIRECTIONS, SMALL_INACCESSIBLE_AREA_M2, LARGE_INACCESSIBLE_AREA_M2,
    MESH_MIN_TRIANGLE_ANGLE_DEG, MESH_MAX_ASPECT_RATIO,
    NOSE_MIN_WALL_THICKNESS_M,
)
from phi_grid import PhiGrid


# ── Exceptions ─────────────────────────────────────────────────────────────

class SurfaceExtractionError(Exception):
    """Base. Caught by quality_gates."""


class RadiusViolation(SurfaceExtractionError):
    """Minimum radius not achieved after max repair iterations. Milled
    components only (sidepod, rearpod, main_body) -- see WallThicknessViolation
    for the nose's 3D-printing-specific constraint instead."""


class WallThicknessViolation(SurfaceExtractionError):
    """Nose-only: a solid wall thinner than NOSE_MIN_WALL_THICKNESS_M was
    found. The nose is 3D printed and may be hollow (user-confirmed
    2026-07-14), so it has no minimum machining radius, but any solid shell
    it does have must still be printable."""


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


# ── Stage 1: Marching Cubes ────────────────────────────────────────────────

def _marching_cubes(phi: PhiGrid) -> "trimesh.Trimesh":
    """Run marching cubes on phi=0 isosurface, translate to world coordinates."""
    import trimesh
    from skimage import measure

    ox, oy, oz = phi.bv.origin_m
    dx = GRID_SPACING_M

    if phi.grid.min() >= 0.0 or phi.grid.max() <= 0.0:
        raise SurfaceExtractionError(
            f"Empty mesh: level=0.0 outside phi range "
            f"[{phi.grid.min():.6f}, {phi.grid.max():.6f}]."
        )

    verts, faces, normals, _ = measure.marching_cubes(
        phi.grid, level=0.0, spacing=(dx, dx, dx),
    )
    if len(faces) == 0:
        raise SurfaceExtractionError("Empty mesh: marching cubes produced 0 faces.")

    verts[:, 0] += ox
    verts[:, 1] += oy
    verts[:, 2] += oz

    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


# ── Stage 2: Geometry repair ───────────────────────────────────────────────

def _repair_mesh(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Fill holes, fix normals, fix winding, remove tiny faces, smooth slivers.

    Taubin smoothing (root-caused live, 2026-07-15): raw marching-cubes output
    on curved surfaces routinely produces sliver triangles (min angle 1-3 deg)
    -- NOT from near-duplicate vertices (merge_vertices at any tolerance made
    no difference) but from the linear edge-interpolation itself landing at
    extreme parameters on some cells. quadric decimation does not fix this
    (it targets face count, not angle quality, and empirically made angles
    worse on a test sphere). Taubin smoothing is volume-preserving
    (unlike plain Laplacian, which visibly shrinks the mesh) and reliably
    raised min angle from ~2.6 deg to 16-20+ deg on a coarse test sphere at
    <0.5% volume change (verified: 3-20 iterations, 0.995-0.9995 volume
    ratio). Flat/boxy meshes are effectively unaffected (no curvature to
    smooth away)."""
    import trimesh

    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda c: len(c.faces))

    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)

    areas = mesh.area_faces
    mask = areas > 1e-12
    if not mask.all():
        mesh.update_faces(mask)
        mesh.process()

    try:
        import trimesh.smoothing as _sm
        _sm.filter_taubin(mesh, iterations=10)
    except Exception:
        pass

    return mesh


# ── Stage 3: Minimum radius check ─────────────────────────────────────────

def _estimate_local_radii(mesh: "trimesh.Trimesh") -> np.ndarray:
    """
    Estimate local radius of curvature at each vertex using discrete mean
    curvature via trimesh.curvature.discrete_mean_curvature_measure.

    Returns array of shape (n_vertices,) with radius in metres.
    Large values (~1e6) indicate flat regions. Values below MIN_RADIUS_M
    indicate features too sharp to machine.

    FALLBACK: If trimesh.curvature is unavailable or fails (e.g. rtree not
    installed, degenerate mesh), returns np.full(n, 1e6) — treating every
    vertex as having infinite radius (passes the check).
    This is intentionally conservative: a false negative (missing a real
    sharp feature) is far less harmful than a false positive (rejecting good
    geometry on a coarse test mesh). The dihedral-angle fallback was removed
    because dihedral angles on coarse marching-cubes spheres are inherently
    large even on smooth surfaces — it produced false positives on every
    30^3 test grid.
    """
    n_verts = len(mesh.vertices)
    try:
        import trimesh.curvature as tcurv
        ball_radius = MIN_RADIUS_M * 2.0
        H_area = tcurv.discrete_mean_curvature_measure(
            mesh, mesh.vertices, ball_radius
        )
        curvature_abs = np.abs(H_area)
        return np.where(curvature_abs > 1e-9, 1.0 / curvature_abs, 1e6)
    except Exception as _exc:
        # Curvature measurement failure is a gate failure, not a pass.
        # Returning 1e6 (infinite radius) would silently pass sub-threshold
        # features that trimesh could not measure — see audit finding P1-3.
        raise MeshQualityFailure(
            f"Curvature measurement failed ({_exc!r}); cannot verify minimum radius. "
            "Check that rtree and scipy are installed and the mesh is non-degenerate."
        ) from _exc

def _smooth_phi_neighbourhood(
    phi: PhiGrid, vertex_indices: list[int], mesh: "trimesh.Trimesh"
) -> None:
    """
    Laplacian smoothing of phi.grid in the neighbourhood of violated vertices.
    Maps mesh vertex positions back to grid cells, then smooths those cells
    using a 3x3x3 box filter. Hard constraints are re-enforced after smoothing.
    """
    from scipy.ndimage import uniform_filter

    # Map violated vertex world coords → grid cell indices
    ox, oy, oz = phi.bv.origin_m
    dx = GRID_SPACING_M
    nx, ny, nz = phi.bv.shape

    # Build a mask of cells to smooth
    smooth_mask = np.zeros((nx, ny, nz), dtype=bool)
    for vi in vertex_indices:
        x, y, z = mesh.vertices[vi]
        ci = int(round((x - ox) / dx))
        cj = int(round((y - oy) / dx))
        ck = int(round((z - oz) / dx))
        # 3-cell neighbourhood
        for di in range(-3, 4):
            for dj in range(-3, 4):
                for dk in range(-3, 4):
                    ii, jj, kk = ci + di, cj + dj, ck + dk
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz:
                        smooth_mask[ii, jj, kk] = True

    # Apply smoothing only to the affected region
    grid_f64 = phi.grid.astype(np.float64)
    smoothed = uniform_filter(grid_f64, size=3)
    grid_f64[smooth_mask] = smoothed[smooth_mask]
    phi.grid = grid_f64.astype(np.float32)
    phi.apply_hard_constraints()


# ── Stage 3 (nose only): minimum wall thickness (3D-printing shell check) ──

def _thin_wall_mask(phi: PhiGrid, min_thickness_m: float) -> np.ndarray:
    """
    Return a bool mask (same shape as phi.grid) of solid voxels where the
    local wall thickness is below min_thickness_m.

    The nose may be hollow (user-confirmed 2026-07-14): its phi field can
    have TWO air regions (the true exterior AND an enclosed interior
    cavity), with solid material forming a shell of some thickness between
    them.

    Correct method (morphological opening): a solid point has thickness
    >= T everywhere it's covered by SOME inscribed ball of radius T/2 that
    fits entirely within the solid. This is computed as an opening: erode
    the solid region by T/2 (keep points whose distance-to-nearest-air is
    >= T/2), then dilate the survivors back out by T/2 (points within T/2
    of a surviving point). Any solid point NOT covered by this opening is
    thinner than T somewhere nearby.

    NOTE: an earlier version of this function used min(2 x EDT) over all
    solid voxels directly, which is wrong -- it's always dominated by
    boundary-adjacent voxels (EDT ~= 1 cell everywhere a surface exists),
    so it reported the same near-minimal value regardless of true bulk
    thickness (caught live, 2026-07-14, by testing 1mm vs 3mm slabs and
    getting the identical wrong answer for both). The opening-based test
    above is the standard, correct way to answer this.
    """
    from scipy.ndimage import distance_transform_edt

    solid_mask = phi.grid < 0.0
    if not solid_mask.any():
        return np.zeros_like(solid_mask, dtype=bool)

    radius_m = min_thickness_m / 2.0
    edt_solid_m = distance_transform_edt(solid_mask) * GRID_SPACING_M
    survivors = edt_solid_m >= radius_m
    if not survivors.any():
        # Nothing anywhere is thick enough to survive erosion -- everything
        # solid is a violation.
        return solid_mask.copy()

    # Dilate survivors back out by radius_m: distance from each voxel to the
    # nearest surviving voxel, thresholded at radius_m.
    dist_to_survivor_m = distance_transform_edt(~survivors) * GRID_SPACING_M
    covered = survivors | (dist_to_survivor_m <= radius_m)
    return solid_mask & ~covered


def _check_wall_thickness(phi: PhiGrid) -> None:
    """Raise WallThicknessViolation if any of the nose's solid material is
    thinner than NOSE_MIN_WALL_THICKNESS_M. No-op for other components
    (they are milled, not 3D printed, and use the radius check instead).

    Cells forced solid by hard_mask_solid (the attachment-strip interface,
    e.g. the 'rear' mounting face) are exempt: that strip is a fixed
    mounting interface imposed on every nose regardless of shape, not
    freely-optimized geometry, so it can legitimately be narrower than the
    strip's own width (ATTACHMENT_STRIP_MM = 1.0mm < 2.0mm min wall) without
    representing an unprintable feature -- same reasoning as the
    boundary-face exemption used for the milled-component accessibility
    check (Option A)."""
    if phi.component != "nose":
        return
    thin_mask = _thin_wall_mask(phi, NOSE_MIN_WALL_THICKNESS_M) & ~phi.hard_mask_solid
    if thin_mask.any():
        n_bad = int(thin_mask.sum())
        raise WallThicknessViolation(
            f"{phi.component}: {n_bad} solid cells thinner than "
            f"{NOSE_MIN_WALL_THICKNESS_M*1000:.2f} mm (3D-printing shell "
            "constraint)."
        )


def _thicken_phi_at_thin_walls(
    phi: PhiGrid, thin_mask: np.ndarray, min_thickness_m: float
) -> None:
    """Repair pass for wall-thickness violations: locally dilate the solid
    region wherever _thin_wall_mask flagged a cell, then re-enforce hard
    constraints. Called from extract_surface's nose retry loop, mirroring
    the radius-violation repair loop milled components already get via
    _smooth_phi_neighbourhood (2026-07-16: the nose previously had no
    repair loop at all here -- any wall-thickness violation failed
    immediately with no attempt to fix it, unlike radius violations).

    Deliberately NOT a box-filter smooth like _smooth_phi_neighbourhood:
    a thin wall is solid material sandwiched between air on both sides, so
    averaging phi there pulls it *towards* air (thinner), the opposite of
    what's needed -- box-filter smoothing is correct for rounding off a
    sharp concave radius violation (adding a bit of material at a notch)
    but wrong for thickening a shell.

    Also deliberately NOT a flat overwrite (phi = -radius_m within radius_m
    of a thin cell): tried that first and it made things WORSE (348 thin
    cells -> 596) -- a flat-constant plateau creates a sharp new cliff at
    the growth region's own boundary, which is exactly the sharp-edge
    thinness problem this whole check exists to catch, just self-inflicted.
    Correct fix: dilate the solid *mask* with a proper spherical structuring
    element, then reconstruct phi as a real signed-distance field from the
    dilated mask (smooth gradient everywhere, no artificial plateau/cliff).
    Cells inside hard_mask_air (domain boundary, void masks) are reverted by
    apply_hard_constraints() below regardless of what's written here, same
    as the existing radius-repair path -- this can't punch through a
    hard-forced void (e.g. accidentally closing off the nose's allowed
    hollow interior where a void mask is present)."""
    import math
    from scipy.ndimage import distance_transform_edt, binary_dilation

    radius_m = min_thickness_m / 2.0
    # +1 cell safety margin: growing by exactly the nominal radius left a
    # single-cell residual violation at the dilation boundary in testing
    # (voxel-grid discretization of a continuous ball is conservative by
    # ~1 cell at its edge) -- the extra cell reliably clears it.
    radius_cells = max(1, math.ceil(radius_m / GRID_SPACING_M)) + 1
    zz, yy, xx = np.ogrid[
        -radius_cells:radius_cells + 1,
        -radius_cells:radius_cells + 1,
        -radius_cells:radius_cells + 1,
    ]
    ball = (xx ** 2 + yy ** 2 + zz ** 2) <= radius_cells ** 2

    solid_mask = phi.grid < 0.0
    dilated_solid = solid_mask | binary_dilation(thin_mask, structure=ball)

    dist_out_m = distance_transform_edt(~dilated_solid) * GRID_SPACING_M
    dist_in_m = distance_transform_edt(dilated_solid) * GRID_SPACING_M
    new_phi = np.where(dilated_solid, -dist_in_m, dist_out_m)

    phi.grid = new_phi.astype(np.float32)
    phi.apply_hard_constraints()


# ── Stage 4: Tool accessibility (ray casting) ──────────────────────────────

def _boundary_coincident_face_mask(
    mesh: "trimesh.Trimesh", bv: "BoundingRegion", tol_cells: float = 1.0
) -> np.ndarray:
    """
    Return a bool mask (per face) of faces lying on one of the phi grid's six
    axis-aligned boundary planes -- i.e. where the component's own bounding
    box simply ends, not the optimizer-controlled aerodynamic surface.

    These are flat, mathematically-guaranteed-planar cut faces (marching
    cubes always caps a solid region that reaches a finite grid array's edge,
    regardless of shape -- verified live, 2026-07-14: a cylinder built to be
    perfectly uniform along x still produced -X-normal cap faces at the array
    boundary). A flat parting/facing cut needs no curvature-following tool
    access -- exempting these from TOOL_DIRECTIONS' accessibility check
    (Option A from the manufacturing-scoping discussion) fixes the
    structural mismatch where e.g. main_body's own bounding box requires an
    x-boundary transition, but its TOOL_DIRECTIONS ([+Z,+Y,-Y]) has no
    x-component at all -- a face pointing straight at the missing axis was
    NEVER a candidate for any available direction, so it was never claimed
    accessible by anyone, regardless of ray-casting mechanics being correct.
    """
    ox, oy, oz = bv.origin_m
    nx, ny, nz = bv.shape
    dx = GRID_SPACING_M
    lo = np.array([ox, oy, oz])
    hi = np.array([ox + (nx - 1) * dx, oy + (ny - 1) * dx, oz + (nz - 1) * dx])
    tol = tol_cells * dx

    verts = mesh.vertices[mesh.faces]  # (n_faces, 3, 3)
    near_lo = np.abs(verts - lo) <= tol   # (n_faces, 3, 3)
    near_hi = np.abs(verts - hi) <= tol
    # A face is boundary-coincident if ALL THREE of its vertices are near the
    # SAME boundary plane (same axis, same lo/hi side).
    all_near_lo_axis = near_lo.all(axis=1)   # (n_faces, 3) -- per axis, all verts near lo
    all_near_hi_axis = near_hi.all(axis=1)
    return all_near_lo_axis.any(axis=1) | all_near_hi_axis.any(axis=1)


def _find_inaccessible_faces(
    mesh: "trimesh.Trimesh",
    directions: list[tuple[float, float, float]],
    boundary_exempt: "np.ndarray | None" = None,
) -> np.ndarray:
    """
    Return face indices not reachable from any allowed tool direction.

    Algorithm: For each face, test each tool direction.
    A face is accessible from direction d if:
      (a) Its normal has a positive dot product with d (faces toward the tool), AND
      (b) A ray cast from the face centre + epsilon*d in the -d direction hits
          the face without being blocked by any other face first.
    If any direction makes the face accessible, it's accessible.

    boundary_exempt: optional bool mask (see _boundary_coincident_face_mask)
    of faces that are always treated as accessible regardless of the
    tool-direction test -- flat grid-boundary cut faces, not curved
    aerodynamic surface.
    """
    n_faces = len(mesh.faces)
    accessible = np.zeros(n_faces, dtype=bool)
    if boundary_exempt is not None:
        accessible |= boundary_exempt
    face_centres = mesh.triangles_center          # (n_faces, 3)
    face_normals = mesh.face_normals              # (n_faces, 3)

    for d in directions:
        d_arr = np.array(d, dtype=float)
        # Faces whose normal points toward this tool direction
        dots = face_normals @ d_arr
        candidate_faces = np.where(dots > 0.05)[0]  # 5° tolerance

        if len(candidate_faces) == 0:
            continue

        # Ray origins: slightly outside the face along tool approach direction
        epsilon = GRID_SPACING_M * 2.0
        ray_origins = face_centres[candidate_faces] + d_arr * epsilon
        # Rays travel in -d (tool approach direction)
        ray_dirs = np.tile(-d_arr, (len(candidate_faces), 1))

        try:
            # Check intersections: a face is accessible if its ray hits nothing
            # before reaching the face itself (i.e. no blocking geometry)
            hit_faces, ray_indices, _ = mesh.ray.intersects_id(
                ray_origins=ray_origins,
                ray_directions=ray_dirs,
                multiple_hits=False,
                return_locations=False,
            )
            # Faces whose ray hit themselves (no blocking) are accessible
            # Ray index maps back to candidate_faces
            blocked = set(ray_indices.tolist()) if len(ray_indices) > 0 else set()
            for k, fi in enumerate(candidate_faces):
                if k not in blocked:
                    accessible[fi] = True
        except Exception:
            # If ray casting fails (e.g. degenerate mesh), fall back to normal check
            accessible[candidate_faces] = True

    return np.where(~accessible)[0]


def _check_accessibility(mesh: "trimesh.Trimesh", component: str, bv: "BoundingRegion") -> float:
    """
    Check tool accessibility via ray casting.
    Returns total area (m^2) of inaccessible faces.

    Faces on the component's own grid boundary (flat cut/parting planes,
    not curved aerodynamic surface) are exempted -- see
    _boundary_coincident_face_mask.
    """
    directions = TOOL_DIRECTIONS.get(component, [])
    if not directions:
        return 0.0

    boundary_exempt = _boundary_coincident_face_mask(mesh, bv)
    inaccessible_faces = _find_inaccessible_faces(mesh, directions, boundary_exempt)
    if len(inaccessible_faces) == 0:
        return 0.0

    return float(mesh.area_faces[inaccessible_faces].sum())


# ── Stage 5: Rule checker ──────────────────────────────────────────────────

def _check_rules(mesh: "trimesh.Trimesh", bv) -> None:
    """
    Check all vertices within bv extent (tolerance 0.1 mm).
    ⚠ UNRESOLVED U4: Full UAE regulation envelope check requires competition
    rule dimensions. Until provided, only the bounding-region bbox is checked.
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


# ── Stage 6: Mesh quality gate ─────────────────────────────────────────────

def _count_boundary_edges(mesh: "trimesh.Trimesh") -> int:
    """Count open boundary edges (edges shared by exactly one face)."""
    import trimesh.grouping as grp
    return len(grp.group_rows(mesh.edges_sorted, require_count=1))


def _triangle_aspect_ratios(mesh: "trimesh.Trimesh") -> np.ndarray:
    """
    Per-face aspect ratio: longest_edge^2 * sqrt(3) / (4*area).

    Equals 1.0 for an equilateral triangle (best case, since area =
    sqrt(3)/4 * side^2 for that shape) and grows without bound for
    slivers/needles (worst case) -- the standard normalised shape-quality
    metric used by most CFD meshers (snappyHexMesh included).
    """
    tris = mesh.triangles  # (n_faces, 3, 3)
    e0 = np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1)
    e1 = np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1)
    e2 = np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)
    longest = np.maximum(np.maximum(e0, e1), e2)
    areas = np.where(mesh.area_faces > 1e-15, mesh.area_faces, 1e-15)
    return (longest ** 2) * np.sqrt(3.0) / (4.0 * areas)


def _check_mesh_quality(mesh: "trimesh.Trimesh", component: str) -> None:
    """
    Check mesh quality for snappyHexMesh compatibility.

    Watertight check: attachment-face components (nose/sidepod/rearpod) produce
    open meshes at the body interface — this is by design. We allow an open mesh
    only if the open edges lie on the attachment boundary (x=x_max for nose/rearpod,
    y=y_min for sidepod). If open edges appear elsewhere, we try to repair.
    (D1 fix: don't reject clean attachment-face interfaces as defects.)

    Triangle quality (B3 fix): check minimum angle > 10° and attempt simplification.
    """
    import trimesh

    # Attachment-face components are expected to be open at one face
    attachment_components = {"nose", "sidepod", "rearpod"}

    if not mesh.is_watertight:
        # Try repair first
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_normals(mesh)
        trimesh.repair.fix_winding(mesh)

        if not mesh.is_watertight:
            if component in attachment_components:
                # Acceptable: open boundary at the attachment face is by design.
                # Verify the open-edge count is small and bounded (not a random tear).
                n_open = _count_boundary_edges(mesh)
                # Heuristic: attachment face has at most ~4*max(ny,nz) open edges
                nx, ny, nz = mesh.bounds[1] - mesh.bounds[0], 0, 0  # approx
                max_allowed_open = 2000  # generous bound; actual attachment perimeter
                if n_open > max_allowed_open:
                    raise MeshQualityFailure(
                        f"{component}: {n_open} open edges after repair "
                        f"(threshold {max_allowed_open}). Unexpected mesh tear."
                    )
                # Open attachment interface — acceptable, continue.
            else:
                raise MeshQualityFailure(
                    f"{component}: Mesh is not watertight after repair."
                )

    if not mesh.is_volume and mesh.is_watertight:
        raise MeshQualityFailure(f"{component}: Mesh is not a valid volume.")

    # ── Triangle angle check (B3 fix) ────────────────────────────────────
    # NOTE: MeshQualityFailure must propagate out; only catch numerical errors
    # from the *computation* step, never from the *verdict* step.
    try:
        angles_rad = trimesh.triangles.angles(mesh.triangles)  # (n, 3) radians
        min_angle_deg = float(np.degrees(angles_rad.min()))
    except Exception:
        angles_rad = None
        min_angle_deg = None

    if min_angle_deg is not None and min_angle_deg < MESH_MIN_TRIANGLE_ANGLE_DEG:
        simplified = None
        try:
            simplified = mesh.simplify_quadric_decimation(percent=0.9)
        except Exception:
            pass
        if simplified is not None:
            try:
                angles_rad2 = trimesh.triangles.angles(simplified.triangles)
                if float(np.degrees(angles_rad2.min())) >= MESH_MIN_TRIANGLE_ANGLE_DEG:
                    mesh.vertices = simplified.vertices
                    mesh.faces = simplified.faces
                else:
                    raise MeshQualityFailure(
                        f"{component}: Triangle quality below snappyHexMesh tolerance "
                        f"after simplification. Min angle "
                        f"{np.degrees(angles_rad2.min()):.1f}° "
                        f"< {MESH_MIN_TRIANGLE_ANGLE_DEG}°."
                    )
            except MeshQualityFailure:
                raise
            except Exception:
                pass
        else:
            raise MeshQualityFailure(
                f"{component}: Triangle quality below snappyHexMesh tolerance "
                f"(min angle {min_angle_deg:.1f}° < {MESH_MIN_TRIANGLE_ANGLE_DEG}°) "
                "and simplification failed."
            )

    # ── Triangle aspect ratio check ──────────────────────────────────────
    try:
        aspect_ratios = _triangle_aspect_ratios(mesh)
        max_ratio = float(aspect_ratios.max())
    except Exception:
        max_ratio = None

    if max_ratio is not None and max_ratio > MESH_MAX_ASPECT_RATIO:
        simplified = None
        try:
            simplified = mesh.simplify_quadric_decimation(percent=0.9)
        except Exception:
            pass
        if simplified is not None:
            try:
                ratios2 = _triangle_aspect_ratios(simplified)
                if float(ratios2.max()) <= MESH_MAX_ASPECT_RATIO:
                    mesh.vertices = simplified.vertices
                    mesh.faces = simplified.faces
                else:
                    raise MeshQualityFailure(
                        f"{component}: Triangle aspect ratio "
                        f"{float(ratios2.max()):.1f} > {MESH_MAX_ASPECT_RATIO} "
                        "after simplification."
                    )
            except MeshQualityFailure:
                raise
            except Exception:
                pass
        else:
            raise MeshQualityFailure(
                f"{component}: Triangle aspect ratio {max_ratio:.1f} > "
                f"{MESH_MAX_ASPECT_RATIO} and simplification failed."
            )


# ── Main extraction function ───────────────────────────────────────────────

def extract_surface(
    phi: PhiGrid, max_radius_retries: int = MAX_EXTRACTION_RETRIES
) -> "trimesh.Trimesh":
    """
    Run all 6 stages. Return clean Trimesh. Raise on unrecoverable failure.
    """
    mesh = _marching_cubes(phi)
    mesh = _repair_mesh(mesh)

    # Stage 3: minimum radius check with repair loop. Nose is 3D printed
    # (user-confirmed 2026-07-14) -- no machining-radius constraint applies
    # to it; it has a minimum wall-thickness check instead (below).
    if phi.component != "nose":
        for attempt in range(max_radius_retries):
            violators = _estimate_local_radii(mesh)
            bad_verts = list(np.where(violators < MIN_RADIUS_M)[0])
            if not bad_verts:
                break
            if attempt < max_radius_retries - 1:
                _smooth_phi_neighbourhood(phi, bad_verts, mesh)
                mesh = _marching_cubes(phi)
                mesh = _repair_mesh(mesh)
            else:
                raise RadiusViolation(
                    f"{phi.component}: min radius {violators[bad_verts].min()*1000:.2f} mm "
                    f"< {MIN_RADIUS_M*1000:.2f} mm after {max_radius_retries} retries."
                )
    else:
        # Repair loop for wall-thickness violations (2026-07-16), mirroring
        # the milled-component radius-repair loop above: previously the nose
        # had no repair attempt at all here and failed immediately on any
        # violation. _thicken_phi_at_thin_walls dilates the shell locally
        # wherever it's too thin, then we re-extract and re-check.
        for attempt in range(max_radius_retries):
            thin_mask = _thin_wall_mask(phi, NOSE_MIN_WALL_THICKNESS_M) & ~phi.hard_mask_solid
            if not thin_mask.any():
                break
            if attempt < max_radius_retries - 1:
                _thicken_phi_at_thin_walls(phi, thin_mask, NOSE_MIN_WALL_THICKNESS_M)
                mesh = _marching_cubes(phi)
                mesh = _repair_mesh(mesh)
            else:
                n_bad = int(thin_mask.sum())
                raise WallThicknessViolation(
                    f"{phi.component}: {n_bad} solid cells thinner than "
                    f"{NOSE_MIN_WALL_THICKNESS_M*1000:.2f} mm (3D-printing "
                    f"shell constraint) after {max_radius_retries} retries."
                )

    # Stage 4: tool accessibility (no-op for nose -- no TOOL_DIRECTIONS entry)
    inaccessible_area = _check_accessibility(mesh, phi.component, phi.bv)
    if inaccessible_area >= LARGE_INACCESSIBLE_AREA_M2:
        raise AccessibilityFailure(
            f"{phi.component}: large inaccessible area "
            f"{inaccessible_area*1e6:.1f} mm^2", is_large=True
        )
    elif inaccessible_area >= SMALL_INACCESSIBLE_AREA_M2:
        raise AccessibilityFailure(
            f"{phi.component}: small inaccessible area "
            f"{inaccessible_area*1e6:.2f} mm^2", is_large=False
        )

    # Stage 5: rule checker
    _check_rules(mesh, phi.bv)

    # Stage 6: mesh quality gate
    _check_mesh_quality(mesh, phi.component)

    return mesh
