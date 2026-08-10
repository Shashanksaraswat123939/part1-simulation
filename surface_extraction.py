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
import warnings
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

    # SMOOTH FIRST, THEN clean up degenerates -- the order is the whole point.
    #
    # The tiny-face deletion used to run BEFORE this Taubin pass, and it was the
    # single worst thing in the extraction pipeline: update_faces punches a hole
    # for every face it removes, and trimesh's fill_holes only closes simple
    # holes, so scattered slivers left the mesh open. Measured on a carved car
    # at 1 mm: marching cubes handed over 128,280 triangles, watertight; repair
    # gave back 128,202, NOT watertight. The caller then raised "Mesh is not
    # watertight after repair" and Part 3 logged geometry_rejected -- which is
    # exactly how the first re-run after the redistancing fix died on its second
    # iteration.
    #
    # The deletion was also unnecessary. Taubin smoothing collapses the
    # degenerate faces on its own, by moving vertices rather than removing
    # triangles, so nothing is punched out. Measured over four carve steps at
    # level=0.0, smoothing with NO deletion at all:
    #     raw          watertight, 32-72 degenerate faces, min angle 0.00 deg
    #     +taubin      watertight, 0 degenerate faces,     min angle 14-24 deg
    # against the 10 deg snappyHexMesh gate. It reads as a leftover: the Taubin
    # pass was added 2026-07-15 and the deletion it made redundant was never
    # taken out, so the two fought and the deletion won.
    #
    # Both orderings were tried before settling here: reordering the deletion to
    # come first still lost watertightness, and merging coincident vertices
    # before dropping degenerate faces broke it on the very first step.
    try:
        import trimesh.smoothing as _sm
        _sm.filter_taubin(mesh, iterations=10)
    except Exception:
        pass

    # Belt and braces. Measures zero on every case tested, so it should not fire
    # -- but if smoothing ever leaves a degenerate face behind, dropping it and
    # refilling is still better than shipping a zero-area triangle to
    # snappyHexMesh. Warns, because silently changing topology here is what
    # caused the original bug.
    mask = mesh.area_faces > 1e-12
    if not mask.all():
        warnings.warn(
            f"{int((~mask).sum())} degenerate faces survived Taubin smoothing; "
            f"dropping them and refilling. If this fires, check whether the "
            f"smoothing iterations are enough for this geometry.",
            RuntimeWarning, stacklevel=2)
        mesh.update_faces(mask)
        mesh.process()
        trimesh.repair.fill_holes(mesh)

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

        # Measured on the OCI VM, 2026-07-26, and both problems are real:
        #
        #   MEMORY. discrete_mean_curvature_measure runs an rtree ball query per
        #   point and materialises every hit. Asking for all vertices at once
        #   threw std::bad_alloc at 15 GB RSS on a 0.5 mm full-car surface; at
        #   the 0.3 mm production spacing it took a 24 GB box down hard enough
        #   to kill sshd and need a hard reset.
        #
        #   TIME. Even chunked, it ran >20 min per call. This gate runs EVERY
        #   inner-loop iteration, so that alone would cost ~67 h over a
        #   200-iteration run -- more than the CFD it is gating.
        #
        # Local test meshes are 2-3 mm and hit neither, which is why nothing
        # caught this before a real VM.
        #
        # Fix: measure on a DECIMATED working copy, then map each original
        # vertex to its nearest working-copy vertex. This is sound because the
        # gate asks one question -- "is anything sharper than the 3.175 mm tool
        # radius?" -- and a feature that sharp spans many facets at any
        # resolution we produce. Sampling the field at ~1.5 mm cannot miss a
        # 3.175 mm-radius feature, while measuring at 0.3 mm facet density is
        # pure waste. Below WORK_FACES nothing changes: the original mesh is
        # used directly.
        WORK_FACES = 30_000
        work = mesh
        if len(mesh.faces) > WORK_FACES:
            try:
                cand = mesh.simplify_quadric_decimation(face_count=WORK_FACES)
                if cand is not None and len(cand.faces) and len(cand.vertices):
                    work = cand
            except Exception:
                work = mesh   # decimator unavailable: fall back to exact

        CHUNK = 20_000
        wv = work.vertices
        if len(wv) <= CHUNK:
            H_area = tcurv.discrete_mean_curvature_measure(work, wv, ball_radius)
        else:
            H_area = np.empty(len(wv), dtype=np.float64)
            for lo in range(0, len(wv), CHUNK):
                hi = min(lo + CHUNK, len(wv))
                H_area[lo:hi] = tcurv.discrete_mean_curvature_measure(
                    work, wv[lo:hi], ball_radius)

        curvature_abs = np.abs(H_area)
        radii_work = np.where(curvature_abs > 1e-9, 1.0 / curvature_abs, 1e6)
        if work is mesh:
            return radii_work
        # Nearest working-copy vertex carries its radius to each original
        # vertex, so the returned array still indexes mesh.vertices exactly --
        # the caller uses those indices to smooth phi.
        from scipy.spatial import cKDTree
        _d, idx = cKDTree(wv).query(mesh.vertices, k=1)
        return radii_work[idx]
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

    Algorithm: SHADOW RAY. A face is accessible from tool direction d if a ray
    starting just off the face (along its own normal, to avoid running in the
    face's own plane) and travelling OUTWARD along +d escapes without hitting
    the mesh. That is exactly "the cutter has a clear straight path in from d".
    A face is inaccessible only if every allowed direction is blocked.

    Two things this replaces, both measured on the 2026-08-10 car (36,260
    faces, 37,950 mm^2):

      * The old normal-dot gate required dot(n, d) > 0.05, so a face had to
        POINT at a tool axis. With tools only along +-y/+-z, every face with a
        pure +-x normal failed that for all four directions and was reported
        unmachinable -- 2,919 faces / 3,330 mm^2, and 100% of the flagged area
        was exactly those x-normal walls. That is a property of the tool set,
        not of the shape: a 3-axis cutter coming down +z machines a vertical
        wall with the side of the tool. The gate was condemning every car for
        having a nose and a tail.

      * The old ray test never ran. intersects_id with return_locations=False
        returns TWO arrays, the call unpacked THREE, and the resulting
        ValueError was caught by a bare `except Exception` whose handler marked
        every candidate accessible. So no occlusion was ever checked; the
        reported area was purely the normal test above. The except no longer
        pretends success -- an unusable ray engine raises.

    MEASURE THIS ON THE FULL CAR, NOT A HALF. With no material at y<0, a
    shadow ray from an undercut face escapes through the missing half and the
    face reads as reachable. Measured on the same geometry at 1.5 mm: the half
    mesh reports 166 mm^2 blocked, the full car 4,762 mm^2 -- a 29x
    under-report, and it is the full-car figure that is stable under
    refinement (4,762 at 1.5 mm, 4,807 at 1.0 mm, 4,942 live at 0.5 mm).
    extract_unified_surface already does the right thing; this note is so the
    next person measuring by hand does not repeat the mistake.

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

    # Lift the ray origin off the surface so it neither self-intersects nor
    # runs inside the face's own plane (the degenerate case for a wall that is
    # parallel to the tool axis, which is the case that matters most here).
    lift = GRID_SPACING_M * 2.0

    for d in directions:
        d_arr = np.array(d, dtype=float)
        todo = np.where(~accessible)[0]
        if len(todo) == 0:
            break

        origins = face_centres[todo] + face_normals[todo] * lift
        ray_dirs = np.tile(d_arr, (len(todo), 1))
        blocked = mesh.ray.intersects_any(ray_origins=origins,
                                          ray_directions=ray_dirs)
        accessible[todo[~np.asarray(blocked, dtype=bool)]] = True

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


def _retry_triangle_quality(mesh):
    """Candidate meshes to try when a triangle-quality gate fails, best first.

    SMOOTHING BEFORE DECIMATION. Both gates below used to attempt only
    `simplify_quadric_decimation(percent=0.9)`, which is the wrong tool by this
    module's own finding -- see the 2026-07-15 note in _repair_mesh: "quadric
    decimation does not fix this (it targets face count, not angle quality, and
    empirically made angles worse on a test sphere)". The same note records what
    does work: Taubin smoothing took min angle from ~2.6 deg to 16-20+ deg at
    under 0.5% volume change, and it is volume-preserving, so it does not shrink
    the car the way plain Laplacian does.

    Found because a carved geometry failed with "Min angle 0.2 deg < 10.0 deg
    after simplification" -- the gate had decimated a mesh whose angles
    decimation cannot repair, then rejected it. Decimation is kept as a
    fallback: it is the right move when the problem is genuinely too many
    triangles rather than badly shaped ones.

    Returns a list of (label, mesh) candidates; each is a copy, so a rejected
    attempt cannot mutate the input.
    """
    out = []
    try:
        import trimesh.smoothing as _sm
        for iters in (10, 30):
            cand = mesh.copy()
            _sm.filter_taubin(cand, iterations=iters)
            out.append((f"taubin x{iters}", cand))
    except Exception:
        pass
    try:
        out.append(("quadric 0.9", mesh.simplify_quadric_decimation(percent=0.9)))
    except Exception:
        pass
    return out


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


# Measured envelope for accepting a mesh snappyHexMesh has to swallow. Module
# level because two places need it: _check_mesh_quality below, and Part 3's
# _decimate_for_cfd, which hands OpenFOAM a DIFFERENT mesh (decimated) that
# never passes through this gate. Duplicating the numbers there is how they
# drift apart. See _check_mesh_quality for the solver runs these come from.
MEASURED_SAFE_MIN_ANGLE_DEG: float = 8.6
MEASURED_SAFE_SLIVER_FRACTION: float = 1.0e-4    # 3/62,500 is 4.8e-5


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

    # MEASURED ENVELOPE for the fall-through below. A handful of sliver
    # triangles is not a mesh snappyHexMesh cannot handle -- that was an
    # assumption, and it was wrong.
    #
    # This gate is a hard MINIMUM over every triangle, so one bad triangle out of
    # sixty thousand rejected the whole car. Carving a real car down and meshing
    # the rejected STLs on the actual solver (openfoam2412, coarse background
    # mesh, the production case builder) measured:
    #
    #     41.6 g   min angle 8.65 deg   1 sliver / 59,308 tris -> Mesh OK, 0 illegal faces
    #     43.4 g   min angle 9.45 deg   3 slivers / 62,500     -> Mesh OK, 0 illegal faces
    #     29.6 g   min angle 14.19 deg  0 slivers / 35,004     -> Mesh OK, 0 illegal faces
    #
    # snappyHexMesh reported "Detected 0 illegal faces" on every one. Worse, the
    # min angle does not degrade as the car carves -- it OSCILLATES, because a
    # sliver is a transient artefact of where the isosurface happens to cut a
    # cell. Down the same ladder it read 9.45, 8.65, 10.49, 9.84, 11.73, 15.05,
    # 12.98, 13.70, 13.23, 14.19 deg while the geometry stayed watertight and
    # single-body throughout. So the gate was halting the optimiser at a random
    # iteration where one triangle happened to be thin, with a perfectly good
    # geometry on either side of it, and costing ~15 g of carving headroom.
    #
    # The repair is still ATTEMPTED first -- Taubin genuinely improves these
    # (9.06 -> 11.28 deg on one measured step), and skipping it to accept a
    # worse mesh would be a regression. Only when every repair fails does the
    # measured envelope decide between warn-and-continue and raise. Outside the
    # envelope it still raises, because nothing has been measured out there: if
    # you want to go lower, mesh a sample first and move these numbers with the
    # evidence rather than assuming, as the 10 deg did.
    _MEASURED_SAFE_MIN_ANGLE_DEG = MEASURED_SAFE_MIN_ANGLE_DEG
    _MEASURED_SAFE_SLIVER_FRACTION = MEASURED_SAFE_SLIVER_FRACTION

    if min_angle_deg is not None and min_angle_deg < MESH_MIN_TRIANGLE_ANGLE_DEG:
        # Seeded with the DO-NOTHING option, so the message cannot claim a
        # repair "reached" a number worse than the mesh it started from.
        # Measured on a car carved to 43.4 g: input 8.7 deg, and every repair
        # candidate came back worse, the best of them 5.6 deg -- which the
        # failure text then reported as what repair "reached", reading as if it
        # had got closer to the 10 deg gate rather than further away.
        best = ("no repair", min_angle_deg, None)
        for label, cand in _retry_triangle_quality(mesh):
            try:
                got = float(np.degrees(trimesh.triangles.angles(cand.triangles).min()))
            except Exception:
                continue
            if best is None or got > best[1]:
                best = (label, got, cand)
            if got >= MESH_MIN_TRIANGLE_ANGLE_DEG:
                mesh.vertices = cand.vertices
                mesh.faces = cand.faces
                break
        else:
            if best[2] is None:
                got = ("no repair improved on it -- every candidate came back "
                       "worse than the unrepaired mesh")
            else:
                got = f"best repair reached {best[1]:.1f}° via {best[0]}"
            # Every repair failed. Inside the measured-safe envelope that is not
            # a reason to throw the car away; outside it, it is.
            _below = float((np.degrees(angles_rad)
                            < MESH_MIN_TRIANGLE_ANGLE_DEG).mean())
            _worst = max(min_angle_deg, best[1])
            if (_worst >= _MEASURED_SAFE_MIN_ANGLE_DEG
                    and _below <= _MEASURED_SAFE_SLIVER_FRACTION):
                warnings.warn(
                    f"{component}: min triangle angle {_worst:.2f}° is below "
                    f"the {MESH_MIN_TRIANGLE_ANGLE_DEG}° gate and {got}, but "
                    f"only {100.0 * _below:.4f}% of triangles are below it and "
                    f"snappyHexMesh was measured to mesh this cleanly at "
                    f"8.65° with this sliver density (0 illegal faces, Mesh "
                    f"OK). Continuing.",
                    RuntimeWarning, stacklevel=2)
            else:
                raise MeshQualityFailure(
                    f"{component}: Triangle quality below snappyHexMesh "
                    f"tolerance. Min angle {min_angle_deg:.1f}° < "
                    f"{MESH_MIN_TRIANGLE_ANGLE_DEG}° across "
                    f"{100.0 * _below:.4f}% of triangles; {got}. Outside the "
                    f"measured-safe envelope (>= {_MEASURED_SAFE_MIN_ANGLE_DEG}° "
                    f"and <= {100.0 * _MEASURED_SAFE_SLIVER_FRACTION:.4f}% "
                    f"slivers), so this is not known to mesh."
                )

    # ── Triangle aspect ratio check ──────────────────────────────────────
    try:
        aspect_ratios = _triangle_aspect_ratios(mesh)
        max_ratio = float(aspect_ratios.max())
    except Exception:
        max_ratio = None

    if max_ratio is not None and max_ratio > MESH_MAX_ASPECT_RATIO:
        best = None
        for label, cand in _retry_triangle_quality(mesh):
            try:
                got = float(_triangle_aspect_ratios(cand).max())
            except Exception:
                continue
            if best is None or got < best[1]:
                best = (label, got, cand)
            if got <= MESH_MAX_ASPECT_RATIO:
                mesh.vertices = cand.vertices
                mesh.faces = cand.faces
                break
        else:
            got = f"{best[1]:.1f} via {best[0]}" if best else "no candidate produced"
            raise MeshQualityFailure(
                f"{component}: Triangle aspect ratio {max_ratio:.1f} > "
                f"{MESH_MAX_ASPECT_RATIO}; best repair reached {got}."
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
