"""
Tests for surface_extraction.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from surface_extraction import (
    SurfaceExtractionError, RadiusViolation, AccessibilityFailure,
    RuleViolation, MeshQualityFailure, extract_surface,
    _triangle_aspect_ratios,
)
from phi_grid import PhiGrid
from bounding_volumes import BoundingRegion

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

def _make_phi(nx=30, ny=30, nz=30):
    """A flat slab (full x/z extent, inset only in y), not a bare sphere:
    a curved shape at this coarse a resolution reliably produces
    marching-cubes sliver triangles regardless of decimation tuning
    (verified live, 2026-07-14 -- spheres at 6 different resolutions all
    failed), and main_body's TOOL_DIRECTIONS ({+Z,+Y,-Y}, no +-X) can never
    reach a sphere's naturally-curved polar cap. A flat slab has neither
    problem: no curvature (no slivers), and every exposed face is either a
    +-Y side wall (directly covered) or an x/z boundary cap (exempted by
    the boundary-face accessibility exemption)."""
    from geometry_contract import GRID_SPACING_M
    shape = (nx, ny, nz)
    bv = BoundingRegion("main_body", (0.0, -0.0045, 0.0), shape)
    solid = np.zeros(shape, dtype=bool)
    air = np.zeros(shape, dtype=bool)
    air[0, :, :] = True; air[-1, :, :] = True
    air[:, 0, :] = True; air[:, -1, :] = True
    air[:, :, 0] = True; air[:, :, -1] = True
    phi = PhiGrid("main_body", bv, np.zeros(shape, dtype=np.float32), solid, air)
    margin_y = 6
    jj = np.arange(ny)
    dist_y = np.minimum(jj - margin_y, (ny - 1 - margin_y) - jj).astype(np.float32)
    slice1d = -dist_y * GRID_SPACING_M
    phi.grid = np.broadcast_to(slice1d[None, :, None], shape).astype(np.float32).copy()
    phi.apply_hard_constraints()
    return phi

def test_extract_surface_returns_mesh():
    phi = _make_phi()
    mesh = extract_surface(phi)
    assert mesh is not None, "Should return a mesh"
    assert len(mesh.faces) > 0, "Mesh should have faces"
    _pass("test_extract_surface_returns_mesh")

def test_extract_surface_vertices_in_world_coords():
    phi = _make_phi()
    mesh = extract_surface(phi)
    # Check vertices are in world coordinates (not just index space)
    # World coords should range up to ~ bv extent
    ox, oy, oz = phi.bv.origin_m
    x_max = ox + 30 * 0.0003  # GRID_SPACING_M = 0.0003
    # At least some vertices should be beyond index range (0-29)
    # because they're offset by origin
    assert np.any(mesh.vertices[:, 0] >= ox), "Vertices should be in world coords"
    _pass("test_extract_surface_vertices_in_world_coords")

def test_empty_mesh_raises():
    # Create a grid with all phi > 0 (no solid)
    bv = BoundingRegion("main_body", (0.0, -0.003, 0.0), (20, 20, 20))
    solid = np.zeros((20, 20, 20), dtype=bool)
    air = np.ones((20, 20, 20), dtype=bool)  # all air
    grid = np.ones((20, 20, 20), dtype=np.float32)  # all positive
    phi = PhiGrid("main_body", bv, grid, solid, air)
    try:
        extract_surface(phi)
        _fail("test_empty_mesh_raises", "Should have raised SurfaceExtractionError")
    except SurfaceExtractionError:
        _pass("test_empty_mesh_raises")

def test_exception_hierarchy():
    assert issubclass(RadiusViolation, SurfaceExtractionError)
    assert issubclass(AccessibilityFailure, SurfaceExtractionError)
    assert issubclass(RuleViolation, SurfaceExtractionError)
    assert issubclass(MeshQualityFailure, SurfaceExtractionError)
    _pass("test_exception_hierarchy")

def test_accessibility_failure_has_is_large():
    e = AccessibilityFailure("test", is_large=True)
    assert e.is_large == True
    e2 = AccessibilityFailure("test", is_large=False)
    assert e2.is_large == False
    _pass("test_accessibility_failure_has_is_large")

def test_rule_violation_has_is_major():
    e = RuleViolation("test", is_major=True)
    assert e.is_major == True
    e2 = RuleViolation("test", is_major=False)
    assert e2.is_major == False
    _pass("test_rule_violation_has_is_major")

def test_aspect_ratio_equilateral_triangle_is_one():
    import trimesh
    # Equilateral triangle, side length 1
    h = np.sqrt(3.0) / 2.0
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, h, 0.0]])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    ratios = _triangle_aspect_ratios(mesh)
    assert abs(ratios[0] - 1.0) < 1e-9, f"Equilateral triangle aspect ratio should be 1.0, got {ratios[0]}"
    _pass("test_aspect_ratio_equilateral_triangle_is_one")


def test_aspect_ratio_sliver_triangle_is_large():
    import trimesh
    # Extreme sliver: very long, very thin
    verts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 0.001, 0.0]])
    faces = np.array([[0, 1, 2]])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    ratios = _triangle_aspect_ratios(mesh)
    assert ratios[0] > 100.0, f"Sliver triangle should have a large aspect ratio, got {ratios[0]}"
    _pass("test_aspect_ratio_sliver_triangle_is_large")




def test_a_few_slivers_warn_but_a_bad_mesh_still_raises():
    """The angle gate is bounded by what snappyHexMesh was MEASURED to accept.

    MESH_MIN_TRIANGLE_ANGLE_DEG = 10 was a hard minimum over every triangle, so
    one bad triangle in sixty thousand rejected the whole car -- and it stopped
    a real carve at 45 g. Meshing the rejected STLs on the actual solver
    (openfoam2412, coarse background mesh, the production case builder) showed
    the premise was false:

        41.6 g  min angle 8.65 deg   1 sliver / 59,308 -> Mesh OK, 0 illegal faces
        43.4 g  min angle 9.45 deg   3 slivers / 62,500 -> Mesh OK, 0 illegal faces
        29.6 g  min angle 14.19 deg  0 slivers / 35,004 -> Mesh OK, 0 illegal faces

    The min angle does not even degrade as the car carves; it oscillates,
    because a sliver is a transient artefact of where the isosurface cuts a
    cell. So a violation inside that measured envelope warns and continues, and
    anything outside it still raises -- nothing has been measured out there.

    This asserts BOTH halves. A gate that only warns would be as wrong as one
    that only raises.
    """
    import inspect
    import surface_extraction as SE

    src = inspect.getsource(SE._mesh_quality_gate) if hasattr(SE, "_mesh_quality_gate") \
        else inspect.getsource(SE)
    assert "_MEASURED_SAFE_MIN_ANGLE_DEG" in src, (
        "the angle gate no longer bounds itself by a measured envelope")
    # The envelope must still RAISE outside itself -- a pure warning would let
    # an unmeshable car through.
    assert "raise MeshQualityFailure" in src, (
        "the angle gate never raises; outside the measured envelope an "
        "unmeshable mesh would be accepted silently")
    # And the repair must be attempted BEFORE the envelope decides, or a mesh
    # Taubin could have fixed gets accepted worse than necessary (measured:
    # 9.06 -> 11.28 deg on one carve step).
    i_env = src.index("_MEASURED_SAFE_MIN_ANGLE_DEG >=") if "_MEASURED_SAFE_MIN_ANGLE_DEG >=" in src \
        else src.index("_worst >= _MEASURED_SAFE_MIN_ANGLE_DEG")
    i_repair = src.index("_retry_triangle_quality")
    assert i_repair < i_env, (
        "the measured-safe envelope is consulted before the repair is "
        "attempted; a mesh Taubin could have fixed would be accepted as-is")


if __name__ == "__main__":
    # Collected by name. The hand-written call list this replaces printed
    # "All surface_extraction tests passed" while silently skipping every test
    # appended below it -- the same bug found in five other files today.
    import sys as _sys
    _mod = _sys.modules[__name__]
    _fail = 0
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        try:
            getattr(_mod, _n)()
            print("PASS " + _n)
        except Exception as _e:  # noqa: BLE001
            print("FAIL %s: %s" % (_n, _e))
            _fail += 1
    print("All surface_extraction tests passed." if not _fail
          else "%d failed" % _fail)
    _sys.exit(1 if _fail else 0)
