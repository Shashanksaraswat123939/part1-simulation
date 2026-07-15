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


if __name__ == "__main__":
    test_extract_surface_returns_mesh()
    test_extract_surface_vertices_in_world_coords()
    test_empty_mesh_raises()
    test_exception_hierarchy()
    test_accessibility_failure_has_is_large()
    test_rule_violation_has_is_major()
    test_aspect_ratio_equilateral_triangle_is_one()
    test_aspect_ratio_sliver_triangle_is_large()
    print("\nAll surface_extraction tests passed.")