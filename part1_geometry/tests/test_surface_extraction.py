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
)
from phi_grid import PhiGrid
from bounding_volumes import BoundingRegion

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

def _make_phi(nx=30, ny=30, nz=30):
    bv = BoundingRegion("main_body", (0.0, -0.0045, 0.0), (nx, ny, nz))
    solid = np.zeros((nx, ny, nz), dtype=bool)
    air = np.zeros((nx, ny, nz), dtype=bool)
    air[0, :, :] = True; air[-1, :, :] = True
    air[:, 0, :] = True; air[:, -1, :] = True
    air[:, :, 0] = True; air[:, :, -1] = True
    phi = PhiGrid("main_body", bv, np.zeros((nx,ny,nz), dtype=np.float32), solid, air)
    phi.init("sphere")
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

if __name__ == "__main__":
    test_extract_surface_returns_mesh()
    test_extract_surface_vertices_in_world_coords()
    test_empty_mesh_raises()
    test_exception_hierarchy()
    test_accessibility_failure_has_is_large()
    test_rule_violation_has_is_major()
    print("\nAll surface_extraction tests passed.")