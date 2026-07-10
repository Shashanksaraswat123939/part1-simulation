"""
Tests for phi_grid.py
"""
import sys, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from phi_grid import PhiGrid
from bounding_volumes import BoundingRegion
from geometry_contract import GRID_SPACING_M

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

def _make_simple_grid(nx=20, ny=20, nz=20, attachment_faces=None):
    bv = BoundingRegion("main_body", (0.0, -0.003, 0.0), (nx, ny, nz))
    if attachment_faces:
        solid, air = PhiGrid.build_hard_masks(bv, [], attachment_faces)
    else:
        solid, air = PhiGrid.build_hard_masks(bv, [], [])
    return PhiGrid("main_body", bv, np.zeros((nx,ny,nz), dtype=np.float32), solid, air)

def test_init_sphere_produces_negative_inside():
    phi = _make_simple_grid()
    phi.init("sphere")
    cx = cy = cz = 10  # centre
    assert phi.grid[cx, cy, cz] < 0, f"Centre should be solid (phi<0), got {phi.grid[cx,cy,cz]}"
    _pass("test_init_sphere_produces_negative_inside")

def test_init_sphere_produces_positive_outside():
    phi = _make_simple_grid()
    phi.init("sphere")
    # Corner should be outside sphere
    assert phi.grid[0, 0, 0] > 0, f"Corner should be air (phi>0), got {phi.grid[0,0,0]}"
    _pass("test_init_sphere_produces_positive_outside")

def test_init_slab():
    phi = _make_simple_grid()
    phi.init("slab")
    assert phi.grid[10, 10, 5] < 0, "Lower half should be solid"
    assert phi.grid[10, 10, 15] > 0, "Upper half should be air"
    _pass("test_init_slab")

def test_init_random_has_noise():
    phi = _make_simple_grid()
    phi.init("random", seed=42)
    # Random should differ from pure sphere
    phi_sphere = _make_simple_grid()
    phi_sphere.init("sphere")
    assert not np.allclose(phi.grid, phi_sphere.grid), "Random init should differ from sphere"
    _pass("test_init_random_has_noise")

def test_hard_constraints_enforced_after_init():
    phi = _make_simple_grid(attachment_faces=["front"])
    phi.init("sphere")
    # Front strip should be solid (interior, away from bbox walls)
    assert np.all(phi.grid[1:4, 5:15, 5:15] < 0), "Front attachment strip interior should be phi < 0"
    _pass("test_hard_constraints_enforced_after_init")

def test_build_hard_masks_basic():
    bv = BoundingRegion("main_body", (0.0, -0.003, 0.0), (20, 20, 20))
    solid, air = PhiGrid.build_hard_masks(bv, [], ["front"])
    assert solid.shape == (20, 20, 20)
    assert air.shape == (20, 20, 20)
    assert solid[:4, 1:-1, 1:-1].all(), "Front 4 cells should be solid (interior)"
    assert solid[4:, :, :].sum() == 0, "Rest should not be solid"
    # Bbox walls should be air EXCEPT where front attachment strip overrides
    # x=0 is front attachment (solid), so air[0] should be False where solid is True
    assert air[-1, :, :].all(), "Rear bbox wall should be air"
    assert air[4:, 0, :].all(), "Y=0 bbox wall should be air (outside front strip)"
    assert air[4:, -1, :].all(), "Y=ny-1 bbox wall should be air (outside front strip)"
    assert air[:, :, 0].all() or air[4:,:,0].all(), "Z=0 bbox wall should be air"
    # Actually z walls don't overlap with front strip in x, so check directly
    assert air[4:, :, 0].all() and air[4:, :, -1].all(), "Z bbox walls should be air outside front strip"
    # x=0 should NOT be air (it's solid via front attachment)
    assert not air[0, :, :].any() or (air[0, :, :] & ~solid[0, :, :]).any(), \
        "x=0 should not have air where solid is"
    _pass("test_build_hard_masks_basic")

def test_build_hard_masks_overlap_resolved():
    bv = BoundingRegion("main_body", (0.0, -0.003, 0.0), (20, 20, 20))
    # void mask that overlaps with front attachment strip
    void = np.zeros((20, 20, 20), dtype=bool)
    void[0, :, :] = True  # overlaps with both front strip and bbox wall
    # Should not raise --- void masks take priority, solid is removed from overlap
    solid, air = PhiGrid.build_hard_masks(bv, [void], ["front"])
    overlap = solid & air
    assert overlap.sum() == 0, f"No cells should be in both masks, got {overlap.sum()}"
    # The void at x=0 should be air (void takes priority over attachment)
    assert air[0, :, :].all(), "Void mask should make x=0 air"
    _pass("test_build_hard_masks_overlap_resolved")

def test_save_and_load_roundtrip():
    phi = _make_simple_grid()
    phi.init("sphere")
    with tempfile.TemporaryDirectory() as d:
        path = phi.save("test_cand", d)
        assert os.path.exists(path), f"File not created at {path}"
        phi2 = _make_simple_grid()
        phi2.load(path)
        assert np.allclose(phi.grid, phi2.grid, atol=1e-6), "Grids should match after save/load"
    _pass("test_save_and_load_roundtrip")

def test_load_wrong_shape_raises():
    phi = _make_simple_grid(20, 20, 20)
    phi.init("sphere")
    with tempfile.TemporaryDirectory() as d:
        path = phi.save("test_cand", d)
        phi2 = _make_simple_grid(30, 20, 20)  # different shape
        try:
            phi2.load(path)
            _fail("test_load_wrong_shape_raises", "should have raised")
        except ValueError:
            _pass("test_load_wrong_shape_raises")

def test_grid_dtype_is_float32():
    phi = _make_simple_grid()
    assert phi.grid.dtype == np.float32
    _pass("test_grid_dtype_is_float32")

def test_post_init_shape_mismatch_raises():
    bv = BoundingRegion("main_body", (0.0, -0.003, 0.0), (20, 20, 20))
    try:
        PhiGrid("main_body", bv, np.zeros((10,10,10), dtype=np.float32),
                np.zeros((20,20,20), dtype=bool), np.zeros((20,20,20), dtype=bool))
        _fail("test_post_init_shape_mismatch_raises", "should have raised")
    except ValueError:
        _pass("test_post_init_shape_mismatch_raises")

if __name__ == "__main__":
    test_init_sphere_produces_negative_inside()
    test_init_sphere_produces_positive_outside()
    test_init_slab()
    test_init_random_has_noise()
    test_hard_constraints_enforced_after_init()
    test_build_hard_masks_basic()
    test_build_hard_masks_overlap_resolved()
    test_save_and_load_roundtrip()
    test_load_wrong_shape_raises()
    test_grid_dtype_is_float32()
    test_post_init_shape_mismatch_raises()
    print("\nAll phi_grid tests passed.")