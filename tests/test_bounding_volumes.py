"""
Tests for bounding_volumes.py.
Use a stub RuleEnvelope with made-up but physically reasonable dimensions.
Once U6 is resolved, update the stub values.
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bounding_volumes import (
    BoundingRegion, BoundingVolumes, RuleEnvelope, compute_bounding_volumes,
    _point_in_polygon_vectorised,
)
from fixed_hardware import ForbiddenCylinder
from geometry_contract import R_WHEEL_M, WHEEL_CLEARANCE_M, mm_to_m
import numpy as np

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

# Stub rule envelope with physically plausible dimensions
STUB_RE = RuleEnvelope(
    y_body_half_m       = 0.030,   # 30 mm half-width
    y_sidepod_inner_m   = 0.030,
    y_sidepod_outer_m   = 0.060,
    z_floor_m           = 0.000,
    z_nose_top_m        = 0.040,
    z_sidepod_top_m     = 0.035,
    z_rearpod_top_m     = 0.035,
    z_body_top_m        = 0.045,
    rearpod_max_length_m= 0.030,
)

def _make_cylinders(W_mm):
    W_m = mm_to_m(W_mm)
    r = R_WHEEL_M + WHEEL_CLEARANCE_M
    front = ForbiddenCylinder(0.0, 0.0, 0.015, r, 0.008)
    rear  = ForbiddenCylinder(W_m, 0.0, 0.015, r, 0.008)
    return front, rear

def test_sidepod_length_decreases_with_W():
    bv120 = compute_bounding_volumes(120.0, 10.0, *_make_cylinders(120.0), STUB_RE)
    bv140 = compute_bounding_volumes(140.0, 10.0, *_make_cylinders(140.0), STUB_RE)
    assert bv140.sidepod_length_m > bv120.sidepod_length_m, (
        f"Sidepod should be longer at W=140 than W=120. "
        f"Got {bv140.sidepod_length_m:.4f} vs {bv120.sidepod_length_m:.4f}"
    )
    _pass("test_sidepod_length_decreases_with_W")

def test_sidepod_length_positive_at_W_min():
    bv = compute_bounding_volumes(120.0, 10.0, *_make_cylinders(120.0), STUB_RE)
    assert bv.sidepod_length_m > 0, f"sidepod_length={bv.sidepod_length_m:.4f}"
    _pass("test_sidepod_length_positive_at_W_min")

def test_nose_length_matches_d_halo():
    from geometry_contract import GRID_SPACING_M
    d_halo_mm = 20.0
    bv = compute_bounding_volumes(130.0, d_halo_mm, *_make_cylinders(130.0), STUB_RE)
    nose_length_m = bv.nose.nx * GRID_SPACING_M
    assert abs(nose_length_m - mm_to_m(d_halo_mm)) <= GRID_SPACING_M, (
        f"Nose length {nose_length_m*1000:.2f} mm should ? d_halo={d_halo_mm} mm"
    )
    _pass("test_nose_length_matches_d_halo")

def test_d_halo_zero_gives_degenerate_nose():
    bv = compute_bounding_volumes(130.0, 0.0, *_make_cylinders(130.0), STUB_RE)
    assert bv.nose.nx == 1, f"nx={bv.nose.nx}, expected 1 for d_halo=0"
    _pass("test_d_halo_zero_gives_degenerate_nose")

def test_W_out_of_range_raises():
    for bad_W in [119.9, 140.1, 0.0]:
        try:
            compute_bounding_volumes(bad_W, 10.0, *_make_cylinders(130.0), STUB_RE)
            _fail("test_W_out_of_range_raises", f"W={bad_W} should raise")
        except ValueError:
            pass
    _pass("test_W_out_of_range_raises")

def test_d_halo_out_of_range_raises():
    try:
        compute_bounding_volumes(130.0, 147.0, *_make_cylinders(130.0), STUB_RE)
        _fail("test_d_halo_out_of_range_raises", "d_halo=147 > W+16=146 should raise")
    except ValueError:
        _pass("test_d_halo_out_of_range_raises")

def test_all_shapes_are_positive_ints():
    bv = compute_bounding_volumes(130.0, 15.0, *_make_cylinders(130.0), STUB_RE)
    for comp in ("nose", "sidepod", "rearpod", "main_body"):
        region = bv.get(comp)
        for dim in region.shape:
            assert isinstance(dim, int) and dim > 0, f"{comp} dim={dim}"
    _pass("test_all_shapes_are_positive_ints")

def test_rearpod_origin_x_equals_W():
    bv = compute_bounding_volumes(130.0, 15.0, *_make_cylinders(130.0), STUB_RE)
    assert abs(bv.rearpod.origin_m[0] - mm_to_m(130.0)) < 1e-9
    _pass("test_rearpod_origin_x_equals_W")

def test_sidepod_is_right_half_only():
    bv = compute_bounding_volumes(130.0, 15.0, *_make_cylinders(130.0), STUB_RE)
    # Sidepod origin y must be >= 0 (right half)
    assert bv.sidepod.origin_m[1] >= 0.0, (
        f"Sidepod y origin={bv.sidepod.origin_m[1]:.4f} --- should be >= 0 (right half only)"
    )
    _pass("test_sidepod_is_right_half_only")

def test_polygon_point_in_polygon():
    # Square polygon: (0,0),(1,0),(1,1),(0,1)
    poly = [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)]
    ys = np.array([0.5, 1.5, -0.1, 0.5])
    zs = np.array([0.5, 0.5,  0.5, 1.5])
    inside = _point_in_polygon_vectorised(ys, zs, poly)
    assert inside[0] == True,  "Centre point should be inside"
    assert inside[1] == False, "Outside y should be outside"
    assert inside[2] == False, "Outside z should be outside"
    assert inside[3] == False, "Outside z should be outside"
    _pass("test_polygon_point_in_polygon")

def test_bounding_region_box_mode_all_valid():
    region = BoundingRegion("nose", (0.0, -0.015, 0.0), (10, 10, 10))
    mask = region.valid_mask()
    assert mask.shape == (10, 10, 10)
    assert mask.all(), "Box mode: all cells should be valid"
    _pass("test_bounding_region_box_mode_all_valid")

def test_bounding_region_polygon_mode():
    # Triangular cross-section in y-z: (0,0),(0.01,0),(0.005,0.01)
    poly = [(0.0,0.0),(0.01,0.0),(0.005,0.01)]
    from geometry_contract import GRID_SPACING_M
    region = BoundingRegion(
        component="sidepod",
        origin_m=(0.0, 0.0, 0.0),
        shape=(5, 50, 50),
        polygon_yz_m=poly,
    )
    mask = region.valid_mask()
    assert mask.shape == (5, 50, 50)
    # Centre of triangle in y-z should be inside
    assert mask.any(), "Some cells should be inside the triangle"
    _pass("test_bounding_region_polygon_mode")

def test_bounding_region_voxel_mode():
    vox = np.zeros((5, 10, 10), dtype=bool)
    vox[2, 5, 5] = True
    region = BoundingRegion("main_body", (0.0,0.0,0.0), (5,10,10), voxel_mask=vox)
    mask = region.valid_mask()
    assert mask[2, 5, 5] == True
    assert mask[0, 0, 0] == False
    _pass("test_bounding_region_voxel_mode")

if __name__ == "__main__":
    test_sidepod_length_decreases_with_W()
    test_sidepod_length_positive_at_W_min()
    test_nose_length_matches_d_halo()
    test_d_halo_zero_gives_degenerate_nose()
    test_W_out_of_range_raises()
    test_d_halo_out_of_range_raises()
    test_all_shapes_are_positive_ints()
    test_rearpod_origin_x_equals_W()
    test_sidepod_is_right_half_only()
    test_polygon_point_in_polygon()
    test_bounding_region_box_mode_all_valid()
    test_bounding_region_polygon_mode()
    test_bounding_region_voxel_mode()
    print("\nAll bounding_volumes tests passed.")