"""
Tests for bounding_volumes.py.
Use a stub RuleEnvelope with made-up but physically reasonable dimensions.
Once U6 is resolved, update the stub values.

Coordinate system: x=0 at nose tip.
front axle at x = x_front_m, rear axle at x = x_front_m + W_m.
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bounding_volumes import (
    BoundingRegion, BoundingVolumes, RuleEnvelope, compute_bounding_volumes,
    _point_in_polygon_vectorised, default_rule_envelope,
)
from geometry_contract import R_WHEEL_M, WHEEL_CLEARANCE_M, mm_to_m
import numpy as np
from dataclasses import dataclass

# Minimal ForbiddenCylinder stub — mirrors fixed_hardware.ForbiddenCylinder interface.
# Defined here so test_bounding_volumes has no dependency on Part 2 (mass_com_ingest).
@dataclass(frozen=True)
class ForbiddenCylinder:
    x_center_m:    float
    y_center_m:    float
    z_center_m:    float
    radius_m:      float
    x_half_width_m: float

    @property
    def x_min_m(self) -> float: return self.x_center_m - self.x_half_width_m
    @property
    def x_max_m(self) -> float: return self.x_center_m + self.x_half_width_m

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

# Real confirmed/derived RuleEnvelope -- see default_rule_envelope() for
# which fields are exact regulation numbers vs. design choices.
STUB_RE = default_rule_envelope()

# Default x_front for tests that don't exercise it specifically.
# 64 mm is valid for all W in [120, 140]: range is [61, 207-W] and 64 ≤ 207-140=67.
DEFAULT_X_FRONT_MM = 64.0

def _make_cylinders(W_mm, x_front_mm=DEFAULT_X_FRONT_MM):
    """Create ForbiddenCylinders in nose-tip coordinate system."""
    W_m = mm_to_m(W_mm)
    x_front_m = mm_to_m(x_front_mm)
    r = R_WHEEL_M + WHEEL_CLEARANCE_M
    front = ForbiddenCylinder(x_front_m,        0.0, 0.015, r, 0.008)
    rear  = ForbiddenCylinder(x_front_m + W_m,  0.0, 0.015, r, 0.008)
    return front, rear

def test_sidepod_length_increases_with_W():
    bv120 = compute_bounding_volumes(120.0, DEFAULT_X_FRONT_MM, 10.0, *_make_cylinders(120.0), STUB_RE)
    bv140 = compute_bounding_volumes(140.0, DEFAULT_X_FRONT_MM, 10.0, *_make_cylinders(140.0), STUB_RE)
    assert bv140.sidepod_length_m > bv120.sidepod_length_m, (
        f"Sidepod should be longer at W=140 than W=120. "
        f"Got {bv140.sidepod_length_m:.4f} vs {bv120.sidepod_length_m:.4f}"
    )
    _pass("test_sidepod_length_increases_with_W")

def test_sidepod_length_positive_at_W_min():
    bv = compute_bounding_volumes(120.0, DEFAULT_X_FRONT_MM, 10.0, *_make_cylinders(120.0), STUB_RE)
    assert bv.sidepod_length_m > 0, f"sidepod_length={bv.sidepod_length_m:.4f}"
    _pass("test_sidepod_length_positive_at_W_min")

def test_nose_length_matches_x_front():
    """Nose extends from x=0 to Ref Plane A = x_front - 16 mm."""
    from geometry_contract import GRID_SPACING_M
    x_front_mm = 75.0
    bv = compute_bounding_volumes(130.0, x_front_mm, 50.0, *_make_cylinders(130.0, x_front_mm), STUB_RE)
    expected_length_m = mm_to_m(x_front_mm - 16.0)   # Ref Plane A from nose tip
    actual_length_m = bv.nose.nx * GRID_SPACING_M
    assert abs(actual_length_m - expected_length_m) <= GRID_SPACING_M, (
        f"Nose length {actual_length_m*1000:.2f} mm should ≈ x_front-16={x_front_mm-16:.1f} mm"
    )
    _pass("test_nose_length_matches_x_front")

def test_nose_origin_at_zero():
    """Nose always starts at x=0 (nose tip) in the new coordinate system."""
    bv = compute_bounding_volumes(130.0, DEFAULT_X_FRONT_MM, 50.0, *_make_cylinders(130.0), STUB_RE)
    assert bv.nose.origin_m[0] == 0.0, f"Nose origin x={bv.nose.origin_m[0]}, expected 0.0"
    _pass("test_nose_origin_at_zero")

def test_W_out_of_range_raises():
    for bad_W in [119.9, 140.1, 0.0]:
        try:
            compute_bounding_volumes(bad_W, DEFAULT_X_FRONT_MM, 10.0, *_make_cylinders(130.0), STUB_RE)
            _fail("test_W_out_of_range_raises", f"W={bad_W} should raise")
        except ValueError:
            pass
    _pass("test_W_out_of_range_raises")

def test_d_halo_out_of_range_raises():
    try:
        compute_bounding_volumes(130.0, DEFAULT_X_FRONT_MM, 100.1, *_make_cylinders(130.0), STUB_RE)
        _fail("test_d_halo_out_of_range_raises", "d_halo=100.1 > min(100,W+16)=100 should raise")
    except ValueError:
        _pass("test_d_halo_out_of_range_raises")

def test_x_front_out_of_range_raises():
    try:
        # x_front=50 < X_FRONT_MIN_MM=61 → should raise
        compute_bounding_volumes(130.0, 50.0, 10.0, *_make_cylinders(130.0, 50.0), STUB_RE)
        _fail("test_x_front_out_of_range_raises", "x_front=50 < 61 should raise")
    except ValueError:
        _pass("test_x_front_out_of_range_raises")

def test_all_shapes_are_positive_ints():
    bv = compute_bounding_volumes(130.0, DEFAULT_X_FRONT_MM, 15.0, *_make_cylinders(130.0), STUB_RE)
    for comp in ("nose", "sidepod", "rearpod", "main_body"):
        region = bv.get(comp)
        for dim in region.shape:
            assert isinstance(dim, int) and dim > 0, f"{comp} dim={dim}"
    _pass("test_all_shapes_are_positive_ints")

def test_rearpod_origin_x_equals_x_front_plus_W():
    """Rearpod starts at rear axle = x_front + W in nose-tip coordinates."""
    x_front_mm = DEFAULT_X_FRONT_MM
    bv = compute_bounding_volumes(130.0, x_front_mm, 15.0, *_make_cylinders(130.0), STUB_RE)
    expected = mm_to_m(x_front_mm + 130.0)
    assert abs(bv.rearpod.origin_m[0] - expected) < 1e-9, (
        f"Rearpod origin x={bv.rearpod.origin_m[0]:.6f}, expected {expected:.6f}"
    )
    _pass("test_rearpod_origin_x_equals_x_front_plus_W")

def test_main_body_origin_x_equals_ref_plane_A():
    """Main body starts at Ref Plane A = x_front - 16 mm."""
    x_front_mm = DEFAULT_X_FRONT_MM
    bv = compute_bounding_volumes(130.0, x_front_mm, 15.0, *_make_cylinders(130.0), STUB_RE)
    expected = mm_to_m(x_front_mm - 16.0)
    assert abs(bv.main_body.origin_m[0] - expected) < 1e-9, (
        f"Main body origin x={bv.main_body.origin_m[0]:.6f}, expected {expected:.6f}"
    )
    _pass("test_main_body_origin_x_equals_ref_plane_A")

def test_sidepod_is_right_half_only():
    bv = compute_bounding_volumes(130.0, DEFAULT_X_FRONT_MM, 15.0, *_make_cylinders(130.0), STUB_RE)
    assert bv.sidepod.origin_m[1] >= 0.0, (
        f"Sidepod y origin={bv.sidepod.origin_m[1]:.4f} --- should be >= 0 (right half only)"
    )
    _pass("test_sidepod_is_right_half_only")

def test_bounding_volumes_stores_x_front():
    bv = compute_bounding_volumes(130.0, DEFAULT_X_FRONT_MM, 15.0, *_make_cylinders(130.0), STUB_RE)
    assert bv.x_front_mm == DEFAULT_X_FRONT_MM
    assert abs(bv.x_front_m - mm_to_m(DEFAULT_X_FRONT_MM)) < 1e-12
    _pass("test_bounding_volumes_stores_x_front")

def test_ref_plane_properties():
    x_front_mm = DEFAULT_X_FRONT_MM
    bv = compute_bounding_volumes(130.0, x_front_mm, 15.0, *_make_cylinders(130.0), STUB_RE)
    assert abs(bv.ref_plane_A_m - mm_to_m(x_front_mm - 16.0)) < 1e-9
    assert abs(bv.ref_plane_B_m - mm_to_m(x_front_mm + 130.0 + 16.0)) < 1e-9
    _pass("test_ref_plane_properties")

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
    test_sidepod_length_increases_with_W()
    test_sidepod_length_positive_at_W_min()
    test_nose_length_matches_x_front()
    test_nose_origin_at_zero()
    test_W_out_of_range_raises()
    test_d_halo_out_of_range_raises()
    test_x_front_out_of_range_raises()
    test_all_shapes_are_positive_ints()
    test_rearpod_origin_x_equals_x_front_plus_W()
    test_main_body_origin_x_equals_ref_plane_A()
    test_sidepod_is_right_half_only()
    test_bounding_volumes_stores_x_front()
    test_ref_plane_properties()
    test_polygon_point_in_polygon()
    test_bounding_region_box_mode_all_valid()
    test_bounding_region_polygon_mode()
    test_bounding_region_voxel_mode()
    print("\nAll bounding_volumes tests passed.")
