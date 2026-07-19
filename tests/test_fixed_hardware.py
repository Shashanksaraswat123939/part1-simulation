"""
Tests for fixed_hardware.py.
These tests use placeholder values to bypass ! UNRESOLVED items where possible,
using the public cylinder and mask builders directly.
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from fixed_hardware import (
    ForbiddenCylinder, _build_cylinder_void_mask, _build_box_void_mask,
    _validate_halo_position, HaloGeometry, _assert_com_in_range,
    WheelDiscZone, _build_wheel_disc_void_mask, _build_four_wheel_zones,
)
from geometry_contract import (
    R_WHEEL_M, WHEEL_CLEARANCE_M, mm_to_m,
    WHEEL_WIDTH_M, FRONT_WHEEL_INNER_Y_M, REAR_WHEEL_INNER_Y_M,
)

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

def test_front_cylinder_x_center_at_x_front():
    """Coordinate system: x=0 at nose tip, front axle at x_front_m."""
    x_front_m = mm_to_m(64.0)
    cyl = ForbiddenCylinder(x_front_m, 0.0, 0.015, R_WHEEL_M+WHEEL_CLEARANCE_M, 0.010)
    assert abs(cyl.x_center_m - x_front_m) < 1e-12
    _pass("test_front_cylinder_x_center_at_x_front")

def test_rear_cylinder_x_center_equals_x_front_plus_W():
    """Rear axle at x_front_m + W_m in nose-tip coordinates."""
    x_front_m = mm_to_m(64.0)
    W_m = mm_to_m(130.0)
    rear_axle_m = x_front_m + W_m
    cyl = ForbiddenCylinder(rear_axle_m, 0.0, 0.015, R_WHEEL_M+WHEEL_CLEARANCE_M, 0.010)
    assert abs(cyl.x_center_m - rear_axle_m) < 1e-12
    _pass("test_rear_cylinder_x_center_equals_x_front_plus_W")

def test_cylinder_contains_point_inside():
    cyl = ForbiddenCylinder(0.0, 0.0, 0.015, 0.020, 0.010)
    assert cyl.contains_point(0.0, 0.0, 0.015)   # centre
    assert cyl.contains_point(0.005, 0.01, 0.015)  # inside radius
    _pass("test_cylinder_contains_point_inside")

def test_cylinder_contains_point_outside():
    cyl = ForbiddenCylinder(0.0, 0.0, 0.015, 0.020, 0.010)
    assert not cyl.contains_point(0.0, 0.05, 0.015)  # outside radius
    assert not cyl.contains_point(0.020, 0.0, 0.015)  # outside x extent
    _pass("test_cylinder_contains_point_outside")

def test_cylinder_void_mask_shape():
    shape = (50, 50, 50)
    origin = (0.0, -0.025, 0.0)
    cyl = ForbiddenCylinder(0.0, 0.0, 0.015, 0.020, 0.010)
    mask = _build_cylinder_void_mask(shape, origin, cyl)
    assert mask.shape == shape
    assert mask.dtype == bool
    _pass("test_cylinder_void_mask_shape")

def test_cylinder_void_mask_centre_is_true():
    # Grid origin at (0,0,0), spacing 0.3mm, centre of cylinder should be masked
    from geometry_contract import GRID_SPACING_M
    shape = (100, 100, 100)
    origin = (-0.015, -0.015, 0.0)
    cyl = ForbiddenCylinder(0.0, 0.0, 0.015, 0.010, 0.005)
    mask = _build_cylinder_void_mask(shape, origin, cyl)
    # Find cell closest to (0, 0, 0.015) --- axle centre
    ci = int(round((0.0 - origin[0]) / GRID_SPACING_M))
    cj = int(round((0.0 - origin[1]) / GRID_SPACING_M))
    ck = int(round((0.015 - origin[2]) / GRID_SPACING_M))
    ci = max(0, min(ci, shape[0]-1))
    cj = max(0, min(cj, shape[1]-1))
    ck = max(0, min(ck, shape[2]-1))
    assert mask[ci, cj, ck], "Axle centre cell should be in void mask"
    _pass("test_cylinder_void_mask_centre_is_true")

def test_wheel_disc_zone_contains_point_at_real_wheel_position():
    """A wheel disc's clearance zone must actually cover the wheel's real
    lateral position (y=19.25-36.5mm for front), not the centreline."""
    r = R_WHEEL_M + WHEEL_CLEARANCE_M
    zone = WheelDiscZone(
        x_center_m=0.070, y_min_m=FRONT_WHEEL_INNER_Y_M,
        y_max_m=FRONT_WHEEL_INNER_Y_M + WHEEL_WIDTH_M,
        z_center_m=R_WHEEL_M, radius_m=r,
    )
    # Point at the wheel's real y-position (inner face + half width), axle x/z.
    y_mid = FRONT_WHEEL_INNER_Y_M + WHEEL_WIDTH_M / 2.0
    assert zone.contains_point(0.070, y_mid, R_WHEEL_M)
    _pass("test_wheel_disc_zone_contains_point_at_real_wheel_position")

def test_wheel_disc_zone_excludes_centreline():
    """The old bug centred the exclusion zone at y=0; a real wheel does not
    reach the centreline at all, so y=0 must be OUTSIDE the zone."""
    r = R_WHEEL_M + WHEEL_CLEARANCE_M
    zone = WheelDiscZone(
        x_center_m=0.070, y_min_m=FRONT_WHEEL_INNER_Y_M,
        y_max_m=FRONT_WHEEL_INNER_Y_M + WHEEL_WIDTH_M,
        z_center_m=R_WHEEL_M, radius_m=r,
    )
    assert not zone.contains_point(0.070, 0.0, R_WHEEL_M)
    _pass("test_wheel_disc_zone_excludes_centreline")

def test_wheel_disc_zone_excludes_outside_x_z_circle():
    r = R_WHEEL_M + WHEEL_CLEARANCE_M
    zone = WheelDiscZone(
        x_center_m=0.070, y_min_m=FRONT_WHEEL_INNER_Y_M,
        y_max_m=FRONT_WHEEL_INNER_Y_M + WHEEL_WIDTH_M,
        z_center_m=R_WHEEL_M, radius_m=r,
    )
    y_mid = FRONT_WHEEL_INNER_Y_M + WHEEL_WIDTH_M / 2.0
    assert not zone.contains_point(0.070 + r + 0.005, y_mid, R_WHEEL_M)  # past disc edge in x
    _pass("test_wheel_disc_zone_excludes_outside_x_z_circle")

def test_wheel_disc_void_mask_shape():
    r = R_WHEEL_M + WHEEL_CLEARANCE_M
    zone = WheelDiscZone(0.070, FRONT_WHEEL_INNER_Y_M, FRONT_WHEEL_INNER_Y_M + WHEEL_WIDTH_M, R_WHEEL_M, r)
    shape = (400, 300, 100)
    origin = (0.0, 0.0, 0.0)
    mask = _build_wheel_disc_void_mask(shape, origin, zone)
    assert mask.shape == shape
    assert mask.dtype == bool
    assert mask.any(), "Wheel disc void mask should carve out some cells"
    _pass("test_wheel_disc_void_mask_shape")

def test_four_wheel_zones_left_right_symmetric():
    """front-left and front-right zones must be mirror images about y=0,
    and neither may include the centreline (regression for the y_center=0 bug)."""
    r = R_WHEEL_M + WHEEL_CLEARANCE_M
    zones = _build_four_wheel_zones(x_front_m=0.070, rear_axle_m=0.200, axle_z_m=R_WHEEL_M, radius_m=r)
    assert len(zones) == 4
    # _build_four_wheel_zones iterates sign in (+1.0, -1.0) per axle, i.e.
    # [front_right, front_left, rear_right, rear_left].
    front_right, front_left, rear_right, rear_left = zones
    assert front_right.y_min_m > 0.0 and front_left.y_max_m < 0.0
    assert abs(front_right.y_min_m - (-front_left.y_max_m)) < 1e-12
    assert abs(front_right.y_max_m - (-front_left.y_min_m)) < 1e-12
    # Neither wheel reaches the centreline.
    for z in zones:
        assert not (z.y_min_m <= 0.0 <= z.y_max_m)
    _pass("test_four_wheel_zones_left_right_symmetric")

def test_box_void_mask_shape():
    shape = (50, 50, 50)
    origin = (0.0, -0.025, 0.0)
    mask = _build_box_void_mask(shape, origin, (0.01, 0.02), (-0.005, 0.005), (0.005, 0.015))
    assert mask.shape == shape
    assert mask.dtype == bool
    _pass("test_box_void_mask_shape")

def test_halo_validation_behind_front_axle():
    # Coordinate system: x=0 at nose tip. front_axle_m=0.064, rear_axle_m=0.194.
    # Valid: halo x_front > front_axle_m and > canister_x
    halo = HaloGeometry(x_front_m=0.070, x_rear_m=0.100)
    _validate_halo_position(halo, canister_x_m=0.020, front_axle_m=0.064, rear_axle_m=0.194)
    _pass("test_halo_validation_behind_front_axle")

def test_halo_validation_allows_at_or_before_front_axle():
    """The real regs have no rule tying halo x-position to the front axle
    (the earlier H2 assumption was removed -- see fixed_hardware.py). A halo
    starting at or even before the front axle (small d_halo) must be allowed."""
    halo = HaloGeometry(x_front_m=0.064, x_rear_m=0.100)
    _validate_halo_position(halo, canister_x_m=0.020, front_axle_m=0.064, rear_axle_m=0.194)
    halo_before = HaloGeometry(x_front_m=0.050, x_rear_m=0.086)
    _validate_halo_position(halo_before, canister_x_m=0.020, front_axle_m=0.064, rear_axle_m=0.194)
    _pass("test_halo_validation_allows_at_or_before_front_axle")

def test_halo_validation_allows_before_canister():
    """No real rule ties halo position to the canister either (H3a removed)."""
    halo = HaloGeometry(x_front_m=0.068, x_rear_m=0.100)
    _validate_halo_position(halo, canister_x_m=0.070, front_axle_m=0.064, rear_axle_m=0.194)
    _pass("test_halo_validation_allows_before_canister")

def test_halo_validation_fails_if_past_rear_axle():
    halo = HaloGeometry(x_front_m=0.070, x_rear_m=0.195)
    try:
        _validate_halo_position(halo, canister_x_m=0.020, front_axle_m=0.064, rear_axle_m=0.194)
        _fail("test_halo_validation_fails_if_past_rear_axle", "should have raised")
    except ValueError:
        _pass("test_halo_validation_fails_if_past_rear_axle")

def test_com_sanity_gate_catches_mm_as_m():
    try:
        _assert_com_in_range("test", (0.050, 0.0, 25.0), rear_axle_m=0.194)  # z=25 m is mm error
        _fail("test_com_sanity_gate_catches_mm_as_m", "should have raised")
    except ValueError:
        _pass("test_com_sanity_gate_catches_mm_as_m")

def test_com_sanity_gate_valid():
    _assert_com_in_range("test", (0.050, 0.0, 0.025), rear_axle_m=0.194)
    _pass("test_com_sanity_gate_valid")

def test_com_sanity_gate_outside_car_length():
    try:
        _assert_com_in_range("test", (0.300, 0.0, 0.025), rear_axle_m=0.194)
        _fail("test_com_sanity_gate_outside_car_length", "should have raised")
    except ValueError:
        _pass("test_com_sanity_gate_outside_car_length")

if __name__ == "__main__":
    test_front_cylinder_x_center_at_x_front()
    test_rear_cylinder_x_center_equals_x_front_plus_W()
    test_cylinder_contains_point_inside()
    test_cylinder_contains_point_outside()
    test_cylinder_void_mask_shape()
    test_cylinder_void_mask_centre_is_true()
    test_wheel_disc_zone_contains_point_at_real_wheel_position()
    test_wheel_disc_zone_excludes_centreline()
    test_wheel_disc_zone_excludes_outside_x_z_circle()
    test_wheel_disc_void_mask_shape()
    test_four_wheel_zones_left_right_symmetric()
    test_box_void_mask_shape()
    test_halo_validation_behind_front_axle()
    test_halo_validation_allows_at_or_before_front_axle()
    test_halo_validation_allows_before_canister()
    test_halo_validation_fails_if_past_rear_axle()
    test_com_sanity_gate_catches_mm_as_m()
    test_com_sanity_gate_valid()
    test_com_sanity_gate_outside_car_length()
    print("\nAll fixed_hardware tests passed.")