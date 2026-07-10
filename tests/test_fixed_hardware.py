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
)
from geometry_contract import R_WHEEL_M, WHEEL_CLEARANCE_M, mm_to_m

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

def test_front_cylinder_x_center_is_zero():
    cyl = ForbiddenCylinder(0.0, 0.0, 0.015, R_WHEEL_M+WHEEL_CLEARANCE_M, 0.010)
    assert cyl.x_center_m == 0.0
    _pass("test_front_cylinder_x_center_is_zero")

def test_rear_cylinder_x_center_equals_W():
    W_m = mm_to_m(130.0)
    cyl = ForbiddenCylinder(W_m, 0.0, 0.015, R_WHEEL_M+WHEEL_CLEARANCE_M, 0.010)
    assert abs(cyl.x_center_m - W_m) < 1e-12
    _pass("test_rear_cylinder_x_center_equals_W")

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

def test_box_void_mask_shape():
    shape = (50, 50, 50)
    origin = (0.0, -0.025, 0.0)
    mask = _build_box_void_mask(shape, origin, (0.01, 0.02), (-0.005, 0.005), (0.005, 0.015))
    assert mask.shape == shape
    assert mask.dtype == bool
    _pass("test_box_void_mask_shape")

def test_halo_validation_behind_front_axle():
    # Valid: halo x_front > 0 and > canister_x
    halo = HaloGeometry(x_front_m=0.010, x_rear_m=0.050)
    _validate_halo_position(halo, canister_x_m=0.005, W_m=0.130)
    _pass("test_halo_validation_behind_front_axle")

def test_halo_validation_fails_if_at_front_axle():
    halo = HaloGeometry(x_front_m=0.0, x_rear_m=0.050)
    try:
        _validate_halo_position(halo, canister_x_m=0.005, W_m=0.130)
        _fail("test_halo_validation_fails_if_at_front_axle", "should have raised")
    except ValueError:
        _pass("test_halo_validation_fails_if_at_front_axle")

def test_halo_validation_fails_if_before_canister():
    halo = HaloGeometry(x_front_m=0.003, x_rear_m=0.050)
    try:
        _validate_halo_position(halo, canister_x_m=0.010, W_m=0.130)
        _fail("test_halo_validation_fails_if_before_canister", "should have raised")
    except ValueError:
        _pass("test_halo_validation_fails_if_before_canister")

def test_halo_validation_fails_if_past_rear_axle():
    halo = HaloGeometry(x_front_m=0.010, x_rear_m=0.135)
    try:
        _validate_halo_position(halo, canister_x_m=0.005, W_m=0.130)
        _fail("test_halo_validation_fails_if_past_rear_axle", "should have raised")
    except ValueError:
        _pass("test_halo_validation_fails_if_past_rear_axle")

def test_com_sanity_gate_catches_mm_as_m():
    try:
        _assert_com_in_range("test", (0.050, 0.0, 25.0), W_m=0.130)  # z=25 m is mm error
        _fail("test_com_sanity_gate_catches_mm_as_m", "should have raised")
    except ValueError:
        _pass("test_com_sanity_gate_catches_mm_as_m")

def test_com_sanity_gate_valid():
    _assert_com_in_range("test", (0.050, 0.0, 0.025), W_m=0.130)
    _pass("test_com_sanity_gate_valid")

def test_com_sanity_gate_outside_car_length():
    try:
        _assert_com_in_range("test", (0.200, 0.0, 0.025), W_m=0.130)
        _fail("test_com_sanity_gate_outside_car_length", "should have raised")
    except ValueError:
        _pass("test_com_sanity_gate_outside_car_length")

if __name__ == "__main__":
    test_front_cylinder_x_center_is_zero()
    test_rear_cylinder_x_center_equals_W()
    test_cylinder_contains_point_inside()
    test_cylinder_contains_point_outside()
    test_cylinder_void_mask_shape()
    test_cylinder_void_mask_centre_is_true()
    test_box_void_mask_shape()
    test_halo_validation_behind_front_axle()
    test_halo_validation_fails_if_at_front_axle()
    test_halo_validation_fails_if_before_canister()
    test_halo_validation_fails_if_past_rear_axle()
    test_com_sanity_gate_catches_mm_as_m()
    test_com_sanity_gate_valid()
    test_com_sanity_gate_outside_car_length()
    print("\nAll fixed_hardware tests passed.")