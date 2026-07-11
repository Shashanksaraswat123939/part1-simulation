"""
Tests for wheel_visibility_zones.py (T7.9 visibility keep-clear zones).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from wheel_visibility_zones import (
    build_t79_forbidden_mask, _rect_forbidden_mask, _wedge_forbidden_mask,
    FRONT_INNER_GAP_MIN_MM, REAR_INNER_GAP_MIN_MM,
    T79_FRONT_LEG_X_MM, T79_REAR_LEG_X_MM, T79_WEDGE_LEG_Y_MM,
)
from geometry_contract import mm_to_m, GRID_SPACING_M

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)


def test_rect_mask_marks_correct_x_band():
    origin = (0.0, 0.0, 0.0)
    shape = (100, 50, 10)
    mask = _rect_forbidden_mask(
        origin, shape,
        x_min_m=mm_to_m(10.0), x_max_m=mm_to_m(15.0),
        y_min_m=mm_to_m(5.0), z_max_m=mm_to_m(65.0),
    )
    # A cell at x=12mm, y=10mm, z=1mm should be forbidden
    xi = round(mm_to_m(12.0) / GRID_SPACING_M)
    yi = round(mm_to_m(10.0) / GRID_SPACING_M)
    zi = round(mm_to_m(1.0) / GRID_SPACING_M)
    assert mask[xi, yi, zi], "Expected cell inside rectangle to be forbidden"
    # A cell at x=2mm (outside band) should not be forbidden
    xi2 = round(mm_to_m(2.0) / GRID_SPACING_M)
    assert not mask[xi2, yi, zi], "Expected cell outside x-band to be clear"
    _pass("test_rect_mask_marks_correct_x_band")


def test_rect_mask_respects_y_min():
    origin = (0.0, 0.0, 0.0)
    shape = (60, 150, 10)
    mask = _rect_forbidden_mask(
        origin, shape,
        x_min_m=mm_to_m(5.0), x_max_m=mm_to_m(10.0),
        y_min_m=mm_to_m(19.0), z_max_m=mm_to_m(65.0),
    )
    xi = round(mm_to_m(7.0) / GRID_SPACING_M)
    zi = round(mm_to_m(1.0) / GRID_SPACING_M)
    yi_below = round(mm_to_m(10.0) / GRID_SPACING_M)   # below y_min -> clear
    yi_above = round(mm_to_m(25.0) / GRID_SPACING_M)   # above y_min -> forbidden
    assert not mask[xi, yi_below, zi], "Cell below y_min should be clear"
    assert mask[xi, yi_above, zi], "Cell above y_min should be forbidden"
    _pass("test_rect_mask_respects_y_min")


def test_wedge_mask_right_angle_corner_is_forbidden():
    """The right-angle corner (at the wheel, outer y) must be inside the wedge."""
    origin = (0.0, 0.0, 0.0)
    shape = (300, 250, 10)
    x_at_wheel = mm_to_m(50.0)
    y_bottom = mm_to_m(19.0)
    y_top = y_bottom + mm_to_m(T79_WEDGE_LEG_Y_MM)
    mask = _wedge_forbidden_mask(
        origin, shape,
        x_at_wheel_m=x_at_wheel, leg_x_m=mm_to_m(T79_FRONT_LEG_X_MM),
        direction="aft", y_bottom_m=y_bottom, y_top_m=y_top,
        z_max_m=mm_to_m(65.0),
    )
    # Right-angle corner: just inside (x_at_wheel + epsilon, y_top - epsilon)
    xi = round((x_at_wheel + mm_to_m(1.0)) / GRID_SPACING_M)
    yi = round((y_top - mm_to_m(1.0)) / GRID_SPACING_M)
    zi = round(mm_to_m(1.0) / GRID_SPACING_M)
    assert mask[xi, yi, zi], "Right-angle corner should be forbidden"
    _pass("test_wedge_mask_right_angle_corner_is_forbidden")


def test_wedge_mask_far_tip_is_clear():
    """Beyond the hypotenuse (far corner) must NOT be forbidden."""
    origin = (0.0, 0.0, 0.0)
    shape = (300, 250, 10)
    x_at_wheel = mm_to_m(50.0)
    y_bottom = mm_to_m(19.0)
    y_top = y_bottom + mm_to_m(T79_WEDGE_LEG_Y_MM)
    mask = _wedge_forbidden_mask(
        origin, shape,
        x_at_wheel_m=x_at_wheel, leg_x_m=mm_to_m(T79_FRONT_LEG_X_MM),
        direction="aft", y_bottom_m=y_bottom, y_top_m=y_top,
        z_max_m=mm_to_m(65.0),
    )
    # Far corner (x_at_wheel + leg_x, y_bottom) is the hypotenuse endpoint --
    # just beyond it (further aft, at y_top) should be clear (outside hypotenuse)
    xi = round((x_at_wheel + mm_to_m(T79_FRONT_LEG_X_MM)) / GRID_SPACING_M)
    yi = round(y_top / GRID_SPACING_M)
    zi = round(mm_to_m(1.0) / GRID_SPACING_M)
    assert not mask[xi, yi, zi], "Point beyond hypotenuse should be clear"
    _pass("test_wedge_mask_far_tip_is_clear")


def test_wedge_direction_fwd_mirrors_aft():
    """direction='fwd' should mirror direction='aft' in the -x direction."""
    origin = (0.0, 0.0, 0.0)
    shape = (300, 250, 10)
    x_at_wheel = mm_to_m(50.0)
    y_bottom = mm_to_m(15.0)
    y_top = y_bottom + mm_to_m(T79_WEDGE_LEG_Y_MM)
    mask = _wedge_forbidden_mask(
        origin, shape,
        x_at_wheel_m=x_at_wheel, leg_x_m=mm_to_m(T79_REAR_LEG_X_MM),
        direction="fwd", y_bottom_m=y_bottom, y_top_m=y_top,
        z_max_m=mm_to_m(65.0),
    )
    # Right-angle corner should be just BEFORE (smaller x) the wheel this time
    xi = round((x_at_wheel - mm_to_m(1.0)) / GRID_SPACING_M)
    yi = round((y_top - mm_to_m(1.0)) / GRID_SPACING_M)
    zi = round(mm_to_m(1.0) / GRID_SPACING_M)
    assert mask[xi, yi, zi], "Right-angle corner (fwd direction) should be forbidden"
    _pass("test_wedge_direction_fwd_mirrors_aft")


def test_build_t79_sidepod_combines_both_wedges():
    W_mm, x_front_mm = 130.0, 64.0
    W_m, x_front_m = mm_to_m(W_mm), mm_to_m(x_front_mm)
    origin = (x_front_m + mm_to_m(10.0), mm_to_m(19.0), 0.0)  # roughly sidepod start
    shape = (400, 150, 10)
    mask = build_t79_forbidden_mask("sidepod", origin, shape, W_mm, x_front_mm)
    assert mask is not None
    assert mask.shape == shape
    assert mask.any(), "Sidepod T7.9 mask should mark some cells forbidden"
    _pass("test_build_t79_sidepod_combines_both_wedges")


def test_build_t79_nose_returns_none():
    origin = (0.0, -0.03, 0.0)
    shape = (50, 50, 50)
    mask = build_t79_forbidden_mask("nose", origin, shape, 130.0, 64.0)
    assert mask is None, "Nose has no T7.9 zone in this coordinate mapping"
    _pass("test_build_t79_nose_returns_none")


def test_build_t79_main_body_forbidden_near_front_wheel():
    W_mm, x_front_mm = 130.0, 64.0
    x_front_m = mm_to_m(x_front_mm)
    origin = (x_front_m - mm_to_m(20.0), -mm_to_m(30.0), 0.0)
    shape = (200, 200, 10)
    mask = build_t79_forbidden_mask("main_body", origin, shape, W_mm, x_front_mm)
    assert mask is not None
    assert mask.any(), "main_body T7.9.1 mask should mark cells forbidden"
    _pass("test_build_t79_main_body_forbidden_near_front_wheel")


def test_build_t79_rearpod_forbidden_near_rear_wheel():
    W_mm, x_front_mm = 130.0, 64.0
    rear_axle_m = mm_to_m(x_front_mm + W_mm)
    origin = (rear_axle_m, -mm_to_m(30.0), 0.0)
    shape = (200, 200, 10)
    mask = build_t79_forbidden_mask("rearpod", origin, shape, W_mm, x_front_mm)
    assert mask is not None
    assert mask.any(), "rearpod T7.9.4 mask should mark cells forbidden"
    _pass("test_build_t79_rearpod_forbidden_near_rear_wheel")


def test_default_gap_constants_match_regs():
    assert FRONT_INNER_GAP_MIN_MM == 38.0   # T7.2.1
    assert REAR_INNER_GAP_MIN_MM == 30.0    # T7.2.2
    _pass("test_default_gap_constants_match_regs")


if __name__ == "__main__":
    test_rect_mask_marks_correct_x_band()
    test_rect_mask_respects_y_min()
    test_wedge_mask_right_angle_corner_is_forbidden()
    test_wedge_mask_far_tip_is_clear()
    test_wedge_direction_fwd_mirrors_aft()
    test_build_t79_sidepod_combines_both_wedges()
    test_build_t79_nose_returns_none()
    test_build_t79_main_body_forbidden_near_front_wheel()
    test_build_t79_rearpod_forbidden_near_rear_wheel()
    test_default_gap_constants_match_regs()
    print("\nAll wheel_visibility_zones tests passed.")
