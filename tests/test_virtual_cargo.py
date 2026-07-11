"""
Tests for virtual_cargo.py (T4.2 virtual cargo placement).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from virtual_cargo import (
    find_cargo_placement, build_virtual_cargo_solid_mask,
    CARGO_LENGTH_MM, CARGO_WIDE_WIDTH_MM, CARGO_NARROW_WIDTH_MM, CARGO_HEIGHT_MM,
)
from geometry_contract import mm_to_m, GRID_SPACING_M

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)


def test_placement_avoids_halo_pocket():
    """Chosen cargo x-range must not overlap the halo pocket's x-range."""
    from halo_pocket import compute_halo_pocket_box_m
    x_front_mm, W_mm, d_halo_mm = 70.0, 130.0, 10.0
    ref_A_m = mm_to_m(x_front_mm - 16.0)
    result = find_cargo_placement(x_front_mm, W_mm, ref_A_m, d_halo_mm, z_floor_m=0.0)
    halo_box = compute_halo_pocket_box_m(ref_A_m, d_halo_mm)
    cargo_x_min = result["x_start_m"]
    cargo_x_max = cargo_x_min + mm_to_m(CARGO_LENGTH_MM)
    overlap = cargo_x_min < halo_box["x_max_m"] and halo_box["x_min_m"] < cargo_x_max
    assert not overlap, "Cargo placement must not overlap halo pocket"
    _pass("test_placement_avoids_halo_pocket")


def test_placement_within_axle_corridor():
    x_front_mm, W_mm, d_halo_mm = 70.0, 130.0, 10.0
    ref_A_m = mm_to_m(x_front_mm - 16.0)
    result = find_cargo_placement(x_front_mm, W_mm, ref_A_m, d_halo_mm, z_floor_m=0.0)
    x_front_m, W_m = mm_to_m(x_front_mm), mm_to_m(W_mm)
    assert result["x_start_m"] >= x_front_m - 1e-9
    assert result["x_start_m"] + mm_to_m(CARGO_LENGTH_MM) <= x_front_m + W_m + 1e-9
    _pass("test_placement_within_axle_corridor")


def test_placement_defaults_to_corridor_centre_when_no_conflict():
    """With no halo nearby, the chosen placement should be the un-shifted default."""
    x_front_mm, W_mm, d_halo_mm = 70.0, 140.0, 0.0   # halo right at Ref Plane A, far from centre
    ref_A_m = mm_to_m(x_front_mm - 16.0)
    result = find_cargo_placement(x_front_mm, W_mm, ref_A_m, d_halo_mm, z_floor_m=0.0)
    assert not result["collided_with_default"], "Expected the default centre placement to be used"
    _pass("test_placement_defaults_to_corridor_centre_when_no_conflict")


def test_placement_shifts_when_default_collides():
    """When the halo pocket sits at the corridor centre, placement must shift away."""
    x_front_mm, W_mm = 70.0, 140.0
    x_front_m, W_m = mm_to_m(x_front_mm), mm_to_m(W_mm)
    ref_A_m = mm_to_m(x_front_mm - 16.0)
    corridor_centre_m = x_front_m + W_m / 2.0
    # Choose d_halo so the halo pocket starts right at the corridor centre
    d_halo_mm = (corridor_centre_m - ref_A_m) * 1000.0
    result = find_cargo_placement(x_front_mm, W_mm, ref_A_m, d_halo_mm, z_floor_m=0.0)
    assert result["collided_with_default"], "Expected placement to shift away from centred halo"
    _pass("test_placement_shifts_when_default_collides")


def test_placement_raises_when_impossible():
    """Halo occupying the middle of a too-tight corridor leaves no room for cargo."""
    x_front_mm, W_mm, d_halo_mm = 64.0, 130.0, 50.0
    ref_A_m = mm_to_m(x_front_mm - 16.0)
    try:
        find_cargo_placement(x_front_mm, W_mm, ref_A_m, d_halo_mm, z_floor_m=0.0)
        _fail("test_placement_raises_when_impossible", "should have raised ValueError")
    except ValueError:
        _pass("test_placement_raises_when_impossible")


def test_solid_mask_wide_end_is_wider_than_narrow_end():
    origin = (0.0, -0.05, 0.0)
    shape = (250, 350, 50)
    x_start_m = mm_to_m(5.0)
    z_base_m = 0.0
    mask = build_virtual_cargo_solid_mask(origin, shape, x_start_m, z_base_m)

    xi_wide = round((x_start_m - origin[0] + mm_to_m(1.0)) / GRID_SPACING_M)     # near wide end
    xi_narrow = round((x_start_m - origin[0] + mm_to_m(CARGO_LENGTH_MM - 1.0)) / GRID_SPACING_M)  # near narrow end
    zi = round(mm_to_m(5.0) / GRID_SPACING_M)   # mid-height

    wide_count = mask[xi_wide, :, zi].sum()
    narrow_count = mask[xi_narrow, :, zi].sum()
    assert wide_count > narrow_count, (
        f"Wide end ({wide_count} cells) should be wider than narrow end ({narrow_count} cells)"
    )
    _pass("test_solid_mask_wide_end_is_wider_than_narrow_end")


def test_solid_mask_respects_height():
    origin = (0.0, -0.05, 0.0)
    shape = (250, 350, 50)
    x_start_m = mm_to_m(5.0)
    z_base_m = 0.0
    mask = build_virtual_cargo_solid_mask(origin, shape, x_start_m, z_base_m)

    xi = round((mm_to_m(30.0)) / GRID_SPACING_M)   # middle of cargo length
    yi = round(0.05 / GRID_SPACING_M)              # y=0 (centreline, origin y=-0.05)
    zi_inside = round(mm_to_m(5.0) / GRID_SPACING_M)     # within [0,10mm]
    zi_outside = round(mm_to_m(13.0) / GRID_SPACING_M)   # above 10mm height

    assert mask[xi, yi, zi_inside], "Cell within cargo height should be solid"
    assert not mask[xi, yi, zi_outside], "Cell above cargo height should be clear"
    _pass("test_solid_mask_respects_height")


def test_regs_dimensions_unchanged():
    assert CARGO_LENGTH_MM == 60.0
    assert CARGO_WIDE_WIDTH_MM == 55.0
    assert CARGO_NARROW_WIDTH_MM == 10.0
    assert CARGO_HEIGHT_MM == 10.0
    _pass("test_regs_dimensions_unchanged")


if __name__ == "__main__":
    test_placement_avoids_halo_pocket()
    test_placement_within_axle_corridor()
    test_placement_defaults_to_corridor_centre_when_no_conflict()
    test_placement_shifts_when_default_collides()
    test_placement_raises_when_impossible()
    test_solid_mask_wide_end_is_wider_than_narrow_end()
    test_solid_mask_respects_height()
    test_regs_dimensions_unchanged()
    print("\nAll virtual_cargo tests passed.")
