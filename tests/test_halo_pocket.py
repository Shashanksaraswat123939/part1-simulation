"""
Tests for halo_pocket.py (T4.4.4 halo mounting pocket geometry).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from halo_pocket import (
    compute_halo_pocket_box_m, build_halo_pocket_forbidden_mask,
    HALO_POCKET_LENGTH_MM, HALO_POCKET_WIDTH_MM, HALO_POCKET_DEPTH_MM,
)
from geometry_contract import mm_to_m, HALO_MIN_Z_M, GRID_SPACING_M

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)


def test_pocket_floor_is_fixed_at_24mm():
    box = compute_halo_pocket_box_m(ref_plane_A_m=0.05, d_halo_mm=20.0)
    assert abs(box["z_min_m"] - HALO_MIN_Z_M) < 1e-12
    assert abs(box["z_min_m"] - 0.024) < 1e-12
    _pass("test_pocket_floor_is_fixed_at_24mm")


def test_pocket_depth_matches_ball_nose_cut():
    box = compute_halo_pocket_box_m(ref_plane_A_m=0.05, d_halo_mm=20.0)
    depth_m = box["z_max_m"] - box["z_min_m"]
    assert abs(depth_m - mm_to_m(HALO_POCKET_DEPTH_MM)) < 1e-12
    _pass("test_pocket_depth_matches_ball_nose_cut")


def test_pocket_x_position_shifts_with_d_halo():
    box0 = compute_halo_pocket_box_m(ref_plane_A_m=0.05, d_halo_mm=0.0)
    box20 = compute_halo_pocket_box_m(ref_plane_A_m=0.05, d_halo_mm=20.0)
    assert abs(box0["x_min_m"] - 0.05) < 1e-12, "d_halo=0 should start exactly at Ref Plane A"
    assert abs(box20["x_min_m"] - (0.05 + 0.020)) < 1e-12, "d_halo=20mm should shift pocket 20mm aft"
    _pass("test_pocket_x_position_shifts_with_d_halo")


def test_pocket_length_matches_regs_appendix():
    box = compute_halo_pocket_box_m(ref_plane_A_m=0.0, d_halo_mm=0.0)
    length_m = box["x_max_m"] - box["x_min_m"]
    assert abs(length_m - mm_to_m(HALO_POCKET_LENGTH_MM)) < 1e-12
    assert abs(length_m - 0.050) < 1e-12
    _pass("test_pocket_length_matches_regs_appendix")


def test_pocket_width_symmetric_about_centerline():
    box = compute_halo_pocket_box_m(ref_plane_A_m=0.0, d_halo_mm=0.0)
    assert abs(box["y_min_m"]) == abs(box["y_max_m"])
    width_m = box["y_max_m"] - box["y_min_m"]
    assert abs(width_m - mm_to_m(HALO_POCKET_WIDTH_MM)) < 1e-12
    _pass("test_pocket_width_symmetric_about_centerline")


def test_forbidden_mask_marks_pocket_cells():
    ref_A_m = 0.05
    origin = (ref_A_m, -0.03, 0.0)
    shape = (300, 200, 200)   # covers x to ref_A+90mm, y +/-30mm, z 0-60mm
    mask = build_halo_pocket_forbidden_mask(origin, shape, ref_A_m, d_halo_mm=10.0)
    # Cell at pocket centre: x=ref_A+10+25mm, y=0, z=25mm (inside 24-27.175mm floor band)
    xi = round((mm_to_m(10.0 + 25.0)) / GRID_SPACING_M)
    yi = round((0.03) / GRID_SPACING_M)   # y=0 -> index for origin y=-0.03
    zi = round((0.025) / GRID_SPACING_M)
    assert mask[xi, yi, zi], "Pocket centre cell should be forbidden"
    _pass("test_forbidden_mask_marks_pocket_cells")


def test_forbidden_mask_clear_outside_pocket_z():
    ref_A_m = 0.05
    origin = (ref_A_m, -0.03, 0.0)
    shape = (300, 200, 200)
    mask = build_halo_pocket_forbidden_mask(origin, shape, ref_A_m, d_halo_mm=10.0)
    # Cell well below the pocket floor (z=5mm) should be clear
    xi = round((mm_to_m(10.0 + 25.0)) / GRID_SPACING_M)
    yi = round((0.03) / GRID_SPACING_M)
    zi_low = round((0.005) / GRID_SPACING_M)
    assert not mask[xi, yi, zi_low], "Cell below pocket floor should be clear"
    _pass("test_forbidden_mask_clear_outside_pocket_z")


def test_d_halo_changes_bounding_volumes_end_to_end():
    """Two different d_halo values must produce different main_body voxel masks."""
    from bounding_volumes import compute_bounding_volumes, default_rule_envelope
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _StubCyl:
        x_center_m: float
        y_center_m: float
        z_center_m: float
        radius_m: float
        x_half_width_m: float
        @property
        def x_min_m(self): return self.x_center_m - self.x_half_width_m
        @property
        def x_max_m(self): return self.x_center_m + self.x_half_width_m

    STUB_RE = default_rule_envelope()
    W_mm, x_front_mm = 130.0, 70.0
    front = _StubCyl(mm_to_m(x_front_mm), 0.0, 0.015, 0.017, 0.008)
    rear = _StubCyl(mm_to_m(x_front_mm + W_mm), 0.0, 0.015, 0.017, 0.008)

    bv_10 = compute_bounding_volumes(W_mm, x_front_mm, 10.0, front, rear, STUB_RE)
    bv_50 = compute_bounding_volumes(W_mm, x_front_mm, 50.0, front, rear, STUB_RE)

    mask_10 = bv_10.main_body.valid_mask()
    mask_50 = bv_50.main_body.valid_mask()
    assert not np.array_equal(mask_10, mask_50), (
        "Different d_halo values must produce different main_body geometry"
    )
    _pass("test_d_halo_changes_bounding_volumes_end_to_end")


if __name__ == "__main__":
    test_pocket_floor_is_fixed_at_24mm()
    test_pocket_depth_matches_ball_nose_cut()
    test_pocket_x_position_shifts_with_d_halo()
    test_pocket_length_matches_regs_appendix()
    test_pocket_width_symmetric_about_centerline()
    test_forbidden_mask_marks_pocket_cells()
    test_forbidden_mask_clear_outside_pocket_z()
    test_d_halo_changes_bounding_volumes_end_to_end()
    print("\nAll halo_pocket tests passed.")
