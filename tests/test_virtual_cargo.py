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


def test_placement_never_shares_volume_with_the_halo_pocket():
    """T4.2: the cargo must not COINCIDE with the halo pocket.

    Coincidence is a 3-D question. Since 2026-07-24 the cargo is pinned to
    14..24 mm (top face on the pocket floor) and the pocket floor is 24 mm, so
    the two are z-disjoint and cannot share volume at ANY x -- the cargo is
    specified to sit directly BENEATH the halo. This test therefore checks the
    thing the reg actually forbids (shared volume), not the x-only proxy it
    used to check, which forbade the intended placement and cost 42% of the
    legal d_halo range.
    """
    from halo_pocket import compute_halo_pocket_box_m
    from virtual_cargo import CARGO_Z_BASE_M
    x_front_mm, W_mm, d_halo_mm = 70.0, 130.0, 20.0
    ref_A_m = mm_to_m(x_front_mm - 16.0)
    result = find_cargo_placement(x_front_mm, W_mm, ref_A_m, d_halo_mm, z_floor_m=0.0)
    halo_box = compute_halo_pocket_box_m(ref_A_m, d_halo_mm)

    cargo_x = (result["x_start_m"], result["x_start_m"] + mm_to_m(CARGO_LENGTH_MM))
    cargo_z = (CARGO_Z_BASE_M, CARGO_Z_BASE_M + mm_to_m(CARGO_HEIGHT_MM))
    ox = cargo_x[0] < halo_box["x_max_m"] and halo_box["x_min_m"] < cargo_x[1]
    oz = cargo_z[0] < halo_box["z_max_m"] and halo_box["z_min_m"] < cargo_z[1]
    assert not (ox and oz), (
        f"cargo x{cargo_x} z{cargo_z} shares volume with the halo pocket "
        f"x[{halo_box['x_min_m']}, {halo_box['x_max_m']}] "
        f"z[{halo_box['z_min_m']}, {halo_box['z_max_m']}]"
    )
    assert not oz, "cargo and halo pocket must stay z-disjoint (top face at 24 mm)"
    _pass("test_placement_never_shares_volume_with_the_halo_pocket")


def test_cargo_sits_directly_under_the_halo():
    """z is FIXED: top face on the halo pocket floor, 14..24 mm above track."""
    from virtual_cargo import CARGO_Z_BASE_MM, CARGO_TOP_Z_MM
    from geometry_contract import HALO_MIN_Z_MM
    assert CARGO_TOP_Z_MM == HALO_MIN_Z_MM, (
        f"cargo top {CARGO_TOP_Z_MM} must equal the halo pocket floor {HALO_MIN_Z_MM}"
    )
    assert abs(CARGO_TOP_Z_MM - CARGO_Z_BASE_MM - CARGO_HEIGHT_MM) < 1e-9
    # z must NOT follow the rule-envelope floor any more, whatever is passed in.
    a = find_cargo_placement(70.0, 130.0, mm_to_m(54.0), 20.0, z_floor_m=0.0)
    b = find_cargo_placement(70.0, 130.0, mm_to_m(54.0), 20.0, z_floor_m=0.05)
    assert a["z_base_m"] == b["z_base_m"] == mm_to_m(CARGO_Z_BASE_MM), (
        f"cargo z must be pinned under the halo, got {a['z_base_m']} / {b['z_base_m']}"
    )
    _pass("test_cargo_sits_directly_under_the_halo")


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


def test_placement_no_longer_shifts_for_a_centred_halo():
    """INVERTED 2026-07-24, deliberately.

    This used to assert the cargo shifts away when the halo pocket sits at the
    corridor centre. That behaviour is gone on purpose: the cargo is now pinned
    to 14..24 mm, z-disjoint from the pocket (floor 24 mm), so sitting under the
    halo shares no volume and T4.2 permits it. Keeping the old x-only screen
    made the specified placement -- directly beneath the halo -- impossible, and
    cost 42% of the legal d_halo range at W=130/x_front=46.
    """
    x_front_mm, W_mm = 70.0, 140.0
    x_front_m, W_m = mm_to_m(x_front_mm), mm_to_m(W_mm)
    ref_A_m = mm_to_m(x_front_mm - 16.0)
    corridor_centre_m = x_front_m + W_m / 2.0
    d_halo_mm = (corridor_centre_m - ref_A_m) * 1000.0
    result = find_cargo_placement(x_front_mm, W_mm, ref_A_m, d_halo_mm, z_floor_m=0.0)
    assert not result["collided_with_default"], (
        "cargo should stay at the corridor centre under the halo; the x-only "
        "halo screen should no longer apply while the two are z-disjoint"
    )
    _pass("test_placement_no_longer_shifts_for_a_centred_halo")


def test_placement_still_raises_when_the_corridor_itself_is_too_short():
    """The corridor length check survives -- it is independent of the halo.

    The old 'halo splits the corridor' failure is gone (see the test above), so
    the remaining way to have no placement is a wheelbase too short to hold the
    60 mm cargo between the axle centre lines at all.
    """
    try:
        find_cargo_placement(70.0, 50.0, mm_to_m(54.0), 20.0, z_floor_m=0.0)
        _fail("test_placement_still_raises_when_the_corridor_itself_is_too_short",
              "should have raised ValueError")
    except ValueError:
        _pass("test_placement_still_raises_when_the_corridor_itself_is_too_short")


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


def test_is_valid_filters_placements_the_builder_would_reject():
    """The halo screen is blind to wheel keep-clears and the other void regions.

    Without an is_valid hook a scored search returns the COM-optimal placement
    with no idea whether it can be built. Measured 2026-07-27 at
    W=130/x_front=46: the scorer chose x_start = 0.046 m, 35 mm forward of the
    geometric default and inside the front wheel keep-clear, eroding 23.7% of
    the cargo. build_unified_geometry refused, evaluate_scalars returned _fail,
    and EVERY Stage-1 candidate died -- run_stage1 gave best=None and the whole
    two-stage run crashed on the handoff.
    """
    from virtual_cargo import find_cargo_placement

    kw = dict(x_front_mm=46.0, W_mm=130.0, ref_plane_A_m=0.030,
              d_halo_mm=20.0, z_floor_m=0.0015)

    unscreened = find_cargo_placement(**kw)

    # Learn the candidate positions actually offered, then reject all but one
    # of them -- a target invented independently need not be on the grid.
    offered = []
    find_cargo_placement(**kw, is_valid=lambda x, f: (offered.append(x), True)[1])
    assert offered, "is_valid was never consulted"
    target = next((x for x in offered
                   if abs(x - unscreened["x_start_m"]) > 1e-9), None)
    assert target is not None, "only one candidate position; cannot test filtering"

    got = find_cargo_placement(**kw,
                               is_valid=lambda x, f: abs(x - target) < 1e-9)
    assert abs(got["x_start_m"] - target) < 1e-9, (
        f"is_valid was ignored: got {got['x_start_m']}, wanted {target}")
    assert abs(got["x_start_m"] - unscreened["x_start_m"]) > 1e-9, (
        "the screened result should differ from the unscreened default")

    # And when nothing is buildable it must raise, not return an unbuildable one.
    try:
        find_cargo_placement(**kw, is_valid=lambda x, f: False)
    except ValueError as exc:
        assert "forced-air" in str(exc)
    else:
        raise AssertionError("no buildable placement should raise, not return one")
    _pass("test_is_valid_filters_placements_the_builder_would_reject")


def test_regs_dimensions_unchanged():
    assert CARGO_LENGTH_MM == 60.0
    assert CARGO_WIDE_WIDTH_MM == 55.0
    assert CARGO_NARROW_WIDTH_MM == 10.0
    assert CARGO_HEIGHT_MM == 10.0
    _pass("test_regs_dimensions_unchanged")


if __name__ == "__main__":
    test_placement_never_shares_volume_with_the_halo_pocket()
    test_cargo_sits_directly_under_the_halo()
    test_placement_within_axle_corridor()
    test_placement_defaults_to_corridor_centre_when_no_conflict()
    test_placement_no_longer_shifts_for_a_centred_halo()
    test_placement_still_raises_when_the_corridor_itself_is_too_short()
    test_solid_mask_wide_end_is_wider_than_narrow_end()
    test_solid_mask_respects_height()
    test_is_valid_filters_placements_the_builder_would_reject()
    test_regs_dimensions_unchanged()
    print("\nAll virtual_cargo tests passed.")
