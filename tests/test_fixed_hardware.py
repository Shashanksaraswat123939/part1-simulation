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



def test_the_canister_void_is_the_bore_and_stays_legal():
    """The chamber is regulated; the wall around it is not part of the hole.

    hardware_cad/co2_canister.stl "already includes the minimum SAFETY ZONE
    around the cartridge" (hardware_geometry.canister_front_x_mm) and measures
    12.00 mm about the bore axis, so a solid-body intersection test against
    that mesh reports overlap wherever the wall is. I read that as the void
    being too small and widened it to 12.125 mm -- which makes the CHAMBER
    24.25 mm across, against T5.1's 18.0-18.5 mm. Illegal, and caught by
    test_cartridge_bore_is_carved_and_open_at_the_rear.

    The annulus between the bore and the assembly's outer face is meant to be
    SOLID: it is T5.5's 3 mm wall. Body material there is correct. (The
    vertices that looked like an intersection sat at radius 9.05-11.91 mm,
    i.e. almost entirely outside the 9.125 mm bore -- they were the wall.)

    So: void = bore, and the wider radius exists only to anchor the loft.
    """
    from fixed_hardware import (CANISTER_CLEARANCE_RADIUS_MM,
                                CANISTER_DIAMETER_MM, CANISTER_SAFETY_ZONE_MM,
                                compute_default_fixed_hardware_inputs)

    inputs = compute_default_fixed_hardware_inputs(
        130.0, 46.0, 20.0, 0.030, 0.176, rear_face_x_m=0.213)
    bore_dia = inputs["canister_radius_mm"] * 2.0
    assert 18.0 <= bore_dia <= 18.5, (
        f"chamber diameter {bore_dia:.2f} mm is outside T5.1's 18.0-18.5 mm -- "
        f"the void has been sized to something other than the bore")
    assert CANISTER_CLEARANCE_RADIUS_MM == (
        CANISTER_DIAMETER_MM / 2.0 + CANISTER_SAFETY_ZONE_MM)
    assert CANISTER_CLEARANCE_RADIUS_MM > inputs["canister_radius_mm"], (
        "the clearance radius must exceed the bore, or the loft anchors on the "
        "chamber and leaves no wall above it")
    _pass("test_the_canister_void_is_the_bore_and_stays_legal")


def test_the_loft_leaves_the_t5_5_wall_above_the_bore():
    """Deck lands on the cartridge's flat face, not on the bore.

    Anchoring the loft's rear end on the bore put the deck at z=44.1 mm, level
    with the top of the chamber, leaving no material over it. Anchoring on the
    assembly's outer face puts it at 47.1 mm, which is the part's flat surface
    AND exactly T5.5's 3 mm of Model Block above the chamber.
    """
    from fixed_hardware import (CANISTER_CLEARANCE_RADIUS_MM,
                                CANISTER_SAFETY_ZONE_MM, CANISTER_DIAMETER_MM)
    gap = CANISTER_CLEARANCE_RADIUS_MM - CANISTER_DIAMETER_MM / 2.0
    assert gap >= CANISTER_SAFETY_ZONE_MM - 1e-9, (
        f"only {gap:.3f} mm between the bore and the loft anchor; T5.5 wants "
        f"{CANISTER_SAFETY_ZONE_MM} mm of material above the chamber")
    _pass("test_the_loft_leaves_the_t5_5_wall_above_the_bore")


def test_wheel_supports_are_not_flipped_end_for_end():
    """The bracket's body end must stay inboard and its wheel end outboard.

    place_supports translated by `-lo[1]` with the comment "inner end (min y)
    -> y=0". But lo[1] is the MINIMUM y, and in the CAD frame that is the
    OUTBOARD end: front_wheel_support spans y[-36.5, 0] with front_wheel at
    y[-36.5, -19.2]. So the translation put the wheel-mounting end on the
    centreline and the body-mounting end out at the wheel -- inside-out, both
    sides, every render.

    The CAD halves are already positioned relative to each other, so the +y
    instance is a pure MIRROR of the CAD, never a translation of it. Centroid y
    discriminates: mirroring preserves |centroid|, the old translate-then-
    mirror does not.
    """
    import sys as _s
    from pathlib import Path as _P
    _s.path.insert(0, str(_P(__file__).resolve().parent.parent / "sandbox"))
    import hardware_assembly as HA

    cad = HA._load("front_wheel_support.stl")
    placed = HA.place_supports(120.0, 36.0)
    left = placed["support_front_left"]

    # x and z are placed; y must be untouched by anything but the mirror.
    assert abs(abs(left.centroid[1]) - abs(cad.centroid[1])) < 1e-6, (
        f"|centroid y| moved from {abs(cad.centroid[1])*1000:.2f} mm to "
        f"{abs(left.centroid[1])*1000:.2f} mm -- the support has been "
        f"translated in y, which flips it end-for-end")

    right = placed["support_front_right"]
    assert abs(left.centroid[1] + right.centroid[1]) < 1e-9, (
        "the two sides are not mirror images of each other")
    assert left.bounds[0][1] < 0 < right.bounds[1][1], (
        "left/right supports are not on their own sides of the centreline")
    _pass("test_wheel_supports_are_not_flipped_end_for_end")


if __name__ == "__main__":
    # Collected BY NAME. The hand-written list that used to live here silently
    # skipped every test added after it was written -- see
    # part3-simulation/tests/test_test_suites_run_what_they_define.py.
    import sys as _sys
    _mod = _sys.modules[__name__]
    _failed = 0
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        _fn = getattr(_mod, _n)
        if not callable(_fn):
            continue
        try:
            _fn()
        except Exception as _exc:  # noqa: BLE001
            print("FAIL %s: %s" % (_n, _exc))
            _failed += 1
    if _failed:
        print("%d fixed_hardware test(s) FAILED." % _failed)
        _sys.exit(1)
    print("All fixed_hardware tests passed.")
