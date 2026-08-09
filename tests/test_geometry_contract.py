"""Tests for geometry_contract.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geometry_contract as gc

def _pass(name): print(f"PASS {name}")
def _fail(name, msg): print(f"FAIL {name}: {msg}"); sys.exit(1)

def test_co2_mass_matches_part2_constant():
    # Part 2's mass_com_ingest.CO2_CARTRIDGE_MASS_KG = 0.023
    # If this test fails, every FixedHardwareSpec construction will raise in Part 2.
    assert abs(gc.CO2_MASS_KG - 0.023) < 1e-12, f"CO2_MASS_KG={gc.CO2_MASS_KG} != 0.023"
    _pass("test_co2_mass_matches_part2_constant")

def test_wheel_constants_match_locked_race_objective():
    """Cross-check Part 1 against PART 2, not against a literal.

    This asserted `R_WHEEL_M == 0.015` -- a copy of the number it was supposed
    to be checking. A copy cannot detect the two drifting apart; it only detects
    Part 1 changing, and then fails even when Part 1 and Part 2 were updated
    together and still agree. That is what happened when the radius moved to the
    measured 14.13 mm. Read Part 2's value and compare.
    """
    assert gc.N_WHEELS == 4, f"N_WHEELS={gc.N_WHEELS} != 4"

    import importlib.util
    import pathlib
    p2 = (pathlib.Path(__file__).resolve().parent.parent.parent
          / "part2-simulation" / "race_objective.py")
    if not p2.exists():
        return _pass("test_wheel_constants_match_locked_race_objective "
                     "(skipped: Part 2 absent)")
    # Read the constant textually -- importing race_objective drags in jax.
    r_wheel = n_wheels = None
    for line in p2.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("R_WHEEL") and "=" in s and not s.startswith("R_WHEEL_M"):
            r_wheel = float(s.split("=", 1)[1].split("#")[0].strip())
        elif s.startswith("N_WHEELS") and "=" in s:
            n_wheels = int(float(s.split("=", 1)[1].split("#")[0].strip()))
    assert r_wheel is not None, f"could not find R_WHEEL in {p2}"
    assert abs(gc.R_WHEEL_M - r_wheel) < 1e-12, (
        f"Part 1 R_WHEEL_M={gc.R_WHEEL_M} but Part 2 R_WHEEL={r_wheel} -- these "
        f"divide the rotational inertia term in m_eff and must be identical")
    if n_wheels is not None:
        assert gc.N_WHEELS == n_wheels, (
            f"Part 1 N_WHEELS={gc.N_WHEELS} but Part 2 N_WHEELS={n_wheels}")
    _pass("test_wheel_constants_match_locked_race_objective")

def test_nose_density_is_1000():
    assert gc.DENSITY_NOSE_KGM3 == 1000.0
    _pass("test_nose_density_is_1000")

def test_nose_density_is_6x_sidepod():
    ratio = gc.DENSITY_NOSE_KGM3 / gc.DENSITY_SIDEPOD_KGM3
    assert abs(ratio - 1000.0/163.0) < 1e-6, f"Ratio={ratio}"
    _pass("test_nose_density_is_6x_sidepod")

def test_all_machined_densities_present():
    for name in ("nose", "sidepod", "rearpod", "main_body"):
        d = gc.get_density(name)
        assert d > 0, f"density of {name} is {d}"
    _pass("test_all_machined_densities_present")

def test_get_density_unknown_raises():
    try:
        gc.get_density("wing")
        _fail("test_get_density_unknown_raises", "should have raised ValueError")
    except ValueError:
        _pass("test_get_density_unknown_raises")

def test_mm_to_m_round_trip():
    for v in [0.0, 1.0, 120.0, 140.0, 0.3, 3.15]:
        assert abs(gc.m_to_mm(gc.mm_to_m(v)) - v) < 1e-9, f"Round trip failed for {v}"
    _pass("test_mm_to_m_round_trip")

def test_gcm3_to_kgm3():
    assert gc.gcm3_to_kgm3(1.0) == 1000.0
    assert abs(gc.gcm3_to_kgm3(0.163) - 163.0) < 1e-9
    _pass("test_gcm3_to_kgm3")

def test_grid_cells_minimum_one():
    assert gc.grid_cells(0.0) == 1
    assert gc.grid_cells(-5.0) == 1
    assert gc.grid_cells(0.3) == 1
    assert gc.grid_cells(0.31) == 2
    assert gc.grid_cells(0.6) == 2
    assert gc.grid_cells(0.61) == 3
    _pass("test_grid_cells_minimum_one")

def test_W_bounds():
    assert gc.W_MIN_MM == 120.0
    assert gc.W_MAX_MM == 140.0
    assert gc.W_MIN_MM < gc.W_MAX_MM
    _pass("test_W_bounds")

def test_validate_W_valid():
    gc.validate_W(120.0)
    gc.validate_W(130.0)
    gc.validate_W(140.0)
    _pass("test_validate_W_valid")

def test_validate_W_invalid():
    for bad in [119.9, 140.1, 0.0, 200.0]:
        try:
            gc.validate_W(bad)
            _fail("test_validate_W_invalid", f"W={bad} should have raised")
        except ValueError:
            pass
    _pass("test_validate_W_invalid")

def test_validate_d_halo_valid():
    # Range is [16, W-34), both bounds physical:
    #   16     = pocket FRONT edge on the front axle line (forward-most travel;
    #            Ref Plane A sits 16 mm ahead of the axle, so d_halo=16 puts the
    #            pocket front exactly on it). Was 0, which allowed the halo up to
    #            16 mm AHEAD of the front axle.
    #   W-34   = pocket REAR edge must not reach the rear axle. At W=130: 96.0,
    #            strict/exclusive.
    gc.validate_d_halo(16.0, 130.0)
    gc.validate_d_halo(95.99, 130.0)
    _pass("test_validate_d_halo_valid")

def test_validate_d_halo_invalid():
    for bad in (
        96.0,    # W-34 = 96, strict upper bound -- 96.0 itself is invalid
        100.1,
        -1.0,
        0.0,     # below the forward-most physical position
        15.99,   # just below it
    ):
        try:
            gc.validate_d_halo(bad, 130.0)
            _fail("test_validate_d_halo_invalid", f"{bad} should have raised")
        except ValueError:
            pass
    _pass("test_validate_d_halo_invalid")

def test_d_halo_forward_limit_puts_pocket_front_on_the_front_axle():
    """d_halo_min is not a tuning value -- it is fixed by the geometry."""
    x_front_mm = 46.0
    ref_A_mm = x_front_mm - gc.D_HALO_REF_A_OFFSET_MM
    pocket_front_mm = ref_A_mm + gc.calibrate_d_halo_min_mm()
    assert abs(pocket_front_mm - x_front_mm) < 1e-9, (
        f"at d_halo_min the pocket front is at {pocket_front_mm} mm, "
        f"expected the front axle line at {x_front_mm} mm"
    )
    _pass("test_d_halo_forward_limit_puts_pocket_front_on_the_front_axle")

def test_d_halo_rear_limit_honours_the_canister_when_supplied():
    """Rear bound = min(axle bound, canister bound); canister wins when tighter."""
    W, x_front = 140.0, 46.0
    axle_only = gc.calibrate_d_halo_max_mm(W)
    # A canister far forward must tighten the bound below the axle limit.
    tight = gc.calibrate_d_halo_max_mm(W, canister_front_x_mm=140.0, x_front_mm=x_front)
    assert tight < axle_only, f"canister bound {tight} did not tighten {axle_only}"
    # pocket rear at the canister front face, exactly
    ref_A = x_front - gc.D_HALO_REF_A_OFFSET_MM
    assert abs((ref_A + tight + gc.D_HALO_POCKET_LENGTH_MM) - 140.0) < 1e-9
    # A canister far aft must NOT loosen it past the axle bound.
    loose = gc.calibrate_d_halo_max_mm(W, canister_front_x_mm=900.0, x_front_mm=x_front)
    assert loose == axle_only, f"{loose} != {axle_only}"
    _pass("test_d_halo_rear_limit_honours_the_canister_when_supplied")

def test_calibrate_d_halo_max_scales_with_W():
    """K-5 fix: bound is W-34 mm (placement-derived, strict/exclusive),
    NOT the old min(100, W+16) which was always exactly 100 across the whole
    W range -- the new bound genuinely scales with W (86 at W=120, 106 at
    W=140), reflecting that the halo pocket's legal room really does depend
    on wheelbase."""
    for W_mm, expected in ((120.0, 86.0), (130.0, 96.0), (140.0, 106.0)):
        got = gc.calibrate_d_halo_max_mm(W_mm)
        assert abs(got - expected) < 1e-9, (
            f"W={W_mm}: expected d_halo max={expected}, got {got}"
        )
    _pass("test_calibrate_d_halo_max_scales_with_W")

def test_halo_z_min():
    assert gc.HALO_MIN_Z_MM == 24.0
    assert abs(gc.HALO_MIN_Z_M - 0.024) < 1e-12
    _pass("test_halo_z_min")

def test_lifecycle_states_count():
    assert len(gc.ALLOWED_LIFECYCLE_STATES) == 8, \
        f"Expected 8 lifecycle states, got {len(gc.ALLOWED_LIFECYCLE_STATES)}"
    _pass("test_lifecycle_states_count")

def test_lifecycle_states_exact_names():
    expected = {
        "valid_simulated", "geometry_repaired", "geometry_rejected",
        "rule_rejected", "machining_rejected", "CFD_failed",
        "objective_failed", "converged",
    }
    assert gc.ALLOWED_LIFECYCLE_STATES == expected, \
        f"Mismatch: {gc.ALLOWED_LIFECYCLE_STATES ^ expected}"
    _pass("test_lifecycle_states_exact_names")

def test_tool_directions_all_components():
    # Nose is 3D printed (user-confirmed 2026-07-14), not CNC-milled -- it
    # has no TOOL_DIRECTIONS entry at all (no directional tool-access
    # constraint applies; it has a minimum wall-thickness constraint
    # instead -- see NOSE_MIN_WALL_THICKNESS_MM).
    for name in ("sidepod", "rearpod", "main_body"):
        assert name in gc.TOOL_DIRECTIONS, f"Missing tool directions for {name}"
        dirs = gc.TOOL_DIRECTIONS[name]
        assert len(dirs) >= 2, f"{name} has only {len(dirs)} tool directions"
    assert "nose" not in gc.TOOL_DIRECTIONS, \
        "nose is 3D printed and should have no TOOL_DIRECTIONS entry"
    _pass("test_tool_directions_all_components")

def test_tool_directions_unit_vectors():
    import math
    for comp, dirs in gc.TOOL_DIRECTIONS.items():
        for d in dirs:
            mag = math.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
            assert abs(mag - 1.0) < 1e-9, f"{comp} direction {d} is not unit vector (mag={mag})"
    _pass("test_tool_directions_unit_vectors")

def test_phi_snapshot_keys():
    assert set(gc.PHI_SNAPSHOT_COMPONENT_KEYS) == {"nose", "sidepod", "rearpod", "main_body"}
    assert len(gc.PHI_SNAPSHOT_COMPONENT_KEYS) == 4
    _pass("test_phi_snapshot_keys")

def test_grid_spacing_consistency():
    assert abs(gc.GRID_SPACING_M - gc.GRID_SPACING_MM / 1000.0) < 1e-15
    _pass("test_grid_spacing_consistency")

def test_min_radius_consistency():
    assert abs(gc.MIN_RADIUS_M - gc.MIN_RADIUS_MM / 1000.0) < 1e-15
    _pass("test_min_radius_consistency")



def test_machining_is_top_bottom_and_sides_only():
    """No +-X tool access, and the bottom must be offered.

    The block is machined from the top, the bottom and the two sides (project
    owner, 2026-08-06). Every entry used to disagree:

        sidepod   [+Y, -X, +Z]        -X is a cut from the front
        rearpod   [+X, +Z, +Y, -Y]    +X is a cut from the rear
        main_body [+Z, +Y, -Y]        no -Z at all

    rearpod's +X is the one with teeth: it declared anything reachable straight
    up the car's axis machinable, which is what let a shroud of material stand
    around the CO2 cartridge -- geometry a tool can only reach from behind. The
    missing -Z was the mirror-image error, rejecting bodywork that a flip of
    the block reaches. Too permissive about the impossible, too strict about
    the routine.
    """
    from geometry_contract import TOOL_DIRECTIONS

    for comp, dirs in TOOL_DIRECTIONS.items():
        for d in dirs:
            assert abs(d[0]) < 1e-12, (
                f"{comp} claims tool access along x ({d}) -- the block is not "
                f"machined from the front or the rear")
        got = {tuple(float(c) for c in d) for d in dirs}
        for need in ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0),
                     (0.0, 1.0, 0.0), (0.0, -1.0, 0.0)):
            assert need in got, f"{comp} is missing tool direction {need}"
    _pass("test_machining_is_top_bottom_and_sides_only")


if __name__ == "__main__":
    # Collected BY NAME -- the hand-written list this replaces would have
    # silently skipped anything added after it was written.
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
        print("%d geometry_contract test(s) FAILED." % _failed)
        _sys.exit(1)
    print("All geometry_contract tests passed.")
