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
    assert gc.N_WHEELS == 4, f"N_WHEELS={gc.N_WHEELS} != 4"
    assert abs(gc.R_WHEEL_M - 0.015) < 1e-12, f"R_WHEEL_M={gc.R_WHEEL_M} != 0.015"
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
    gc.validate_d_halo(0.0, 130.0)
    gc.validate_d_halo(100.0, 130.0)   # min(100, W+16=146) = 100
    _pass("test_validate_d_halo_valid")

def test_validate_d_halo_invalid():
    try:
        gc.validate_d_halo(100.1, 130.0)   # min(100, W+16=146)=100, so 100.1 is invalid
        _fail("test_validate_d_halo_invalid", "should have raised")
    except ValueError:
        pass
    try:
        gc.validate_d_halo(-1.0, 130.0)
        _fail("test_validate_d_halo_invalid", "should have raised")
    except ValueError:
        pass
    _pass("test_validate_d_halo_invalid")

def test_calibrate_d_halo_max_always_100_in_W_range():
    """For W in [120,140], min(100, W+16) always equals exactly 100."""
    for W_mm in (120.0, 130.0, 140.0):
        assert gc.calibrate_d_halo_max_mm(W_mm) == 100.0, (
            f"W={W_mm}: expected d_halo max=100.0, got {gc.calibrate_d_halo_max_mm(W_mm)}"
        )
    _pass("test_calibrate_d_halo_max_always_100_in_W_range")

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
    for name in ("nose", "sidepod", "rearpod", "main_body"):
        assert name in gc.TOOL_DIRECTIONS, f"Missing tool directions for {name}"
        dirs = gc.TOOL_DIRECTIONS[name]
        assert len(dirs) >= 2, f"{name} has only {len(dirs)} tool directions"
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

if __name__ == "__main__":
    test_co2_mass_matches_part2_constant()
    test_wheel_constants_match_locked_race_objective()
    test_nose_density_is_1000()
    test_nose_density_is_6x_sidepod()
    test_all_machined_densities_present()
    test_get_density_unknown_raises()
    test_mm_to_m_round_trip()
    test_gcm3_to_kgm3()
    test_grid_cells_minimum_one()
    test_W_bounds()
    test_validate_W_valid()
    test_validate_W_invalid()
    test_validate_d_halo_valid()
    test_validate_d_halo_invalid()
    test_calibrate_d_halo_max_always_100_in_W_range()
    test_halo_z_min()
    test_lifecycle_states_count()
    test_lifecycle_states_exact_names()
    test_tool_directions_all_components()
    test_tool_directions_unit_vectors()
    test_phi_snapshot_keys()
    test_grid_spacing_consistency()
    test_min_radius_consistency()
    print("\nAll geometry_contract tests passed.")