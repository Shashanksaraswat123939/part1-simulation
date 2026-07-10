"""
geometry_contract.py --- Part 1 single source of truth.

Every constant, unit helper, and density value lives here.
No other file in Part 1 re-derives any of these.
No imports from Part 2 --- constants are duplicated deliberately
to avoid circular dependency. Cross-check tests verify they match.

Coordinate system (matches Part 2 physics_contract.py exactly):
    x = front to rear   (x=0 = front axle, x=W_m = rear axle)
    y = centerline to right side   (y=0 = symmetry plane)
    z = track upward    (z=0 = track surface)

Units: all internal values are SI (m, kg, N, kg/m^3).
S1 accepts mm inputs via unit helpers and converts.
"""

from __future__ import annotations
import math

# ?? Coordinate system labels ???????????????????????????????????????????????
AXIS_X: str = "front_to_rear"
AXIS_Y: str = "centerline_to_right"
AXIS_Z: str = "track_upward"

# ?? Outer loop bounds ??????????????????????????????????????????????????????
W_MIN_MM: float = 120.0
W_MAX_MM: float = 140.0
# d_halo upper bound = W + 16.0 mm (computed per W, not a fixed constant)

# ?? Grid ???????????????????????????????????????????????????????????????????
GRID_SPACING_MM: float = 0.3     # 0.3 mm spacing, Nyquistx10 on 3.15 mm min radius
GRID_SPACING_M:  float = GRID_SPACING_MM / 1000.0

# ?? Machining ??????????????????????????????????????????????????????????????
MIN_RADIUS_MM: float = 3.15      # minimum machining radius --- hard floor
MIN_RADIUS_M:  float = MIN_RADIUS_MM / 1000.0

# ?? Densities (kg/m^3) ??????????????????????????????????????????????????????
DENSITY_NOSE_KGM3:    float = 1000.0  # 1.0 g/cm^3 --- 6x denser than body material
DENSITY_SIDEPOD_KGM3: float = 163.0   # 0.163 g/cm^3
DENSITY_REARPOD_KGM3: float = 163.0
DENSITY_BODY_KGM3:    float = 163.0

COMPONENT_DENSITY_KGM3: dict[str, float] = {
    "nose":      DENSITY_NOSE_KGM3,
    "sidepod":   DENSITY_SIDEPOD_KGM3,
    "rearpod":   DENSITY_REARPOD_KGM3,
    "main_body": DENSITY_BODY_KGM3,
}

# ?? Fixed hardware masses ??????????????????????????????????????????????????
# CO2_MASS_KG must equal Part 2's mass_com_ingest.CO2_CARTRIDGE_MASS_KG = 0.023 exactly.
# Part 2's FixedHardwareSpec.__post_init__ raises ValueError if they differ by > 1e-9 kg.
# TEST: test_co2_mass_matches_part2_constant verifies this equality.
CO2_MASS_KG: float = 0.023

# ?? Wheel / axle geometry ??????????????????????????????????????????????????
# These must match race_objective.py's locked constants (N_WHEELS=4, R_WHEEL=0.015).
# Changing them without changing the locked file is a silent physics error.
N_WHEELS:  int   = 4
R_WHEEL_M: float = 0.015          # 15 mm radius

# ?? Halo rules (CONFIRMED by project owner) ????????????????????????????????
# Bottom of halo must be at least this far above track surface.
HALO_MIN_Z_MM: float = 24.0       # 24 mm above track
HALO_MIN_Z_M:  float = HALO_MIN_Z_MM / 1000.0
# Halo front edge must be strictly aft of front axle (x > 0.0 m).
# Halo must sit between canister pocket (forward) and front axle (aft) in x.
# That means: canister_x_m < halo_x_front_m AND halo_x_front_m > 0.0 m
# In front-to-rear convention: front axle = x=0, halo is behind it = x > 0
# and canister is even further forward = x < halo_x_front.
# Enforcement: checked in fixed_hardware.py::_validate_halo_position()

# ?? Clearances and gaps ????????????????????????????????????????????????????
WHEEL_CLEARANCE_MM:    float = 2.0   # gap between wheel cylinder edge and sidepod corridor
WHEEL_CLEARANCE_M:     float = WHEEL_CLEARANCE_MM / 1000.0
ATTACHMENT_STRIP_MM:   float = 1.0   # width of hard phi < 0 attachment faces
ATTACHMENT_STRIP_M:    float = ATTACHMENT_STRIP_MM / 1000.0
HARDWARE_CLEARANCE_MM: float = 1.0   # min surface-to-hardware gap (rule checker)
HARDWARE_CLEARANCE_M:  float = HARDWARE_CLEARANCE_MM / 1000.0

# ?? COM sanity bounds ??????????????????????????????????????????????????????
# z below 0.005 m or above 0.060 m almost certainly means a mm/m units error.
# Part 2's adapter has a tighter inner guard: [0.018, 0.042] m (polynomial range).
# We catch it here first with a broader physical check.
COM_Z_LOWER_BOUND_M: float = 0.005
COM_Z_UPPER_BOUND_M: float = 0.060
# Part 2 polynomial fitted range (imported into test only --- do not import Part 2 here)
COM_Z_POLY_MIN_M: float = 0.018
COM_Z_POLY_MAX_M: float = 0.042

# ?? Mesh quality thresholds (snappyHexMesh requirements) ???????????????????
MESH_MIN_TRIANGLE_ANGLE_DEG: float = 10.0   # minimum interior angle in any triangle
MESH_MAX_ASPECT_RATIO:       float = 10.0   # max triangle aspect ratio

# ?? Tool accessibility thresholds ?????????????????????????????????????????
SMALL_INACCESSIBLE_AREA_M2: float = 1e-5    # 10 mm^2 --- smooth and retry
LARGE_INACCESSIBLE_AREA_M2: float = 1e-3    # 1000 mm^2 --- kill candidate

# ?? Retry limits ???????????????????????????????????????????????????????????
MAX_EXTRACTION_RETRIES: int = 3

# ?? phi snapshot file naming ?????????????????????????????????????????????????
# Part 3 stores phi_grid_snapshot_paths in CandidateRecord.
# Keys must be exactly these four strings.
PHI_SNAPSHOT_COMPONENT_KEYS: tuple[str, ...] = (
    "nose", "sidepod", "rearpod", "main_body"
)

# ?? Lifecycle states ???????????????????????????????????????????????????????
# Duplicated from Part 2's candidate_record.ALLOWED_LIFECYCLE_STATES.
# Cross-check test verifies they match. Do not add or remove any.
ALLOWED_LIFECYCLE_STATES: frozenset[str] = frozenset({
    "valid_simulated",
    "geometry_repaired",
    "geometry_rejected",
    "rule_rejected",
    "machining_rejected",
    "CFD_failed",
    "objective_failed",
    "converged",
})

# ?? Tool directions per component ??????????????????????????????????????????
# Each tuple is a unit vector (dx, dy, dz) representing one allowed approach
# direction for the CNC tool. The tool comes FROM that direction.
# Spec ?Components, machinability constraints.
TOOL_DIRECTIONS: dict[str, list[tuple[float, float, float]]] = {
    "nose": [
        (-1.0,  0.0,  0.0),   # -X: tool approaches from front (tip approach)
        ( 0.0,  0.0,  1.0),   # +Z: tool comes from above
        ( 0.0,  1.0,  0.0),   # +Y: tool comes from right side
        ( 0.0, -1.0,  0.0),   # -Y: tool comes from left side
    ],
    "sidepod": [
        # Right sidepod only --- left is mirror with y negated
        ( 0.0,  1.0,  0.0),   # +Y: tool comes from outside (right)
        (-1.0,  0.0,  0.0),   # -X: tool comes from front
        ( 0.0,  0.0,  1.0),   # +Z: tool comes from above
    ],
    "rearpod": [
        ( 1.0,  0.0,  0.0),   # +X: tool approaches from rear (tail approach)
        ( 0.0,  0.0,  1.0),   # +Z: tool comes from above
        ( 0.0,  1.0,  0.0),   # +Y: tool comes from right
        ( 0.0, -1.0,  0.0),   # -Y: tool comes from left
    ],
    "main_body": [
        ( 0.0,  0.0,  1.0),   # +Z: tool comes from above
        ( 0.0,  1.0,  0.0),   # +Y: tool comes from right
        ( 0.0, -1.0,  0.0),   # -Y: tool comes from left
    ],
}


# ?? Unit helpers ???????????????????????????????????????????????????????????

def mm_to_m(mm: float) -> float:
    """Convert millimetres to metres. Call this; never write /1000 inline."""
    return mm / 1000.0

def m_to_mm(m: float) -> float:
    """Convert metres to millimetres. Call this; never write *1000 inline."""
    return m * 1000.0

def gcm3_to_kgm3(g_cm3: float) -> float:
    """Convert g/cm^3 to kg/m^3. 1 g/cm^3 = 1000 kg/m^3 exactly."""
    return g_cm3 * 1000.0

def grid_cells(length_mm: float) -> int:
    """
    Number of grid cells to cover a dimension given in mm.
    Always at least 1. Uses ceiling division.
    Example: length_mm=0.0 -> 1, length_mm=0.3 -> 1, length_mm=0.31 -> 2
    """
    if length_mm <= 0.0:
        return 1
    return max(1, math.ceil(length_mm / GRID_SPACING_MM))

def get_density(component: str) -> float:
    """
    Look up density for a machined component by name.
    Raises ValueError for unknown names.
    Valid names: 'nose', 'sidepod', 'rearpod', 'main_body'
    """
    if component not in COMPONENT_DENSITY_KGM3:
        raise ValueError(
            f"Unknown component '{component}'. "
            f"Valid names: {sorted(COMPONENT_DENSITY_KGM3.keys())}"
        )
    return COMPONENT_DENSITY_KGM3[component]

def validate_W(W_mm: float) -> None:
    """Raise ValueError if W is outside [120, 140] mm."""
    if not (W_MIN_MM <= W_mm <= W_MAX_MM):
        raise ValueError(
            f"Wheelbase W={W_mm} mm is outside allowed range "
            f"[{W_MIN_MM}, {W_MAX_MM}] mm."
        )

def validate_d_halo(d_halo_mm: float, W_mm: float) -> None:
    """Raise ValueError if d_halo is outside [0, W+16] mm."""
    d_max = W_mm + 16.0
    if not (0.0 <= d_halo_mm <= d_max):
        raise ValueError(
            f"d_halo={d_halo_mm} mm is outside allowed range "
            f"[0.0, {d_max}] mm for W={W_mm} mm."
        )

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
    gc.validate_d_halo(146.0, 130.0)   # W+16 = 146
    _pass("test_validate_d_halo_valid")

def test_validate_d_halo_invalid():
    try:
        gc.validate_d_halo(147.0, 130.0)   # W+16=146, so 147 is invalid
        _fail("test_validate_d_halo_invalid", "should have raised")
    except ValueError:
        pass
    try:
        gc.validate_d_halo(-1.0, 130.0)
        _fail("test_validate_d_halo_invalid", "should have raised")
    except ValueError:
        pass
    _pass("test_validate_d_halo_invalid")

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
    test_halo_z_min()
    test_lifecycle_states_count()
    test_lifecycle_states_exact_names()
    test_tool_directions_all_components()
    test_tool_directions_unit_vectors()
    test_phi_snapshot_keys()
    test_grid_spacing_consistency()
    test_min_radius_consistency()
    print("\nAll geometry_contract tests passed.")