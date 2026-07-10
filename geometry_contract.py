"""
geometry_contract.py --- Part 1 single source of truth.

Every constant, unit helper, and density value lives here.
No other file in Part 1 re-derives any of these.
No imports from Part 2 — constants are duplicated deliberately
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

# ── Coordinate system labels ───────────────────────────────────────────────
AXIS_X: str = "front_to_rear"
AXIS_Y: str = "centerline_to_right"
AXIS_Z: str = "track_upward"

# ── Outer loop bounds ──────────────────────────────────────────────────────
W_MIN_MM: float = 120.0
W_MAX_MM: float = 140.0
# d_halo upper bound = W + 16.0 mm (computed per W, not a fixed constant)

# ── Grid ───────────────────────────────────────────────────────────────────
GRID_SPACING_MM: float = 0.3     # 0.3 mm spacing, Nyquist×10 on 3.15 mm min radius
GRID_SPACING_M:  float = GRID_SPACING_MM / 1000.0

# ── Machining ──────────────────────────────────────────────────────────────
MIN_RADIUS_MM: float = 3.15      # minimum machining radius — hard floor
MIN_RADIUS_M:  float = MIN_RADIUS_MM / 1000.0

# ── Densities (kg/m^3) ────────────────────────────────────────────────────
DENSITY_NOSE_KGM3:    float = 1000.0  # 1.0 g/cm^3 — 6x denser than body material
DENSITY_SIDEPOD_KGM3: float = 163.0   # 0.163 g/cm^3
DENSITY_REARPOD_KGM3: float = 163.0
DENSITY_BODY_KGM3:    float = 163.0

COMPONENT_DENSITY_KGM3: dict[str, float] = {
    "nose":      DENSITY_NOSE_KGM3,
    "sidepod":   DENSITY_SIDEPOD_KGM3,
    "rearpod":   DENSITY_REARPOD_KGM3,
    "main_body": DENSITY_BODY_KGM3,
}

# ── Fixed hardware masses ──────────────────────────────────────────────────
# CO2_MASS_KG must equal Part 2's mass_com_ingest.CO2_CARTRIDGE_MASS_KG = 0.023 exactly.
# Part 2's FixedHardwareSpec.__post_init__ raises ValueError if they differ by > 1e-9 kg.
# TEST: test_co2_mass_matches_part2_constant verifies this equality.
CO2_MASS_KG: float = 0.023

# ── Wheel / axle geometry ──────────────────────────────────────────────────
# These must match race_objective.py's locked constants (N_WHEELS=4, R_WHEEL=0.015).
N_WHEELS:  int   = 4
R_WHEEL_M: float = 0.015          # 15 mm radius

# ── Halo rules (CONFIRMED by project owner) ────────────────────────────────
HALO_MIN_Z_MM: float = 24.0       # 24 mm above track — confirmed rule
HALO_MIN_Z_M:  float = HALO_MIN_Z_MM / 1000.0
# Halo front edge must be strictly aft of front axle (x > 0.0 m).
# Halo must sit between canister pocket and front axle in x.
# Enforcement: checked in fixed_hardware.py::_validate_halo_position()

# ── Clearances and gaps ────────────────────────────────────────────────────
WHEEL_CLEARANCE_MM:    float = 2.0
WHEEL_CLEARANCE_M:     float = WHEEL_CLEARANCE_MM / 1000.0
ATTACHMENT_STRIP_MM:   float = 1.0
ATTACHMENT_STRIP_M:    float = ATTACHMENT_STRIP_MM / 1000.0
HARDWARE_CLEARANCE_MM: float = 1.0
HARDWARE_CLEARANCE_M:  float = HARDWARE_CLEARANCE_MM / 1000.0

# ── COM sanity bounds ──────────────────────────────────────────────────────
COM_Z_LOWER_BOUND_M: float = 0.005
COM_Z_UPPER_BOUND_M: float = 0.060
COM_Z_POLY_MIN_M:    float = 0.018   # Part 2 polynomial range lower
COM_Z_POLY_MAX_M:    float = 0.042   # Part 2 polynomial range upper

# ── Mesh quality thresholds (snappyHexMesh requirements) ──────────────────
MESH_MIN_TRIANGLE_ANGLE_DEG: float = 10.0
MESH_MAX_ASPECT_RATIO:       float = 10.0

# ── Tool accessibility thresholds ─────────────────────────────────────────
SMALL_INACCESSIBLE_AREA_M2: float = 1e-5    # 10 mm^2 — smooth and retry
LARGE_INACCESSIBLE_AREA_M2: float = 1e-3    # 1000 mm^2 — kill candidate

# ── Retry limits ───────────────────────────────────────────────────────────
MAX_EXTRACTION_RETRIES: int = 3

# ── Phi snapshot file naming ───────────────────────────────────────────────
PHI_SNAPSHOT_COMPONENT_KEYS: tuple[str, ...] = (
    "nose", "sidepod", "rearpod", "main_body"
)

# ── Lifecycle states ───────────────────────────────────────────────────────
# Duplicated from Part 2's candidate_record.ALLOWED_LIFECYCLE_STATES.
# Cross-check test (test_integration_part1_part2.py) verifies they match.
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

# ── Tool directions per component ──────────────────────────────────────────
TOOL_DIRECTIONS: dict[str, list[tuple[float, float, float]]] = {
    "nose": [
        (-1.0,  0.0,  0.0),
        ( 0.0,  0.0,  1.0),
        ( 0.0,  1.0,  0.0),
        ( 0.0, -1.0,  0.0),
    ],
    "sidepod": [
        ( 0.0,  1.0,  0.0),
        (-1.0,  0.0,  0.0),
        ( 0.0,  0.0,  1.0),
    ],
    "rearpod": [
        ( 1.0,  0.0,  0.0),
        ( 0.0,  0.0,  1.0),
        ( 0.0,  1.0,  0.0),
        ( 0.0, -1.0,  0.0),
    ],
    "main_body": [
        ( 0.0,  0.0,  1.0),
        ( 0.0,  1.0,  0.0),
        ( 0.0, -1.0,  0.0),
    ],
}


# ── Unit helpers ───────────────────────────────────────────────────────────

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
    """
    if length_mm <= 0.0:
        return 1
    return max(1, math.ceil(length_mm / GRID_SPACING_MM))

def get_density(component: str) -> float:
    """Look up density (kg/m^3) for a machined component. Raises ValueError for unknown names."""
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
