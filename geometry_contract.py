"""
geometry_contract.py --- Part 1 single source of truth.

Every constant, unit helper, and density value lives here.
No other file in Part 1 re-derives any of these.
No imports from Part 2 — constants are duplicated deliberately
to avoid circular dependency. Cross-check tests verify they match.

Coordinate system (matches Part 2 physics_contract.py exactly):
    x = front to rear   (x=0 = nose tip, car extends in +x direction)
    y = centerline to right side   (y=0 = symmetry plane)
    z = track upward    (z=0 = track surface)

Key x positions (all derived from outer-loop scalars W and x_front):
    nose tip         = 0
    Ref Plane A      = x_front_m - 0.016   (16 mm ahead of front axle, T1.17)
    front axle       = x_front_m
    rear axle        = x_front_m + W_m
    Ref Plane B      = x_front_m + W_m + 0.016
    halo centre (x)  = Ref_A + d_halo_m

Units: all internal values are SI (m, kg, N, kg/m^3).
S1 accepts mm inputs via unit helpers and converts.
"""

from __future__ import annotations
import math
from typing import Optional

# ── Coordinate system labels ───────────────────────────────────────────────
AXIS_X: str = "front_to_rear"
AXIS_Y: str = "centerline_to_right"
AXIS_Z: str = "track_upward"

# ── Outer loop bounds ──────────────────────────────────────────────────────
W_MIN_MM: float = 120.0
W_MAX_MM: float = 140.0

# x_front: front axle position from nose tip (mm).
#
# The nose occupies x in [0, Ref Plane A] = [0, x_front - 16], so
#     nose_overhang_mm == x_front - 16.
#
# Max: T8.2 (regs p35, verified 2026-07-20) -- "From the Reference plane A the
#      nose cone overhang is 40mm maximum", measured to the extreme front of
#      the fully assembled car. So x_front - 16 <= 40  ->  x_front <= 56 mm.
#      Also bounded by the model block: x_front + W + 16 <= 223 -> x_front <= 207 - W,
#      which at W=140 gives 67 mm and is therefore never the binding constraint.
#
# Min: a design choice, NOT a regulation. The regs set no minimum nose length.
#      NOSE_OVERHANG_MIN_MM below is a build-practicality floor.
#
# HISTORY (fixed 2026-07-20): this bound was previously X_FRONT_MIN_MM = 61.0,
# derived from "nose section must fit CO2 cartridge depth (T5.3: 45 mm) forward
# of Ref Plane A". That premise is false -- the cartridge chamber is REAR of
# Ref Plane A, not in the nose. Regs p22: "A single continuous piece of CNC
# manufactured Model Block material must exist rear of the reference plane A,
# encompassing both the virtual cargo and power unit cartridge chamber."
# fixed_hardware.compute_default_fixed_hardware_inputs has always placed the
# canister at the rear, so the two disagreed.
#
# The old bound was not merely redundant, it was ILLEGAL: it forced
# nose_overhang >= 45 mm against T8.2's 40 mm maximum, so *every* car the
# search could propose violated T8.2 by at least 5 mm (19 mm at the commonly
# used x_front=75). The feasible x_front interval under T8.2 is [36, 56]; the
# old interval [61, min(90, 207-W)] does not intersect it at all.
NOSE_OVERHANG_MAX_MM: float = 40.0   # T8.2, hard regulation
NOSE_OVERHANG_MIN_MM: float = 20.0   # design choice -- no regulation minimum exists
REF_PLANE_A_OFFSET_MM: float = 16.0  # T1.17: Ref Plane A is 16 mm ahead of front axle

X_FRONT_MIN_MM: float = REF_PLANE_A_OFFSET_MM + NOSE_OVERHANG_MIN_MM   # 36.0
X_FRONT_ABS_MAX_MM: float = REF_PLANE_A_OFFSET_MM + NOSE_OVERHANG_MAX_MM  # 56.0 (T8.2)

# d_halo upper bound = W + 16.0 mm (computed per W, not a fixed constant)
# d_halo lower bound: 0.0 (halo at Ref Plane A; nose is degenerate but not an error)

# ── Grid ───────────────────────────────────────────────────────────────────
GRID_SPACING_MM: float = 0.3     # 0.3 mm spacing, Nyquist×10 on 3.15 mm min radius
GRID_SPACING_M:  float = GRID_SPACING_MM / 1000.0

# ── Machining ──────────────────────────────────────────────────────────────
# Applies to the three MILLED components (sidepod, rearpod, main_body) only.
# The nose is 3D printed (user-confirmed 2026-07-14), not CNC-milled -- it has
# no minimum machining radius or tool-direction constraint at all. Instead it
# has a minimum WALL THICKNESS (a 3D-printing shell constraint, since the nose
# is allowed to be hollow) -- see NOSE_MIN_WALL_THICKNESS_MM below.
MIN_RADIUS_MM: float = 3.15      # minimum machining radius — hard floor (milled components only)
MIN_RADIUS_M:  float = MIN_RADIUS_MM / 1000.0

# 3D-printing shell constraint for the nose only. The nose may be hollow
# (user-confirmed) but any solid wall must be at least this thick everywhere
# to print reliably.
NOSE_MIN_WALL_THICKNESS_MM: float = 2.0
NOSE_MIN_WALL_THICKNESS_M:  float = NOSE_MIN_WALL_THICKNESS_MM / 1000.0

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

# ── Wheel disc lateral geometry (measured from provided CAD: "front/rear
# wheel itself.stl" + "Front/Rear Wheel Support.stl", see hardware_geometry.py)
# ────────────────────────────────────────────────────────────────────────────
# Fixes sandbox/README.md finding #12: the old wheel exclusion zone was a
# ForbiddenCylinder aligned along x and centred on y=0 (the centreline) --
# the wrong axis (a wheel spins about y, its circular face is in the x-z
# plane) AND nowhere near where the wheels actually sit (y=19-29mm / front,
# y=16-24mm / rear). Measured, ~50% of the front wheel and ~58% of the rear
# wheel ended up inside solid bodywork as a result.
WHEEL_WIDTH_MM: float = 17.25              # measured, front and rear wheels identical
WHEEL_WIDTH_M:  float = WHEEL_WIDTH_MM / 1000.0

# Inner (track-contact) face y-offset from centreline. Both measured design
# choices from the CAD -- comfortably above T7.2's legal MINIMUM half-gaps
# (front 19.0mm, rear 15.0mm); the regs set a floor, not an exact value.
FRONT_WHEEL_INNER_Y_MM: float = 19.25      # >= T7.2.1 min half-gap 19.0mm
REAR_WHEEL_INNER_Y_MM:  float = 16.25      # >= T7.2.2 min half-gap 15.0mm
FRONT_WHEEL_INNER_Y_M:  float = FRONT_WHEEL_INNER_Y_MM / 1000.0
REAR_WHEEL_INNER_Y_M:   float = REAR_WHEEL_INNER_Y_MM / 1000.0

# Fore-aft (x) clearance radius for both the sidepod corridor boundary and
# the wheel-disc void mask. A disc's x-extent at any y within its width is
# its full diameter, so this must derive from the wheel's actual radius
# (R_WHEEL_M) plus clearance -- not the old separate, arbitrary 8mm
# "axle hub half-width" guess, which was less than half the size actually
# needed (30mm-diameter wheel needs >=15mm of radius clearance, not 8mm).
WHEEL_X_CLEARANCE_HALF_WIDTH_M:  float = R_WHEEL_M + WHEEL_CLEARANCE_M
WHEEL_X_CLEARANCE_HALF_WIDTH_MM: float = WHEEL_X_CLEARANCE_HALF_WIDTH_M * 1000.0

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
# No entry for "nose": it is 3D printed (user-confirmed 2026-07-14), not
# CNC-milled, so it has no directional tool-access constraint at all.
# surface_extraction._check_accessibility already treats a missing/empty
# entry as "no constraint" (returns 0.0 inaccessible area).
# The block is machined from the TOP, the BOTTOM, and the two SIDES, and from
# nowhere else (project owner, 2026-08-06). So the tool set is +-Z and +-Y for
# every milled component, and there is no +-X access at all.
#
# Every entry here was wrong, in both directions:
#
#   sidepod   [+Y, -X, +Z]        -X is a cut from the front. Not available.
#   rearpod   [+X, +Z, +Y, -Y]    +X is a cut from the rear. Not available.
#   main_body [+Z, +Y, -Y]        no -Z: the bottom was never offered.
#
# rearpod's +X is the consequential one. It made every surface reachable by
# looking straight up the car's axis "accessible", which is what licensed the
# shroud of material standing around the CO2 cartridge -- geometry you can only
# get a tool to from behind, i.e. you cannot. And the missing -Z made the gate
# reject bodywork that a simple flip of the block reaches, so it was
# simultaneously too permissive about the impossible and too strict about the
# routine.
#
# Kept per-component rather than collapsed to one list: the labels also select
# density and the gate dispatches per face, so a future component that really
# does have different access (a printed part, say) still has somewhere to say
# so. They just all happen to agree today.
_MILL_TOP_BOTTOM_SIDES: list[tuple[float, float, float]] = [
    ( 0.0,  0.0,  1.0),      # top
    ( 0.0,  0.0, -1.0),      # bottom
    ( 0.0,  1.0,  0.0),      # side +y
    ( 0.0, -1.0,  0.0),      # side -y
]

TOOL_DIRECTIONS: dict[str, list[tuple[float, float, float]]] = {
    "sidepod":   list(_MILL_TOP_BOTTOM_SIDES),
    "rearpod":   list(_MILL_TOP_BOTTOM_SIDES),
    "main_body": list(_MILL_TOP_BOTTOM_SIDES),
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

def calibrate_x_front_bounds(W_mm: float) -> tuple[float, float]:
    """
    Return (x_front_min_mm, x_front_max_mm) for the given wheelbase.

    Min: 36 mm — NOSE_OVERHANG_MIN_MM (20 mm, design choice) + Ref Plane A offset.
    Max: min(56, 207 - W) mm — 56 is T8.2's 40 mm nose overhang ceiling plus the
         Ref Plane A offset; 207 - W keeps Ref Plane B inside the 223 mm model
         block. At W=140 the block bound gives 67 mm, so T8.2 always binds and
         the interval is [36, 56] for every legal W.

    See the X_FRONT_MIN_MM block above for why the old 61 mm floor was wrong
    (and illegal under T8.2).
    """
    x_min = X_FRONT_MIN_MM
    x_max = min(X_FRONT_ABS_MAX_MM, 207.0 - W_mm)
    x_max = max(x_max, x_min + 1.0)   # always at least 1 mm of search range
    return x_min, x_max

def validate_x_front(x_front_mm: float, W_mm: float) -> None:
    """Raise ValueError if x_front is outside its W-dependent bounds."""
    x_min, x_max = calibrate_x_front_bounds(W_mm)
    if not (x_min <= x_front_mm <= x_max):
        raise ValueError(
            f"x_front={x_front_mm} mm is outside allowed range "
            f"[{x_min}, {x_max}] mm for W={W_mm} mm."
        )

# d_halo upper bound is derived from the halo pocket placement constraint:
#   pocket_rear = (x_front - REF_A_OFFSET) + d_halo + POCKET_LENGTH
#              must be < rear_axle = x_front + W
#   → d_halo < W - (POCKET_LENGTH - REF_A_OFFSET) = W - (50 - 16) = W - 34
# Constants match halo_pocket.HALO_POCKET_LENGTH_MM and the 16mm Ref Plane A offset.
D_HALO_POCKET_LENGTH_MM: float = 50.0   # T4.4.4 Appendix ix
D_HALO_REF_A_OFFSET_MM: float = 16.0   # Ref Plane A is 16mm ahead of front axle
# Strict exclusive upper bound: d_halo < W - 34.  The placement check in
# fixed_hardware._validate_halo_position uses ">=" so the limit is strict.
_D_HALO_PLACEMENT_MARGIN_MM: float = D_HALO_POCKET_LENGTH_MM - D_HALO_REF_A_OFFSET_MM  # 34 mm

# --- d_halo travel limits, physical (project owner, 2026-07-24) ------------
# d_halo is measured from Ref Plane A to the halo pocket's FRONT edge. Its
# travel is bounded by where the halo can physically go:
#
#   FORWARD-most : pocket front edge on the FRONT AXLE LINE.
#                  pocket_front = x_front, and pocket_front = ref_A + d_halo
#                  with ref_A = x_front - 16, so d_halo = 16 mm. Exactly.
#                  Independent of W. Replaces the old floor of 0, which put the
#                  halo up to 16 mm AHEAD of the front axle.
#   REARWARD-most : pocket REAR edge touching the CO2 canister.
#                  d_halo = (canister_front_x - 50) - ref_A.
#
# ⚠ The rearward bound is PROVISIONAL. It currently references the cartridge
# BORE front face (rear_face - bore_depth). The halo is specified to loft to a
# minimum-SAFETY-ZONE block around the canister, and that block does not exist
# in the model yet -- once it does, point D_HALO_REAR_REFERENCE at its front
# face. The safety zone is forward of the bore front, so this bound will get
# SMALLER, never larger; nothing built against it becomes illegal.
D_HALO_MIN_MM: float = D_HALO_REF_A_OFFSET_MM   # 16.0 — pocket front on front axle


def calibrate_d_halo_min_mm() -> float:
    """Forward-most halo position: pocket front edge on the front axle line."""
    return D_HALO_MIN_MM


def calibrate_d_halo_max_mm(W_mm: float, canister_front_x_mm: Optional[float] = None,
                            x_front_mm: Optional[float] = None) -> float:
    """
    Return the exclusive d_halo upper bound.

    With canister_front_x_mm and x_front_mm supplied, this is the PHYSICAL
    limit the project owner specified: the pocket's rear edge touching the
    canister (its minimum safety zone once that exists) --
        d_halo_max = (canister_front_x - POCKET_LENGTH) - ref_A
                   = canister_front_x - 50 - (x_front - 16)

    Without them it falls back to the older placement-derived bound
    `W - 34` (pocket rear must not reach the rear axle), which every existing
    caller and test still uses. Both are returned as STRICT upper bounds.
    The effective limit is the MINIMUM of the two -- the halo may neither reach
    the canister nor the rear axle.
    """
    axle_bound = W_mm - _D_HALO_PLACEMENT_MARGIN_MM
    if canister_front_x_mm is None or x_front_mm is None:
        return axle_bound
    ref_A_mm = x_front_mm - D_HALO_REF_A_OFFSET_MM
    canister_bound = canister_front_x_mm - D_HALO_POCKET_LENGTH_MM - ref_A_mm
    return min(axle_bound, canister_bound)


def validate_d_halo(d_halo_mm: float, W_mm: float) -> None:
    """Raise ValueError if d_halo is outside [16, W-34) mm (strict upper bound).

    Lower bound is the forward-most physical position (pocket front on the
    front axle line), not 0 -- see D_HALO_MIN_MM.
    """
    d_max = calibrate_d_halo_max_mm(W_mm)
    if not (D_HALO_MIN_MM <= d_halo_mm < d_max):
        raise ValueError(
            f"d_halo={d_halo_mm} mm is outside allowed range "
            f"[{D_HALO_MIN_MM}, {d_max:.1f}) mm for W={W_mm} mm. "
            f"Lower bound {D_HALO_MIN_MM} mm = halo pocket front edge on the front "
            f"axle line (forward-most physical position). "
            f"Upper bound is W-34 mm (pocket rear must not reach the rear axle); "
            f"the canister/safety-zone limit applies on top of it."
        )
