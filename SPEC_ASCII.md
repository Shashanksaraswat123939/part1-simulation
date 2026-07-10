# Part 1: Generative Geometry Designer --- Exhaustive Build Specification

**Version:** Final  
**Governing specs:** `01_generative_geometry_1_.md`, `02_simulation_setups_1_.md`, `03_optimizer_workflow_1_.md`  
**Part 2 status:** Fully implemented, 82 tests pass. Every type, function, and constant referenced below is real and locked.  
**Part 3 status:** Spec read. All interface points derived from it.  
**CFD symmetry:** CONFIRMED half-car (right side only). Adjoint sensitivity comes from right-half domain. U3 is RESOLVED.  
**Halo rules:** CONFIRMED. Bottom >= 24 mm above track. Front edge must be aft of front axle (x > 0). Halo must sit between canister pocket and front axle in x (canister_x < halo_x_front < 0... wait --- front axle is at x=0 and halo is BEHIND front axle meaning halo_x_front > 0 in front-to-rear convention). See Section 8 for full derivation.

**How to use this document:** Every section is complete and self-contained. The coding model reads each section and writes exactly what is described. No design decisions are left to the coder. If something says "write this code", write exactly that code. If something says "raise NotImplementedError", raise exactly that. Do not invent anything not written here.

---

## 0. Non-Negotiable Ground Rules

These apply to every single line of code in Part 1. Violating any of them produces silent, catastrophic bugs.

### Rule 1 --- Coordinate system
```
x = front to rear        (x=0 is front axle, x=W is rear axle, x increases going backward)
y = centerline to right  (y=0 is car centerline, y>0 is right side, y<0 is left side)
z = track upward         (z=0 is track surface, z increases going up)
```
This matches Part 2's `physics_contract.py` docstring **word for word**. A sign flip anywhere silently gives wrong COM, wrong drag direction, wrong adjoint gradient.

### Rule 2 --- Units
- **Inside Part 1 (S2 onward): everything is metres, kilograms, newtons, kg/m^3**
- S1 defines conversion helpers. Call them. Never write `/ 1000` or `* 1000` anywhere except in S1's helpers.
- Part 2 enforces SI. It will silently produce wrong results if you pass mm where m is expected.

### Rule 3 --- No circular imports
- Part 1 imports from Part 2 only at two points: S5 uses `ComponentMassCOM`, S6 uses `FixedHardwareSpec`.
- Part 2 never imports from Part 1.
- Within Part 1: S1 imports nothing. S6 imports from S1. S2 imports from S1+S6. S3 imports from S1+S2+S6. S4 imports from S1+S3. S5 imports from S1+S3. S7 imports from S4. S9 imports from S4+S7. S8 imports from S1+S3.

### Rule 4 --- Hard constraints are sacred
Hard masks in every phi grid are enforced by calling `apply_hard_constraints()`:
- After `__init__` / `init()`
- After every `hj_update()` (inside `phi_updater.py`)
- After every `reinitialise_sdf()` (inside `phi_updater.py`)  
- After every `remap()` (inside `phi_grid.py`)
- After every `load()` (inside `phi_grid.py`)

**Never skip this call.** The optimizer will violate every physical constraint if you do.

### Rule 5 --- Arbitrary region shapes
Bounding regions are NOT simple axis-aligned boxes. They are defined by a list of half-space constraints (planes) or a voxel mask. The system must support any convex or non-convex polygon cross-section extruded in one direction, and any voxel-painted 3D region. See Section 9 (BoundingRegion class).

### Rule 6 --- Failed candidates keep their phi snapshots
`run_quality_gates()` saves phi snapshots **before** attempting surface extraction. Even if a candidate is killed in the first stage, its phi snapshot exists on disk. Part 3's evolutionary outer loop reads these to avoid re-exploring bad regions.

### Rule 7 --- Test runner is reliable
Use `Path(__file__).resolve().parent / "tests"`. Check the directory exists. Check files exist. Surface stderr. Exit code 2 on runner infrastructure failure, 1 on test failures, 0 on all pass. Never silently eat errors. (Part 2 audit finding D1 showed how a broken runner reported "0 passed, 0 failed" with no error --- do not repeat this.)

### Rule 8 --- Every placeholder is loud
If a value is not yet known (halo cross-section, canister coordinate, UAE envelope dimensions, rear wing position), the code raises `NotImplementedError` with a message beginning `? UNRESOLVED:` and explaining exactly what is needed and where to put it. The comment above the raise must also start with `# ? UNRESOLVED`.

---

## 1. Repository Layout

Exact directory structure. Create every file listed. Do not add extra files.

```
part1_geometry/
??? geometry_contract.py          S1
??? fixed_hardware.py             S6
??? bounding_volumes.py           S2
??? phi_grid.py                   S3
??? mass_com_calculator.py        S5
??? surface_extraction.py         S4
??? quality_gates.py              S9
??? stl_assembler.py              S7
??? phi_updater.py                S8
??? tests/
?   ??? __init__.py               (empty file --- required for imports in test files)
?   ??? test_geometry_contract.py
?   ??? test_fixed_hardware.py
?   ??? test_bounding_volumes.py
?   ??? test_phi_grid.py
?   ??? test_mass_com_calculator.py
?   ??? test_surface_extraction.py
?   ??? test_quality_gates.py
?   ??? test_stl_assembler.py
?   ??? test_phi_updater.py
?   ??? test_integration_part1_part2.py
??? run_all_tests.py
```

---

## 2. Known Values, Placeholders, and Open Items

This is the master list of every value in the system. If a value is known, it is stated. If unknown, the placeholder strategy is stated.

### 2.1 Confirmed Values (hard-code these exactly)

| Constant | Value | Unit | Source |
|---|---|---|---|
| W minimum | 120.0 | mm | Spec ?Inputs |
| W maximum | 140.0 | mm | Spec ?Inputs |
| d_halo minimum | 0.0 | mm | Spec ?Inputs |
| d_halo maximum | W + 16.0 | mm | Spec ?Inputs (depends on W) |
| Grid spacing | 0.3 | mm | Spec ?phi Field Representation |
| Min machining radius | 3.15 | mm | Spec ?Surface Extraction |
| Nose density | 1000.0 | kg/m^3 | Spec ?Mass and COM table |
| Sidepod density | 163.0 | kg/m^3 | Spec ?Mass and COM table |
| Rearpod density | 163.0 | kg/m^3 | Spec ?Mass and COM table |
| Main body density | 163.0 | kg/m^3 | Spec ?Mass and COM table |
| CO2 mass | 0.023 | kg | Spec + Part 2 `CO2_CARTRIDGE_MASS_KG` |
| Number of wheels | 4 | --- | Part 2 locked `race_objective.py` |
| Wheel radius | 0.015 | m | Part 2 locked `race_objective.py` |
| Wheel clearance (forbidden zone padding) | 2.0 | mm | Design decision --- do not change without updating sidepod corridor |
| Attachment strip width | 1.0 | mm | Design decision |
| Hardware void clearance | 1.0 | mm | Design decision |
| COM z lower bound (sanity gate) | 0.005 | m | Physical --- below this means units bug |
| COM z upper bound (sanity gate) | 0.060 | m | Physical --- above this means units bug |
| COM z polynomial range (Part 2 guard) | [0.018, 0.042] | m | Part 2 audit finding 3.1 |
| Halo bottom height above track | 24.0 | mm | **CONFIRMED by user** |
| Halo front edge constraint | Must be aft of front axle | --- | **CONFIRMED by user** (halo x_front > 0.0 m) |
| Halo position constraint | Must be between canister pocket and front axle in x | --- | **CONFIRMED by user** |
| CFD symmetry | Right half only | --- | **CONFIRMED by user** |
| Adjoint domain | Right half (same as CFD) | --- | **CONFIRMED by user --- U3 RESOLVED** |
| Small inaccessible area threshold | 1e-5 | m^2 | 10 mm^2 --- below this: try to fix |
| Large inaccessible area threshold | 1e-3 | m^2 | 1000 mm^2 --- above this: kill candidate |
| Max repair retries | 3 | --- | Spec ?Failure and Recovery |

### 2.2 Placeholder Values (raise NotImplementedError with ? UNRESOLVED label)

| ID | What is missing | Where placeholder goes | Message to write |
|---|---|---|---|
| U1 | Halo cross-section shape in y-z plane (the exact polygon or voxel mask of the halo tube) | `fixed_hardware.py::place_halo_void()` | `"? UNRESOLVED U1: Halo cross-section shape (y-z polygon vertices in mm) not provided. Measure physical halo hardware and call place_halo_void(cross_section_yz_mm=...)."` |
| U2 | CO2 canister legal position (x, y, z in mm) | `fixed_hardware.py::place_canister_void()` | `"? UNRESOLVED U2: CO2 canister legal position not confirmed from competition rules. Provide canister_com_mm=(x,y,z) from the official STEM Racing rule sheet."` |
| U4 | UAE regulation envelope polygon | `surface_extraction.py::_stage5_rule_checker()` | `"? UNRESOLVED U4: UAE competition regulation envelope polygon not provided. Rule checker cannot validate legal envelope until these dimensions are supplied."` |
| U5 | Rear wing fixed position (x, y, z in mm) | `fixed_hardware.py::place_fixed_hardware()` | `"? UNRESOLVED U5: Rear wing fixed position coordinate not confirmed. Provide rear_wing_com_mm=(x,y,z) from competition rules."` |
| U6 | Absolute bounding box dimensions (y-extents, z-extents, rearpod length) for all components | `bounding_volumes.py::compute_bounding_volumes()` | `"? UNRESOLVED U6: Car envelope dimensions (y-half-width, z-floor, z-top per component, rearpod max length) not provided from UAE competition rules."` |

---

## 3. Outer Loop Scalars

Part 3 sets these before each outer iteration. Part 1 recomputes all geometry from them.

```
W      in [120.0, 140.0] mm     wheelbase
                                 Coarse sweep: 21 values, W = 120, 121, ..., 140 mm
                                 Refined sweep: 0.5 mm steps near top 5 candidates
                                 Front axle at x = 0.0 m (always)
                                 Rear axle at x = W/1000 m

d_halo in [0.0, W + 16.0] mm   halo-to-canister fore-aft distance
                                 When d_halo = 0: nose volume is degenerate (1 cell thick)
                                 This is NOT an error. Handle it.
```

When W changes:
1. Rear axle moves to new x position
2. Both forbidden-zone cylinders shift (rear one moves)
3. Sidepod corridor x_min and x_max both recompute
4. All four phi grids remap via `PhiGrid.remap(new_bv)`
5. Hard constraints re-enforced in all grids after remap

---

## 4. Component Summary

| Component | Has phi grid | Density | Which half | Key constraint |
|---|---|---|---|---|
| Nose cone | Yes | 1000.0 kg/m^3 | Full (both y sides, symmetric) | Rear face phi < 0 (attaches to body); bbox phi > 0 |
| Right sidepod | Yes (right half only) | 163.0 kg/m^3 | Right (y >= 0) | Inner face phi < 0 (attaches to body); wheel forbidden zones phi > 0 |
| Rearpod | Yes | 163.0 kg/m^3 | Full (symmetric) | Front face phi < 0 (attaches to body); bbox phi > 0 |
| Main body | Yes | 163.0 kg/m^3 | Full (symmetric) | Hardware voids phi > 0; sidepod walls phi < 0 |
| CO2 cartridge | No | 23 g fixed | --- | Fixed mass, fixed position |
| Rear wing | No | known kg fixed | --- | Fixed mass, fixed position |
| Wheels + axles | No | known kg fixed | --- | Fixed mass, fixed geometry |

The left sidepod is always the right sidepod reflected across y=0. It is never stored as a separate grid. It is generated at STL assembly time.

---

## 5. Stage Map and Build Order

Build in this exact order. Do not skip stages. Do not build out of order.

```
BUILD ORDER:

Step 1   Write S1: geometry_contract.py
         Reason: everything imports from here; no dependencies

Step 2   Write S6: fixed_hardware.py
         Reason: forbidden cylinders must exist before bounding volumes

Step 3   Write S2: bounding_volumes.py
         Reason: volumes must exist before grids can be sized

Step 4   Write S3: phi_grid.py
         Reason: grid class needed by S5, S4, S8

Step 5   Write S5: mass_com_calculator.py
         Reason: needed by integration test; pure phi consumer

Step 6   Write S4: surface_extraction.py (Stages 1+2 first, then 3+4+5+6)
         Reason: S9 and S7 depend on it

Step 7   Write S9: quality_gates.py
         Reason: calls S4 and S7; drives lifecycle states

Step 8   Write S7: stl_assembler.py
         Reason: depends on S4 meshes

Step 9   Write integration test
         Reason: confirms S1->S7 chain works end-to-end with Part 2's real types

Step 10  Write S8: phi_updater.py
         Reason: nothing in S1-S9 calls it; it is the Part 3 interface only
```

---

## 6. S1 --- `geometry_contract.py` (Complete File Specification)

This file has zero imports from anywhere except Python's standard `math` module. Write it exactly as shown.

### 6.1 Full File Contents

```python
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
```

### 6.2 Tests for S1 (`tests/test_geometry_contract.py`)

Write every test below. Each test is independent. Run with `python tests/test_geometry_contract.py`. Tests pass with zero output and exit code 0. Tests fail by printing "FAIL: <message>" and exiting with code 1.

```python
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
    test_halo_z_min()
    test_lifecycle_states_count()
    test_lifecycle_states_exact_names()
    test_tool_directions_all_components()
    test_tool_directions_unit_vectors()
    test_phi_snapshot_keys()
    test_grid_spacing_consistency()
    test_min_radius_consistency()
    print("\nAll geometry_contract tests passed.")
```

---

## 7. BoundingRegion --- Arbitrary Shape Support

This is a standalone class that must be implemented BEFORE `bounding_volumes.py` (S2) because S2 uses it. It lives in `bounding_volumes.py`.

### 7.1 Why Arbitrary Shapes

The car body cross-sections are NOT rectangular. Sidepods taper. The nose has a curved profile. Rearpods have rounded cross-sections. Constraining everything to axis-aligned boxes (a) over-constrains the optimizer (cells that are technically inside the box but outside the legal cross-section won't be used, wasting grid resolution) and (b) produces incorrect forbidden zones if the wheel void mask is a cylinder (not a box).

The system must support:
- **Axis-aligned box**: special case, always works
- **Convex polygon cross-section extruded in x**: e.g. a sidepod corridor that is trapezoidal in y-z
- **Arbitrary voxel mask**: the most general case --- a boolean numpy array of exactly the grid shape, True where the region is valid, False where it is forbidden

### 7.2 `BoundingRegion` Class

Write this class inside `bounding_volumes.py`. It is imported and used by `phi_grid.py` and `fixed_hardware.py`.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class BoundingRegion:
    """
    A 3D region definition for one phi grid component.

    Supports three modes:
      1. BOX mode: axis-aligned bounding box (simplest)
      2. POLYGON mode: convex polygon cross-section in y-z plane, extruded along x
      3. VOXEL mode: arbitrary boolean mask (most general)

    Only one mode is active per instance. The mode is determined by which
    constructor arguments are non-None.

    In all modes:
      - origin_m: (x0, y0, z0) coordinates of grid index [0,0,0] in metres
      - shape: (nx, ny, nz) number of grid cells in each axis
      - All coordinates are in the car coordinate system (x=front-to-rear, y=CL-to-right, z=up)

    Cell centre of grid index (i, j, k) is at:
      x = origin_m[0] + i * GRID_SPACING_M
      y = origin_m[1] + j * GRID_SPACING_M
      z = origin_m[2] + k * GRID_SPACING_M
    """

    component: str
    origin_m: tuple[float, float, float]   # (x0, y0, z0) of cell [0,0,0] centre
    shape: tuple[int, int, int]            # (nx, ny, nz) --- number of cells

    # BOX mode: None means use full grid extent (same as axis-aligned box)
    # If all three polygon_* and voxel_mask are None, this is BOX mode.

    # POLYGON mode: convex polygon in y-z plane, constant along x
    # polygon_yz_m: list of (y, z) vertices in metres, counterclockwise order
    # The polygon is the same cross-section at every x slice.
    polygon_yz_m: Optional[list[tuple[float, float]]] = None

    # VOXEL mode: explicit boolean mask, shape must equal self.shape
    # True = valid (inside region), False = forbidden (outside region)
    # If provided, polygon_yz_m is ignored.
    voxel_mask: Optional[np.ndarray] = None

    # Cached valid mask (computed on first access, then cached)
    _cached_valid_mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def nx(self) -> int: return self.shape[0]
    @property
    def ny(self) -> int: return self.shape[1]
    @property
    def nz(self) -> int: return self.shape[2]

    @property
    def mode(self) -> str:
        """Returns 'voxel', 'polygon', or 'box'."""
        if self.voxel_mask is not None:
            return "voxel"
        if self.polygon_yz_m is not None:
            return "polygon"
        return "box"

    def valid_mask(self) -> np.ndarray:
        """
        Return boolean array of shape (nx, ny, nz).
        True = this cell is inside the valid region.
        False = this cell is outside the valid region (should be treated as phi > 0).

        In BOX mode: all True (entire grid is valid).
        In POLYGON mode: True where cell centre (y, z) is inside the polygon.
        In VOXEL mode: directly returns the provided voxel_mask.

        Result is cached after first computation.
        """
        if self._cached_valid_mask is not None:
            return self._cached_valid_mask

        if self.mode == "voxel":
            if self.voxel_mask.shape != self.shape:
                raise ValueError(
                    f"{self.component}: voxel_mask shape {self.voxel_mask.shape} "
                    f"!= grid shape {self.shape}."
                )
            self._cached_valid_mask = self.voxel_mask.astype(bool)

        elif self.mode == "polygon":
            self._cached_valid_mask = self._compute_polygon_mask()

        else:  # BOX mode
            self._cached_valid_mask = np.ones(self.shape, dtype=bool)

        return self._cached_valid_mask

    def _compute_polygon_mask(self) -> np.ndarray:
        """
        Compute boolean mask for polygon cross-section.

        Algorithm: for each grid cell, compute (y, z) of its centre.
        Test whether (y, z) is inside the polygon using the ray-casting method.
        Result is the same at every x slice (polygon is extruded along x).

        The polygon_yz_m must be a convex or non-convex simple polygon.
        Vertices can be in any order --- the code handles both CW and CCW.
        """
        from geometry_contract import GRID_SPACING_M

        nx, ny, nz = self.shape
        ox, oy, oz = self.origin_m
        dx = GRID_SPACING_M

        # Build y and z coordinate arrays for all grid cells
        ys = oy + np.arange(ny) * dx    # shape (ny,)
        zs = oz + np.arange(nz) * dx    # shape (nz,)
        # Y[j, k] = y coordinate of cell (*, j, k)
        # Z[j, k] = z coordinate of cell (*, j, k)
        Y, Z = np.meshgrid(ys, zs, indexing="ij")   # shape (ny, nz)

        inside_yz = _point_in_polygon_vectorised(
            Y.ravel(), Z.ravel(), self.polygon_yz_m
        ).reshape(ny, nz)    # shape (ny, nz)

        # Broadcast to all x slices: shape (nx, ny, nz)
        mask = np.broadcast_to(inside_yz[np.newaxis, :, :], (nx, ny, nz)).copy()
        return mask

    def x_max_m(self) -> float:
        return self.origin_m[0] + self.nx * GRID_SPACING_M

    def y_max_m(self) -> float:
        return self.origin_m[1] + self.ny * GRID_SPACING_M

    def z_max_m(self) -> float:
        return self.origin_m[2] + self.nz * GRID_SPACING_M


def _point_in_polygon_vectorised(
    ys: np.ndarray,
    zs: np.ndarray,
    polygon_yz: list[tuple[float, float]],
) -> np.ndarray:
    """
    Ray-casting polygon test for N points.

    Returns boolean array of shape (N,).
    True = point is inside the polygon.

    Algorithm: cast a ray in the +y direction from each test point.
    Count how many polygon edges it crosses. Odd = inside, even = outside.
    This handles both convex and non-convex polygons correctly.

    polygon_yz: list of (y, z) vertex coordinates.
    """
    n_pts = len(ys)
    inside = np.zeros(n_pts, dtype=bool)

    verts = list(polygon_yz)
    n_verts = len(verts)

    for i in range(n_verts):
        y1, z1 = verts[i]
        y2, z2 = verts[(i + 1) % n_verts]

        # Edge crosses z-coordinate of test point (z1 to z2 straddles test z)
        cond_z = ((z1 <= zs) & (zs < z2)) | ((z2 <= zs) & (zs < z1))

        # x-intercept of edge at test point's z coordinate
        # (if z1 == z2 this is a horizontal edge; cond_z handles it correctly)
        dz = z2 - z1
        safe = np.where(np.abs(dz) > 1e-14, dz, 1.0)
        y_intercept = y1 + (zs - z1) * (y2 - y1) / safe

        # Ray in +y direction: does intercept lie to the right of (or at) test point?
        cond_y = ys < y_intercept

        inside ^= (cond_z & cond_y)

    return inside
```

---

## 8. S6 --- `fixed_hardware.py` (Complete File Specification)

### 8.1 Purpose and Outputs

Place the four fixed inputs. Produce:
1. phi void masks (boolean numpy arrays) for the main body grid --- these force certain cells to be air in S3
2. Forbidden cylinders for S2 (sidepod corridor calculation)
3. A `FixedHardwareSpec` object for Part 2

### 8.2 Imports

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'part2_simulation'))
from mass_com_ingest import FixedHardwareSpec   # Part 2 type
from geometry_contract import (
    CO2_MASS_KG, GRID_SPACING_M, WHEEL_CLEARANCE_M,
    HARDWARE_CLEARANCE_M, HALO_MIN_Z_M, mm_to_m
)
```

### 8.3 Supporting Types

```python
@dataclass(frozen=True)
class ForbiddenCylinder:
    """
    Wheel/axle exclusion zone. A cylinder aligned with the x-axis.

    The cylinder is infinite in y-z (circular cross-section in y-z plane)
    and has finite extent in x (from x_center - x_half_width to x_center + x_half_width).

    All values in metres.
    """
    x_center_m: float       # axle position in x (0.0 = front axle, W_m = rear axle)
    y_center_m: float       # always 0.0 --- axle is on centerline
    z_center_m: float       # axle height above track surface (z=0)
    radius_m:   float       # wheel radius + clearance = R_WHEEL_M + WHEEL_CLEARANCE_M
    x_half_width_m: float   # half-width of wheel+axle assembly in x direction

    @property
    def x_min_m(self) -> float:
        return self.x_center_m - self.x_half_width_m

    @property
    def x_max_m(self) -> float:
        return self.x_center_m + self.x_half_width_m

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Returns True if point (x,y,z) is inside this cylinder."""
        if not (self.x_min_m <= x <= self.x_max_m):
            return False
        r2 = (y - self.y_center_m)**2 + (z - self.z_center_m)**2
        return r2 <= self.radius_m ** 2


@dataclass
class HaloGeometry:
    """
    Physical halo hardware geometry.

    The halo is placed in the main body phi grid.
    Its void region forces phi > 0 so the optimizer cannot fill the halo mount.

    Coordinate system: all values in metres, car coordinate system.

    The halo sits between the canister pocket and the front axle in x.
    Its bottom must be at z >= HALO_MIN_Z_M (24 mm above track).
    Its front edge must be at x > 0.0 m (behind front axle).

    cross_section_yz_m: list of (y, z) polygon vertices defining the halo
        tube cross-section in the y-z plane. This is the shape that gets
        extruded along x from x_front to x_rear.
        ? UNRESOLVED U1: These vertices must be measured from the physical hardware.
    """
    x_front_m: float           # fore-most edge of halo void in x
    x_rear_m:  float           # aft-most edge of halo void in x
    # ? UNRESOLVED U1: cross-section shape not yet provided
    cross_section_yz_m: Optional[list[tuple[float, float]]] = None


@dataclass(frozen=True)
class FixedHardwareResult:
    """All outputs from fixed hardware placement."""
    # Bool void masks --- shape == main body grid shape (nx, ny, nz)
    # True = cell is forced phi > 0 (void --- hardware occupies this space)
    halo_void_mask:      np.ndarray
    canister_void_mask:  np.ndarray
    front_axle_void_mask: np.ndarray
    rear_axle_void_mask:  np.ndarray

    # Forbidden cylinders --- used by S2 to compute sidepod corridor
    front_cylinder: ForbiddenCylinder
    rear_cylinder:  ForbiddenCylinder

    # Combined void mask (union of all four) --- convenience, fed to PhiGrid
    combined_void_mask: np.ndarray

    # Part 2 interface
    fixed_hardware_spec: FixedHardwareSpec
```

### 8.4 Halo Position Validation

This is the most critical validation in S6 because the halo rules are confirmed.

```python
def _validate_halo_position(
    halo: HaloGeometry,
    canister_x_m: float,
    W_m: float,
) -> None:
    """
    Enforce all three confirmed halo rules:

    Rule H1: Halo bottom must be at z >= 24 mm = 0.024 m above track.
             We check the bottom of the cross-section (min z vertex).
             ? UNRESOLVED U1: Until cross_section_yz_m is provided, we cannot
             check the exact bottom z. We can only check that x_front is valid.

    Rule H2: Halo front edge must be strictly aft of front axle.
             Front axle is at x = 0.0 m. "Aft" means larger x (front-to-rear convention).
             Therefore: halo.x_front_m > 0.0 m.

    Rule H3: Halo must sit between canister pocket and front axle in x.
             canister is forward (smaller x), halo is between canister and front axle.
             Therefore: canister_x_m < halo.x_front_m  AND  halo.x_front_m > 0.0 m
             (Rule H2 already enforces the second condition.)
             Also: halo must not extend past the rear axle: halo.x_rear_m < W_m
    """
    # Rule H2: halo front must be behind front axle (x > 0)
    if halo.x_front_m <= 0.0:
        raise ValueError(
            f"Halo front edge at x={halo.x_front_m:.4f} m must be strictly "
            f"aft of front axle (x=0.0 m). In front-to-rear convention, "
            f"'behind front axle' means x > 0. Got x={halo.x_front_m:.4f} m."
        )

    # Rule H3a: halo must be aft of canister pocket
    if halo.x_front_m <= canister_x_m:
        raise ValueError(
            f"Halo front edge (x={halo.x_front_m:.4f} m) must be aft of "
            f"canister pocket (x={canister_x_m:.4f} m). "
            f"'Between canister and front axle' means canister_x < halo_x_front."
        )

    # Rule H3b: halo must not extend past rear axle
    if halo.x_rear_m >= W_m:
        raise ValueError(
            f"Halo rear edge (x={halo.x_rear_m:.4f} m) extends past or to "
            f"rear axle (x={W_m:.4f} m). Halo must fit within the car body."
        )

    # Rule H1: check z bottom if cross-section is known
    if halo.cross_section_yz_m is not None:
        z_bottom = min(z for _, z in halo.cross_section_yz_m)
        if z_bottom < HALO_MIN_Z_M - 1e-6:
            raise ValueError(
                f"Halo cross-section bottom at z={z_bottom*1000:.2f} mm "
                f"is below minimum {HALO_MIN_Z_M*1000:.1f} mm above track. "
                f"Halo must clear the track by at least 24 mm."
            )
    # If cross_section_yz_m is None (? U1), we cannot check Rule H1 on z.
    # The NotImplementedError in place_halo_void() below handles this.
```

### 8.5 COM Sanity Validation

```python
def _assert_com_in_range(
    label: str,
    com_m: tuple[float, float, float],
    W_m: float,
) -> None:
    """
    Validate that a COM coordinate is physically plausible.

    This catches mm-vs-m units bugs before they reach Part 2's polynomial,
    which would produce race time values of 10^15 seconds (Part 2 audit finding 3.1).

    Rules:
      x in [-0.01, W_m + 0.01]   within car length (+/-10 mm tolerance)
      y in [-0.05, 0.05]          within +/-50 mm of centerline
      z in [COM_Z_LOWER_BOUND_M, COM_Z_UPPER_BOUND_M]  = [0.005, 0.060] m

    If z is in mm instead of m, it would be ~25 m, which is obviously caught.
    """
    from geometry_contract import COM_Z_LOWER_BOUND_M, COM_Z_UPPER_BOUND_M
    x, y, z = com_m
    if not (-0.01 <= x <= W_m + 0.01):
        raise ValueError(
            f"{label}: com_x={x:.6f} m is outside car length range "
            f"[-0.01, {W_m+0.01:.4f}] m. "
            f"If this looks like mm, you forgot to divide by 1000."
        )
    if not (-0.05 <= y <= 0.05):
        raise ValueError(
            f"{label}: com_y={y:.6f} m is outside +/-50 mm range. "
            f"Check units --- expected metres, not mm."
        )
    if not (COM_Z_LOWER_BOUND_M <= z <= COM_Z_UPPER_BOUND_M):
        raise ValueError(
            f"{label}: com_z={z:.6f} m is outside physical range "
            f"[{COM_Z_LOWER_BOUND_M}, {COM_Z_UPPER_BOUND_M}] m. "
            f"If z={z*1000:.1f} looks right in mm, you forgot mm_to_m()."
        )
```

### 8.6 Void Mask Builders

```python
def _build_cylinder_void_mask(
    grid_shape: tuple[int, int, int],
    grid_origin_m: tuple[float, float, float],
    cylinder: ForbiddenCylinder,
) -> np.ndarray:
    """
    Returns bool array shape (nx, ny, nz).
    True where grid cell centre is inside the cylinder.

    The cylinder's circular cross-section is in the y-z plane.
    Its x extent is [cylinder.x_min_m, cylinder.x_max_m].
    """
    nx, ny, nz = grid_shape
    ox, oy, oz = grid_origin_m
    dx = GRID_SPACING_M

    xs = ox + np.arange(nx) * dx   # shape (nx,)
    ys = oy + np.arange(ny) * dx   # shape (ny,)
    zs = oz + np.arange(nz) * dx   # shape (nz,)

    X = xs[:, np.newaxis, np.newaxis]  # (nx, 1, 1)
    Y = ys[np.newaxis, :, np.newaxis]  # (1, ny, 1)
    Z = zs[np.newaxis, np.newaxis, :]  # (1, 1, nz)

    in_x = (X >= cylinder.x_min_m) & (X <= cylinder.x_max_m)
    r2 = (Y - cylinder.y_center_m)**2 + (Z - cylinder.z_center_m)**2
    in_r = r2 <= cylinder.radius_m**2

    return (in_x & in_r).astype(bool)


def _build_polygon_void_mask(
    grid_shape: tuple[int, int, int],
    grid_origin_m: tuple[float, float, float],
    x_min_m: float,
    x_max_m: float,
    polygon_yz_m: list[tuple[float, float]],
) -> np.ndarray:
    """
    Returns bool array shape (nx, ny, nz).
    True where grid cell centre is inside the polygon cross-section AND within [x_min, x_max].

    Used for halo void (polygon cross-section of the halo tube extruded in x).
    """
    from bounding_volumes import _point_in_polygon_vectorised

    nx, ny, nz = grid_shape
    ox, oy, oz = grid_origin_m
    dx = GRID_SPACING_M

    xs = ox + np.arange(nx) * dx
    ys = oy + np.arange(ny) * dx
    zs = oz + np.arange(nz) * dx

    # in_x: shape (nx,)
    in_x = (xs >= x_min_m) & (xs <= x_max_m)

    # in_yz: shape (ny, nz)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    n_pts = ny * nz
    in_yz = _point_in_polygon_vectorised(
        Y.ravel(), Z.ravel(), polygon_yz_m
    ).reshape(ny, nz)

    # Broadcast: shape (nx, ny, nz)
    mask = in_x[:, np.newaxis, np.newaxis] & in_yz[np.newaxis, :, :]
    return mask.astype(bool)


def _build_box_void_mask(
    grid_shape: tuple[int, int, int],
    grid_origin_m: tuple[float, float, float],
    x_range_m: tuple[float, float],
    y_range_m: tuple[float, float],
    z_range_m: tuple[float, float],
) -> np.ndarray:
    """
    Returns bool array shape (nx, ny, nz).
    True where grid cell centre is inside the axis-aligned box.

    Used for canister void (simple box --- shape pending U2).
    """
    nx, ny, nz = grid_shape
    ox, oy, oz = grid_origin_m
    dx = GRID_SPACING_M

    xs = ox + np.arange(nx) * dx
    ys = oy + np.arange(ny) * dx
    zs = oz + np.arange(nz) * dx

    in_x = (xs >= x_range_m[0]) & (xs <= x_range_m[1])
    in_y = (ys >= y_range_m[0]) & (ys <= y_range_m[1])
    in_z = (zs >= z_range_m[0]) & (zs <= z_range_m[1])

    X = in_x[:, np.newaxis, np.newaxis]
    Y = in_y[np.newaxis, :, np.newaxis]
    Z = in_z[np.newaxis, np.newaxis, :]

    return (X & Y & Z).astype(bool)
```

### 8.7 Main Entry Point

```python
def place_fixed_hardware(
    W_mm: float,
    halo_geometry: HaloGeometry,
    canister_com_mm: Optional[tuple[float, float, float]],   # ? U2: None until confirmed
    canister_box_half_size_mm: float,                         # half-size of canister void box
    wheel_axle_mass_kg: float,
    wheel_axle_com_mm: tuple[float, float, float],
    wheel_x_half_width_mm: float,                             # half-width of wheel assembly in x
    wheel_axle_z_mm: float,                                   # axle height above track in mm
    rear_wing_mass_kg: float,
    rear_wing_com_mm: Optional[tuple[float, float, float]],  # ? U5: None until confirmed
    body_grid_shape: tuple[int, int, int],
    body_grid_origin_m: tuple[float, float, float],
) -> FixedHardwareResult:
    """
    Place all fixed hardware. Validate positions. Build void masks. Construct FixedHardwareSpec.

    Args:
        W_mm: wheelbase in mm
        halo_geometry: halo dimensions (x_front_m, x_rear_m, cross_section_yz_m)
        canister_com_mm: (x, y, z) of CO2 canister centre in mm, or None (? U2)
        canister_box_half_size_mm: half-size of cubic void box around canister in mm
        wheel_axle_mass_kg: total mass of all 4 wheels + axles combined, in kg
        wheel_axle_com_mm: (x, y, z) of combined wheels+axles COM in mm
        wheel_x_half_width_mm: half-width of wheel+axle assembly in x (for forbidden zone)
        wheel_axle_z_mm: height of axle centreline above track in mm
        rear_wing_mass_kg: rear wing mass in kg
        rear_wing_com_mm: (x, y, z) of rear wing COM in mm, or None (? U5)
        body_grid_shape: (nx, ny, nz) of the main body phi grid
        body_grid_origin_m: (x0, y0, z0) of the main body grid in metres

    Returns:
        FixedHardwareResult with all void masks, forbidden cylinders, and FixedHardwareSpec
    """
    from geometry_contract import validate_W, R_WHEEL_M, WHEEL_CLEARANCE_M

    validate_W(W_mm)
    W_m = mm_to_m(W_mm)

    # ?? Front and rear forbidden cylinders ????????????????????????????????
    axle_z_m = mm_to_m(wheel_axle_z_mm)
    axle_x_half_m = mm_to_m(wheel_x_half_width_mm)
    cylinder_radius_m = R_WHEEL_M + WHEEL_CLEARANCE_M

    front_cylinder = ForbiddenCylinder(
        x_center_m     = 0.0,
        y_center_m     = 0.0,
        z_center_m     = axle_z_m,
        radius_m       = cylinder_radius_m,
        x_half_width_m = axle_x_half_m,
    )
    rear_cylinder = ForbiddenCylinder(
        x_center_m     = W_m,
        y_center_m     = 0.0,
        z_center_m     = axle_z_m,
        radius_m       = cylinder_radius_m,
        x_half_width_m = axle_x_half_m,
    )

    # ?? Canister position ?????????????????????????????????????????????????
    # ? UNRESOLVED U2: CO2 canister legal position not confirmed from competition rules.
    # Provide canister_com_mm=(x,y,z) from the official STEM Racing rule sheet.
    if canister_com_mm is None:
        raise NotImplementedError(
            "? UNRESOLVED U2: CO2 canister legal position not confirmed. "
            "Provide canister_com_mm=(x_mm, y_mm, z_mm) from competition rules. "
            "The canister is at the front of the car (small x value)."
        )
    canister_com_m = tuple(mm_to_m(v) for v in canister_com_mm)
    _assert_com_in_range("CO2 canister", canister_com_m, W_m)

    # ?? Halo position validation ??????????????????????????????????????????
    _validate_halo_position(halo_geometry, canister_com_m[0], W_m)

    # ?? Wheel+axle COM ????????????????????????????????????????????????????
    wheel_axle_com_m = tuple(mm_to_m(v) for v in wheel_axle_com_mm)
    _assert_com_in_range("Wheels+axles", wheel_axle_com_m, W_m)

    # ?? Rear wing COM ?????????????????????????????????????????????????????
    # ? UNRESOLVED U5: Rear wing fixed position coordinate not confirmed.
    if rear_wing_com_mm is None:
        raise NotImplementedError(
            "? UNRESOLVED U5: Rear wing fixed position not confirmed from competition rules. "
            "Provide rear_wing_com_mm=(x_mm, y_mm, z_mm)."
        )
    rear_wing_com_m = tuple(mm_to_m(v) for v in rear_wing_com_mm)
    _assert_com_in_range("Rear wing", rear_wing_com_m, W_m)

    # ?? Build void masks ??????????????????????????????????????????????????
    front_axle_mask = _build_cylinder_void_mask(
        body_grid_shape, body_grid_origin_m, front_cylinder
    )
    rear_axle_mask = _build_cylinder_void_mask(
        body_grid_shape, body_grid_origin_m, rear_cylinder
    )

    # Canister void: simple box around canister COM
    cs_half = mm_to_m(canister_box_half_size_mm)
    cx, cy, cz = canister_com_m
    canister_mask = _build_box_void_mask(
        body_grid_shape, body_grid_origin_m,
        x_range_m=(cx - cs_half, cx + cs_half),
        y_range_m=(cy - cs_half, cy + cs_half),
        z_range_m=(cz - cs_half, cz + cs_half),
    )

    # Halo void
    # ? UNRESOLVED U1: Halo cross-section shape (y-z polygon vertices) not provided.
    if halo_geometry.cross_section_yz_m is None:
        raise NotImplementedError(
            "? UNRESOLVED U1: Halo cross-section shape (y-z polygon vertices in mm) "
            "not provided. Measure physical halo hardware and supply "
            "HaloGeometry(cross_section_yz_m=[(y1,z1),(y2,z2),...]) in metres. "
            "The polygon defines the halo tube cross-section at each x slice."
        )
    halo_mask = _build_polygon_void_mask(
        body_grid_shape, body_grid_origin_m,
        x_min_m=halo_geometry.x_front_m,
        x_max_m=halo_geometry.x_rear_m,
        polygon_yz_m=halo_geometry.cross_section_yz_m,
    )

    # Combined mask: union of all four voids
    combined = front_axle_mask | rear_axle_mask | canister_mask | halo_mask

    # ?? Construct FixedHardwareSpec (Part 2 type) ?????????????????????????
    spec = FixedHardwareSpec(
        co2_cartridge_mass_kg = CO2_MASS_KG,          # exactly 0.023 --- Part 2 validates this
        co2_cartridge_com     = canister_com_m,        # (x, y, z) in metres
        rear_wing_mass_kg     = rear_wing_mass_kg,
        rear_wing_com         = rear_wing_com_m,
        wheels_axles_mass_kg  = wheel_axle_mass_kg,
        wheels_axles_com      = wheel_axle_com_m,
    )
    # Part 2's __post_init__ raises ValueError if co2_cartridge_mass_kg != 0.023.
    # If that raise fires, it means CO2_MASS_KG drifted from Part 2's constant --- fix S1.

    return FixedHardwareResult(
        halo_void_mask       = halo_mask,
        canister_void_mask   = canister_mask,
        front_axle_void_mask = front_axle_mask,
        rear_axle_void_mask  = rear_axle_mask,
        front_cylinder       = front_cylinder,
        rear_cylinder        = rear_cylinder,
        combined_void_mask   = combined,
        fixed_hardware_spec  = spec,
    )
```

### 8.8 Tests for S6 (`tests/test_fixed_hardware.py`)

```python
"""
Tests for fixed_hardware.py.
These tests use placeholder values to bypass ? UNRESOLVED items where possible,
using the public cylinder and mask builders directly.
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from fixed_hardware import (
    ForbiddenCylinder, _build_cylinder_void_mask, _build_box_void_mask,
    _validate_halo_position, HaloGeometry, _assert_com_in_range,
)
from geometry_contract import R_WHEEL_M, WHEEL_CLEARANCE_M, mm_to_m

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

def test_front_cylinder_x_center_is_zero():
    cyl = ForbiddenCylinder(0.0, 0.0, 0.015, R_WHEEL_M+WHEEL_CLEARANCE_M, 0.010)
    assert cyl.x_center_m == 0.0
    _pass("test_front_cylinder_x_center_is_zero")

def test_rear_cylinder_x_center_equals_W():
    W_m = mm_to_m(130.0)
    cyl = ForbiddenCylinder(W_m, 0.0, 0.015, R_WHEEL_M+WHEEL_CLEARANCE_M, 0.010)
    assert abs(cyl.x_center_m - W_m) < 1e-12
    _pass("test_rear_cylinder_x_center_equals_W")

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

def test_box_void_mask_shape():
    shape = (50, 50, 50)
    origin = (0.0, -0.025, 0.0)
    mask = _build_box_void_mask(shape, origin, (0.01, 0.02), (-0.005, 0.005), (0.005, 0.015))
    assert mask.shape == shape
    assert mask.dtype == bool
    _pass("test_box_void_mask_shape")

def test_halo_validation_behind_front_axle():
    # Valid: halo x_front > 0 and > canister_x
    halo = HaloGeometry(x_front_m=0.010, x_rear_m=0.050)
    _validate_halo_position(halo, canister_x_m=0.005, W_m=0.130)
    _pass("test_halo_validation_behind_front_axle")

def test_halo_validation_fails_if_at_front_axle():
    halo = HaloGeometry(x_front_m=0.0, x_rear_m=0.050)
    try:
        _validate_halo_position(halo, canister_x_m=0.005, W_m=0.130)
        _fail("test_halo_validation_fails_if_at_front_axle", "should have raised")
    except ValueError:
        _pass("test_halo_validation_fails_if_at_front_axle")

def test_halo_validation_fails_if_before_canister():
    halo = HaloGeometry(x_front_m=0.003, x_rear_m=0.050)
    try:
        _validate_halo_position(halo, canister_x_m=0.010, W_m=0.130)
        _fail("test_halo_validation_fails_if_before_canister", "should have raised")
    except ValueError:
        _pass("test_halo_validation_fails_if_before_canister")

def test_halo_validation_fails_if_past_rear_axle():
    halo = HaloGeometry(x_front_m=0.010, x_rear_m=0.135)
    try:
        _validate_halo_position(halo, canister_x_m=0.005, W_m=0.130)
        _fail("test_halo_validation_fails_if_past_rear_axle", "should have raised")
    except ValueError:
        _pass("test_halo_validation_fails_if_past_rear_axle")

def test_com_sanity_gate_catches_mm_as_m():
    try:
        _assert_com_in_range("test", (0.050, 0.0, 25.0), W_m=0.130)  # z=25 m is mm error
        _fail("test_com_sanity_gate_catches_mm_as_m", "should have raised")
    except ValueError:
        _pass("test_com_sanity_gate_catches_mm_as_m")

def test_com_sanity_gate_valid():
    _assert_com_in_range("test", (0.050, 0.0, 0.025), W_m=0.130)
    _pass("test_com_sanity_gate_valid")

def test_com_sanity_gate_outside_car_length():
    try:
        _assert_com_in_range("test", (0.200, 0.0, 0.025), W_m=0.130)
        _fail("test_com_sanity_gate_outside_car_length", "should have raised")
    except ValueError:
        _pass("test_com_sanity_gate_outside_car_length")

if __name__ == "__main__":
    test_front_cylinder_x_center_is_zero()
    test_rear_cylinder_x_center_equals_W()
    test_cylinder_contains_point_inside()
    test_cylinder_contains_point_outside()
    test_cylinder_void_mask_shape()
    test_cylinder_void_mask_centre_is_true()
    test_box_void_mask_shape()
    test_halo_validation_behind_front_axle()
    test_halo_validation_fails_if_at_front_axle()
    test_halo_validation_fails_if_before_canister()
    test_halo_validation_fails_if_past_rear_axle()
    test_com_sanity_gate_catches_mm_as_m()
    test_com_sanity_gate_valid()
    test_com_sanity_gate_outside_car_length()
    print("\nAll fixed_hardware tests passed.")
```

---

## 9. S2 --- `bounding_volumes.py` (Complete File Specification)

### 9.1 Purpose

Compute the legal bounding region for each of the four phi-grid components given W, d_halo, and the forbidden cylinders from S6. The sidepod corridor is the critical computation --- it changes with every W step.

### 9.2 Sidepod Corridor Derivation (Most Important Geometry)

```
In front-to-rear (x) direction:

  Front wheel forbidden cylinder:
    x_max = front_cylinder.x_max_m  (furthest-back edge of front wheel assembly)

  Rear wheel forbidden cylinder:
    x_min = rear_cylinder.x_min_m   (furthest-forward edge of rear wheel assembly)

  Sidepod corridor:
    sidepod_x_min = front_cylinder.x_max_m + WHEEL_CLEARANCE_M
    sidepod_x_max = rear_cylinder.x_min_m  - WHEEL_CLEARANCE_M
    sidepod_length = sidepod_x_max - sidepod_x_min

  As W increases (rear axle moves back):
    rear_cylinder.x_min_m increases -> sidepod_x_max increases -> corridor gets longer

  As W decreases (rear axle moves forward):
    rear_cylinder.x_min_m decreases -> sidepod_x_max decreases -> corridor gets shorter

  At W=120 mm, sidepod_length must be > 0 (otherwise the rules are violated).
  If sidepod_length <= 0, raise ValueError immediately.
```

### 9.3 Full File

```python
"""
bounding_volumes.py --- Legal bounding regions for all phi-grid components.

Contains:
  - BoundingRegion class (arbitrary shape support)
  - _point_in_polygon_vectorised (ray-casting polygon test)
  - BoundingVolumes dataclass (all four volumes for one W, d_halo)
  - compute_bounding_volumes() entry point
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import math

from geometry_contract import (
    GRID_SPACING_M, GRID_SPACING_MM, WHEEL_CLEARANCE_M,
    mm_to_m, grid_cells, validate_W, validate_d_halo,
)
from fixed_hardware import ForbiddenCylinder


# ?? BoundingRegion and polygon helpers ????????????????????????????????????
# (Paste the full BoundingRegion class and _point_in_polygon_vectorised
#  function from Section 7.2 here --- they live in this file.)


@dataclass(frozen=True)
class BoundingVolumes:
    """
    All four bounding regions for one (W, d_halo) configuration.

    Each region defines the legal 3D space for one component's phi grid.
    The region knows its own shape (nx, ny, nz) and origin in metres.

    sidepod_x_min_m and sidepod_x_max_m are stored separately for use
    by PhiGrid's hard constraint setup and phi_updater's symmetry logic.
    """
    nose:      BoundingRegion
    sidepod:   BoundingRegion   # RIGHT half only (y >= 0 side)
    rearpod:   BoundingRegion
    main_body: BoundingRegion

    # Sidepod corridor limits --- stored for external use
    sidepod_x_min_m: float
    sidepod_x_max_m: float

    # The W and d_halo that produced this configuration
    W_mm: float
    d_halo_mm: float

    @property
    def sidepod_length_m(self) -> float:
        return self.sidepod_x_max_m - self.sidepod_x_min_m

    @property
    def W_m(self) -> float:
        return mm_to_m(self.W_mm)

    def get(self, component: str) -> BoundingRegion:
        """Look up a region by component name."""
        mapping = {
            "nose": self.nose,
            "sidepod": self.sidepod,
            "rearpod": self.rearpod,
            "main_body": self.main_body,
        }
        if component not in mapping:
            raise ValueError(f"Unknown component '{component}'. Valid: {list(mapping)}")
        return mapping[component]


def compute_bounding_volumes(
    W_mm: float,
    d_halo_mm: float,
    front_cylinder: ForbiddenCylinder,
    rear_cylinder: ForbiddenCylinder,
    rule_envelope: "RuleEnvelope",
) -> BoundingVolumes:
    """
    Compute all four bounding volumes given outer loop scalars and wheel geometry.

    Args:
        W_mm: wheelbase in mm. Must be in [120, 140].
        d_halo_mm: halo-canister distance in mm. Must be in [0, W+16].
        front_cylinder: from S6, defines front wheel forbidden zone
        rear_cylinder: from S6, defines rear wheel forbidden zone
        rule_envelope: RuleEnvelope containing absolute car dimensions.
                       ? UNRESOLVED U6 until UAE rule dimensions are confirmed.

    Returns:
        BoundingVolumes with all four regions.
    """
    validate_W(W_mm)
    validate_d_halo(d_halo_mm, W_mm)

    # ? UNRESOLVED U6: Absolute envelope dimensions not yet provided.
    # The RuleEnvelope must supply:
    #   y_body_half_m: half-width of car body in y (both sides)
    #   y_sidepod_inner_m: inner wall of sidepod in y (outer edge of body)
    #   y_sidepod_outer_m: outer edge of sidepod in y
    #   z_floor_m: bottom of all volumes (should be 0.0 or close to track surface)
    #   z_nose_top_m: maximum z of nose volume
    #   z_sidepod_top_m: maximum z of sidepod volume
    #   z_rearpod_top_m: maximum z of rearpod volume
    #   z_body_top_m: maximum z of main body volume
    #   rearpod_max_length_m: maximum x-length of rearpod from rear axle
    if rule_envelope is None:
        raise NotImplementedError(
            "? UNRESOLVED U6: UAE competition regulation envelope dimensions not provided. "
            "Create a RuleEnvelope object with y_body_half_m, y_sidepod_inner_m, "
            "y_sidepod_outer_m, z_floor_m, z_nose_top_m, z_sidepod_top_m, "
            "z_rearpod_top_m, z_body_top_m, rearpod_max_length_m --- all in metres."
        )

    W_m = mm_to_m(W_mm)
    d_halo_m = mm_to_m(d_halo_mm)

    # ?? Sidepod corridor (THE W-DEPENDENT COMPUTATION) ????????????????????
    sidepod_x_min = front_cylinder.x_max_m + WHEEL_CLEARANCE_M
    sidepod_x_max = rear_cylinder.x_min_m  - WHEEL_CLEARANCE_M
    sidepod_length = sidepod_x_max - sidepod_x_min

    if sidepod_length <= 0.0:
        raise ValueError(
            f"Sidepod corridor has zero or negative length at W={W_mm} mm: "
            f"x_min={sidepod_x_min:.5f} m, x_max={sidepod_x_max:.5f} m, "
            f"length={sidepod_length*1000:.2f} mm. "
            f"Check wheel clearance ({WHEEL_CLEARANCE_M*1000:.1f} mm) and "
            f"wheel half-width settings. The rules require a positive sidepod corridor."
        )

    re = rule_envelope   # shorthand

    # ?? Nose volume ????????????????????????????????????????????????????????
    # x: from 0 (front axle) to d_halo_m
    # y: from -y_body_half to +y_body_half (full width, symmetric)
    # z: from z_floor to z_nose_top
    nose_nx = grid_cells(d_halo_mm)   # might be 1 if d_halo=0
    nose_ny = grid_cells((re.y_body_half_m * 2) * 1000)
    nose_nz = grid_cells((re.z_nose_top_m - re.z_floor_m) * 1000)
    nose = BoundingRegion(
        component = "nose",
        origin_m  = (0.0, -re.y_body_half_m, re.z_floor_m),
        shape     = (nose_nx, nose_ny, nose_nz),
        # No polygon or voxel override --- nose uses full box
        polygon_yz_m = None,
        voxel_mask   = None,
    )

    # ?? Sidepod volume (right half only, y >= 0) ???????????????????????????
    # x: from sidepod_x_min to sidepod_x_max (W-dependent!)
    # y: from y_sidepod_inner to y_sidepod_outer (right side only, y > 0)
    # z: from z_floor to z_sidepod_top
    # The polygon_yz_m can be set to a non-rectangular cross-section if needed.
    # For now, use box mode. Once UAE envelope gives a non-rectangular profile,
    # replace with polygon_yz_m=[(y1,z1), ...].
    sp_x_length_mm = (sidepod_x_max - sidepod_x_min) * 1000
    sp_y_length_mm = (re.y_sidepod_outer_m - re.y_sidepod_inner_m) * 1000
    sp_z_length_mm = (re.z_sidepod_top_m - re.z_floor_m) * 1000
    sidepod = BoundingRegion(
        component = "sidepod",
        origin_m  = (sidepod_x_min, re.y_sidepod_inner_m, re.z_floor_m),
        shape     = (
            grid_cells(sp_x_length_mm),
            grid_cells(sp_y_length_mm),
            grid_cells(sp_z_length_mm),
        ),
        polygon_yz_m = None,   # Switch to polygon for non-rectangular cross-sections
        voxel_mask   = None,
    )

    # ?? Rearpod volume ?????????????????????????????????????????????????????
    # x: from W_m (rear axle) to W_m + rearpod_max_length_m
    # y: from -y_body_half to +y_body_half
    # z: from z_floor to z_rearpod_top
    rp_x_mm = re.rearpod_max_length_m * 1000
    rp_y_mm = re.y_body_half_m * 2 * 1000
    rp_z_mm = (re.z_rearpod_top_m - re.z_floor_m) * 1000
    rearpod = BoundingRegion(
        component = "rearpod",
        origin_m  = (W_m, -re.y_body_half_m, re.z_floor_m),
        shape     = (
            grid_cells(rp_x_mm),
            grid_cells(rp_y_mm),
            grid_cells(rp_z_mm),
        ),
        polygon_yz_m = None,
        voxel_mask   = None,
    )

    # ?? Main body volume ???????????????????????????????????????????????????
    # x: from 0 (front axle) to W_m + rearpod_max_length_m
    # y: from -y_body_half to +y_body_half
    # z: from z_floor to z_body_top
    # Note: this x range overlaps with nose and rearpod.
    # That is intentional --- the main body grid covers the full car length.
    # Hard constraints in S3 handle the attachment at those interfaces.
    body_x_mm = (W_m + re.rearpod_max_length_m) * 1000
    body_y_mm = re.y_body_half_m * 2 * 1000
    body_z_mm = (re.z_body_top_m - re.z_floor_m) * 1000
    main_body = BoundingRegion(
        component = "main_body",
        origin_m  = (0.0, -re.y_body_half_m, re.z_floor_m),
        shape     = (
            grid_cells(body_x_mm),
            grid_cells(body_y_mm),
            grid_cells(body_z_mm),
        ),
        polygon_yz_m = None,
        voxel_mask   = None,
    )

    return BoundingVolumes(
        nose             = nose,
        sidepod          = sidepod,
        rearpod          = rearpod,
        main_body        = main_body,
        sidepod_x_min_m  = sidepod_x_min,
        sidepod_x_max_m  = sidepod_x_max,
        W_mm             = W_mm,
        d_halo_mm        = d_halo_mm,
    )


@dataclass
class RuleEnvelope:
    """
    ? UNRESOLVED U6: Absolute car envelope dimensions from UAE competition rules.

    All values in metres. Must be provided before bounding volumes can be computed.
    Fill these in once UAE rules are confirmed.

    Until then, any call to compute_bounding_volumes() raises NotImplementedError.
    """
    y_body_half_m:       float   # Half-width of car body in y (e.g. 0.030 m = 30 mm)
    y_sidepod_inner_m:   float   # Inner y-edge of sidepod corridor (outer edge of body)
    y_sidepod_outer_m:   float   # Outer y-edge of sidepod
    z_floor_m:           float   # Bottom of all components (usually 0.0 m = track surface)
    z_nose_top_m:        float   # Maximum z of nose volume
    z_sidepod_top_m:     float   # Maximum z of sidepod volume
    z_rearpod_top_m:     float   # Maximum z of rearpod volume
    z_body_top_m:        float   # Maximum z of main body volume
    rearpod_max_length_m: float  # Max x-extent of rearpod from rear axle
```

### 9.4 Tests for S2 (`tests/test_bounding_volumes.py`)

```python
"""
Tests for bounding_volumes.py.
Use a stub RuleEnvelope with made-up but physically reasonable dimensions.
Once U6 is resolved, update the stub values.
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bounding_volumes import (
    BoundingRegion, BoundingVolumes, RuleEnvelope, compute_bounding_volumes,
    _point_in_polygon_vectorised,
)
from fixed_hardware import ForbiddenCylinder
from geometry_contract import R_WHEEL_M, WHEEL_CLEARANCE_M, mm_to_m
import numpy as np

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

# Stub rule envelope with physically plausible dimensions
STUB_RE = RuleEnvelope(
    y_body_half_m       = 0.030,   # 30 mm half-width
    y_sidepod_inner_m   = 0.030,
    y_sidepod_outer_m   = 0.060,
    z_floor_m           = 0.000,
    z_nose_top_m        = 0.040,
    z_sidepod_top_m     = 0.035,
    z_rearpod_top_m     = 0.035,
    z_body_top_m        = 0.045,
    rearpod_max_length_m= 0.030,
)

def _make_cylinders(W_mm):
    W_m = mm_to_m(W_mm)
    r = R_WHEEL_M + WHEEL_CLEARANCE_M
    front = ForbiddenCylinder(0.0, 0.0, 0.015, r, 0.008)
    rear  = ForbiddenCylinder(W_m, 0.0, 0.015, r, 0.008)
    return front, rear

def test_sidepod_length_decreases_with_W():
    bv120 = compute_bounding_volumes(120.0, 10.0, *_make_cylinders(120.0), STUB_RE)
    bv140 = compute_bounding_volumes(140.0, 10.0, *_make_cylinders(140.0), STUB_RE)
    assert bv140.sidepod_length_m > bv120.sidepod_length_m, (
        f"Sidepod should be longer at W=140 than W=120. "
        f"Got {bv140.sidepod_length_m:.4f} vs {bv120.sidepod_length_m:.4f}"
    )
    _pass("test_sidepod_length_decreases_with_W")

def test_sidepod_length_positive_at_W_min():
    bv = compute_bounding_volumes(120.0, 10.0, *_make_cylinders(120.0), STUB_RE)
    assert bv.sidepod_length_m > 0, f"sidepod_length={bv.sidepod_length_m:.4f}"
    _pass("test_sidepod_length_positive_at_W_min")

def test_nose_length_matches_d_halo():
    from geometry_contract import GRID_SPACING_M
    d_halo_mm = 20.0
    bv = compute_bounding_volumes(130.0, d_halo_mm, *_make_cylinders(130.0), STUB_RE)
    nose_length_m = bv.nose.nx * GRID_SPACING_M
    assert abs(nose_length_m - mm_to_m(d_halo_mm)) <= GRID_SPACING_M, (
        f"Nose length {nose_length_m*1000:.2f} mm should ? d_halo={d_halo_mm} mm"
    )
    _pass("test_nose_length_matches_d_halo")

def test_d_halo_zero_gives_degenerate_nose():
    bv = compute_bounding_volumes(130.0, 0.0, *_make_cylinders(130.0), STUB_RE)
    assert bv.nose.nx == 1, f"nx={bv.nose.nx}, expected 1 for d_halo=0"
    _pass("test_d_halo_zero_gives_degenerate_nose")

def test_W_out_of_range_raises():
    for bad_W in [119.9, 140.1, 0.0]:
        try:
            compute_bounding_volumes(bad_W, 10.0, *_make_cylinders(130.0), STUB_RE)
            _fail("test_W_out_of_range_raises", f"W={bad_W} should raise")
        except ValueError:
            pass
    _pass("test_W_out_of_range_raises")

def test_d_halo_out_of_range_raises():
    try:
        compute_bounding_volumes(130.0, 147.0, *_make_cylinders(130.0), STUB_RE)
        _fail("test_d_halo_out_of_range_raises", "d_halo=147 > W+16=146 should raise")
    except ValueError:
        _pass("test_d_halo_out_of_range_raises")

def test_all_shapes_are_positive_ints():
    bv = compute_bounding_volumes(130.0, 15.0, *_make_cylinders(130.0), STUB_RE)
    for comp in ("nose", "sidepod", "rearpod", "main_body"):
        region = bv.get(comp)
        for dim in region.shape:
            assert isinstance(dim, int) and dim > 0, f"{comp} dim={dim}"
    _pass("test_all_shapes_are_positive_ints")

def test_rearpod_origin_x_equals_W():
    bv = compute_bounding_volumes(130.0, 15.0, *_make_cylinders(130.0), STUB_RE)
    assert abs(bv.rearpod.origin_m[0] - mm_to_m(130.0)) < 1e-9
    _pass("test_rearpod_origin_x_equals_W")

def test_sidepod_is_right_half_only():
    bv = compute_bounding_volumes(130.0, 15.0, *_make_cylinders(130.0), STUB_RE)
    # Sidepod origin y must be >= 0 (right half)
    assert bv.sidepod.origin_m[1] >= 0.0, (
        f"Sidepod y origin={bv.sidepod.origin_m[1]:.4f} --- should be >= 0 (right half only)"
    )
    _pass("test_sidepod_is_right_half_only")

def test_polygon_point_in_polygon():
    # Square polygon: (0,0),(1,0),(1,1),(0,1)
    poly = [(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)]
    ys = np.array([0.5, 1.5, -0.1, 0.5])
    zs = np.array([0.5, 0.5,  0.5, 1.5])
    inside = _point_in_polygon_vectorised(ys, zs, poly)
    assert inside[0] == True,  "Centre point should be inside"
    assert inside[1] == False, "Outside y should be outside"
    assert inside[2] == False, "Outside z should be outside"
    assert inside[3] == False, "Outside z should be outside"
    _pass("test_polygon_point_in_polygon")

def test_bounding_region_box_mode_all_valid():
    region = BoundingRegion("nose", (0.0, -0.015, 0.0), (10, 10, 10))
    mask = region.valid_mask()
    assert mask.shape == (10, 10, 10)
    assert mask.all(), "Box mode: all cells should be valid"
    _pass("test_bounding_region_box_mode_all_valid")

def test_bounding_region_polygon_mode():
    # Triangular cross-section in y-z: (0,0),(0.01,0),(0.005,0.01)
    poly = [(0.0,0.0),(0.01,0.0),(0.005,0.01)]
    from geometry_contract import GRID_SPACING_M
    region = BoundingRegion(
        component="sidepod",
        origin_m=(0.0, 0.0, 0.0),
        shape=(5, 50, 50),
        polygon_yz_m=poly,
    )
    mask = region.valid_mask()
    assert mask.shape == (5, 50, 50)
    # Centre of triangle in y-z should be inside
    assert mask.any(), "Some cells should be inside the triangle"
    _pass("test_bounding_region_polygon_mode")

def test_bounding_region_voxel_mode():
    vox = np.zeros((5, 10, 10), dtype=bool)
    vox[2, 5, 5] = True
    region = BoundingRegion("main_body", (0.0,0.0,0.0), (5,10,10), voxel_mask=vox)
    mask = region.valid_mask()
    assert mask[2, 5, 5] == True
    assert mask[0, 0, 0] == False
    _pass("test_bounding_region_voxel_mode")

if __name__ == "__main__":
    test_sidepod_length_decreases_with_W()
    test_sidepod_length_positive_at_W_min()
    test_nose_length_matches_d_halo()
    test_d_halo_zero_gives_degenerate_nose()
    test_W_out_of_range_raises()
    test_d_halo_out_of_range_raises()
    test_all_shapes_are_positive_ints()
    test_rearpod_origin_x_equals_W()
    test_sidepod_is_right_half_only()
    test_polygon_point_in_polygon()
    test_bounding_region_box_mode_all_valid()
    test_bounding_region_polygon_mode()
    test_bounding_region_voxel_mode()
    print("\nAll bounding_volumes tests passed.")
```

---

## 10. S3 --- `phi_grid.py` --- Key Specifications

(Full implementation follows S1/S6/S2 patterns above. Every method listed here must be implemented with the exact signature shown.)

### 10.1 Class Signature

```python
@dataclass
class PhiGrid:
    component: str                  # "nose"|"sidepod"|"rearpod"|"main_body"
    bv: BoundingRegion              # from S2 --- defines shape and valid region
    grid: np.ndarray                # float32, shape (nx, ny, nz)
    hard_mask_solid: np.ndarray     # bool, shape (nx,ny,nz) --- forced phi < 0
    hard_mask_air:   np.ndarray     # bool, shape (nx,ny,nz) --- forced phi > 0
    # hard_mask_air includes: bbox walls + forbidden zones + hardware voids + invalid region
```

### 10.2 Initialization

```python
def init(self, mode: str = "sphere") -> None:
    """
    Initialize phi field.

    mode="sphere":
        Compute signed distance to sphere inscribed in bounding volume.
        Centre at (nx/2, ny/2, nz/2) in grid index space.
        Radius = 0.7 x min(nx, ny, nz) / 2 x GRID_SPACING_M
        phi[i,j,k] = sqrt((i-cx)^2 + (j-cy)^2 + (k-cz)^2) x GRID_SPACING_M - radius
        Result: phi < 0 inside sphere, phi > 0 outside, phi = 0 on surface.
        dtype: float32.

    mode="slab":
        phi < 0 in lower half of z dimension, phi > 0 in upper half.
        Used for sidepod test fixtures.

    mode="random":
        sphere phi field + smooth random noise at amplitude 0.5 x GRID_SPACING_M.
        Noise is generated with np.random.default_rng(seed=42 unless seed provided).
        Used by Part 3's evolutionary perturbation.

    After computing initial values, MUST call apply_hard_constraints().
    """
```

### 10.3 Hard Constraint Setup

Call `build_hard_masks()` once at construction time. This computes the `hard_mask_solid` and `hard_mask_air` arrays from the bounding region and void masks.

```python
@staticmethod
def build_hard_masks(
    bv: BoundingRegion,
    void_masks: list[np.ndarray],     # list of bool arrays, each shape (nx,ny,nz)
                                       # cells where phi must be > 0 (hardware voids etc.)
    attachment_faces: list[str],       # which faces have attachment strips
                                       # e.g. ["rear", "front", "inner_y"]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (hard_mask_solid, hard_mask_air).

    hard_mask_solid: cells forced phi < 0 (attachment faces)
    hard_mask_air:   cells forced phi > 0 = union of:
                     - 1-cell border around the bounding volume
                     - all void_masks (hardware voids, forbidden zones)
                     - cells outside bv.valid_mask() (non-rectangular region)

    Attachment face strips (ATTACHMENT_STRIP_MM wide = ceil(1.0/0.3) = 4 cells):
        "rear"    -> x = nx-1 to nx-1-3  (4 cells from rear face, all j,k)
        "front"   -> x = 0 to 3           (4 cells from front face)
        "inner_y" -> y = 0 to 3           (4 cells from y=0 face, for sidepod)

    Bounding box walls (1-cell border):
        x = 0, x = nx-1
        y = 0, y = ny-1
        z = 0, z = nz-1

    Call _validate_masks() after building to confirm no overlap.
    """
```

```python
def _validate_masks(self) -> None:
    """
    Raise ValueError if any cell is in both hard_mask_solid and hard_mask_air.
    This indicates a setup error (e.g. attachment face inside a void region).
    """
    overlap = self.hard_mask_solid & self.hard_mask_air
    n_overlap = int(overlap.sum())
    if n_overlap > 0:
        raise ValueError(
            f"{self.component}: {n_overlap} cells are in BOTH solid and air masks. "
            f"This is a constraint setup error. Check: do any void masks overlap "
            f"with attachment face strips? Check attachment face coordinates vs "
            f"hardware void positions."
        )
```

### 10.4 Hard Constraint Table Per Component

| Component | hard_mask_solid faces | hard_mask_air additions |
|---|---|---|
| nose | rear face (x = nx-4 to nx-1, all y,z) | bbox border + `bv.valid_mask()` complement |
| sidepod | inner y face (y = 0 to 3, all x,z) | bbox border + wheel forbidden zone masks + `bv.valid_mask()` complement |
| rearpod | front face (x = 0 to 3, all y,z) | bbox border + `bv.valid_mask()` complement |
| main_body | sidepod attachment walls (thin y strip at y_sidepod_inner grid index) | bbox border + halo void + canister void + front axle void + rear axle void + `bv.valid_mask()` complement |

### 10.5 Save and Load

```python
def save(self, candidate_id: str, out_dir: str) -> str:
    """
    Save grid as .npy file.
    Filename: phi_{component}_{candidate_id}.npy
    Returns: absolute path string.
    This path goes into CandidateRecord.phi_grid_snapshot_paths[component].
    """
    path = Path(out_dir) / f"phi_{self.component}_{candidate_id}.npy"
    np.save(str(path), self.grid)
    return str(path.resolve())

def load(self, path: str) -> None:
    """
    Load grid from .npy file. Re-enforce hard constraints after load.
    Raise ValueError if loaded shape != self.bv.shape.
    """
    loaded = np.load(path).astype(np.float32)
    if loaded.shape != self.bv.shape:
        raise ValueError(
            f"{self.component}: loaded grid shape {loaded.shape} != "
            f"expected {self.bv.shape}. Wrong file or wrong bounding volume."
        )
    self.grid = loaded
    self.apply_hard_constraints()
```

### 10.6 Warm-Start Remap

```python
def remap(self, new_bv: BoundingRegion, new_hard_masks: tuple[np.ndarray, np.ndarray]) -> "PhiGrid":
    """
    Trilinear interpolation of self.grid into a new bounding volume.
    Returns a NEW PhiGrid. self is not modified.

    Steps:
    1. Compute coordinate mapping from new grid indices -> old grid indices
    2. scipy.ndimage.map_coordinates(self.grid, coords, order=1, mode='nearest')
    3. Construct new PhiGrid with new_bv and new hard masks
    4. Set new_phi.grid = interpolated result (float32)
    5. Call new_phi.apply_hard_constraints()
    6. Return new_phi

    Edge case: if new_bv.shape == self.bv.shape, still run the interpolation
    (origin may have shifted). Do not short-circuit with a copy.
    """
```

---

## 11. S5 --- `mass_com_calculator.py` --- Key Specifications

### 11.1 Core Volume Integral

```python
def compute_component_mass_com(
    phi: PhiGrid,
    density_kgm3: float,
) -> ComponentMassCOM:
    """
    Compute mass and COM from phi grid.

    Solid cells: phi.grid < 0.0
    Cell volume: GRID_SPACING_M^3 metres^3

    volume = count(solid cells) x GRID_SPACING_M^3
    mass = volume x density_kgm3

    COM x = origin_x + mean(ix) x GRID_SPACING_M   (for all solid cell indices ix)
    COM y = origin_y + mean(iy) x GRID_SPACING_M
    COM z = origin_z + mean(iz) x GRID_SPACING_M

    If no solid cells: return zero mass, COM at grid origin. Log a warning.
    After computing COM: call _assert_com_physical() to catch units bugs.
    Return ComponentMassCOM(name=phi.component, mass_kg=..., com_x_m=..., com_y_m=..., com_z_m=...)
    """
```

### 11.2 Sidepod Pair

```python
def compute_sidepod_pair_mass_com(right_phi: PhiGrid) -> ComponentMassCOM:
    """
    The right sidepod phi grid represents the right half only.
    The left sidepod is its y-mirror.
    The pair mass = 2 x right mass.
    The pair COM: x = right COM x, y = 0.0 (cancels by symmetry), z = right COM z.

    Returns ComponentMassCOM(name="sidepod", mass_kg=2*right_mass, com_x_m=right_x,
                              com_y_m=0.0, com_z_m=right_z)
    """
```

### 11.3 Full Component List

```python
def compute_all_machined_components(
    nose_phi: PhiGrid,
    sidepod_phi: PhiGrid,   # right half
    rearpod_phi: PhiGrid,
    body_phi: PhiGrid,
) -> list[ComponentMassCOM]:
    """
    Returns list of exactly 4 ComponentMassCOM objects:
    [nose, sidepod_pair, rearpod, main_body]
    in this exact order.
    The list is passed directly to Part 2's mass_com_ingest.ingest_mass_com().
    """
    return [
        compute_component_mass_com(nose_phi,   get_density("nose")),
        compute_sidepod_pair_mass_com(sidepod_phi),
        compute_component_mass_com(rearpod_phi, get_density("rearpod")),
        compute_component_mass_com(body_phi,    get_density("main_body")),
    ]
```

---

## 12. S4 --- `surface_extraction.py` --- Key Specifications

### 12.1 Exception Hierarchy

```python
class SurfaceExtractionError(Exception):
    """Base. Caught by S9."""

class RadiusViolation(SurfaceExtractionError):
    """Minimum radius not achieved after max repair iterations."""

class AccessibilityFailure(SurfaceExtractionError):
    is_large: bool  # True -> kill; False -> retry with smoothing
    def __init__(self, msg: str, is_large: bool):
        super().__init__(msg)
        self.is_large = is_large

class RuleViolation(SurfaceExtractionError):
    is_major: bool  # True -> hard penalty, kill; False -> project and retry
    def __init__(self, msg: str, is_major: bool):
        super().__init__(msg)
        self.is_major = is_major

class MeshQualityFailure(SurfaceExtractionError):
    """Mesh fails gate after simplification attempt."""
```

### 12.2 Pipeline Entry Point

```python
def extract_surface(phi: PhiGrid, max_radius_retries: int = MAX_EXTRACTION_RETRIES) -> trimesh.Trimesh:
    """
    Run all 6 stages. Return clean Trimesh. Raise on unrecoverable failure.

    Stage 1: Marching Cubes on phi.grid, level=0.0, spacing=(GRID_SPACING_M,)*3
             Translate vertices from index space to world coordinates using phi.bv.origin_m
             If 0 faces: raise SurfaceExtractionError("Empty mesh")

    Stage 2: Geometry repair
             trimesh.util.concatenate([max component by face count])
             trimesh.repair.fill_holes(mesh)
             trimesh.repair.fix_normals(mesh)
             trimesh.repair.fix_winding(mesh)
             Remove faces with area < 1e-12 m^2

    Stage 3: Minimum radius check (loop up to max_radius_retries)
             Estimate local radius from vertex neighbour angles
             If violations present: smooth phi.grid via Laplacian smoothing
             on neighbourhood of violated vertices, apply_hard_constraints(),
             re-run Stage 1+2, try again
             If still violated after max_radius_retries: raise RadiusViolation

    Stage 4: Tool accessibility
             Use TOOL_DIRECTIONS[phi.component]
             For each direction: find faces whose normal has dot > 0 with direction
             Cast ray from face centre along approach direction
             Face is accessible if ray does not re-enter mesh
             Sum area of inaccessible faces
             If area < SMALL_INACCESSIBLE_AREA_M2: raise AccessibilityFailure(is_large=False)
             If area >= LARGE_INACCESSIBLE_AREA_M2: raise AccessibilityFailure(is_large=True)

    Stage 5: Rule checker
             Check all vertices within bv extent (tolerance 0.1 mm)
             ? UNRESOLVED U4: Full UAE envelope check not implemented
             Raise RuleViolation(is_major=True) on envelope breach

    Stage 6: Mesh quality gate
             if not mesh.is_watertight: repair, retry, raise MeshQualityFailure if still bad
             if not mesh.is_volume: raise MeshQualityFailure
             Triangle angles: min angle > MESH_MIN_TRIANGLE_ANGLE_DEG (10 deg)
             If below threshold: mesh.simplify_quadric_decimation(percent=0.9), re-check
             If still bad: raise MeshQualityFailure

    Return cleaned mesh in metres.
    """
```

---

## 13. S9 --- `quality_gates.py` --- Key Specifications

### 13.1 GateResult

```python
@dataclass(frozen=True)
class GateResult:
    lifecycle_state: str       # Must be in ALLOWED_LIFECYCLE_STATES
    meshes: Optional[dict[str, trimesh.Trimesh]]  # None if failed before assembly
    phi_snapshot_paths: dict[str, str]             # ALWAYS populated
    stl_path: Optional[str]    # None if failed
    stl_half_path: Optional[str]  # Right-half STL for CFD; None if failed
    failure_reason: Optional[str]

    def __post_init__(self):
        if self.lifecycle_state not in ALLOWED_LIFECYCLE_STATES:
            raise ValueError(
                f"lifecycle_state='{self.lifecycle_state}' not in allowed set "
                f"{sorted(ALLOWED_LIFECYCLE_STATES)}. This is a Part 1 bug."
            )
        if not self.phi_snapshot_paths:
            raise ValueError("phi_snapshot_paths must always be populated, even on failure.")
```

### 13.2 Gate Runner Logic

```python
def run_quality_gates(
    phi_grids: dict[str, PhiGrid],
    candidate_id: str,
    out_dir: str,
    max_retries: int = MAX_EXTRACTION_RETRIES,
) -> GateResult:
    """
    STEP 0 (ALWAYS FIRST): Save phi snapshots before any extraction attempt.
                            Even if everything fails, snapshots exist.

    phi_paths = {name: phi.save(candidate_id, out_dir) for name, phi in phi_grids.items()}

    STEP 1-N: Retry loop (up to max_retries):
        Try extract_surface() on all components.
        On success: assemble STL, return "valid_simulated" (or "geometry_repaired" if attempt > 0).
        On RadiusViolation: if attempt < max_retries-1: continue; else return "geometry_rejected"
        On AccessibilityFailure(is_large=False): if attempt < max_retries-1: continue; else return "geometry_rejected"
        On AccessibilityFailure(is_large=True): return "machining_rejected" immediately (no retry)
        On RuleViolation(is_major=False): if attempt < max_retries-1: continue; else return "rule_rejected"
        On RuleViolation(is_major=True): return "rule_rejected" immediately (no retry)
        On MeshQualityFailure: if attempt < max_retries-1: continue; else return "geometry_rejected"
        On SurfaceExtractionError (base): return "geometry_rejected"

    ALL returned GateResults must have phi_snapshot_paths populated.
    stl_path is None on any failure.
    stl_half_path is None on any failure.
    failure_reason contains str(exception) on failure, None on success.
    """
```

---

## 14. S7 --- `stl_assembler.py` --- Key Specifications

### 14.1 Left Sidepod Mirror

```python
def _mirror_right_to_left(right_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Mirror right sidepod to create left sidepod.

    Steps (order matters):
    1. left = right_mesh.copy()
    2. left.vertices[:, 1] *= -1.0      # flip y coordinate
    3. left.invert()                    # flip face winding (normals reversed by mirror)
    4. trimesh.repair.fix_normals(left) # ensure consistent outward normals
    5. return left
    """
```

### 14.2 Assembly and Export

```python
def assemble_stl(
    meshes: dict[str, trimesh.Trimesh],
    candidate_id: str,
    out_dir: str,
) -> tuple[str, str]:
    """
    Returns (full_stl_path, half_stl_path).

    Full car:
    1. Mirror right sidepod -> left sidepod
    2. trimesh.util.concatenate([nose, sidepod_right, sidepod_left, rearpod, main_body])
    3. trimesh.repair.fix_winding(full_car)
    4. trimesh.repair.fix_normals(full_car)
    5. if not full_car.is_watertight: raise MeshQualityFailure("full car not watertight")
    6. full_car.export(out_dir/car_{candidate_id}_full.stl)

    Right-half STL (for Part 2 CFD --- CONFIRMED half-car simulation):
    1. trimesh.intersections.slice_mesh_plane(full_car, plane_normal=(0,1,0),
                                               plane_origin=(0,0,0), cap=True)
    2. if not right_half.is_watertight: raise MeshQualityFailure("right half not watertight")
    3. Verify: all vertices have y >= -1e-6 (numerical tolerance)
    4. right_half.export(out_dir/car_{candidate_id}_half.stl)

    Return (full_path, half_path) as absolute path strings.
    """
```

---

## 15. S8 --- `phi_updater.py` --- Key Specifications

### 15.1 Adjoint Symmetry (RESOLVED)

OpenFOAM adjoint runs on the right-half domain (CONFIRMED). Sensitivity field covers right-half surface only.

```python
def apply_adjoint_sensitivity_symmetric(
    phi_grids: dict[str, PhiGrid],
    right_half_sensitivity: np.ndarray,   # surface sensitivity on right-half mesh vertices
    right_half_mesh: trimesh.Trimesh,     # the right-half mesh (same as CFD mesh)
    dt: float,
    gradient_weights: dict[str, float],
) -> None:
    """
    Apply adjoint sensitivity to all phi grids.

    For nose, rearpod, main_body (symmetric components):
        Sensitivity applies to the full grid (both y sides).
        Map surface sensitivity -> volume velocity field using extend_velocity().
        Apply hj_update() to the full grid.

    For sidepod (right half only):
        Sensitivity applies to right-half sidepod only.
        The left sidepod is always the mirror --- it does not have its own phi grid.
        Apply hj_update() to the right sidepod grid directly.
        The left sidepod's geometry updates automatically when S7 mirrors at next extraction.

    CFD runs on right half, adjoint gives dObjective/dSurface on right-half mesh.
    For symmetric components, the right-half sensitivity is mirrored analytically:
        - x and z components of sensitivity: same on both sides (symmetric)
        - y component: negated on left side
    Since the main_body, nose, and rearpod grids cover the full car (both y sides),
    the sensitivity must be mapped to the volume considering symmetry.
    Implementation: compute the volume velocity field from right-half sensitivity,
    then apply it to the full grid (the symmetry is implicit in the phi grid structure).
    """
```

### 15.2 Hamilton-Jacobi Update

```python
def hj_update(phi: PhiGrid, velocity: np.ndarray, dt: float) -> None:
    """
    phi_new = phi_old - dt x F x |?phi|

    |?phi| computed via Godunov upwind scheme:
        For each interior cell (i,j,k):
            grad_x = Godunov_x(phi.grid, i, j, k)
            grad_y = Godunov_y(phi.grid, i, j, k)
            grad_z = Godunov_z(phi.grid, i, j, k)
            |?phi| = sqrt(grad_x^2 + grad_y^2 + grad_z^2)

    Godunov scheme for x-component (same for y and z by symmetry):
        D_minus = (phi[i,j,k] - phi[i-1,j,k]) / GRID_SPACING_M
        D_plus  = (phi[i+1,j,k] - phi[i,j,k]) / GRID_SPACING_M
        if phi[i,j,k] > 0:
            G = max(max(D_minus, 0)^2, min(D_plus, 0)^2)
        else:
            G = max(min(D_minus, 0)^2, max(D_plus, 0)^2)
        grad_x = sqrt(G)  [then combine all three for |?phi|]

    After computing: phi.grid -= dt * velocity * grad_mag
    After update: call phi.apply_hard_constraints()

    Boundary cells (i=0, i=nx-1, etc.) are handled by hard constraints,
    so they do not need one-sided differences --- the constraint enforcer overwrites them.
    """

def reinitialise_sdf(phi: PhiGrid, n_steps: int = 20, dt_reinit: float = 0.3) -> None:
    """
    Reinitialise phi as a signed distance field.
    Solve ?phi/?? + sign(phi)(|?phi| - 1) = 0 for n_steps pseudo-time steps.
    Use Godunov scheme for |?phi| (same as hj_update).
    After all steps: call phi.apply_hard_constraints().

    After reinit: |?phi| should be ? 1.0 everywhere (check: RMS should be in [0.95, 1.05]).
    """

def extend_velocity(
    phi_grid: np.ndarray,
    surface_velocity: np.ndarray,
    n_steps: int = 10,
    dt_ext: float = 0.1,
) -> np.ndarray:
    """
    Propagate surface velocity into the volume.
    Solve: ?F/?? + sign(phi) ?phi ? ?F = 0 for n_steps pseudo-time steps.
    Returns F: velocity field defined everywhere in the volume.

    surface_velocity: shape (nx, ny, nz). Non-zero only near phi=0 surface.
                      Usually computed by mapping adjoint sensitivity from mesh
                      vertices to nearby grid cells (nearest-cell assignment or
                      trilinear splatting).
    """

def combine_gradients(
    aero_gradient: np.ndarray,
    mass_gradient: np.ndarray,
    com_gradient:  np.ndarray,
    mfg_gradient:  np.ndarray,
    w_aero: float,
    w_mass: float,
    w_com:  float,
    w_mfg:  float,
) -> np.ndarray:
    """
    Normalize each gradient to unit RMS, then weighted sum.

    def _normalize(g):
        rms = sqrt(mean(g^2))
        return g / rms if rms > 1e-12 else np.zeros_like(g)

    return (_normalize(aero)*w_aero + _normalize(mass)*w_mass +
            _normalize(com)*w_com + _normalize(mfg)*w_mfg)

    This is required because gradient terms have different units and scales.
    Naive addition (without normalization) produces optimizer bias toward
    whichever gradient has the largest magnitude --- typically aero.
    """
```

---

## 16. Part 2 Interface Contract (Exact Handshake)

Every row is a real function/type that exists in Part 2 right now.

| Part 1 output | Part 2 receiver | Python type | Hard constraint |
|---|---|---|---|
| 4x `ComponentMassCOM` | `mass_com_ingest.ingest_mass_com(machined_components, fixed_hardware)` | `list[ComponentMassCOM]` from `physics_contract` | Names: `"nose"`, `"sidepod"`, `"rearpod"`, `"main_body"`. Sidepod = pair (massx2, com_y=0.0). |
| `FixedHardwareSpec` | `mass_com_ingest.ingest_mass_com(..., fixed_hardware)` | `FixedHardwareSpec` from `mass_com_ingest` | `co2_cartridge_mass_kg` MUST be `0.023` +/- 1e-9. Part 2 raises on any other value. |
| `com_z_m` from full mass report | `race_objective_adapter._assert_physical_inputs` | `float` metres | Must be in `[0.018, 0.042]` m. Wider Part 1 guard: `[0.005, 0.060]`. Violation = 10^15 s race time. |
| Right-half STL | `cfd_wrapper.run_half_car_cfd(stl_path)` | `str` path | Must pass `trimesh.is_watertight`. All vertices have y >= -1e-6. Part 2 raises `CFDRunError` otherwise. |
| `phi_grid_snapshot_paths` | `CandidateRecord.phi_grid_snapshot_paths` | `dict[str,str]` | Keys exactly: `{"nose","sidepod","rearpod","main_body"}`. Values are `.npy` absolute paths. |
| `lifecycle_state` | `CandidateRecord.lifecycle_state` | `str` | Must be one of the 8 strings in `ALLOWED_LIFECYCLE_STATES`. Part 2 raises at construction if invalid. |
| Adjoint sensitivity | `adjoint_contract.package_gradient_bundle()` -> S8 | `np.ndarray` | Right-half surface only (CONFIRMED). Part 3 routes from Part 2 adjoint -> S8. |

---

## 17. Part 3 Interface Contract (Exact Calls)

| Part 3 inner loop step | Part 1 call | Notes |
|---|---|---|
| Outer loop: new W | `compute_bounding_volumes(W_mm, d_halo_mm, front_cyl, rear_cyl, rule_env)` | Every outer iteration |
| Warm-start | `phi.remap(new_bv, new_hard_masks)` for all 4 grids | Before first inner iter of new W |
| Step 2: hard constraints | `phi.apply_hard_constraints()` for all 4 grids | Top of every inner iter |
| Step 3: extract STL | `extract_surface(phi)` x 4, then `assemble_stl(meshes, id, dir)` | Every inner iter |
| Step 4: gates | `run_quality_gates(phi_grids, id, dir)` | After extraction |
| Step 7: mass + COM | `compute_all_machined_components(nose_phi, sidepod_phi, rearpod_phi, body_phi)` | Every inner iter |
| Step 11-14: phi update | `update_phi(phi_grids, sensitivity, dt, weights)` | After adjoint run |
| Snapshot save | `phi.save(candidate_id, out_dir)` x 4 | After every inner iter, including failed |
| Snapshot load | `phi.load(path)` | Warm-start restoration |
| Evolutionary perturbation | `phi.init(mode="random")` on a copy | Every mutation step |

---

## 18. Open Items Summary (? UNRESOLVED)

These appear as `# ? UNRESOLVED UN:` comments in source. Code raises `NotImplementedError` at the call site.

| ID | File | What is needed | Consequence if skipped |
|---|---|---|---|
| U1 | `fixed_hardware.py` | Halo cross-section polygon vertices (y,z) in metres | Halo void mask is wrong -> optimizer fills the halo mount |
| U2 | `fixed_hardware.py` | CO2 canister legal position (x,y,z) in mm from competition rules | Part 2 gets wrong canister COM -> 10^15 s race time bug |
| U4 | `surface_extraction.py` | UAE regulation envelope polygon | Rule checker cannot enforce legal boundaries |
| U5 | `fixed_hardware.py` | Rear wing fixed position (x,y,z) in mm | `FixedHardwareSpec.rear_wing_com` is wrong |
| U6 | `bounding_volumes.py` | y-extents, z-extents, rearpod length from UAE rules | Cannot compute any grid sizes |

**Resolved:**
- U3 (adjoint half/full scaling): **RESOLVED** --- OpenFOAM adjoint runs on right-half domain, matching CFD setup. Sensitivities applied to right-half surface, symmetric components handled by full-grid phi update.

---

## 19. Dependencies

```
numpy>=1.24          array math, phi grids, masks, void masks
scipy>=1.10          ndimage.map_coordinates for trilinear remap
scikit-image>=0.20   skimage.measure.marching_cubes for Stage 1
trimesh>=3.22        mesh repair, watertight check, ray casting, STL export
pathlib              standard library --- file paths

From Part 2 (import at interface points only):
  sys.path.insert(0, path_to_part2_simulation)
  from physics_contract import ComponentMassCOM
  from mass_com_ingest import FixedHardwareSpec, CO2_CARTRIDGE_MASS_KG

No JAX anywhere in Part 1. JAX lives in Part 2's race_objective.py only.
Part 1 is pure NumPy throughout.
```

Install command:
```
pip install numpy scipy scikit-image trimesh
```

---

## 20. `run_all_tests.py` (Complete File)

```python
#!/usr/bin/env python3
"""
Run all Part 1 tests.
Exit 0: all pass.
Exit 1: one or more test failures.
Exit 2: runner infrastructure error (directory missing, no test files, etc.)

Mirrors the fix applied to Part 2's test runner (audit finding D1).
Never silently reports 0/0.
"""
import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent / "tests"

def main() -> int:
    if not TESTS_DIR.exists():
        print(f"ERROR: tests/ directory not found at {TESTS_DIR}", file=sys.stderr)
        print("Create the tests/ directory and add test_*.py files.", file=sys.stderr)
        return 2

    test_files = sorted(TESTS_DIR.glob("test_*.py"))
    if not test_files:
        print(f"ERROR: No test_*.py files found in {TESTS_DIR}", file=sys.stderr)
        return 2

    passed = 0
    failed = 0

    for tf in test_files:
        if not tf.is_file():
            print(f"SKIP {tf.name} (not a file)")
            continue

        result = subprocess.run(
            [sys.executable, str(tf)],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            passed += 1
            print(f"PASS {tf.name}")
        else:
            failed += 1
            print(f"FAIL {tf.name}")
            if result.stdout.strip():
                print(result.stdout)
            if result.stderr.strip():
                print(result.stderr, file=sys.stderr)

    total = passed + failed
    print(f"\nTOTAL: {passed} passed, {failed} failed ({total} test files)")

    if total == 0:
        print("ERROR: No test files ran.", file=sys.stderr)
        return 2

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## 21. Build Sequence (Exact Order)

Do not deviate from this order. Each step's tests must pass before moving to the next.

```
STEP 1 --- S1: geometry_contract.py
  Write the complete file from Section 6.1.
  Run tests/test_geometry_contract.py.
  Expected: 22 tests, all PASS.
  Gate: Do not proceed until all 22 pass.

STEP 2 --- S6: fixed_hardware.py (partial --- no place_fixed_hardware yet)
  Write: ForbiddenCylinder, HaloGeometry, FixedHardwareResult
  Write: _validate_halo_position, _assert_com_in_range
  Write: _build_cylinder_void_mask, _build_box_void_mask, _build_polygon_void_mask
  Leave place_fixed_hardware() raising NotImplementedError for U1/U2/U5.
  Run tests/test_fixed_hardware.py.
  Expected: 14 tests, all PASS (tests bypass the NotImplementedError calls).

STEP 3 --- S2: bounding_volumes.py
  Write: _point_in_polygon_vectorised (exact code from Section 7.2)
  Write: BoundingRegion class (exact code from Section 7.2)
  Write: RuleEnvelope dataclass
  Write: BoundingVolumes dataclass
  Write: compute_bounding_volumes() --- with NotImplementedError for U6 when rule_envelope is None
  Run tests/test_bounding_volumes.py.
  Expected: 13 tests, all PASS (tests use STUB_RE which is not None).

STEP 4 --- S3: phi_grid.py
  Write: PhiGrid dataclass
  Write: build_hard_masks() static method
  Write: _validate_masks()
  Write: apply_hard_constraints()
  Write: init(mode="sphere") --- sphere mode only first
  Write: save() and load()
  Defer: init("random"), init("slab"), remap() --- add in Step 10.
  Run tests/test_phi_grid.py (first subset --- no remap tests yet).
  Expected: 8 tests pass.

STEP 5 --- S5: mass_com_calculator.py
  Write: _assert_com_physical()
  Write: COMRangeError exception
  Write: compute_component_mass_com()
  Write: compute_sidepod_pair_mass_com()
  Write: compute_all_machined_components()
  Run tests/test_mass_com_calculator.py.
  Expected: 10 tests, all PASS.

STEP 6 --- S4: surface_extraction.py (Stages 1 and 2 only)
  Write: all exception classes
  Write: _stage1_marching_cubes()
  Write: _stage2_repair()
  Write: extract_surface() --- call only stages 1+2, raise NotImplementedError for 3-6
  Run tests/test_surface_extraction.py (first 3 tests only --- clean sphere, empty, repair).

STEP 7 --- S4: surface_extraction.py (Stages 3-6)
  Write: _estimate_local_radii()
  Write: _smooth_phi_near_violations()
  Write: _stage3_radius_check()
  Write: _find_inaccessible_faces() using trimesh ray casting
  Write: _stage4_accessibility()
  Write: _stage5_rule_checker() --- bbox check + NotImplementedError for U4
  Write: _stage6_quality_gate()
  Update extract_surface() to call all 6 stages.
  Run tests/test_surface_extraction.py (all tests).
  Expected: all PASS.

STEP 8 --- S9: quality_gates.py
  Write: GateResult dataclass
  Write: run_quality_gates()
  Run tests/test_quality_gates.py.
  Expected: 9 tests, all PASS.
  Critical check: ALL emitted lifecycle states must be in ALLOWED_LIFECYCLE_STATES.

STEP 9 --- S7: stl_assembler.py
  Write: _mirror_right_to_left()
  Write: _cut_right_half()
  Write: assemble_stl()
  Run tests/test_stl_assembler.py.
  Expected: 7 tests, all PASS.
  Critical: right-half STL must be watertight, no y < 0 vertices.

STEP 10 --- Integration test
  Write tests/test_integration_part1_part2.py.
  Run it. Expected: 5 tests, all PASS.
  This is the first test that imports Part 2 code directly.
  If any Part 2 test breaks: stop and fix before continuing.

STEP 11 --- S3: phi_grid.py remap + remaining init modes
  Add init("slab") and init("random").
  Add remap().
  Run full tests/test_phi_grid.py.
  Expected: all tests PASS including remap tests.

STEP 12 --- S8: phi_updater.py
  Write: combine_gradients()
  Write: extend_velocity()
  Write: _godunov_grad_mag()
  Write: hj_update()
  Write: reinitialise_sdf()
  Write: apply_adjoint_sensitivity_symmetric()
  Write: update_phi() (master entry point)
  Run tests/test_phi_updater.py.
  Expected: 8 tests, all PASS.

FINAL --- Run all tests
  python run_all_tests.py
  Expected: TOTAL: 10 passed, 0 failed (10 test files)
  Expected: All 82 Part 2 tests still pass (run Part 2's run_all_tests.py too --- no regression).
```

---

## 22. Quick Reference: Every Function Signature

```python
# geometry_contract.py
mm_to_m(mm: float) -> float
m_to_mm(m: float) -> float
gcm3_to_kgm3(g_cm3: float) -> float
grid_cells(length_mm: float) -> int
get_density(component: str) -> float
validate_W(W_mm: float) -> None
validate_d_halo(d_halo_mm: float, W_mm: float) -> None

# fixed_hardware.py
place_fixed_hardware(W_mm, halo_geometry, canister_com_mm, canister_box_half_size_mm,
                     wheel_axle_mass_kg, wheel_axle_com_mm, wheel_x_half_width_mm,
                     wheel_axle_z_mm, rear_wing_mass_kg, rear_wing_com_mm,
                     body_grid_shape, body_grid_origin_m) -> FixedHardwareResult
_validate_halo_position(halo, canister_x_m, W_m) -> None
_assert_com_in_range(label, com_m, W_m) -> None
_build_cylinder_void_mask(grid_shape, grid_origin_m, cylinder) -> np.ndarray
_build_box_void_mask(grid_shape, grid_origin_m, x_range_m, y_range_m, z_range_m) -> np.ndarray
_build_polygon_void_mask(grid_shape, grid_origin_m, x_min_m, x_max_m, polygon_yz_m) -> np.ndarray

# bounding_volumes.py
_point_in_polygon_vectorised(ys, zs, polygon_yz) -> np.ndarray
BoundingRegion.valid_mask() -> np.ndarray
compute_bounding_volumes(W_mm, d_halo_mm, front_cylinder, rear_cylinder, rule_envelope) -> BoundingVolumes
BoundingVolumes.get(component: str) -> BoundingRegion

# phi_grid.py
PhiGrid.init(mode: str = "sphere") -> None
PhiGrid.build_hard_masks(bv, void_masks, attachment_faces) -> tuple[np.ndarray, np.ndarray]
PhiGrid._validate_masks() -> None
PhiGrid.apply_hard_constraints() -> None
PhiGrid.save(candidate_id: str, out_dir: str) -> str
PhiGrid.load(path: str) -> None
PhiGrid.remap(new_bv, new_hard_masks) -> PhiGrid

# mass_com_calculator.py
compute_component_mass_com(phi: PhiGrid, density_kgm3: float) -> ComponentMassCOM
compute_sidepod_pair_mass_com(right_phi: PhiGrid) -> ComponentMassCOM
compute_all_machined_components(nose_phi, sidepod_phi, rearpod_phi, body_phi) -> list[ComponentMassCOM]

# surface_extraction.py
extract_surface(phi: PhiGrid, max_radius_retries: int = 3) -> trimesh.Trimesh

# quality_gates.py
run_quality_gates(phi_grids: dict, candidate_id: str, out_dir: str, max_retries: int = 3) -> GateResult

# stl_assembler.py
assemble_stl(meshes: dict, candidate_id: str, out_dir: str) -> tuple[str, str]

# phi_updater.py
combine_gradients(aero_gradient, mass_gradient, com_gradient, mfg_gradient,
                  w_aero, w_mass, w_com, w_mfg) -> np.ndarray
extend_velocity(phi_grid, surface_velocity, n_steps=10, dt_ext=0.1) -> np.ndarray
hj_update(phi: PhiGrid, velocity: np.ndarray, dt: float) -> None
reinitialise_sdf(phi: PhiGrid, n_steps=20, dt_reinit=0.3) -> None
update_phi(phi_grids, right_half_sensitivity, right_half_mesh, dt, gradient_weights) -> None
```
