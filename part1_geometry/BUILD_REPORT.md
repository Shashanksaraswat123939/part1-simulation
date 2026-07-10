# BUILD_REPORT.md — Part 1 Geometry: Full Audit and Build Report

**Date:** 2026-07-10
**Status:** Part 1 is FUNCTIONALLY COMPLETE with placeholders. All 10 test suites pass.

---

## Executive Summary

Part 1 (part1_geometry) implements the geometry pipeline for a STEM Racing car
optimization system. It takes outer-loop scalars (wheelbase W, halo distance d_halo)
and produces:
1. Bounding volumes for 4 machined components (nose, sidepod, rearpod, main_body)
2. Level-set phi grids with hard constraints
3. Mass and centre-of-mass calculations
4. Surface extraction (marching cubes → mesh repair → quality gates)
5. STL assembly (full car + right-half for CFD)
6. Quality gate runner with lifecycle states

**All 10 test suites pass (10/10).** All 6 stub modules have been implemented.
12 placeholder items remain — see PLACEHOLDERS.md for details.

---

## Module Status

| Module               | Status      | Tests | Notes                                    |
|----------------------|-------------|-------|------------------------------------------|
| geometry_contract.py | ✅ Complete | 22    | Single source of truth. No placeholders.  |
| bounding_volumes.py  | ✅ Complete | 13    | Fixed: missing imports (validate_W, etc) |
| fixed_hardware.py    | ✅ Complete | 14    | Fixed: missing imports (COM_Z bounds, etc)|
| phi_grid.py          | ✅ Complete | 11    | Implemented from stub                     |
| mass_com_calculator.py| ✅ Complete| 7     | Implemented from stub                     |
| phi_updater.py       | ✅ Complete | 8     | Implemented from stub                     |
| surface_extraction.py| ✅ Complete | 6     | Implemented from stub                     |
| stl_assembler.py     | ✅ Complete | 4     | Implemented from stub                     |
| quality_gates.py     | ✅ Complete | 5     | Implemented from stub                     |

**Total tests:** 90 individual test assertions across 10 test files.

---

## Bugs Fixed During Audit

### 1. bounding_volumes.py — Missing imports (CRITICAL)
**Bug:** `validate_W`, `validate_d_halo`, `mm_to_m`, `grid_cells`, `WHEEL_CLEARANCE_M`
were used but never imported. `ForbiddenCylinder` was referenced in type hints without import.
**Fix:** Added import block from `geometry_contract` + `TYPE_CHECKING` guard for
`ForbiddenCylinder` (avoids circular import).

### 2. fixed_hardware.py — Missing imports (CRITICAL)
**Bug:** `COM_Z_LOWER_BOUND_M`, `COM_Z_UPPER_BOUND_M`, `R_WHEEL_M`, `validate_W` were
referenced but not imported from `geometry_contract`.
**Fix:** Added all missing names to the existing `from geometry_contract import (...)` block.

### 3. phi_grid.py — Hard constraint idempotency
**Bug:** `apply_hard_constraints()` used `phi = -|phi| - dx` which changes the value
every call (not idempotent). Save/load roundtrip test failed because load calls
apply_hard_constraints again.
**Fix:** Changed to: only modify cells that violate the constraint (phi >= 0 for solid,
phi <= 0 for air). Set to ±GRID_SPACING_M.

### 4. phi_grid.py — Attachment strip / bbox wall overlap
**Bug:** Attachment strips (4 cells wide) overlapped with bbox border walls (1 cell)
at shared faces, causing a ValueError.
**Fix:** Bbox walls are skipped on faces that have attachment strips. Remaining overlaps
between void masks and attachment strips are resolved by giving air (void) priority
over solid (attachment).

---

## Modules Implemented From Scratch

### phi_grid.py
- `PhiGrid` dataclass with component, bv, grid, hard_mask_solid, hard_mask_air
- `init(mode="sphere"|"slab"|"random")` — initial phi field
- `apply_hard_constraints()` — idempotent enforcement of solid/air masks
- `build_hard_masks()` — static method constructing masks from bv + voids + attachment faces
- `save()` / `load()` — .npy persistence
- `remap()` — trilinear interpolation to new bounding volume via scipy.ndimage

### mass_com_calculator.py
- `ComponentMassCOM` dataclass
- `compute_component_mass_com()` — mass and COM from solid cells in phi grid
- `compute_sidepod_pair_mass_com()` — doubles right half, zeroes y COM by symmetry
- `compute_all_machined_components()` — returns list of 4 ComponentMassCOM

### phi_updater.py
- `_godunov_gradient()` — Godunov upwind scheme for one axis
- `_grad_magnitude()` — full 3D gradient magnitude
- `hj_update()` — Hamilton-Jacobi update: phi -= dt * F * |grad phi|
- `reinitialise_sdf()` — SDF reinitialisation via pseudo-time stepping
- `extend_velocity()` — surface-to-volume velocity extension (simplified)
- `combine_gradients()` — RMS-normalized weighted sum of 4 gradient fields
- `apply_adjoint_sensitivity_symmetric()` — applies CFD adjoint to phi grids

### surface_extraction.py
- Exception hierarchy: SurfaceExtractionError → RadiusViolation, AccessibilityFailure,
  RuleViolation, MeshQualityFailure
- `_marching_cubes()` — skimage marching cubes on phi=0, world coordinate translation
- `_repair_mesh()` — keep largest component, fill holes, fix normals/winding
- `_check_min_radius()` — PLACEHOLDER: returns no violators
- `_check_accessibility()` — simplified normal-based heuristic
- `_check_rules()` — bounding region extent check (U4: full UAE rules not implemented)
- `_check_mesh_quality()` — watertight + is_volume checks (angles/aspect ratios: placeholder)
- `extract_surface()` — full 6-stage pipeline with retry loop

### stl_assembler.py
- `_mirror_right_to_left()` — y-flip + winding invert + normal fix
- `assemble_stl()` — concatenate all components + mirror sidepod + slice at y=0
  → returns (full_stl_path, half_stl_path)

### quality_gates.py
- `GateResult` dataclass with lifecycle validation
- `run_quality_gates()` — save phi snapshots first, then retry loop with
  exception-driven lifecycle state assignment

---

## Test Files Created

| Test File                        | Tests | Covers                           |
|----------------------------------|-------|----------------------------------|
| test_geometry_contract.py        | 22    | Constants, unit helpers, validate_W/d_halo |
| test_bounding_volumes.py         | 13    | BoundingRegion, BoundingVolumes, compute_bounding_volumes, polygon test |
| test_fixed_hardware.py           | 14    | ForbiddenCylinder, void masks, halo validation, COM sanity gate |
| test_phi_grid.py                 | 11    | init modes, hard constraints, build_hard_masks, save/load, remap |
| test_mass_com_calculator.py      | 7     | Mass/COM computation, sidepod pair, all 4 components |
| test_phi_updater.py              | 8     | Godunov gradient, HJ update, SDF reinit, velocity extension, gradient combination |
| test_surface_extraction.py       | 6     | Mesh extraction, empty mesh, exception hierarchy |
| test_stl_assembler.py            | 4     | Mirror, assemble, half STL y>=0, missing component error |
| test_quality_gates.py            | 5     | GateResult validation, run_quality_gates success, phi snapshots |
| test_integration_part1_part2.py  | 5     | Cross-module: CO2 mass, ComponentMassCOM shape, FixedHardwareSpec, lifecycle states, ingest_mass_com |

---

## Dependencies Installed

- `mapbox-earcut` — triangulation engine for trimesh `slice_mesh_plane(cap=True)`

---

## Cross-Module Compatibility

Part 1 integrates with Part 2 (part2_simulation):
- `CO2_MASS_KG` (0.023) matches Part 2's `CO2_CARTRIDGE_MASS_KG` ✅
- `ComponentMassCOM` fields match Part 2's `physics_contract.ComponentMassCOM` ✅
- `FixedHardwareSpec` from Part 2 accepts Part 1's output values ✅
- `ALLOWED_LIFECYCLE_STATES` cross-checks match ✅
- `ingest_mass_com()` successfully consumes Part 1's 4-component output ✅

---

## What Part 1 Cannot Do Yet (Blocking Items)

These items prevent the full Part 1 → Part 2 pipeline from running end-to-end:

1. **U1: Halo cross-section shape** — `place_fixed_hardware()` raises
   `NotImplementedError` without halo polygon vertices. **BLOCKING.**

2. **U2: CO2 canister position** — `place_fixed_hardware()` raises
   `NotImplementedError` without canister COM coordinates. **BLOCKING.**

3. **U5: Rear wing COM** — `place_fixed_hardware()` raises
   `NotImplementedError` without rear wing COM. **BLOCKING.**

4. **U6: UAE rule envelope dimensions** — `compute_bounding_volumes()` uses
   placeholder `RuleEnvelope` values. Not blocking (tests use stub), but all
   geometry is sized to placeholder dimensions. **NON-BLOCKING but incorrect.**

All other placeholder items (minimum radius check, accessibility check, mesh
quality, velocity extension, adjoint mapping) are non-blocking — they affect
quality and optimization accuracy but don't prevent the pipeline from running.

---

## Final Test Results

```
PASS test_bounding_volumes.py
PASS test_fixed_hardware.py
PASS test_geometry_contract.py
PASS test_integration_part1_part2.py
PASS test_mass_com_calculator.py
PASS test_phi_grid.py
PASS test_phi_updater.py
PASS test_quality_gates.py
PASS test_stl_assembler.py
PASS test_surface_extraction.py

TOTAL: 10 passed, 0 failed (10 test files)
```

---

## Next Steps to Fully Complete Part 1

1. **Provide UAE competition rule dimensions** → fill `RuleEnvelope` (resolves U6)
2. **Measure halo hardware cross-section** → fill `HaloGeometry.cross_section_yz_m` (resolves U1)
3. **Confirm CO2 canister legal position** → provide `canister_com_mm` (resolves U2)
4. **Confirm rear wing position** → provide `rear_wing_com_mm` (resolves U5)
5. **Implement proper minimum radius check** (resolves placeholder #6)
6. **Implement full tool accessibility ray-casting** (resolves placeholder #7)
7. **Implement triangle angle/aspect ratio checks** (resolves placeholder #8)
8. **Implement mesh-to-grid sensitivity mapping** (resolves placeholder #9)

Items 1-4 require external input (rule sheets, physical measurements).
Items 5-8 are algorithmic work that can proceed independently.