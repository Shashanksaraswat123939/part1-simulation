# PLACEHOLDERS.md — Part 1 Geometry: All Placeholder Values and Reasons

This file lists every placeholder/stub value used in Part 1, what it represents,
and why it couldn't be filled with a real value. Each item is tagged with its
original UNRESOLVED ID from the source code where applicable.

---

## 1. U6 — RuleEnvelope dimensions (bounding_volumes.py)

**What:** The `RuleEnvelope` dataclass contains absolute car envelope dimensions
(y_body_half_m, y_sidepod_inner_m, y_sidepod_outer_m, z_floor_m, z_nose_top_m,
z_sidepod_top_m, z_rearpod_top_m, z_body_top_m, rearpod_max_length_m).

**Placeholder used in tests (test_bounding_volumes.py → STUB_RE):**
```python
STUB_RE = RuleEnvelope(
    y_body_half_m       = 0.030,   # 30 mm half-width — PLACEHOLDER
    y_sidepod_inner_m   = 0.030,   # same as body half-width — PLACEHOLDER
    y_sidepod_outer_m   = 0.060,   # 60 mm outer edge — PLACEHOLDER
    z_floor_m           = 0.000,   # track surface — reasonable default
    z_nose_top_m        = 0.040,   # 40 mm — PLACEHOLDER
    z_sidepod_top_m     = 0.035,   # 35 mm — PLACEHOLDER
    z_rearpod_top_m     = 0.035,   # 35 mm — PLACEHOLDER
    z_body_top_m        = 0.045,   # 45 mm — PLACEHOLDER
    rearpod_max_length_m= 0.030,   # 30 mm — PLACEHOLDER
)
```

**Why:** The UAE STEM Racing competition rule sheet has not been provided.
These dimensions define the legal envelope within which the car geometry must
fit. Until the official rule dimensions are confirmed, these values are
physically plausible guesses based on typical STEM Racing car sizes.

**Impact:** Bounding volumes for all four components (nose, sidepod, rearpod,
main_body) use these dimensions. Any geometry produced will be sized to the
placeholder envelope, not the real one.

---

## 2. U1 — Halo cross-section shape (fixed_hardware.py)

**What:** `HaloGeometry.cross_section_yz_m` — the y-z polygon vertices defining
the halo tube's cross-sectional shape.

**Placeholder:** `None` (not provided).

**Why:** The physical halo hardware has not been measured. The cross-section is
needed to build the halo void mask in the phi grid. Without these vertices,
`place_fixed_hardware()` raises `NotImplementedError` when it reaches the halo
void construction.

**Impact:** The halo void mask cannot be built. Fixed hardware placement
function cannot complete. This blocks the full Part 1 → Part 2 pipeline.

**To resolve:** Measure the physical halo tube cross-section and provide vertices
as `[(y1, z1), (y2, z2), ...]` in metres, counterclockwise.

---

## 3. U2 — CO2 canister position (fixed_hardware.py)

**What:** `canister_com_mm` — the (x, y, z) position of the CO2 canister centre
in millimetres.

**Placeholder:** `None` (not provided).

**Why:** The legal canister position from the UAE competition rules has not been
confirmed. The canister sits at the front of the car (small x value), but the
exact coordinates are unknown.

**Impact:** `place_fixed_hardware()` raises `NotImplementedError` when
`canister_com_mm` is `None`. The canister void mask cannot be built.

**To resolve:** Provide `canister_com_mm=(x_mm, y_mm, z_mm)` from the official
rule sheet.

---

## 4. U5 — Rear wing COM position (fixed_hardware.py)

**What:** `rear_wing_com_mm` — the (x, y, z) centre of mass of the rear wing.

**Placeholder:** `None` (not provided).

**Why:** The rear wing's fixed position has not been confirmed from the
competition rules.

**Impact:** `place_fixed_hardware()` raises `NotImplementedError` when
`rear_wing_com_mm` is `None`.

**To resolve:** Provide `rear_wing_com_mm=(x_mm, y_mm, z_mm)` from the official
rule sheet or physical measurement.

---

## 5. U4 — Full UAE envelope rule check (surface_extraction.py, Stage 5)

**What:** The rule checker in `_check_rules()` currently only verifies that mesh
vertices are within the bounding region extent (±0.1 mm tolerance).

**Placeholder:** Simple bounding-box check instead of full UAE envelope
compliance (e.g. maximum car length, width, height, specific feature rules).

**Why:** The full UAE competition regulation envelope has not been provided.
The current check catches gross violations but misses rule-specific constraints.

**Impact:** Geometry that violates UAE rules (but fits within the bounding
region) would pass the quality gate incorrectly.

**To resolve:** Implement full rule checking once the UAE regulation document
is available.

---

## 6. Minimum radius check (surface_extraction.py, Stage 3)

**What:** `_check_min_radius()` — the local radius estimation that checks
whether the machined surface respects `MIN_RADIUS_MM = 3.15 mm`.

**Placeholder:** Returns empty list `[]` (no violators). The function is a stub.

**Why:** Proper local radius estimation requires curvature analysis from
dihedral angles and vertex neighbourhood topology. The simple neighbour-distance
heuristic originally attempted was too crude and produced false positives on
small test grids.

**Impact:** No geometry will be rejected for minimum radius violations. Parts
with features smaller than 3.15 mm radius will pass through undetected.

**To resolve:** Implement a proper curvature-based local radius estimator
(e.g. using mesh vertex normals and dihedral angle analysis).

---

## 7. Tool accessibility check (surface_extraction.py, Stage 4)

**What:** `_check_accessibility()` — determines whether a CNC tool can physically
reach all faces of the extracted mesh from the allowed `TOOL_DIRECTIONS`.

**Placeholder:** Simple normal-based heuristic (dot product > 0.1 with tool
direction) instead of full ray-casting accessibility analysis.

**Why:** Full ray-casting from each face centre along each tool direction,
checking whether the ray re-enters the mesh, requires `trimesh.intersections`
or `trimesh.ray` operations that are computationally expensive and need careful
implementation to handle edge cases (grazing rays, thin features).

**Impact:** The check may produce false positives (faces that appear accessible
by normal direction but are blocked by intervening geometry) or false negatives.

**To resolve:** Implement proper ray-casting accessibility check using
`trimesh.ray.ray_pyembree` or equivalent.

---

## 8. Mesh quality gate — triangle angles and aspect ratios (surface_extraction.py, Stage 6)

**What:** `_check_mesh_quality()` — full triangle quality checks.

**Placeholder:** Only checks watertight and is_volume. Missing: minimum interior
angle > 10°, maximum aspect ratio < 10, and simplification fallback.

**Why:** `trimesh` does not provide a direct triangle angle/aspect ratio
function in its stable API. Computing these requires manual triangle edge
length and angle calculations.

**Impact:** Degenerate triangles (slivers, needles) won't be caught. snappyHexMesh
may fail on such meshes.

**To resolve:** Implement manual triangle angle and aspect ratio computation,
plus `mesh.simplify_quadric_decimation()` fallback for failing meshes.

---

## 9. Adjoint sensitivity mesh-to-grid mapping (phi_updater.py)

**What:** `apply_adjoint_sensitivity_symmetric()` — mapping surface sensitivity
from CFD mesh vertices to phi grid cells.

**Placeholder:** If sensitivity array shape matches phi grid shape, use it
directly. Otherwise, skip the component (no-op).

**Why:** Proper mesh-to-grid interpolation (nearest-cell or trilinear splatting)
requires knowledge of the CFD mesh structure and the mapping between mesh
vertices and grid cells. This depends on Part 2/Part 3's CFD mesh format.

**Impact:** Adjoint-driven shape optimization will not work properly. The phi
updater won't respond to CFD sensitivity feedback.

**To resolve:** Implement nearest-cell or trilinear splatting from mesh vertices
to grid cells once the CFD mesh interface is finalized.

---

## 10. Fixed hardware input values (test files / integration)

**What:** The `place_fixed_hardware()` function requires several physical
hardware parameters:
- `wheel_axle_mass_kg` — total mass of all 4 wheels + axles
- `wheel_axle_com_mm` — COM of wheels+axles assembly
- `wheel_x_half_width_mm` — half-width of wheel assembly in x
- `wheel_axle_z_mm` — axle height above track
- `rear_wing_mass_kg` — rear wing mass
- `canister_box_half_size_mm` — half-size of canister void box

**Placeholder values used in test files:**
```python
# test_bounding_volumes.py
front = ForbiddenCylinder(0.0, 0.0, 0.015, r, 0.008)  # z=15mm, half-width=8mm — PLACEHOLDER
rear  = ForbiddenCylinder(W_m, 0.0, 0.015, r, 0.008)   # same — PLACEHOLDER
```

**Why:** Physical hardware specifications (wheel mass, axle dimensions, wing mass)
have not been provided. The placeholders use physically plausible values based
on typical STEM Racing car components.

**Impact:** Mass/COM calculations will use placeholder values, producing
incorrect race time predictions in Part 2.

**To resolve:** Measure or obtain from spec sheets: wheel+axle mass, COM,
half-width, axle height, rear wing mass and COM, canister box dimensions.

---

## 11. STL slicing watertightness (stl_assembler.py)

**What:** The right-half STL produced by slicing the full car at y=0 may not
be watertight after `trimesh.intersections.slice_mesh_plane()`.

**Placeholder:** If the sliced mesh is not watertight after repair attempts,
the code continues without raising an error (logs nothing, exports best-effort).

**Why:** `trimesh`'s `slice_mesh_plane(cap=True)` can fail to produce a watertight
result for complex meshes due to triangulation engine limitations. A more robust
slicing approach (Open3D, CGAL) may be needed.

**Impact:** Non-watertight half STLs will be flagged by Part 2's mesh validation
downstream, but Part 1 won't catch it early.

**To resolve:** Install/enable a more robust triangulation engine (triangle,
mapbox-earcut installed), or use Open3D for slicing. Verify watertightness and
raise `MeshQualityFailure` if still broken after repair.

---

## 12. Velocity extension algorithm (phi_updater.py)

**What:** `extend_velocity()` — propagates surface velocity into the volume
using the PDE: dF/dtau + sign(phi) * grad(phi) . grad(F) = 0.

**Placeholder:** Simplified first-order upwind scheme that divides the gradient
by 3 (rough averaging). Not the proper Godunov-based extension.

**Why:** The full velocity extension requires careful upwind scheme selection
based on the sign of phi and the direction of grad(phi). The simplified version
is functional but not numerically accurate for production optimization.

**Impact:** Shape optimization convergence may be slow or unstable.

**To resolve:** Implement the full upwind velocity extension following
Osher & Fedkiw (2003) or similar level-set methods reference.