# PLACEHOLDERS.md — Part 1 Geometry: All Placeholder Values and Reasons

This file lists every placeholder/stub value used in Part 1, what it represents,
and why it couldn't be filled with a real value. Each item is tagged with its
original UNRESOLVED ID from the source code where applicable.

---

## 1. U6 — RuleEnvelope dimensions (bounding_volumes.py)

**Partially resolved (2026-07-11):** `y_sidepod_outer_m` should be **0.0325 m
(32.5 mm)** — user's confirmed design target for max half-width everywhere on
the car (the legal minimum per T3.4 is a 65mm total width / 32.5mm half-width;
the team is choosing to build to that legal minimum rather than the 42.5mm
legal maximum, for reduced frontal area/drag). This is a design choice, not
itself a hard regulation ceiling — T3.4's actual legal range is 32.5-42.5mm
half-width.

**Still unresolved:** `y_body_half_m`, `y_sidepod_inner_m`, `z_floor_m`,
`z_nose_top_m`, `z_sidepod_top_m`, `z_rearpod_top_m`, `z_body_top_m`,
`rearpod_max_length_m` remain placeholders. Note `y_sidepod_inner_m` must be
set smaller than `y_sidepod_outer_m=0.0325` with enough margin for a
manufacturable sidepod width (current stub value of 0.030 leaves only 2.5mm,
too thin to be meaningful) — update both together once resolved.

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

## 6. RESOLVED: Minimum radius check (surface_extraction.py, Stage 3)

`_estimate_local_radii()` uses `trimesh.curvature.discrete_mean_curvature_measure`
for real curvature-based local radius estimation, with a documented
conservative fallback (treat as infinite radius, i.e. pass) if curvature
computation fails on a degenerate mesh. `extract_surface()`'s retry loop
calls `_smooth_phi_neighbourhood()` to locally smooth violating regions and
re-extracts, raising `RadiusViolation` only after `MAX_EXTRACTION_RETRIES`
attempts. This was already implemented before 2026-07-11 (see git history:
"fix: replace broken dihedral-radius fallback with conservative pass-all");
this file's earlier description of it as a stub was stale documentation, not
the actual code state.

---

## 7. RESOLVED: Tool accessibility check (surface_extraction.py, Stage 4)

`_find_inaccessible_faces()` performs real ray-casting via `mesh.ray.intersects_id()`:
for each allowed tool direction, casts a ray from each candidate face (normal
pointing toward the tool) back toward the mesh, and only counts a face as
accessible if its own ray isn't blocked by other geometry first. Same stale-
documentation situation as item 6 above — already implemented, not a stub.

---

## 8. Mesh quality gate — triangle angles and aspect ratios (surface_extraction.py, Stage 6)

**Partially stale, now fully resolved (2026-07-11):** the triangle minimum-angle
check (`MESH_MIN_TRIANGLE_ANGLE_DEG`, with `simplify_quadric_decimation()`
fallback) was already implemented. The aspect ratio check was genuinely
missing — `MESH_MAX_ASPECT_RATIO` wasn't even imported into the file. Added
`_triangle_aspect_ratios()` (longest_edge^2 * sqrt(3) / (4*area), the standard
CFD-mesher shape metric — 1.0 for equilateral, unbounded for slivers/needles)
with the same simplify-then-recheck fallback pattern as the angle check.
Verified against a hand-built equilateral triangle (ratio == 1.0 exactly) and
a hand-built sliver (ratio > 100) in `tests/test_surface_extraction.py`.

---

## 9. RESOLVED: Adjoint sensitivity mesh-to-grid mapping (phi_updater.py)

`apply_adjoint_sensitivity_symmetric()` + `_splat_vertex_sensitivity_to_grid()`
already implement real nearest-cell splatting (with averaging when multiple
mesh vertices round to the same cell), symmetry mirroring for symmetric
components (nose/rearpod/main_body get contributions from both the right-half
mesh directly and its y-mirrored image; sidepod uses the right-half directly
since its grid only covers y>=0), and velocity extension via the (also real)
Godunov `extend_velocity()` before the HJ update. This was stale documentation
from before 2026-07-11's fixes, not the actual code state — same pattern as
items 6/7 above. Added direct test coverage in `tests/test_phi_updater.py`
(previously zero tests exercised these functions despite them being fully
implemented).

**Found and fixed while adding that coverage:** `bayesian_outer_search.py`'s
evolution loop (`_level2_evaluate`, `n_iters > 0` branch) called
`hj_update(pg.grid, ...)` and `reinitialise_sdf(pg.grid)` — passing the raw
array. Both functions actually take the `PhiGrid` object (mutate in place,
return `None`). This crashed with `AttributeError: 'numpy.ndarray' object has
no attribute 'grid'` the moment anyone set `n_iters > 0` — untested until now
because every earlier evaluation in this session used `n_iters=0`. Fixed to
pass `pg` directly; regression test added
(`test_level2_evolution_loop_runs_without_crashing`).

**Still a real scope boundary (not a Part 1 bug):** mass/COM/manufacturing
gradients are combined as zero placeholders in `combine_gradients()` — the
locked race objective's analytical gradients for these are explicitly Part
2/3's responsibility, not something Part 1 computes.

---

## 10. RESOLVED (2026-07-11): Fixed hardware input values wired end-to-end

`fixed_hardware.compute_default_fixed_hardware_inputs()` now supplies all of
these (canister position/size, rear wing mass+COM, wheel/axle mass+COM+
geometry, halo cross-section) as documented design defaults within legal T5/
T9 bounds, and `bayesian_outer_search._level2_evaluate()` calls
`place_fixed_hardware()` + Part 2's real `ingest_mass_com()` with them every
evaluation (see item 14/15-style entries below, and the "Design defaults"
section at the bottom of `fixed_hardware.py` for exact values and rationale).

**Still genuinely open:** these are starting points within the legal range,
not measured/confirmed final values — same caveat as U2/U5 always had.
Override with real measurements once available: wheel+axle mass, COM, canister
exact depth/diameter choice, rear wing mass and mounting position.

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

## 12. RESOLVED: Velocity extension algorithm (phi_updater.py)

`extend_velocity()` already implements the proper Godunov-style upwind
extension: computes the unit surface normal from `_godunov_gradient()`, picks
the upwind finite difference per axis based on the sign of
`sign(phi) * normal_component` (characteristics propagate away from the
zero level set), and iterates the PDE `dF/dtau + sign(phi) * n . grad(F) = 0`
following Osher & Fedkiw (2003). This was stale documentation, not the actual
code state — same pattern as items 6/7/9 above.

---

## 13. T7.9 wedge angle labels vs linear dimensions (wheel_visibility_zones.py)

**What:** T7.9.2/T7.9.3 (the sidepod's wheel-visibility keep-clear wedges) are
defined in the regs diagram (page 32) by two linear dimensions plus an angle
label each: front wedge 15.0mm x 30.0mm at a labelled ~60°; rear wedge 5.0mm x
30.0mm at a labelled ~45°.

**Placeholder/assumption:** The two linear dimensions do not reconcile exactly
with the labelled angle (15/30 implies ~63.4°, not 60°; 5/30 implies ~80.5°,
not 45°) — consistent with rounding in a hand-annotated diagram.
`build_t79_forbidden_mask()` treats the **linear millimetre dimensions as
authoritative** (what a caliper/feeler-gauge scrutineering check actually
measures) and ignores the angle labels.

**Impact:** If STEM Racing's actual scrutineering gauge is manufactured to the
angle instead (e.g. an exact 45°/45° isoceles wedge for the rear), the modelled
forbidden zone would differ slightly from the physical one — current model is
larger/more conservative on the front wedge (real gauge closer to 60°/26mm),
and the rear wedge shape especially should be double-checked.

**To resolve:** Confirm with STEM Racing/Yas in Schools which reading is
authoritative, or measure the physical scrutineering gauge if available.

---

## 14. RESOLVED (2026-07-11): d_halo was not wired into any geometry

**What was wrong:** `d_halo` was validated (`validate_d_halo`) and stored on
`BoundingVolumes.d_halo_mm`, but nothing used it to place anything. The halo's
position was set independently via a hand-built `HaloGeometry(x_front_m=...,
x_rear_m=...)`, completely disconnected from `d_halo`. `_level2_evaluate()`
never called `place_fixed_hardware()` at all. Net effect: every value of
`d_halo` the Bayesian search proposed would have produced an *identical*
mass/COM/race-time — the optimizer would see zero signal on that axis.

**Fix:** New module `halo_pocket.py` computes the halo's mounting-pocket
bounding box (50mm x 25mm x 3.175mm deep, per regs Appendix ix) positioned at
`Ref Plane A + d_halo`, floor pinned at `HALO_MIN_Z_M=24mm` (T4.4.4, fixed).
`compute_bounding_volumes()` now forces this box to `phi > 0` (air) within
`main_body`'s voxel mask, alongside the existing T7.9 wheel-visibility zones.
Verified end-to-end: two evaluations differing only in `d_halo` now produce
different `main_body` masks, different mass, and different race-time proxy
(`tests/test_halo_pocket.py::test_d_halo_changes_bounding_volumes_end_to_end`).

**d_halo bound also corrected:** was a flat `W+16` (136-156mm depending on W);
now `min(100, W+16)` per project owner confirmation — always exactly 100mm
for `W` in `[120,140]`. `calibrate_d_halo_max_mm(W_mm)` in `geometry_contract.py`.

**Still approximate:** the pocket is modelled as its full bounding rectangle,
not the tapered/rounded outline in the diagram (conservative — excludes a bit
more volume than strictly required, never less). See item 13 for the same
caveat pattern applied to T7.9.

---

## 15. RESOLVED (2026-07-11): virtual cargo (T4.2) was not implemented at all

**What was wrong:** zero references to "cargo" anywhere in the codebase.
Nothing generated, positioned, or enforced the T4.2 mandatory minimum-solid
region.

**Fix:** New module `virtual_cargo.py`. The cargo is a tapered wedge (60mm
long, 10mm tall constant, tapering 55mm→10mm wide — read from the T4.2
diagram) that must exist somewhere between the axle centrelines. Position is
NOT dictated by the regs beyond "between axles" + "not overlapping the halo
pocket" (T4.3's example dimensions are one team's illustrative choice, not a
rule) — so this is a genuine design decision, not a lookup.

**Architecture decision (confirmed with project owner):** cargo position is
NOT a Bayesian search dimension. Unlike W/x_front/d_halo it never touches the
exterior surface or aerodynamics — it's a purely interior constraint, so
evaluating a candidate position is a cheap direct mass/COM calculation, no
CFD needed. `find_cargo_placement()` runs once per Level 2 evaluation: scans
candidate x-positions in the axle corridor (preferring centre), skips any
that overlap that evaluation's halo pocket box, and forces the winning
placement to `phi < 0` (solid) via a new `solid_masks` parameter added to
`PhiGrid.build_hard_masks()` (mirrors the existing `void_masks` parameter,
just forcing solid instead of air).

**Real finding from implementation:** for some (W, x_front, d_halo) combos,
the halo pocket can sit in the middle of the axle corridor and split it into
two segments, neither large enough alone to hold the 60mm cargo, even though
the combined free space would be enough. This is now correctly caught and
returns `lifecycle="geometry_rejected"` rather than silently placing the
cargo somewhere illegal. Confirmed via
`tests/test_virtual_cargo.py::test_placement_raises_when_impossible`.

**Still approximate:** modelled as the exact tapered wedge (not a bounding
box, unlike items 13/14 above) — accuracy matters more here because
over-forcing solid volume beyond the regulatory minimum directly adds
unwanted mass, unlike the forbidden-zone cases where over-exclusion is safe.