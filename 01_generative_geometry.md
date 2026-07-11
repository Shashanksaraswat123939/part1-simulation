# Part 1: Generative Geometry Designer

## Purpose

Generate legal, manufacturable STEM Racing car geometry from φ (level-set)
fields. This part owns the shape representation: what the optimizer is allowed
to change, what must remain fixed, and how each component's φ field becomes a
clean STL ready for CFD.

The goal is topology-free organic geometry. No shape family is assumed.
The physics finds the shape.

---

## Three-Level Optimization Structure

Part 1 operates across three nested levels:

```text
Level 1 (Bayesian outer search):
  Search over (W, x_front, d_halo) using BoTorch
  ~80-100 evaluations total
  Each evaluation triggers one full Level 2 run

Level 2 (φ field optimization per outer point):
  Adjoint-driven level-set optimization of all component shapes
  Runs to convergence for fixed (W, x_front, d_halo)
  ~50-100 CFD+adjoint iterations

Level 3 (surface extraction per inner iteration):
  Marching Cubes → repair → quality gates → STL
  Runs every Level 2 iteration
```

Level 1 finds the best overall car configuration.
Level 2 finds the best shape within that configuration.
Level 3 converts the current φ fields into a valid mesh.

---

## Inputs

### Level 1: Bayesian Outer Search Variables

Three coupled scalars searched jointly by BoTorch:

```text
W        ∈ [120, 140] mm      wheelbase (T7.3)
x_front  ∈ [x_min, x_max] mm  front axle position from car datum
d_halo   ∈ [d_min, d_max] mm  halo-to-canister distance
```

x_front bounds (derived from geometry, not directly stated in regs):

```text
x_front_min: enough nose space for cartridge chamber depth (min 45mm, T5.3)
             plus cartridge to front axle clearance
x_front_max: model block length (223mm) minus wheelbase minus rearpod overhang
             (max 40mm from Ref Plane B, T9.4.2) minus Ref Plane B offset (16mm)
```

d_halo bounds (depend on W and x_front — recomputed each Bayesian sample):

```text
d_halo_min: minimum physical clearance between canister rear and halo pocket
d_halo_max: W + 16mm (per project design rule)
```

Any Bayesian sample that violates derived bounds is rejected before
running the inner loop.

Why all three are outer variables: changing any one remaps all bounding
boxes, invalidates the current φ fields, and requires a fresh Level 2 run.
They cannot be optimized inside the Level 2 adjoint loop.

### Level 2: φ Field Variables (per component)

Updated by the adjoint each inner iteration:

```text
nose φ grid        — 3D scalar field within nose bounding volume
sidepod φ grid     — 3D scalar field within sidepod corridor
rearpod φ grid     — 3D scalar field within rearpod bounding volume
main body φ grid   — 3D scalar field within full body envelope
```

All four grids update simultaneously from one full-car CFD+adjoint run.
Inter-component flow interactions are captured naturally.

### Fixed Inputs (never optimized)

```text
CO2 cartridge:  23 g, fixed legal position (T5.2: 30-40mm from track surface)
Rear wing:      fixed mass, fixed position (rear of Ref Plane B)
Wheels+axles:   fixed mass, fixed geometry
Halo:           fixed geometry (T4.4.1, downloaded from Yas in Schools)
Helmet:         fixed geometry (T4.5)
Tether guides:  fixed position (within 10mm of each axle, T6.1)
Component densities: fixed (see Mass and COM section)
Rule envelopes and forbidden zones: fixed from rulebook
```

---

## Outputs

```text
full-car STL for CFD (or right-half STL for symmetric runs)
component STL files for inspection
component volumes (integrated from φ grids)
component masses (per component)
component COMs (per component, x/y/z)
total mass
total COM (h_com, x_com)
rule/manufacturing status report
```

---

## Coordinate System

```text
x = front to rear  (nose tip at x=0, car extends in +x direction)
y = centerline to outside  (y=0 is symmetry plane)
z = track upward  (track surface at z=0)
```

All component placement, forbidden zones, exclusion zones, and tool
directions use this shared coordinate system.

Reference planes (T1.17):

```text
Ref Plane A: x = x_front - 16mm  (16mm in front of front axle)
Ref Plane B: x = x_front + W + 16mm  (16mm behind rear axle)
```

Nose cone exists forward of Ref Plane A.
Body exists rear of Ref Plane A.
Rear wing exists rear of Ref Plane B.

---

## Regulatory Constraints on Geometry

All hard limits from UAE Technical Regulations 2025-26:

### Overall Car

| Regulation | Dimension | Value |
|---|---|---|
| T3.4 | Total width | 65.0–85.0 mm |
| T3.5 | Total height | max 65.0 mm |
| T3.6 | Total weight | min 48.0 g |
| T3.7 | Track clearance (non-wheel) | min 1.5 mm |
| T7.3 | Wheelbase | 120.0–140.0 mm |

### Model Block

```text
Total block: 223mm × 65mm × 50mm
Body must be machined from this block (T3.1.2)
All body geometry is bounded by block dimensions
```

### Cartridge Chamber

| Regulation | Dimension | Value |
|---|---|---|
| T5.1 | Diameter | 18.0–18.5 mm |
| T5.2 | Distance from track | 30.0–40.0 mm |
| T5.3 | Depth | 45.0–58.0 mm |
| T5.4 | Max angle | ±3° |
| T5.5 | Safety zone wall | min 3.0 mm |
| T5.6 | Protrusion from rear | min 5.0 mm |

### Nose and Front

| Regulation | Dimension | Value |
|---|---|---|
| T8.2 | Nose overhang from Ref Plane A | max 40.0 mm |
| T8.5.1 | Nose/wing support height | max 25.0 mm |
| T8.5.2 | Front wing height (>15mm wide) | max 20.0 mm |

### Rear

| Regulation | Dimension | Value |
|---|---|---|
| T9.4.2 | Rear overhang from Ref Plane B | max 40.0 mm |
| T9.4.3 | Rear overhang height | max 65.0 mm |

### Wheels and Exclusion Zones

| Regulation | Dimension | Value |
|---|---|---|
| T7.2.1 | Front wheel inner gap | min 38.0 mm |
| T7.2.2 | Rear wheel inner gap | min 30.0 mm |
| T7.5 | Wheel diameter | 28.0–32.0 mm |
| T7.9 | Wheel visibility zones | see T7.9 diagram |
| T7.11 | Front wheel obstruction height | max 20.0 mm |

Exclusion zones (forbidden sidepod regions) translate rigidly with
wheel positions as W and x_front change.

### Halo

```text
T4.4.1: Halo geometry fixed — no dimensional changes allowed
T4.4.4: Circular notch centre at 34.0 ± 1.0 mm above track
        → halo pocket bottom must be 24.0 mm above track
T4.4.2/3: Halo must be visible in front, side, and top views
```

### Tether Guides

```text
T6.1: Front guide within 10mm of front axle centre line
      Rear guide within 10mm of rear axle centre line
T6.2: Internal dimension 3.5–6.0 mm
```

---

## φ Field Representation

Each component occupies a 3D bounding volume. Every grid point stores φ:

```text
φ < 0 = solid (inside the car)
φ = 0 = surface
φ > 0 = air (outside)
```

Grid resolution: ~0.3 mm spacing. Required to reliably represent the
3.15 mm minimum machining radius (Nyquist margin ×10).

Hard constraints baked in permanently — never modified by optimizer:

```text
Attachment faces:     φ forced < 0 (always solid, always connected)
Bounding box walls:   φ forced > 0 (shape cannot escape legal envelope)
Exclusion zones:      φ forced > 0 (wheel/axle forbidden regions)
Hardware voids:       φ forced > 0 (cartridge chamber, halo pocket, axle holes)
```

Surface extraction: Marching Cubes walks the grid, finds every edge where
φ crosses zero, and produces a triangle mesh for CFD.

---

## How Bounding Boxes Derive from Outer Variables

Every time the Bayesian outer search proposes a new (W, x_front, d_halo),
all bounding boxes are recomputed before the Level 2 loop starts:

```text
front_axle_x  = x_front
rear_axle_x   = x_front + W
ref_plane_A_x = x_front - 16
ref_plane_B_x = x_front + W + 16

nose bounding volume:
  x: [0, ref_plane_A_x]        (forward of Ref Plane A)
  y: [-42.5, 42.5]             (within total width 85mm)
  z: [1.5, 65]                 (track clearance to max height)

sidepod corridor:
  x_min = rear face of front exclusion zone
  x_max = front face of rear exclusion zone
  (both exclusion zones positioned from front_axle_x and rear_axle_x)

main body bounding volume:
  x: [ref_plane_A_x, ref_plane_B_x]
  y: [-42.5, 42.5]
  z: [1.5, 65]

rearpod bounding volume:
  x: [rear_axle_x, ref_plane_B_x + 40]   (max 40mm overhang from Ref B)
  y: [-42.5, 42.5]
  z: [1.5, 65]

halo position:
  x: ref_plane_A_x + d_halo   (forward of Ref Plane A by d_halo)
  z: pocket_bottom = 24.0 mm above track (fixed by T4.4.4)
```

φ grids from the previous Bayesian sample are warm-started into the new
bounding volumes where possible, and re-initialized where the new box
differs substantially.

---

## Components

### Nose Cone

```text
density = 1.0 g/cm³ = 1000 kg/m³
bounding volume: forward of Ref Plane A, max overhang 40mm (T8.2)
length varies with x_front
```

The nose is 6× denser than every other machined component. Unnecessary
nose volume costs disproportionately in mass and shifts COM forward.
The optimizer discovers this naturally through the mass → COM → race time
gradient chain.

Hard constraints in φ grid:

```text
rear face (at Ref Plane A):  φ forced < 0 (attachment to main body)
bounding box walls:          φ forced > 0
cartridge chamber void:      φ forced > 0 (correct diameter and depth, T5.1-T5.5)
```

Machinability (enforced post-extraction each iteration):

```text
allowed tool directions: -X from front, +Z from top, ±Y from sides
minimum radius 3.15 mm everywhere
tool accessibility from all allowed directions
no inaccessible concave regions
```

### Sidepods

```text
density = 0.163 g/cm³ = 163 kg/m³
bounding volume: sidepod corridor between exclusion zones — depends on W and x_front
```

The corridor shifts and shrinks with both W and x_front. As W decreases,
exclusion zones move closer together and corridor shrinks. As x_front
changes, the entire corridor translates fore-aft.

Left and right sidepods are mirrored — only the right-half φ grid is
optimized. Left is reflected across y=0. This halves variable count and
matches the half-car CFD setup.

Hard constraints in φ grid:

```text
inner face (main body wall): φ forced < 0 (attachment strip)
bounding box walls:          φ forced > 0
exclusion zones:             φ forced > 0
```

Machinability (enforced post-extraction each iteration):

```text
right sidepod: +Y from outside, -X from front, +Z from top
left sidepod:  -Y from outside, -X from front, +Z from top
minimum radius 3.15 mm everywhere
no undercuts, hidden pockets, or reverse overhangs
```

### Main Body

```text
density = 0.163 g/cm³ = 163 kg/m³
bounding volume: between Ref Plane A and Ref Plane B, within model block
```

The main body holds all fixed hardware positions as hard voids.
The halo-canister loft region sits within the main body φ grid —
d_halo determines where the halo void appears along x.

Hard constraints in φ grid:

```text
cartridge chamber void:   φ forced > 0 (position from x_front and T5.2)
halo pocket void:         φ forced > 0 (position from d_halo, depth from T4.4.4)
axle hole voids:          φ forced > 0 (at x_front and x_front+W)
tether guide voids:       φ forced > 0 (within 10mm of each axle, T6.1)
virtual cargo void:       φ forced < 0 (min 60×55×10mm between axle lines, T4.2)
sidepod attachment walls: φ forced < 0 (solid connection guaranteed)
outer envelope:           φ forced > 0
```

### Rearpod

```text
density = 0.163 g/cm³ = 163 kg/m³
bounding volume: aft of rear axle, max 40mm overhang from Ref Plane B (T9.4.2)
length varies with x_front and W
```

Hard constraints in φ grid:

```text
front face (at rear axle): φ forced < 0 (attachment to main body)
bounding box:              φ forced > 0
```

Machinability (enforced post-extraction each iteration):

```text
allowed tool directions: +X from rear, +Z from top, ±Y from sides
minimum radius 3.15 mm everywhere
tool accessibility from all allowed directions
```

---

## Surface Extraction Pipeline

Run after every φ grid update (Level 3):

```text
1. Marching Cubes → triangle mesh from φ = 0 surface (all components)

2. Geometry repair:
     remove floating disconnected pieces
     fill holes
     fix non-manifold edges
     remove degenerate triangles

3. Minimum radius check (3.15 mm):
     compute curvature at every surface point
     smooth violations via mean curvature flow
     re-extract until clean

4. Tool accessibility check:
     for each surface point: sample approach vectors
     flag inaccessible regions
     smooth or fill small inaccessible regions
     penalize or reject large ones

5. Rule checker (UAE 2025-26 regs):
     total width 65–85mm (T3.4)
     total height max 65mm (T3.5)
     track clearance min 1.5mm (T3.7)
     cartridge chamber dimensions (T5.1–T5.5)
     cartridge protrusion min 5mm (T5.6)
     nose overhang max 40mm from Ref Plane A (T8.2)
     rear overhang max 40mm from Ref Plane B (T9.4.2)
     halo pocket at correct height (T4.4.4)
     wheel exclusion zones clear
     tether guide positions (T6.1)
     virtual cargo present and dimensioned (T4.2)

6. Mesh quality gate:
     watertight surface
     valid outward normals
     no self-intersections
     triangle quality within snappyHexMesh tolerance
```

Only surfaces that pass all six stages enter CFD.

---

## Mass and COM

Computed directly from φ grids each iteration — no separate geometry pass.

```text
volume_c = (cells where φ < 0) × cell_volume
mass_c   = volume_c × density_c
COM_c    = centroid of φ < 0 region (volume-weighted)

COM_total = Σ(mass_c × COM_c) / Σ(mass_c)
```

Component densities:

| Component | Density | Type |
|---|---|---|
| Nose cone | 1000 kg/m³ | Computed from φ grid |
| Sidepods | 163 kg/m³ | Computed from φ grid |
| Rearpod | 163 kg/m³ | Computed from φ grid |
| Main body | 163 kg/m³ | Computed from φ grid |
| CO2 cartridge | 23 g fixed | Fixed mass, fixed position |
| Rear wing | known mass | Fixed mass, fixed position |
| Wheels + axles | known mass | Fixed mass, fixed geometry |
| Halo | known mass | Fixed mass, position from d_halo |
| Helmet | known mass | Fixed mass, position from halo |

CO2 cartridge is the heaviest single fixed component. Omitting it
makes every COM calculation wrong.

Outputs to Part 2:

```text
m_total   — total car mass [kg]
h_com     — COM height above track [m]   (z coordinate of COM_total)
x_com     — COM fore-aft from front axle [m]  (x coordinate of COM_total)
```

---

## Quality Gates Summary

```text
legal envelope pass              (all reg dimensions)
forbidden zone pass              (exclusion zones clear)
minimum radius pass              (3.15 mm everywhere)
tool accessibility pass          (direction sampling, all components)
virtual cargo pass               (present, correctly dimensioned, T4.2)
halo pocket pass                 (correct position and depth, T4.4.4)
cartridge chamber pass           (correct diameter, depth, angle, T5.1-T5.5)
watertight mesh pass
normal orientation pass
self-intersection pass
component connection pass        (no floating pieces)
total weight pass                (≥ 48g, T3.6)
mass/COM report generated
```

Minimum radius and tool accessibility are separate gates. A surface can
pass the radius check and still be unreachable by the CNC tool. Both
must pass independently.

---

## Failure and Recovery

```text
Surface extraction fails        → repair and retry (max 3 attempts)
Radius violation persists       → increase smoothing strength, retry
Accessibility failure (small)   → smooth/fill region, retry
Accessibility failure (large)   → assign manufacturing penalty, continue
Rule violation (minor)          → project back into legal region, retry
Rule violation (major)          → assign hard penalty, kill candidate
Mesh quality fails              → simplify surface, retry
Weight below 48g                → flag for ballast, continue (ballast in halo pocket)
All retries exhausted           → assign failure penalty, kill candidate
```

---

## Bayesian Outer Search Integration

Part 1 is called by the BoTorch outer loop in Part 3. The interface:

```text
Input:  (W, x_front, d_halo) from BoTorch
Output: best race time T_raw found by Level 2 for this combination
        + φ field snapshots for warm-starting adjacent samples
```

BoTorch maintains a surrogate model (Gaussian process) over the
(W, x_front, d_halo) → race time mapping. After each evaluation it
updates the model and picks the next combination that maximises the
Expected Improvement acquisition function — balancing exploitation
of known good regions and exploration of uncertain ones.

Warm-starting: when the next Bayesian sample is geometrically close
to a previous one, the converged φ fields from that sample initialise
the Level 2 loop instead of random initialisation. This cuts inner
iterations significantly for adjacent samples.

Total outer evaluations: ~80–100 before convergence.
