"""
unified_phi.py --- ONE level-set field for the whole car, with component labels.

WHY THIS EXISTS
---------------
The original representation gave each of the four components its own phi grid
on its own axis-aligned box, extracted each with its own marching-cubes pass,
and concatenated the four meshes (`stl_assembler.assemble_stl` is literally
`trimesh.util.concatenate`). Three consequences, all structural:

  1. NO SURFACE COULD CROSS A COMPONENT BOUNDARY. The halo-canister loft --
     the continuous top surface running from the cartridge chamber at the rear,
     forward over the halo mount and down toward Ref Plane A -- spans main_body
     and rearpod. It was not representable at all. 01_generative_geometry.md
     says "the halo-canister loft region sits within the main body phi grid",
     but nothing implemented a loft and the boxes forbade one.

  2. COMPONENTS NEEDED ATTACHMENT-FACE HACKS TO STAY CONNECTED. Each grid
     forced a strip of cells solid at whichever face touched a neighbour, and
     the extraction still routinely produced disconnected pieces (sandbox
     finding 7: `init="sphere"` gave four separate lumps totalling 8.5 g against
     a 48 g minimum, and they passed the gates). With one field, connectivity is
     a property of the field, not a constraint bolted onto four of them.

  3. main_body AND rearpod OVERLAPPED (main_body ran to rear_axle +
     rearpod_max_length, rearpod started at rear_axle + wheel clearance), so
     cells in the overlap were counted toward BOTH components' mass.

This module replaces that with a single phi field over the whole car envelope
plus a per-cell `labels` array. A label carries what the separate grids used to
carry structurally: which density the cell contributes at, and which tool
directions apply to it. The shape itself is free to be whatever the field wants.

WHAT DOES *NOT* CHANGE
----------------------
The step at Ref Plane A is NOT a modelling artifact and is not removed here.
T8.5.1 requires the nose and front wing support structure to be "no more than
25.0mm above the track surface and no wider than 15mm either side of the centre
line reference plane", while the body aft of Ref Plane A may be the full 65 mm
(T3.5). That discontinuity is legally mandated. What unification buys is
continuity *aft* of Ref Plane A -- main_body <-> sidepod <-> rearpod -- which is
exactly where the loft lives.

MEMORY
------
The unified envelope is bigger than any single old grid but smaller than their
sum plus the overlap. At the 0.3 mm spec spacing a full envelope is ~40M cells,
which is not interactively runnable -- same caveat as before, see
sandbox/coarse.py. Use coarse.use_spacing() for exploration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from bounding_volumes import (
    BoundingRegion,
    BoundingVolumes,
    RuleEnvelope,
    compute_bounding_volumes,
    default_rule_envelope,
)
from geometry_contract import (
    GRID_SPACING_M,
    R_WHEEL_M,
    WHEEL_CLEARANCE_M,
    WHEEL_X_CLEARANCE_HALF_WIDTH_MM,
    get_density,
    grid_cells,
    mm_to_m,
)
from halo_pocket import (
    build_halo_pocket_forbidden_mask,
    build_ballast_container_forbidden_mask,
)
from phi_grid import PhiGrid
from wheel_visibility_zones import build_t79_forbidden_mask

# Label values. 0 is "not part of any component" -- outside the legal envelope
# for every component, and therefore permanently air.
# Extra keep-out around each wheel, on top of the 2 mm assembly clearance, so
# the body stands clearly off the wheels (outrigger look, unambiguous top-view
# visibility) instead of crowding right against them.
WHEEL_KEEPOUT_MARGIN_M: float = 0.004   # 4 mm

LABEL_NONE: int = 0
LABEL_NOSE: int = 1
LABEL_SIDEPOD: int = 2
LABEL_REARPOD: int = 3
LABEL_MAIN_BODY: int = 4

LABEL_IDS: dict[str, int] = {
    "nose": LABEL_NOSE,
    "sidepod": LABEL_SIDEPOD,
    "rearpod": LABEL_REARPOD,
    "main_body": LABEL_MAIN_BODY,
}
LABEL_NAMES: dict[int, str] = {v: k for k, v in LABEL_IDS.items()}

# T3.1.2's model block bounds the MILLED components only. The nose is 3D
# printed (geometry_contract, user-confirmed 2026-07-14) and so is not block
# material -- confirmed with the project owner 2026-07-20. This is why the
# block is 223 x 65 x 50 mm while T3.5 permits a 65 mm tall car: the block
# constrains what is machined, not the assembled car.
MODEL_BLOCK_LENGTH_MM: float = 223.0
MILLED_COMPONENTS: frozenset[str] = frozenset({"sidepod", "rearpod", "main_body"})


@dataclass
class UnifiedGeometry:
    """One phi field for the whole car, plus the labels that give it meaning."""

    phi: PhiGrid                 # component name is "car"
    labels: np.ndarray           # uint8, same shape as phi.grid
    region: BoundingRegion
    landmarks: dict[str, float]  # metres: ref_plane_A, ref_plane_B, front_axle, ...
    bv: BoundingVolumes          # retained: callers need the per-component extents
    W_mm: float
    x_front_mm: float
    d_halo_mm: float
    fixed_hardware: object = None   # FixedHardwareResult, or None if Part 2 absent
    # Grid spacing this geometry was BUILT at, in metres. Recorded because
    # GRID_SPACING_M is a module global that sandbox/coarse.use_spacing rewrites
    # at runtime -- so by the time a geometry is remapped onto a finer grid, the
    # global no longer describes the geometry in hand. remap_geometry needs both.
    spacing_m: float = 0.0

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.region.shape

    def component_mask(self, component: str) -> np.ndarray:
        """Bool mask of cells labelled as `component` (regardless of solid/air)."""
        if component not in LABEL_IDS:
            raise ValueError(
                f"Unknown component '{component}'. Valid: {sorted(LABEL_IDS)}"
            )
        return self.labels == LABEL_IDS[component]

    def solid_mask(self, component: Optional[str] = None) -> np.ndarray:
        """Solid cells (phi < 0), optionally restricted to one component."""
        solid = self.phi.grid < 0.0
        if component is None:
            return solid
        return solid & self.component_mask(component)

    def machined_length_mm(self) -> float:
        """x-extent of SOLID milled material, in mm.

        This is the quantity T3.1.2's 223 mm applies to -- see
        MODEL_BLOCK_LENGTH_MM. Returns 0.0 if no milled cell is solid.
        """
        milled = np.zeros(self.shape, dtype=bool)
        for name in MILLED_COMPONENTS:
            milled |= self.component_mask(name)
        solid_milled = milled & (self.phi.grid < 0.0)
        if not solid_milled.any():
            return 0.0
        xs = np.where(solid_milled.any(axis=(1, 2)))[0]
        return float((xs[-1] - xs[0] + 1) * GRID_SPACING_M * 1000.0)


# How much of the T4.2 virtual cargo may be eaten by REAL forced-air regions
# before the geometry is rejected.
#
# The one-cell border seal (hard_air[:,:,0] etc., which exists so the phi=0
# surface always closes) is excluded from this measurement, because the cargo
# sits on the track-clearance floor and always loses its bottom layer to it.
# That loss is structural, and crucially it is SPACING-DEPENDENT as a fraction:
# measured 2.94% at the 0.3 mm production spacing but 17.8% at 3 mm, where the
# 10 mm-tall wedge is only ~3 cells high. A flat fraction over the raw overlap
# would therefore pass production and reject every coarse-spacing run --
# including Stage 1, which evaluates at 2 mm. So we count only overlap with
# INTERIOR voids: wheel keep-clear, T7.9 zones, cartridge bore, ballast slot.
# Any of those is a genuine placement clash that leaves the mandatory keep-solid
# volume unfilled, and find_cargo_placement guards none of them (only the halo
# pocket).
CARGO_MAX_ERODED_FRACTION = 0.05


def _mirror_right_onto_left(a: np.ndarray) -> np.ndarray:
    """Copy the y>=0 half of `a` onto the y<0 half, in place. Returns `a`.

    Every constraint in this module is symmetric about the centreline by
    construction (the T7.9 zones, wheel voids, halo pocket and cartridge bore
    are all defined on +/-y), but they are *rasterised* by float comparisons
    against cell coordinates, and those are not exactly symmetric: a cell at
    y=-15.0 mm and its partner at y=+15.0 mm can land on opposite sides of a
    `<= 0.015` test because -0.036 + 34*0.0015 != 0.015 in binary floating
    point. Measured effect before this was applied: the nose zone lost one of
    its two boundary columns and reported com_y = +0.75 mm instead of 0.

    Mirroring the masks makes the symmetry exact rather than approximate, so
    enforce_symmetry's field mirror survives apply_hard_constraints.
    """
    mid = a.shape[1] // 2
    a[:, : mid + 1, :] = a[:, mid:, :][:, ::-1, :]
    return a


def wheel_keepclear_mask(region, W_mm: float, x_front_mm: float) -> np.ndarray:
    """Cells the body must vacate around the four wheels.

    This enforces TWO regulations at once:

      - T7.2 lateral gap: the body must not touch a wheel. A wheel is a disc
        (radius R_WHEEL, circular in x-z) spinning about y, its inner face at
        the T7.2 minimum gap.
      - T7.9 wheel visibility: each wheel must be visible from the top AND
        bottom views, so NO bodywork may sit in a wheel's x-y footprint at ANY
        height -- not above it, not below it.

    The exclusion is therefore a FULL-HEIGHT COLUMN over each wheel's footprint
    (x within R_WHEEL+clearance of the axle, |y| across the wheel's y-band
    widened by clearance, every z), not merely the wheel's own disc volume.

    An earlier version carved only the x-z disc, which left the body free to
    arch OVER the wheels: 114 body cells sat directly above the rear wheel,
    hiding it from the top view. Carving the full column is what makes the
    wheels read as separate outriggers and keeps them visible.

    The T7.9 wedge masks (built separately) extend the keep-clear FORWARD of
    the front wheel and BEHIND the rear wheel; this column covers the wheel
    footprint itself. Together they are the sidepod/wheel exclusion zones.
    """
    from geometry_contract import (
        FRONT_WHEEL_INNER_Y_M, REAR_WHEEL_INNER_Y_M, WHEEL_CLEARANCE_M,
        WHEEL_WIDTH_M, R_WHEEL_M,
    )

    # A dedicated keep-out ON TOP of the 2 mm assembly clearance. With only the
    # 2 mm clearance the body sat right against every wheel, so the init="full"
    # block looked like the wheels were embedded in a slab rather than standing
    # on outriggers. This margin necks the body clearly away from each wheel
    # (inboard AND fore/aft), so the wheels read as separate and stay visible
    # from the top with unambiguous daylight around them.
    keepout = WHEEL_KEEPOUT_MARGIN_M

    nx, ny, nz = region.shape
    ox, oy, oz = region.origin_m
    dx = GRID_SPACING_M
    xs = ox + np.arange(nx) * dx
    ys = oy + np.arange(ny) * dx

    x_front_m = x_front_mm / 1000.0
    rear_axle_m = x_front_m + W_mm / 1000.0
    r_clear = R_WHEEL_M + WHEEL_CLEARANCE_M + keepout

    mask = np.zeros(region.shape, dtype=bool)
    axles = (
        (x_front_m, FRONT_WHEEL_INNER_Y_M),
        (rear_axle_m, REAR_WHEEL_INNER_Y_M),
    )
    for axle_x, inner_y in axles:
        # x-footprint of the wheel disc, plus clearance + keep-out margin
        in_x = (np.abs(xs - axle_x) <= r_clear)[:, None, None]
        # y-band: necked well INSIDE the inner face, out past the wheel
        y0 = inner_y - WHEEL_CLEARANCE_M - keepout
        y1 = inner_y + WHEEL_WIDTH_M + WHEEL_CLEARANCE_M
        in_y = ((np.abs(ys) >= y0) & (np.abs(ys) <= y1))[None, :, None]
        # full height: nothing above or below the wheel (top/bottom visibility)
        mask |= in_x & in_y
    return mask


def _zone_masks(
    region: BoundingRegion,
    bv: BoundingVolumes,
    re: RuleEnvelope,
) -> dict[str, np.ndarray]:
    """Per-component zone masks over the unified grid.

    A zone is where a component is ALLOWED to place material. Zones are
    mutually exclusive by construction (see the precedence note below) and
    their union is the legal envelope; everything outside is permanently air.

    Precedence, applied in order, so each test only sees cells no earlier zone
    claimed:
      nose      -- forward of Ref Plane A, inside T8.5.1's 15 mm half-width and
                   25 mm height limits
      sidepod   -- outboard of the body wall, within the wheel-free corridor
      rearpod   -- aft of the rear wheel's trailing edge
      main_body -- everything else between Ref Plane A and the rearpod
    """
    nx, ny, nz = region.shape
    ox, oy, oz = region.origin_m
    dx = GRID_SPACING_M

    xs = ox + np.arange(nx) * dx
    ys = oy + np.arange(ny) * dx
    zs = oz + np.arange(nz) * dx

    X = xs[:, None, None]
    absY = np.abs(ys)[None, :, None]
    Z = zs[None, None, :]

    ref_A = bv.ref_plane_A_m
    rearpod_x_start = bv.rearpod.origin_m[0]
    rearpod_x_end = bv.rearpod.x_max_m()

    zones: dict[str, np.ndarray] = {}

    zones["nose"] = (
        (X >= ox) & (X < ref_A)
        & (absY <= re.y_nose_half_m)
        & (Z >= re.z_floor_m) & (Z <= re.z_nose_top_m)
    )

    claimed = zones["nose"].copy()

    zones["sidepod"] = (
        ~claimed
        & (X >= bv.sidepod_x_min_m) & (X <= bv.sidepod_x_max_m)
        & (absY > re.y_sidepod_inner_m) & (absY <= re.y_sidepod_outer_m)
        & (Z >= re.z_floor_m) & (Z <= re.z_sidepod_top_m)
    )
    claimed |= zones["sidepod"]

    zones["rearpod"] = (
        ~claimed
        & (X >= rearpod_x_start) & (X <= rearpod_x_end)
        & (absY <= re.y_body_half_m)
        & (Z >= re.z_floor_m) & (Z <= re.z_rearpod_top_m)
    )
    claimed |= zones["rearpod"]

    zones["main_body"] = (
        ~claimed
        & (X >= ref_A) & (X < rearpod_x_start)
        & (absY <= re.y_body_half_m)
        & (Z >= re.z_floor_m) & (Z <= re.z_body_top_m)
    )

    return zones


def measure_cargo_erosion(cargo_solid, hard_air, region, ref_plane_A_m,
                          d_halo_mm) -> tuple[int, int]:
    """(eaten, requested) interior cargo cells lost to forced-air regions.

    Extracted so the FEASIBILITY SCREEN and the BUILD-TIME GUARD measure the
    same thing. They previously could not: find_cargo_placement screened only
    the halo pocket, while this measurement -- the actual authority -- also sees
    wheel keep-clears, T7.9 zones, the cartridge bore and the ballast slot. The
    screen therefore returned placements the builder then refused, and with a
    cargo scorer driving placement forward into the front wheel keep-clear that
    was EVERY candidate: Stage 1 returned best=None and the whole two-stage run
    died at `'NoneType' object has no attribute 'W_mm'`.

    The grid border seal is excluded (it is not a collision), and ballast
    overlap is exempt: T4.2 says the cargo "may coincide with the legal ballast
    container but not the halo pocket". With the cargo pinned to 14..24 mm and
    the ballast slot at 17.65..24 mm they overlap 6.35 mm BY DESIGN, so counting
    that as erosion would reject every car.
    """
    interior = np.ones(region.shape, dtype=bool)
    interior[0, :, :] = interior[-1, :, :] = False
    interior[:, 0, :] = interior[:, -1, :] = False
    interior[:, :, 0] = interior[:, :, -1] = False
    allowed = build_ballast_container_forbidden_mask(
        region.origin_m, region.shape, ref_plane_A_m, d_halo_mm,
    )
    _mirror_right_onto_left(allowed)
    requested = int((cargo_solid & interior).sum())
    eaten = int((cargo_solid & hard_air & interior & ~allowed).sum())
    return eaten, requested


def cargo_placement_is_buildable(geom_without_cargo, ref_plane_A_m, d_halo_mm,
                                 x_start_m, z_base_m, flip) -> bool:
    """Would this cargo placement survive build_unified_geometry's guard?

    Uses the forced-air mask of an already-built cargo-free geometry, so a
    caller that has one (stage1_search builds exactly that for its scorer) can
    screen candidate placements for a few mask operations instead of a full
    rebuild each.
    """
    # Local import: virtual_cargo <-> unified_phi would be circular at module
    # level, which is why build_unified_geometry imports it here too.
    from virtual_cargo import build_virtual_cargo_solid_mask

    region = geom_without_cargo.region
    z_base = max(z_base_m, region.origin_m[2])
    cargo = build_virtual_cargo_solid_mask(
        region.origin_m, region.shape, x_start_m, z_base, flip=flip,
    )
    _mirror_right_onto_left(cargo)
    eaten, requested = measure_cargo_erosion(
        cargo, geom_without_cargo.phi.hard_mask_air, region,
        ref_plane_A_m, d_halo_mm,
    )
    if not requested:
        return False
    return eaten / requested <= CARGO_MAX_ERODED_FRACTION


def build_unified_geometry(
    W_mm: float,
    x_front_mm: float,
    d_halo_mm: float,
    rule_envelope: Optional[RuleEnvelope] = None,
    wheel_x_half_width_mm: float = WHEEL_X_CLEARANCE_HALF_WIDTH_MM,
    init_mode: str = "full",
    seed: int = 42,
    with_cargo: bool = True,
    cargo_placement: Optional[dict] = None,
) -> UnifiedGeometry:
    """Build the single labelled phi field for one (W, x_front, d_halo).

    init_mode defaults to "full" (everything inside the envelope starts solid).
    Topology optimisation should start full and carve away; the old "sphere"
    default inscribed a tiny sphere in each box that never reached the
    attachment faces, producing four disconnected lumps (sandbox finding 7).

    with_cargo=False drops the mandatory T4.2 solid region. The resulting
    geometry is NOT competition-legal; it exists to unblock exploration, same
    contract as sandbox/coarse.disable_virtual_cargo.
    """
    from fixed_hardware import (
        ForbiddenCylinder,
        _build_cylinder_void_mask,
        compute_default_fixed_hardware_inputs,
        place_fixed_hardware,
    )
    from virtual_cargo import build_virtual_cargo_solid_mask, find_cargo_placement

    re = rule_envelope or default_rule_envelope()

    # Reuse the existing derivation for every landmark, rather than
    # re-deriving them here and risking drift from bounding_volumes.
    axle_z_m = R_WHEEL_M
    cyl_r_m = R_WHEEL_M + WHEEL_CLEARANCE_M
    x_half_m = mm_to_m(wheel_x_half_width_mm)
    front_cyl = ForbiddenCylinder(
        mm_to_m(x_front_mm), 0.0, axle_z_m, cyl_r_m, x_half_m
    )
    rear_cyl = ForbiddenCylinder(
        mm_to_m(x_front_mm) + mm_to_m(W_mm), 0.0, axle_z_m, cyl_r_m, x_half_m
    )
    bv = compute_bounding_volumes(
        W_mm, x_front_mm, d_halo_mm, front_cyl, rear_cyl, re,
        wheel_x_half_width_mm=wheel_x_half_width_mm,
    )

    # ── The single envelope ────────────────────────────────────────────────
    # x: nose tip to the rearmost machined face.
    # y: FULL width. The old sidepod grid was right-half-only with the left
    #    mirrored at STL time; carrying a half-width convention for one label
    #    inside a shared field would be a trap. Symmetry is enforced on the
    #    field instead (see enforce_symmetry).
    # z: track clearance floor to the T3.5 ceiling.
    x_span_mm = bv.rearpod.x_max_m() * 1000.0
    z_span_mm = (re.z_body_top_m - re.z_floor_m) * 1000.0

    # The y axis MUST be symmetric about the centreline with a cell centred
    # exactly at y=0. Two things depend on it and both fail silently otherwise:
    # the half-car slice at y=0 (Part 2's contract), and the sidepod pair's
    # com_y=0 convention that Part 2's mass_com_ingest expects. A naive
    # origin=-y_outer with an even cell count puts no cell at y=0 and biases
    # every COM by up to half a cell (measured: com_y up to -1.2 mm at 1.5 mm
    # spacing, which is not small next to a 71 mm wide car).
    n_half = int(np.ceil(re.y_sidepod_outer_m / GRID_SPACING_M))
    ny = 2 * n_half + 1

    region = BoundingRegion(
        component="car",
        origin_m=(0.0, -n_half * GRID_SPACING_M, re.z_floor_m),
        shape=(grid_cells(x_span_mm), ny, grid_cells(z_span_mm)),
    )

    # ── Labels ─────────────────────────────────────────────────────────────
    zones = _zone_masks(region, bv, re)
    labels = np.full(region.shape, LABEL_NONE, dtype=np.uint8)
    for name, mask in zones.items():
        labels[mask] = LABEL_IDS[name]
    _mirror_right_onto_left(labels)

    # ── Hard air ───────────────────────────────────────────────────────────
    # Everything not claimed by a zone is permanently outside the car.
    hard_air = labels == LABEL_NONE

    # T7.9 wheel-visibility keep-clear zones. Each builder positions its zone
    # in absolute coordinates, so all three can be rasterised onto the one grid.
    for comp in ("main_body", "sidepod", "rearpod"):
        zone = build_t79_forbidden_mask(
            comp, region.origin_m, region.shape,
            W_mm, x_front_mm, wheel_x_half_width_mm,
        )
        if zone is not None:
            hard_air |= zone

    # T7.2 lateral wheel exclusion: keep the body clear of all four wheels.
    hard_air |= wheel_keepclear_mask(region, W_mm, x_front_mm)

    # T4.4.4 halo mounting pocket.
    hard_air |= build_halo_pocket_forbidden_mask(
        region.origin_m, region.shape, bv.ref_plane_A_m, d_halo_mm,
    )

    # Legal ballast container (Appendix ix): a mandatory empty slot under the
    # halo aperture, always present. Sits directly below the pocket floor so it
    # merges with the pocket recess (open to the top, no sealed cavity).
    hard_air |= build_ballast_container_forbidden_mask(
        region.origin_m, region.shape, bv.ref_plane_A_m, d_halo_mm,
    )

    # Fixed hardware voids: axle holes, wheel discs, cartridge bore, halo.
    # place_fixed_hardware rasterises onto whatever grid it is handed, so the
    # unified region goes in where the main-body grid used to. This is also
    # what finally lets the cartridge bore carve main_body and rearpod as one
    # hole instead of two grids each carving their own half.
    fixed_hardware = None
    try:
        hw_inputs = compute_default_fixed_hardware_inputs(
            W_mm, x_front_mm, d_halo_mm,
            bv.ref_plane_A_m, bv.ref_plane_B_m,
            rear_face_x_m=bv.rearpod.x_max_m(),
        )
        hw_inputs["body_grid_shape"] = region.shape
        hw_inputs["body_grid_origin_m"] = region.origin_m
        fixed_hardware = place_fixed_hardware(
            W_mm=W_mm, x_front_mm=x_front_mm, **hw_inputs
        )
        hard_air |= fixed_hardware.combined_void_mask
        # T4.4.2 / T4.4.3: the halo must be visible in the front, side and top
        # views. Modelled as hard AIR over the halo's shadow in each view
        # direction, so the optimiser cannot fill it at any point in the
        # descent -- a rule that binds every candidate rather than one audited
        # after the fact. Measured before this existed, on a carved 79 g car:
        # 100% of halo cells obstructed in top view, 100% in front, 64.7% in
        # side. The rule was comprehensively violated and nothing forbade it.
        from fixed_hardware import halo_visibility_air_mask
        hard_air |= halo_visibility_air_mask(
            fixed_hardware.halo_void_mask,
            z_origin_m=region.origin_m[2],
            dz_m=GRID_SPACING_M,
        )

        # The halo->canister loft. Making a surface across the component
        # boundary REPRESENTABLE (this module's whole point) did not make one
        # get built: the visibility mask bounds only the halo's own shadow, so
        # aft of the halo the field filled to the envelope roof and left a
        # 28 mm cliff behind the halo. The loft is a ceiling, not a shape --
        # the optimiser still chooses everything under it.
        from fixed_hardware import halo_canister_loft_air_mask
        hard_air |= halo_canister_loft_air_mask(
            fixed_hardware.halo_void_mask,
            getattr(fixed_hardware, "canister_cylinder", None),
            x_origin_m=region.origin_m[0],
            y_origin_m=region.origin_m[1],
            z_origin_m=region.origin_m[2],
            d_m=GRID_SPACING_M,
        )
    except ImportError:
        # Part 2 supplies FixedHardwareSpec. Without it there are no hardware
        # voids -- loud in the returned object (fixed_hardware is None), not
        # silently substituted.
        pass

    _mirror_right_onto_left(hard_air)

    # One-cell border so the phi=0 surface is always closed. No attachment-face
    # strips: with a single field, connectivity is the field's own property.
    hard_air[0, :, :] = True
    hard_air[-1, :, :] = True
    hard_air[:, 0, :] = True
    hard_air[:, -1, :] = True
    hard_air[:, :, 0] = True
    hard_air[:, :, -1] = True

    # ── Hard solid ─────────────────────────────────────────────────────────
    hard_solid = np.zeros(region.shape, dtype=bool)
    if with_cargo:
        # cargo_placement lets Stage 1 inject a COM-scored (position, flip);
        # None falls back to the geometric default (corridor-centre, wide-fwd).
        placement = cargo_placement or find_cargo_placement(
            x_front_mm, W_mm, bv.ref_plane_A_m, d_halo_mm, re.z_floor_m,
        )
        # Clamp the cargo base to the grid floor. find_cargo_placement derives
        # z_base from the rule envelope's z_floor + margin (0.001 m), but the
        # grid ORIGIN is at z_floor itself (0.0015 m), so an unclamped base sat
        # below the first cell layer and the wedge's bottom row was silently
        # clipped by the border seal below (measured: 3.1% of cargo cells).
        z_base = max(placement["z_base_m"], region.origin_m[2])
        hard_solid |= build_virtual_cargo_solid_mask(
            region.origin_m, region.shape,
            placement["x_start_m"], z_base,
            flip=placement.get("flip", False),
        )

    # The lofted deck itself. The ceiling above (halo_canister_loft_air_mask)
    # only bounds the surface; with no aero term in Stage 1's proxy, mass
    # minimisation then pulls the deck well below it -- measured flat at 38.5 mm
    # against a 44.1 mm canister top, running past the cartridge instead of
    # arriving at it. "Lofts to the canister" describes a surface that exists,
    # so the skin is required material.
    if fixed_hardware is not None:
        try:
            from fixed_hardware import halo_canister_loft_solid_mask
            hard_solid |= halo_canister_loft_solid_mask(
                fixed_hardware.halo_void_mask,
                getattr(fixed_hardware, "canister_cylinder", None),
                x_origin_m=region.origin_m[0],
                y_origin_m=region.origin_m[1],
                z_origin_m=region.origin_m[2],
                d_m=GRID_SPACING_M,
            )
        except ImportError:
            pass

    _mirror_right_onto_left(hard_solid)

    # Air wins on overlap, matching PhiGrid.build_hard_masks' resolution.
    # This is where T4.2's MANDATORY keep-solid cargo can be silently deleted:
    # find_cargo_placement only avoids the halo pocket, so a placement
    # overlapping any other void (wheel keep-clear, T7.9 zones, cartridge bore,
    # ballast slot) is eaten here with no diagnostic. PhiGrid's own overlap
    # check is dead code by this point — the `&=` has already run. Measure the
    # loss BEFORE resolving, and refuse to build a car that doesn't contain the
    # cargo the regs require.
    if with_cargo:
        eaten, requested = measure_cargo_erosion(
            hard_solid, hard_air, region, bv.ref_plane_A_m, d_halo_mm,
        )
        if requested and eaten / requested > CARGO_MAX_ERODED_FRACTION:
            raise ValueError(
                f"virtual cargo (T4.2) is {100 * eaten / requested:.1f}% eroded by "
                f"interior forced-air regions at W={W_mm}, x_front={x_front_mm}, "
                f"d_halo={d_halo_mm} ({eaten}/{requested} cells, excluding the "
                "grid border seal). find_cargo_placement only avoids the halo "
                "pocket; this placement collides with a wheel keep-clear, T7.9 "
                "zone, cartridge bore or ballast slot. The car would not contain "
                "the mandatory cargo volume."
            )
    hard_solid &= ~hard_air

    phi = PhiGrid("car", region, np.zeros(region.shape, dtype=np.float32),
                  hard_solid, hard_air)
    _init_field(phi, init_mode, seed)
    phi.apply_hard_constraints()

    return UnifiedGeometry(
        phi=phi,
        labels=labels,
        region=region,
        landmarks={
            "ref_plane_A_m": bv.ref_plane_A_m,
            "ref_plane_B_m": bv.ref_plane_B_m,
            "front_axle_m": mm_to_m(x_front_mm),
            "rear_axle_m": mm_to_m(x_front_mm) + mm_to_m(W_mm),
            "rear_face_m": bv.rearpod.x_max_m(),
        },
        bv=bv,
        W_mm=W_mm,
        x_front_mm=x_front_mm,
        d_halo_mm=d_halo_mm,
        fixed_hardware=fixed_hardware,
        spacing_m=GRID_SPACING_M,
    )


def remap_geometry(
    prev_geom: "UnifiedGeometry",
    W_mm: float = None,
    x_front_mm: float = None,
    d_halo_mm: float = None,
    cargo_placement: dict = None,
    with_cargo: bool = True,
) -> "UnifiedGeometry":
    """Carry an evolved shape onto a grid at the CURRENT GRID_SPACING_M.

    This is what makes coarse-to-fine optimisation possible, and it is the
    thing warm_start_phi_grids explicitly does NOT do: that rebuilds a fresh
    field and throws the evolved shape away.

    Why it matters. Surface travel per Hamilton-Jacobi iteration is
    `CFL x grid_spacing`, while phi cost scales with spacing^-3. So spacing is a
    LINEAR lever on how far the shape can move and a CUBIC one on what that
    costs:

        0.5 mm, CFL 0.3   0.15 mm/iter   ~200 iters to move 30 mm   8.50M cells
        2.0 mm, CFL 0.9   1.80 mm/iter   ~17  iters to move 30 mm   0.13M cells

    Carving the gross shape on a coarse grid and then polishing on a fine one is
    therefore ~12x fewer iterations AND ~60x cheaper per iteration than doing it
    all at 0.5 mm. Without a remap you cannot do that at all, because dropping to
    a finer grid means starting from a brick again.

    Resampling is trilinear on the signed distance field, then the hard masks
    are re-applied and the field is re-distanced. Reinitialisation matters: an
    interpolated SDF no longer satisfies |grad phi| = 1, and every HJ step
    assumes it does.

    Args:
        prev_geom: the evolved geometry to carry over. Its own spacing is read
            from prev_geom.spacing_m, not from the module global -- the global
            has usually already been changed to the target by the caller.
        W_mm / x_front_mm / d_halo_mm: default to prev_geom's own scalars, so
            the common case (same car, finer grid) needs no arguments.

    Returns a new UnifiedGeometry at the current spacing whose zero level set
    matches prev_geom's, with all hard constraints freshly applied.
    """
    from scipy.ndimage import map_coordinates
    from phi_updater import reinitialise_sdf

    src_dx = prev_geom.spacing_m or GRID_SPACING_M
    W_mm = prev_geom.W_mm if W_mm is None else W_mm
    x_front_mm = prev_geom.x_front_mm if x_front_mm is None else x_front_mm
    d_halo_mm = prev_geom.d_halo_mm if d_halo_mm is None else d_halo_mm

    new = build_unified_geometry(
        W_mm, x_front_mm, d_halo_mm, init_mode="full",
        cargo_placement=cargo_placement, with_cargo=with_cargo,
    )
    dst_dx = new.spacing_m or GRID_SPACING_M

    # World coordinates of every destination cell centre, expressed in SOURCE
    # index space, so map_coordinates can sample the source field there.
    src_o = np.asarray(prev_geom.region.origin_m, dtype=np.float64)
    dst_o = np.asarray(new.region.origin_m, dtype=np.float64)
    idx = np.indices(new.region.shape, dtype=np.float32)
    coords = np.empty_like(idx)
    for ax in range(3):
        coords[ax] = ((dst_o[ax] + idx[ax] * dst_dx) - src_o[ax]) / src_dx

    resampled = map_coordinates(
        prev_geom.phi.grid.astype(np.float32), coords,
        order=1, mode="nearest",       # outside the old box -> nearest edge value
    ).astype(np.float32)

    new.phi.grid = resampled
    new.phi.apply_hard_constraints()
    reinitialise_sdf(new.phi)          # restore |grad phi| = 1 after interpolation
    new.phi.apply_hard_constraints()   # reinit can nudge cells across the masks
    enforce_symmetry(new)
    return new


def _init_field(phi: PhiGrid, mode: str, seed: int) -> None:
    """Initialise the unified field.

    "full" is the mode the other three do not offer and the one topology
    optimisation actually wants: start with every cell inside the envelope
    solid and let the optimiser carve. PhiGrid.init's own modes are delegated
    to for the rest.
    """
    if mode != "full":
        phi.init(mode, seed=seed)
        return
    # Signed distance to the envelope boundary would be ideal; a constant
    # negative interior is enough to start, and reinitialise_sdf will
    # redistance it on the first update.
    phi.grid = np.full(phi.bv.shape, -GRID_SPACING_M, dtype=np.float32)


def enforce_machinability(geom: "UnifiedGeometry") -> int:
    """Fill void no tool can reach. Returns how many cells were filled.

    The block is milled from the top, the bottom and the two sides only, so a
    pocket of air is REAL only if a straight run along +z, -z, +y or -y gets
    from it to the outside without passing through solid. Anything else is a
    cavity you would have to machine from inside a closed shell.

    THE FACE GATE DOES NOT CATCH THIS. _check_accessibility ray-casts surface
    normals, so on the 2026-08-06 car it flagged 239 faces / 96 mm^2 -- and all
    239 were +-x-facing, i.e. it was reporting "this face points down an axis
    that has no tool", which is trivially true of every x-normal face and says
    nothing about whether the shape can be cut. Meanwhile 8,310 air CELLS had
    no clear run out in any of the four real directions, spanning x 91-204 mm.
    Two orders of magnitude of unmakeable void, invisible to a normals test.

    So this is a projection applied during the descent, not a gate after it:
    unreachable void goes back to solid, the optimiser sees the mass it just
    regained, and carves again from a direction that exists. Same shape of fix
    as the halo visibility masks -- make the illegal state unrepresentable
    rather than detectable.

    Hardware voids are exempt. The cartridge bore is drilled along x and the
    halo pocket is a placed part; neither is milled from +-Y/+-Z and neither
    should be filled. They live in hard_mask_air, so exempting that covers the
    bore, the halo pocket, the axle zones, the halo visibility shadow and the
    loft ceiling in one go.
    """
    solid = geom.phi.grid < 0

    def _clear_run(axis: int, positive: bool) -> np.ndarray:
        """True where no solid lies strictly beyond this cell along the axis."""
        s = np.flip(solid, axis=axis) if positive else solid
        acc = np.cumsum(s, axis=axis) - s          # excludes the cell itself
        return np.flip(acc, axis=axis) == 0 if positive else acc == 0

    reachable = (_clear_run(2, True) | _clear_run(2, False)
                 | _clear_run(1, True) | _clear_run(1, False))
    stuck = (~solid) & (~reachable) & (~geom.phi.hard_mask_air)
    n = int(stuck.sum())
    if n:
        # One cell inside the surface, so reinitialisation has a sign to work
        # with rather than a plateau at exactly zero.
        geom.phi.grid[stuck] = -GRID_SPACING_M
        geom.phi.apply_hard_constraints()
    return n


def enforce_symmetry(geom: UnifiedGeometry) -> None:
    """Mirror the right half (y >= 0) onto the left, in place.

    The old representation got symmetry for free by optimising a right-half
    sidepod grid and reflecting it. With one full-width field, symmetry has to
    be imposed on the field. Doing it here keeps the half-car CFD contract
    honest: the y<0 half is exactly the mirror of y>0, so slicing at y=0 loses
    nothing.

    The y axis is built with an odd cell count centred on y=0 (see
    build_unified_geometry), so the mirror of index j is exactly ny-1-j and no
    interpolation is involved.
    """
    grid = geom.phi.grid
    ny = grid.shape[1]
    mid = ny // 2          # the cell sitting exactly at y = 0
    right = grid[:, mid:, :]
    grid[:, : mid + 1, :] = right[:, ::-1, :]
    geom.phi.apply_hard_constraints()


def density_field(geom: UnifiedGeometry) -> np.ndarray:
    """Per-cell density (kg/m^3) from component labels.

    The shape gradient multiplies dT/dmass by density; on the unified field the
    density varies by component (nose 1000, milled 163), so it must be an array.
    Cells outside every component (LABEL_NONE) get 0.
    """
    rho = np.zeros(geom.shape, dtype=np.float64)
    for lid, name in LABEL_NAMES.items():
        rho[geom.labels == lid] = get_density(name)
    return rho


def extract_half_surface(geom: UnifiedGeometry):
    """Watertight right-half (y >= 0) car mesh for Part 2's half-car CFD.

    Part 2 requires the half-car STL to be watertight AND have all vertices
    y >= -1e-6 (closed on the symmetry plane). Slicing the full mesh at y=0 and
    capping does NOT reliably produce a watertight result (trimesh's cap
    triangulation fails on this geometry -- the long-standing slice-watertight
    problem, PLACEHOLDERS.md item 11).

    Instead we build the half from the FIELD: copy it, force every cell with
    y < 0 to air, and marching-cube that. The zero level set then closes itself
    against the forced air right at the symmetry plane, giving a naturally
    watertight half with a flat y=0 cap -- no fragile mesh boolean. Residual
    sub-cell negative-y vertices are snapped to 0.
    """
    import surface_extraction as SE
    from phi_grid import PhiGrid

    ny = geom.shape[1]
    oy = geom.region.origin_m[1]
    ys = oy + np.arange(ny) * GRID_SPACING_M

    half_grid = geom.phi.grid.copy()
    air_value = float(np.abs(half_grid).max()) + GRID_SPACING_M
    half_grid[:, ys < -1e-9, :] = air_value          # everything left of y=0 -> air

    clone = PhiGrid("car", geom.region, half_grid.astype(np.float32),
                    geom.phi.hard_mask_solid, geom.phi.hard_mask_air)
    mesh = SE._repair_mesh(SE._marching_cubes(clone))
    # Flatten any sub-cell overshoot onto the symmetry plane (y >= 0).
    mesh.vertices[mesh.vertices[:, 1] < 0.0, 1] = 0.0
    mesh.merge_vertices()
    return mesh


def slice_right_half(mesh):
    """Deprecated: use extract_half_surface(geom). Kept for callers that only
    have a mesh -- slices at y=0, but does NOT guarantee watertightness."""
    import trimesh
    half = trimesh.intersections.slice_mesh_plane(
        mesh, plane_normal=[0.0, 1.0, 0.0], plane_origin=[0.0, 0.0, 0.0], cap=True
    )
    half.merge_vertices()
    return half


def nearest_labels(geom: UnifiedGeometry) -> np.ndarray:
    """`geom.labels` with LABEL_NONE cells filled from their nearest labelled cell.

    Mesh vertices and face centroids sit ON the phi=0 surface, which can round
    to a cell just outside every component zone (LABEL_NONE). Those points still
    belong to a component for gate purposes -- a face on the main_body's outer
    skin is a main_body face. Filling by nearest non-NONE label resolves that
    without inventing a rule.
    """
    from scipy.ndimage import distance_transform_edt

    unlabelled = geom.labels == LABEL_NONE
    if not unlabelled.any():
        return geom.labels
    _, inds = distance_transform_edt(unlabelled, return_indices=True)
    return geom.labels[tuple(inds)]


def _points_to_labels(
    geom: UnifiedGeometry, points: np.ndarray, filled: np.ndarray
) -> np.ndarray:
    """Label each world-space point by the grid cell containing it."""
    origin = np.asarray(geom.region.origin_m)
    idx = np.rint((points - origin) / GRID_SPACING_M).astype(np.int64)
    for axis, n in enumerate(geom.shape):
        np.clip(idx[:, axis], 0, n - 1, out=idx[:, axis])
    return filled[idx[:, 0], idx[:, 1], idx[:, 2]]


def extract_unified_surface(
    geom: UnifiedGeometry,
    max_retries: int = 3,
    allow_inaccessible: bool = False,
):
    """Run the six extraction stages on the unified field, per-label.

    surface_extraction.extract_surface dispatches its manufacturing gates on
    `phi.component`, which worked when each component had its own grid and its
    own mesh. One mesh now spans several components, so the dispatch has to move
    from the mesh to the FACE: each face and vertex is attributed to a component
    by label, and each component's rules are applied only to its own geometry.

    What that changes, concretely:
      - the 3.15 mm machining radius is checked on milled vertices only; the
        nose is 3D printed and exempt (it gets the wall-thickness check instead)
      - tool accessibility is tested per component against ITS tool directions,
        but ray occlusion is computed against the WHOLE car -- a cutter reaching
        a main_body face can be blocked by the nose, which four separate meshes
        could not have discovered
      - fewer boundary exemptions. The four-box path exempted every face lying
        on a component's grid wall, which included the flat inter-component cut
        planes. Those planes no longer exist, so only the true outer envelope is
        exempt and the check is correspondingly stricter.

    Returns (mesh, report). `report` carries per-component inaccessible area so
    a large failure can be surfaced as a manufacturing penalty rather than
    silently swallowed -- 01_generative_geometry.md's failure table asks for
    "assign manufacturing penalty, continue" on large failures, which the
    original code never did (it raised on both sizes).
    """
    import surface_extraction as SE
    from geometry_contract import (
        LARGE_INACCESSIBLE_AREA_M2, MIN_RADIUS_M, NOSE_MIN_WALL_THICKNESS_M,
        TOOL_DIRECTIONS,
    )

    filled = nearest_labels(geom)
    milled_ids = {LABEL_IDS[n] for n in MILLED_COMPONENTS}

    mesh = SE._repair_mesh(SE._marching_cubes(geom.phi))

    # ── Stage 3a: machining radius, milled vertices only ───────────────────
    for attempt in range(max_retries):
        vlabels = _points_to_labels(geom, np.asarray(mesh.vertices), filled)
        is_milled = np.isin(vlabels, list(milled_ids))
        radii = SE._estimate_local_radii(mesh)
        bad = np.where(is_milled & (radii < MIN_RADIUS_M))[0]
        if bad.size == 0:
            break
        if attempt == max_retries - 1:
            raise SE.RadiusViolation(
                f"car: min radius {radii[bad].min()*1000:.2f} mm < "
                f"{MIN_RADIUS_M*1000:.2f} mm on {bad.size} milled vertices "
                f"after {max_retries} retries."
            )
        SE._smooth_phi_neighbourhood(geom.phi, list(bad), mesh)
        mesh = SE._repair_mesh(SE._marching_cubes(geom.phi))

    # ── Stage 3b: nose wall thickness, nose cells only ─────────────────────
    nose_cells = geom.component_mask("nose")
    for attempt in range(max_retries):
        thin = (
            SE._thin_wall_mask(geom.phi, NOSE_MIN_WALL_THICKNESS_M)
            & nose_cells
            & ~geom.phi.hard_mask_solid
        )
        if not thin.any():
            break
        if attempt == max_retries - 1:
            raise SE.WallThicknessViolation(
                f"nose: {int(thin.sum())} solid cells thinner than "
                f"{NOSE_MIN_WALL_THICKNESS_M*1000:.2f} mm (3D-printing shell "
                f"constraint) after {max_retries} retries."
            )
        SE._thicken_phi_at_thin_walls(geom.phi, thin, NOSE_MIN_WALL_THICKNESS_M)
        mesh = SE._repair_mesh(SE._marching_cubes(geom.phi))

    # ── Stage 4: accessibility, per component ──────────────────────────────
    flabels = _points_to_labels(geom, np.asarray(mesh.triangles_center), filled)
    boundary_exempt = SE._boundary_coincident_face_mask(mesh, geom.region)
    areas: dict[str, float] = {}
    for name, directions in TOOL_DIRECTIONS.items():
        own = flabels == LABEL_IDS[name]
        if not own.any():
            areas[name] = 0.0
            continue
        blocked = SE._find_inaccessible_faces(mesh, directions, boundary_exempt)
        if len(blocked) == 0:
            areas[name] = 0.0
            continue
        blocked_own = np.asarray(blocked)[own[np.asarray(blocked)]]
        areas[name] = float(mesh.area_faces[blocked_own].sum())

    worst = max(areas.values()) if areas else 0.0
    if worst >= LARGE_INACCESSIBLE_AREA_M2 and not allow_inaccessible:
        offender = max(areas, key=areas.get)
        raise SE.AccessibilityFailure(
            f"{offender}: large inaccessible area {worst*1e6:.1f} mm^2",
            is_large=True,
        )

    # ── Stages 5 and 6 ─────────────────────────────────────────────────────
    SE._check_rules(mesh, geom.region)
    SE._check_mesh_quality(mesh, "car")

    report = {
        "inaccessible_area_mm2": {k: v * 1e6 for k, v in areas.items()},
        "connected_bodies": len(mesh.split(only_watertight=False)),
        "watertight": bool(mesh.is_watertight),
        "volume_cm3": float(mesh.volume * 1e6),
    }
    return mesh, report


def compute_mass_com(geom: UnifiedGeometry) -> list:
    """Per-component ComponentMassCOM from the one field, via labels.

    Same arithmetic as mass_com_calculator.compute_component_mass_com, but the
    component split comes from `labels` rather than from separate grids. Two
    real differences from the old path:

      - No double counting. main_body and rearpod used to overlap in x, so
        cells in the overlap contributed to both components' mass.
      - No sidepod doubling. The old code computed a right-half sidepod and
        multiplied its mass by 2; here both sidepods are actual cells.
    """
    from mass_com_calculator import ComponentMassCOM

    dx = GRID_SPACING_M
    ox, oy, oz = geom.region.origin_m
    out = []

    for name in ("nose", "sidepod", "rearpod", "main_body"):
        solid = geom.solid_mask(name)
        n = int(solid.sum())
        if n == 0:
            out.append(ComponentMassCOM(name=name, mass_kg=0.0,
                                        com_x_m=ox, com_y_m=oy, com_z_m=oz))
            continue
        ix, iy, iz = np.where(solid)
        out.append(ComponentMassCOM(
            name=name,
            mass_kg=n * dx ** 3 * get_density(name),
            com_x_m=ox + float(np.mean(ix)) * dx,
            com_y_m=oy + float(np.mean(iy)) * dx,
            com_z_m=oz + float(np.mean(iz)) * dx,
        ))
    return out
