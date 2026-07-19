"""
phi_grid_factory.py — build all four ready-to-use PhiGrid objects for one
(W, x_front, d_halo) candidate.

This is the missing piece audit finding P1-17 identified: "There is no
single function anywhere in Part 1 that takes (W, x_front, d_halo) and
returns ready phi grids -- every caller must hand-assemble bounding volumes,
cylinders, hardware, masks, grids." Part 3's `pipeline_interface.real_bindings.
initialize_phi_fields` was a hard `? UNRESOLVED NotImplementedError` stub
waiting on exactly this.

Wires together, in order:
  1. compute_bounding_volumes  (S2) -- four BoundingRegions + forbidden cylinders
  2. compute_default_fixed_hardware_inputs + place_fixed_hardware  (S6) --
     halo pocket, CO2 canister, wheel/axle void masks (main_body grid only --
     FixedHardwareResult's void masks are shaped to the main body grid; the
     other three components have no hardware voids, only attachment-face
     hard-solid strips)
  3. PhiGrid.build_hard_masks + PhiGrid(...).init(...)  (S3) per component

Attachment faces per component (verified against surface_extraction.py's
_check_mesh_quality, which documents exactly which face of each attachment
component is expected to be open -- nose and rearpod at their LOCAL grid
"rear" face (x=x_max in that component's own index space), sidepod at its
"inner_y" face (y=y_min, the centreline side); main_body has none, since it
is the fully closed component the other three attach TO.

What this does NOT do (scoped out, not silently pretended to be handled):
  - Virtual cargo solid-mask placement (T4.2, virtual_cargo.py) -- an
    optional mass-distribution refinement, not required for the grids to
    exist and pass gates. Callers that need it can pass solid_masks in.
  - Warm-start remapping onto a new (W, d_halo) -- see
    warm_start_phi_grids below, which is a separate, simpler operation
    (re-run this factory fresh; true incremental remap is future work).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from bounding_volumes import (
    BoundingVolumes,
    RuleEnvelope,
    compute_bounding_volumes,
    default_rule_envelope,
)
from fixed_hardware import (
    ForbiddenCylinder,
    compute_default_fixed_hardware_inputs,
    place_fixed_hardware,
)
from geometry_contract import (
    GRID_SPACING_M, R_WHEEL_M, WHEEL_CLEARANCE_M, mm_to_m,
    WHEEL_X_CLEARANCE_HALF_WIDTH_MM,
)
from phi_grid import PhiGrid

# Local-grid attachment face per component (see module docstring for why).
_ATTACHMENT_FACES: dict[str, list[str]] = {
    "nose": ["rear"],
    "sidepod": ["inner_y"],
    "rearpod": ["rear"],
    "main_body": [],
}

_COMPONENTS = ("nose", "sidepod", "rearpod", "main_body")


def _default_forbidden_cylinders(
    W_mm: float, x_front_mm: float, wheel_x_half_width_mm: float
) -> tuple[ForbiddenCylinder, ForbiddenCylinder]:
    """Front/rear wheel exclusion cylinders at the design-default axle height
    (wheel radius above track) — same convention
    compute_default_fixed_hardware_inputs uses for wheel_axle_z_mm."""
    W_m = mm_to_m(W_mm)
    x_front_m = mm_to_m(x_front_mm)
    axle_z_m = R_WHEEL_M
    cylinder_radius_m = R_WHEEL_M + WHEEL_CLEARANCE_M
    x_half_m = mm_to_m(wheel_x_half_width_mm)
    front = ForbiddenCylinder(
        x_center_m=x_front_m, y_center_m=0.0, z_center_m=axle_z_m,
        radius_m=cylinder_radius_m, x_half_width_m=x_half_m,
    )
    rear = ForbiddenCylinder(
        x_center_m=x_front_m + W_m, y_center_m=0.0, z_center_m=axle_z_m,
        radius_m=cylinder_radius_m, x_half_width_m=x_half_m,
    )
    return front, rear


def build_phi_grids_for_candidate(
    W_mm: float,
    x_front_mm: float,
    d_halo_mm: float,
    rule_envelope: Optional[RuleEnvelope] = None,
    wheel_x_half_width_mm: float = WHEEL_X_CLEARANCE_HALF_WIDTH_MM,
    init_mode: str = "sphere",
    seed: int = 42,
    solid_masks: Optional[dict[str, np.ndarray]] = None,
) -> tuple[dict[str, PhiGrid], BoundingVolumes]:
    """Build all four PhiGrid objects for one (W, x_front, d_halo) candidate.

    Args:
        W_mm, x_front_mm, d_halo_mm: outer-loop scalars (validated internally
            by compute_bounding_volumes/place_fixed_hardware).
        rule_envelope: legal envelope dimensions. Defaults to
            bounding_volumes.default_rule_envelope() (U6 confirmed values).
        wheel_x_half_width_mm: half-width of the wheel+axle assembly in x.
        init_mode: PhiGrid.init() mode ("sphere" | "slab" | "random").
        seed: passed to PhiGrid.init() for the "random" mode.
        solid_masks: optional {component: bool array matching that
            component's bv.shape} of extra interior regions to force solid
            (e.g. virtual cargo placements) — passed straight through to
            PhiGrid.build_hard_masks. Components not present in this dict
            get no extra solid mask.

    Returns:
        (phi_grids, bounding_volumes) — phi_grids is
        {"nose": PhiGrid, "sidepod": PhiGrid, "rearpod": PhiGrid,
        "main_body": PhiGrid}, each already initialised and with hard
        constraints applied. bounding_volumes is returned too since callers
        (mass/COM calculation, STL assembly) need it downstream and
        recomputing it would silently risk drifting from the exact one used
        to build these grids.

    Invalid input behavior:
        Raises ValueError for invalid W/x_front/d_halo (via
        compute_bounding_volumes/place_fixed_hardware's own validation), or
        if the sidepod corridor collapses to zero/negative length at this
        (W, x_front) combination. Raises whatever
        place_fixed_hardware/_validate_halo_position raises for a d_halo that
        passes validate_d_halo but fails placement (see audit K-5 — this is
        why K-5's fix aligned the two bounds, but a caller-supplied
        rule_envelope could still theoretically produce a placement failure).
    """
    rule_envelope = rule_envelope or default_rule_envelope()
    solid_masks = solid_masks or {}

    front_cylinder, rear_cylinder = _default_forbidden_cylinders(
        W_mm, x_front_mm, wheel_x_half_width_mm
    )
    bv = compute_bounding_volumes(
        W_mm, x_front_mm, d_halo_mm,
        front_cylinder, rear_cylinder, rule_envelope,
        wheel_x_half_width_mm=wheel_x_half_width_mm,
    )

    hw_inputs = compute_default_fixed_hardware_inputs(
        W_mm, x_front_mm, d_halo_mm, bv.ref_plane_A_m, bv.ref_plane_B_m,
    )
    # P1-17's exact bug: compute_default_fixed_hardware_inputs's returned
    # dict is missing these two required place_fixed_hardware kwargs. Filled
    # in here from the main_body region computed above, which is the single
    # source of truth for the body grid's shape/origin.
    hw_inputs["body_grid_shape"] = bv.main_body.shape
    hw_inputs["body_grid_origin_m"] = bv.main_body.origin_m

    hw_result = place_fixed_hardware(W_mm=W_mm, x_front_mm=x_front_mm, **hw_inputs)

    phi_grids: dict[str, PhiGrid] = {}
    for component in _COMPONENTS:
        region = bv.get(component)
        void_masks = [hw_result.combined_void_mask] if component == "main_body" else []
        extra_solid = [solid_masks[component]] if component in solid_masks else None
        hard_solid, hard_air = PhiGrid.build_hard_masks(
            region, void_masks, _ATTACHMENT_FACES[component], solid_masks=extra_solid,
        )
        grid = np.zeros(region.shape, dtype=np.float32)
        phi = PhiGrid(component, region, grid, hard_solid, hard_air)
        phi.init(init_mode, seed=seed)
        phi.apply_hard_constraints()
        phi_grids[component] = phi

    return phi_grids, bv


def warm_start_phi_grids(
    prev_phi_grids: dict[str, PhiGrid],
    W_mm: float,
    x_front_mm: float,
    d_halo_mm: float,
    rule_envelope: Optional[RuleEnvelope] = None,
    wheel_x_half_width_mm: float = WHEEL_X_CLEARANCE_HALF_WIDTH_MM,
) -> tuple[dict[str, PhiGrid], BoundingVolumes]:
    """Warm-start onto a new (W, x_front, d_halo).

    Scoped-down version of true warm-starting: rebuilds fresh grids at the
    new geometry (via build_phi_grids_for_candidate) rather than remapping
    prev_phi_grids' actual field values onto the new bounding volumes. A
    real incremental remap (interpolating the previous phi field onto the
    new grid, only reinitialising cells outside the old bounding volume) is
    future work -- PhiGrid has no `.remap()` method yet, and building one
    correctly (resampling a signed-distance field across a resized,
    re-origined grid without corrupting the |grad phi|=1 property) is a
    separate, nontrivial task, not a small addition. Not silently pretending
    this already does true warm-starting is the point.

    Only the previous grids' shapes/components are used, to confirm this is
    being called with a matching PhiGrid set (fail loudly if not); the
    values in prev_phi_grids are otherwise unused by this scoped-down
    version.
    """
    missing = set(_COMPONENTS) - set(prev_phi_grids)
    if missing:
        raise ValueError(f"prev_phi_grids is missing components: {sorted(missing)}")
    return build_phi_grids_for_candidate(
        W_mm, x_front_mm, d_halo_mm,
        rule_envelope=rule_envelope,
        wheel_x_half_width_mm=wheel_x_half_width_mm,
    )
