"""
coarse.py -- run the Part 1 geometry pipeline at a coarse grid spacing.

WHY THIS EXISTS
---------------
geometry_contract.GRID_SPACING_MM is 0.3 mm (Nyquist x10 on the 3.15 mm
minimum machining radius). That is the right number for a production run and
a catastrophic number for interactive experimentation:

    component    cells @ 0.3mm    cells @ 1.5mm
    main_body    22,470,000       ~186,000
    rearpod       5,300,000        ~44,000
    sidepod       1,240,000        ~11,000
    nose          1,550,000        ~13,000

Marching cubes on a 22M-cell field produces millions of triangles, and
trimesh's curvature estimation and ray-casting on that mesh allocate several
GB. Running all four components in one process is enough to OOM a 16 GB
machine.

This module lets you dial the spacing down for exploration, and refuses to
proceed if the resulting grids are still large enough to be dangerous.

HONEST CAVEAT
-------------
Coarse spacing changes what the gates mean. At 1.5 mm you cannot meaningfully
resolve the 3.15 mm minimum radius (2 cells) or the 2.0 mm nose wall
thickness (1.3 cells). Results at coarse spacing tell you whether the
PIPELINE IS WIRED CORRECTLY and roughly what shape comes out. They do NOT
tell you whether the geometry is legal. Re-run the winner at 0.3 mm, one
component at a time, to answer that.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the part1 modules importable when running from sandbox/.
_PART1 = Path(__file__).resolve().parent.parent
if str(_PART1) not in sys.path:
    sys.path.insert(0, str(_PART1))

# Part 2 lives in a sibling directory whose name uses a hyphen, but
# fixed_hardware.py's default guess uses an underscore (part2_simulation).
# Set PART2_PATH here so the sandbox works without the caller exporting it.
import os
_PART2 = _PART1.parent / "part2-simulation"
os.environ.setdefault("PART2_PATH", str(_PART2))

# Modules that did `from geometry_contract import GRID_SPACING_M` and so hold
# their own module-level copy of the value. Patching geometry_contract alone
# is not enough -- each of these bindings must be updated too.
_SPACING_CONSUMERS = (
    "bounding_volumes",
    "fixed_hardware",
    "halo_pocket",
    "mass_com_calculator",
    "phi_grid",
    "phi_grid_factory",
    "phi_updater",
    "surface_extraction",
    "virtual_cargo",
    "wheel_visibility_zones",
)

# Refuse to build grids totalling more than this many cells. 3M cells of
# float32 phi + several bool masks is a few hundred MB -- survivable. The
# 0.3mm default is ~30M cells, which is not.
MAX_TOTAL_CELLS = 3_000_000


def use_spacing(spacing_mm: float) -> None:
    """Patch GRID_SPACING_MM/GRID_SPACING_M across every module that uses it.

    Must be called BEFORE compute_bounding_volumes / build_phi_grids_for_candidate.
    Grid shapes are computed from the spacing at call time, so changing it
    after grids exist would silently desynchronise shapes from coordinates.
    """
    import importlib

    import geometry_contract

    geometry_contract.GRID_SPACING_MM = float(spacing_mm)
    geometry_contract.GRID_SPACING_M = float(spacing_mm) / 1000.0

    for name in _SPACING_CONSUMERS:
        module = importlib.import_module(name)
        if hasattr(module, "GRID_SPACING_M"):
            module.GRID_SPACING_M = geometry_contract.GRID_SPACING_M
        if hasattr(module, "GRID_SPACING_MM"):
            module.GRID_SPACING_MM = geometry_contract.GRID_SPACING_MM


def disable_virtual_cargo() -> None:
    """Remove the T4.2 virtual-cargo requirement entirely.

    Patches virtual_cargo so that placement never fails and the cargo solid
    mask is empty. This is the single cause of every geometry rejection in the
    outer search (the halo pocket splits the axle corridor so the 60mm wedge
    won't fit), so turning it off opens up the whole space.

    THE RESULTING GEOMETRY IS NOT COMPETITION-LEGAL. T4.2 mandates that solid
    region; a car built from these grids would fail scrutineering. This exists
    to let you see what the phi fields and the search do when they are not
    blocked, not to produce a design.
    """
    import numpy as np
    import virtual_cargo

    def _no_cargo_placement(x_front_mm, W_mm, ref_plane_A_m, d_halo_mm,
                            z_floor_m, z_margin_m=0.001) -> dict:
        return {"x_start_m": mm_to_m_local(x_front_mm), "z_base_m": z_floor_m,
                "collided_with_default": False, "disabled": True}

    def _empty_mask(origin_m, shape, x_start_m, z_base_m):
        return np.zeros(shape, dtype=bool)

    virtual_cargo.find_cargo_placement = _no_cargo_placement
    virtual_cargo.build_virtual_cargo_solid_mask = _empty_mask

    # _level2_evaluate imported these by name at module load, so its own
    # bindings need replacing too -- patching virtual_cargo alone is not enough.
    try:
        import bayesian_outer_search
        bayesian_outer_search.find_cargo_placement = _no_cargo_placement
        bayesian_outer_search.build_virtual_cargo_solid_mask = _empty_mask
    except ImportError:
        pass


def mm_to_m_local(mm: float) -> float:
    return mm / 1000.0


def relax_accessibility() -> list:
    """Stop the tool-accessibility gate from killing candidates.

    Returns a list that accumulates (component, area_mm2) for every component
    that WOULD have failed, so the penalty is reported rather than hidden.

    Rationale: 01_generative_geometry.md's Failure and Recovery table says a
    small accessibility failure should "smooth/fill region, retry" and a large
    one should "assign manufacturing penalty, continue". surface_extraction
    raises on both instead, and quality_gates' retry re-extracts from the same
    unmodified grid, so it can never clear. Recording the area and continuing
    is the spec's large-failure behaviour.
    """
    import surface_extraction

    recorded: list = []
    original = surface_extraction._check_accessibility

    def _record_and_pass(mesh, component, bv) -> float:
        area = original(mesh, component, bv)
        if area > 0:
            recorded.append((component, area * 1e6))
        return 0.0

    surface_extraction._check_accessibility = _record_and_pass
    return recorded


def check_size(bv, max_total_cells: int = MAX_TOTAL_CELLS) -> dict[str, int]:
    """Return {component: cell_count}; raise MemoryError if the total is unsafe.

    Call this after compute_bounding_volumes and before anything that runs
    marching cubes. Raising here costs nothing; discovering the problem inside
    trimesh costs you the machine.
    """
    counts = {}
    for name in ("nose", "sidepod", "rearpod", "main_body"):
        shape = bv.get(name).shape
        counts[name] = int(shape[0]) * int(shape[1]) * int(shape[2])

    total = sum(counts.values())
    if total > max_total_cells:
        detail = "\n".join(
            f"    {n:<10s} {c:>12,d} cells  {tuple(bv.get(n).shape)}"
            for n, c in counts.items()
        )
        raise MemoryError(
            f"Grids total {total:,d} cells, over the {max_total_cells:,d} limit.\n"
            f"{detail}\n"
            f"  Surface extraction on grids this size can allocate several GB\n"
            f"  per component. Use a coarser --spacing, or raise --max-cells\n"
            f"  deliberately if you know you have the RAM for it."
        )
    return counts
