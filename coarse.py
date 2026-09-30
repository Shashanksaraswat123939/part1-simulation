"""
coarse.py -- run the geometry pipeline at another grid spacing.

geometry_contract.GRID_SPACING_MM is 0.3 mm, the production spacing; the
search and the tests run at 1-2 mm. Several modules copy the spacing at import
time, so changing it means patching every copy: use_spacing does that. Call it
BEFORE building any geometry -- grid shapes are computed from the spacing at
call time, so changing it afterwards desynchronises shapes from coordinates.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PART1 = Path(__file__).resolve().parent
if str(_PART1) not in sys.path:
    sys.path.insert(0, str(_PART1))
os.environ.setdefault("PART2_PATH", str(_PART1.parent / "part2-simulation"))

_SPACING_CONSUMERS = (
    "bounding_volumes",
    "fixed_hardware",
    "halo_pocket",
    "mass_com_calculator",
    "phi_grid",
    "phi_updater",
    "surface_extraction",
    "unified_phi",
    "virtual_cargo",
    "wheel_visibility_zones",
)


def use_spacing(spacing_mm: float) -> None:
    """Set GRID_SPACING_MM / GRID_SPACING_M in every module that holds a copy."""
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
