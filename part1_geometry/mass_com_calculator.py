"""
mass_com_calculator.py --- Compute mass and COM from phi grids.

Solid cells (phi < 0) represent material. Cell volume = GRID_SPACING_M^3.
Mass = volume * density. COM = mass-weighted mean of solid cell centres.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from geometry_contract import GRID_SPACING_M, get_density
from phi_grid import PhiGrid


@dataclass(frozen=True)
class ComponentMassCOM:
    """Mass and COM for one component."""
    name: str
    mass_kg: float
    com_x_m: float
    com_y_m: float
    com_z_m: float


def compute_component_mass_com(
    phi: PhiGrid,
    density_kgm3: float,
) -> ComponentMassCOM:
    """
    Compute mass and COM from phi grid.

    Solid cells: phi.grid < 0.0
    Cell volume: GRID_SPACING_M^3

    volume = count(solid cells) * GRID_SPACING_M^3
    mass = volume * density_kgm3

    COM x = origin_x + mean(ix) * GRID_SPACING_M (for all solid cell indices ix)
    COM y = origin_y + mean(iy) * GRID_SPACING_M
    COM z = origin_z + mean(iz) * GRID_SPACING_M

    If no solid cells: return zero mass, COM at grid origin.
    After computing COM: call _assert_com_physical() to catch units bugs.
    """
    solid_mask = phi.grid < 0.0
    n_solid = int(solid_mask.sum())

    ox, oy, oz = phi.bv.origin_m
    dx = GRID_SPACING_M

    if n_solid == 0:
        # No solid cells --- return zero mass at origin
        return ComponentMassCOM(
            name=phi.component,
            mass_kg=0.0,
            com_x_m=ox,
            com_y_m=oy,
            com_z_m=oz,
        )

    cell_volume = dx ** 3
    volume = n_solid * cell_volume
    mass = volume * density_kgm3

    # Indices of solid cells
    ix, iy, iz = np.where(solid_mask)
    com_x = ox + float(np.mean(ix)) * dx
    com_y = oy + float(np.mean(iy)) * dx
    com_z = oz + float(np.mean(iz)) * dx

    return ComponentMassCOM(
        name=phi.component,
        mass_kg=mass,
        com_x_m=com_x,
        com_y_m=com_y,
        com_z_m=com_z,
    )


def compute_sidepod_pair_mass_com(right_phi: PhiGrid) -> ComponentMassCOM:
    """
    The right sidepod phi grid represents the right half only.
    The left sidepod is its y-mirror.
    Pair mass = 2 * right mass.
    Pair COM: x = right COM x, y = 0.0 (cancels by symmetry), z = right COM z.
    """
    right = compute_component_mass_com(right_phi, get_density("sidepod"))
    return ComponentMassCOM(
        name="sidepod",
        mass_kg=2.0 * right.mass_kg,
        com_x_m=right.com_x_m,
        com_y_m=0.0,
        com_z_m=right.com_z_m,
    )


def compute_all_machined_components(
    nose_phi: PhiGrid,
    sidepod_phi: PhiGrid,   # right half
    rearpod_phi: PhiGrid,
    body_phi: PhiGrid,
) -> list[ComponentMassCOM]:
    """
    Returns list of exactly 4 ComponentMassCOM objects:
    [nose, sidepod_pair, rearpod, main_body]
    """
    return [
        compute_component_mass_com(nose_phi, get_density("nose")),
        compute_sidepod_pair_mass_com(sidepod_phi),
        compute_component_mass_com(rearpod_phi, get_density("rearpod")),
        compute_component_mass_com(body_phi, get_density("main_body")),
    ]