"""
ballast.py -- legal ballast (T1.22, Appendix ix), shared by Stage 1 and Stage 2.

WHY THIS EXISTS
---------------
The pipeline had no ballast. When a car was under T3.6's 48.0 g, the only way
back was to GROW FOAM (0.163 g/cm3), so the mass barrier pushed the outer skin
outward and the optimiser treated ~200 cm3 of foam as ballast. That mass term
carried ~94 % of every shape update, which is why the aero gradient could not
steer.

The regulations allow ballast in one place: the capsule under the halo
aperture (12.7 x 20 x 6.35 mm, ~1.39 cm3). This module models it.

THE RULE
--------
Ballast fills whatever the car is short of the target, up to capacity:

    b = clamp(target - m_without_ballast, 0, capacity)

so the regimes seen by the SHAPE update are

    0 < b < capacity   body mass is free: the ballast absorbs any change, and
                       dT/d(body mass) = 0. The skin answers to drag alone.
    b = 0              the car is heavier than the target even with no ballast:
                       the physics gradient dT/dm applies (remove material).
    b = capacity       the container is full and the car is still light:
                       a barrier pushes material back (the old behaviour).

Measured on a real car (rnd/lean_body, 2026-09-25): with lead in the capsule the
same competition mass needs 45 % less foam; with tungsten alloy 75 % less.
Whether a smaller body is FASTER is an aerodynamic question -- CFD with wheels
showed a mass-only lean body is slower -- so this module deliberately removes
the mass push rather than replacing it with a shrink.
"""
from __future__ import annotations

import math

from geometry_contract import mm_to_m

# T3.6: competition mass floor, EXCLUDING the CO2 cartridge.
T36_FLOOR_KG: float = 0.048
# One scale digit (T2.8: 47.9 fails) plus machining scatter.
TARGET_MARGIN_KG: float = 0.0002
CARTRIDGE_KG: float = 0.023
# Ballast kept in hand for the build: printed parts and glue come out heavier
# than CAD, and a car with no ballast left is over 48.2 g and ~17 ms/g slower
# (race objective, 2026-09-26). The body keeps carving until at least this much
# ballast is needed; within it the ballast absorbs changes as before.
#
# 0.5 g from the part5 sweep (round 8, medium mesh, each value run twice and
# repeating to 0.2 %): reserve 0.5 g -> D20 0.3887 N, 3 g 0.3937, 5 g 0.400,
# 8 g 0.3995, 1.5 g 0.4046. The carve depth changes the body shape, and not
# monotonically, so the value is measured, not reasoned; it leaves ~1.1 g of
# ballast in hand.
BUILD_RESERVE_KG: float = 0.0005

# Appendix ix capsule. Duplicated from halo_pocket so this module has no grid
# dependency; test_ballast asserts they agree.
SLOT_WIDTH_MM: float = 12.7
SLOT_LENGTH_MM: float = 20.0
SLOT_DEPTH_MM: float = 6.35

DENSITY_KGM3: dict[str, float] = {
    "none": 0.0,
    "lead": 11340.0,
    "tungsten_alloy": 18000.0,      # typical 90-95 % W heavy alloy, 17-18.5 g/cm3
}
DEFAULT_MATERIAL: str = "lead"

# Barrier used only when the container is full and the car is still light.
_BARRIER_WEIGHT: float = 100.0


def capsule_volume_m3() -> float:
    r = SLOT_WIDTH_MM / 2.0
    straight = SLOT_LENGTH_MM - 2.0 * r
    mm3 = (straight * SLOT_WIDTH_MM + math.pi * r * r) * SLOT_DEPTH_MM
    return mm3 * 1e-9


def capacity_kg(material: str = DEFAULT_MATERIAL) -> float:
    if material not in DENSITY_KGM3:
        raise ValueError(f"unknown ballast material {material!r}; "
                         f"known: {sorted(DENSITY_KGM3)}")
    return DENSITY_KGM3[material] * capsule_volume_m3()


def target_competition_kg() -> float:
    return T36_FLOOR_KG + TARGET_MARGIN_KG


def ballast_kg(total_without_ballast_kg: float,
               material: str = DEFAULT_MATERIAL,
               includes_cartridge: bool = True) -> float:
    """Ballast needed to reach the target, clamped to [0, capacity]."""
    comp = total_without_ballast_kg - (CARTRIDGE_KG if includes_cartridge else 0.0)
    need = target_competition_kg() - comp
    return min(max(need, 0.0), capacity_kg(material))


def slot_centroid_m(ref_plane_A_m: float, d_halo_mm: float) -> tuple[float, float]:
    """(x, z) of the capsule centroid for a given halo placement."""
    import halo_pocket as hp
    box = hp.compute_halo_pocket_box_m(ref_plane_A_m, d_halo_mm)
    x = box["x_min_m"] + mm_to_m(hp.BALLAST_CENTRE_FROM_POCKET_FRONT_MM)
    z = box["z_min_m"] - mm_to_m(hp.BALLAST_DEPTH_MM) / 2.0
    return x, z


def add_to_state(state: dict, ref_plane_A_m: float, d_halo_mm: float,
                 material: str = DEFAULT_MATERIAL) -> dict:
    """Return a copy of a {total_mass_kg, com_x_m, com_z_m} state with ballast in.

    `state` is the car WITHOUT ballast, cartridge included. The returned dict
    carries `ballast_kg` and `mass_without_ballast_kg` so callers can report it.
    """
    b = ballast_kg(state["total_mass_kg"], material)
    out = dict(state)
    out["ballast_kg"] = b
    out["mass_without_ballast_kg"] = state["total_mass_kg"]
    if b <= 0.0:
        return out
    bx, bz = slot_centroid_m(ref_plane_A_m, d_halo_mm)
    M = state["total_mass_kg"] + b
    out["total_mass_kg"] = M
    out["com_x_m"] = (state["com_x_m"] * state["total_mass_kg"] + bx * b) / M
    out["com_z_m"] = (state["com_z_m"] * state["total_mass_kg"] + bz * b) / M
    return out


def shape_dT_dmass(dT_dmass_physics: float, total_without_ballast_kg: float,
                   material: str = DEFAULT_MATERIAL) -> float:
    """The dT/d(body mass) the SHAPE update should use, given the ballast.

    See the module docstring for the three regimes. `dT_dmass_physics` is the
    race objective's own derivative at the operating point.
    """
    comp = total_without_ballast_kg - CARTRIDGE_KG
    target = target_competition_kg()
    cap = capacity_kg(material)
    if comp >= target - BUILD_RESERVE_KG:   # heavy, or inside the build reserve
        return dT_dmass_physics
    if comp + cap >= target:                # ballast absorbs the difference
        return 0.0
    deficit = target - (comp + cap)         # container full, still light
    return -(2.0 * _BARRIER_WEIGHT * deficit / T36_FLOOR_KG ** 2)


def describe(total_without_ballast_kg: float, material: str = DEFAULT_MATERIAL) -> dict:
    b = ballast_kg(total_without_ballast_kg, material)
    comp = total_without_ballast_kg - CARTRIDGE_KG + b
    return {
        "material": material,
        "capacity_g": capacity_kg(material) * 1e3,
        "ballast_g": b * 1e3,
        "competition_mass_g": comp * 1e3,
        "legal_T36": comp >= T36_FLOOR_KG - 1e-9,
        "regime": ("full" if abs(b - capacity_kg(material)) < 1e-12
                   else "heavy" if b < BUILD_RESERVE_KG - 1e-12 else "absorbing"),
    }
