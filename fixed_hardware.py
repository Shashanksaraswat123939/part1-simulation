from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import sys, os
from pathlib import Path

# Resolve Part 2 path: env var PART2_PATH overrides default (sibling directory).
# This avoids hardcoded absolute paths that break on every non-developer machine.
_part2_path = os.environ.get(
    "PART2_PATH",
    str(Path(__file__).resolve().parent.parent / "part2_simulation"),
)
if _part2_path not in sys.path:
    sys.path.insert(0, _part2_path)

from mass_com_ingest import FixedHardwareSpec   # Part 2 type
from geometry_contract import (
    CO2_MASS_KG, GRID_SPACING_M, WHEEL_CLEARANCE_M,
    HARDWARE_CLEARANCE_M, HALO_MIN_Z_M, mm_to_m,
    R_WHEEL_M, validate_W,
    COM_Z_LOWER_BOUND_M, COM_Z_UPPER_BOUND_M,
)

@dataclass(frozen=True)
class ForbiddenCylinder:
    """
    Wheel/axle exclusion zone. A cylinder aligned with the x-axis.

    The cylinder is infinite in y-z (circular cross-section in y-z plane)
    and has finite extent in x (from x_center - x_half_width to x_center + x_half_width).

    All values in metres.
    """
    x_center_m: float       # axle position in x (0.0 = front axle, W_m = rear axle)
    y_center_m: float       # always 0.0 --- axle is on centerline
    z_center_m: float       # axle height above track surface (z=0)
    radius_m:   float       # wheel radius + clearance = R_WHEEL_M + WHEEL_CLEARANCE_M
    x_half_width_m: float   # half-width of wheel+axle assembly in x direction

    @property
    def x_min_m(self) -> float:
        return self.x_center_m - self.x_half_width_m

    @property
    def x_max_m(self) -> float:
        return self.x_center_m + self.x_half_width_m

    def contains_point(self, x: float, y: float, z: float) -> bool:
        """Returns True if point (x,y,z) is inside this cylinder."""
        if not (self.x_min_m <= x <= self.x_max_m):
            return False
        r2 = (y - self.y_center_m)**2 + (z - self.z_center_m)**2
        return r2 <= self.radius_m ** 2


@dataclass
class HaloGeometry:
    """
    Physical halo hardware geometry.

    The halo is placed in the main body phi grid.
    Its void region forces phi > 0 so the optimizer cannot fill the halo mount.

    Coordinate system: all values in metres, car coordinate system.

    The halo sits between the canister pocket and the front axle in x.
    Its bottom must be at z >= HALO_MIN_Z_M (24 mm above track).
    Its front edge must be at x > 0.0 m (behind front axle).

    cross_section_yz_m: list of (y, z) polygon vertices defining the halo
        tube cross-section in the y-z plane. This is the shape that gets
        extruded along x from x_front to x_rear.
        ! UNRESOLVED U1: These vertices must be measured from the physical hardware.
    """
    x_front_m: float           # fore-most edge of halo void in x
    x_rear_m:  float           # aft-most edge of halo void in x
    # ! UNRESOLVED U1: cross-section shape not yet provided
    cross_section_yz_m: Optional[list[tuple[float, float]]] = None


@dataclass(frozen=True)
class FixedHardwareResult:
    """All outputs from fixed hardware placement."""
    # Bool void masks --- shape == main body grid shape (nx, ny, nz)
    # True = cell is forced phi > 0 (void --- hardware occupies this space)
    halo_void_mask:      np.ndarray
    canister_void_mask:  np.ndarray
    front_axle_void_mask: np.ndarray
    rear_axle_void_mask:  np.ndarray

    # Forbidden cylinders --- used by S2 to compute sidepod corridor
    front_cylinder: ForbiddenCylinder
    rear_cylinder:  ForbiddenCylinder

    # Combined void mask (union of all four) --- convenience, fed to PhiGrid
    combined_void_mask: np.ndarray

    # Part 2 interface
    fixed_hardware_spec: FixedHardwareSpec

def _validate_halo_position(
    halo: HaloGeometry,
    canister_x_m: float,
    W_m: float,
) -> None:
    """
    Enforce all three confirmed halo rules:

    Rule H1: Halo bottom must be at z >= 24 mm = 0.024 m above track.
             We check the bottom of the cross-section (min z vertex).
             ! UNRESOLVED U1: Until cross_section_yz_m is provided, we cannot
             check the exact bottom z. We can only check that x_front is valid.

    Rule H2: Halo front edge must be strictly aft of front axle.
             Front axle is at x = 0.0 m. "Aft" means larger x (front-to-rear convention).
             Therefore: halo.x_front_m > 0.0 m.

    Rule H3: Halo must sit between canister pocket and front axle in x.
             canister is forward (smaller x), halo is between canister and front axle.
             Therefore: canister_x_m < halo.x_front_m  AND  halo.x_front_m > 0.0 m
             (Rule H2 already enforces the second condition.)
             Also: halo must not extend past the rear axle: halo.x_rear_m < W_m
    """
    # Rule H2: halo front must be behind front axle (x > 0)
    if halo.x_front_m <= 0.0:
        raise ValueError(
            f"Halo front edge at x={halo.x_front_m:.4f} m must be strictly "
            f"aft of front axle (x=0.0 m). In front-to-rear convention, "
            f"'behind front axle' means x > 0. Got x={halo.x_front_m:.4f} m."
        )

    # Rule H3a: halo must be aft of canister pocket
    if halo.x_front_m <= canister_x_m:
        raise ValueError(
            f"Halo front edge (x={halo.x_front_m:.4f} m) must be aft of "
            f"canister pocket (x={canister_x_m:.4f} m). "
            f"'Between canister and front axle' means canister_x < halo_x_front."
        )

    # Rule H3b: halo must not extend past rear axle
    if halo.x_rear_m >= W_m:
        raise ValueError(
            f"Halo rear edge (x={halo.x_rear_m:.4f} m) extends past or to "
            f"rear axle (x={W_m:.4f} m). Halo must fit within the car body."
        )

    # Rule H1: check z bottom if cross-section is known
    if halo.cross_section_yz_m is not None:
        z_bottom = min(z for _, z in halo.cross_section_yz_m)
        if z_bottom < HALO_MIN_Z_M - 1e-6:
            raise ValueError(
                f"Halo cross-section bottom at z={z_bottom*1000:.2f} mm "
                f"is below minimum {HALO_MIN_Z_M*1000:.1f} mm above track. "
                f"Halo must clear the track by at least 24 mm."
            )
    # If cross_section_yz_m is None (! U1), we cannot check Rule H1 on z.
    # The NotImplementedError in place_halo_void() below handles this.

def _assert_com_in_range(
    label: str,
    com_m: tuple[float, float, float],
    W_m: float,
) -> None:
    """
    Validate that a COM coordinate is physically plausible.

    This catches mm-vs-m units bugs before they reach Part 2's polynomial,
    which would produce race time values of 10^15 seconds (Part 2 audit finding 3.1).

    Rules:
      x in [-0.01, W_m + 0.01]   within car length
      y in [-0.05, 0.05]         within car width
      z in [COM_Z_LOWER_BOUND_M, COM_Z_UPPER_BOUND_M]  physical COM height
    """
    x, y, z = com_m
    if not (-0.01 <= x <= W_m + 0.01):
        raise ValueError(f"{label}: x={x:.6f} outside [-0.01, {W_m + 0.01:.6f}]")
    if not (-0.05 <= y <= 0.05):
        raise ValueError(f"{label}: y={y:.6f} outside [-0.05, 0.05]")
    if not (COM_Z_LOWER_BOUND_M <= z <= COM_Z_UPPER_BOUND_M):
        raise ValueError(
            f"{label}: z={z:.6f} outside [{COM_Z_LOWER_BOUND_M}, {COM_Z_UPPER_BOUND_M}]"
        )
def _build_cylinder_void_mask(
    grid_shape: tuple[int, int, int],
    grid_origin_m: tuple[float, float, float],
    cylinder: ForbiddenCylinder,
) -> np.ndarray:
    """
    Returns bool array shape (nx, ny, nz).
    True where grid cell centre is inside the cylinder.

    The cylinder's circular cross-section is in the y-z plane.
    Its x extent is [cylinder.x_min_m, cylinder.x_max_m].
    """
    nx, ny, nz = grid_shape
    ox, oy, oz = grid_origin_m
    dx = GRID_SPACING_M

    xs = ox + np.arange(nx) * dx   # shape (nx,)
    ys = oy + np.arange(ny) * dx   # shape (ny,)
    zs = oz + np.arange(nz) * dx   # shape (nz,)

    X = xs[:, np.newaxis, np.newaxis]  # (nx, 1, 1)
    Y = ys[np.newaxis, :, np.newaxis]  # (1, ny, 1)
    Z = zs[np.newaxis, np.newaxis, :]  # (1, 1, nz)

    in_x = (X >= cylinder.x_min_m) & (X <= cylinder.x_max_m)
    r2 = (Y - cylinder.y_center_m)**2 + (Z - cylinder.z_center_m)**2
    in_r = r2 <= cylinder.radius_m**2

    return (in_x & in_r).astype(bool)


def _build_polygon_void_mask(
    grid_shape: tuple[int, int, int],
    grid_origin_m: tuple[float, float, float],
    x_min_m: float,
    x_max_m: float,
    polygon_yz_m: list[tuple[float, float]],
) -> np.ndarray:
    """
    Returns bool array shape (nx, ny, nz).
    True where grid cell centre is inside the polygon cross-section AND within [x_min, x_max].

    Used for halo void (polygon cross-section of the halo tube extruded in x).
    """
    from bounding_volumes import _point_in_polygon_vectorised

    nx, ny, nz = grid_shape
    ox, oy, oz = grid_origin_m
    dx = GRID_SPACING_M

    xs = ox + np.arange(nx) * dx
    ys = oy + np.arange(ny) * dx
    zs = oz + np.arange(nz) * dx

    # in_x: shape (nx,)
    in_x = (xs >= x_min_m) & (xs <= x_max_m)

    # in_yz: shape (ny, nz)
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    n_pts = ny * nz
    in_yz = _point_in_polygon_vectorised(
        Y.ravel(), Z.ravel(), polygon_yz_m
    ).reshape(ny, nz)

    # Broadcast: shape (nx, ny, nz)
    mask = in_x[:, np.newaxis, np.newaxis] & in_yz[np.newaxis, :, :]
    return mask.astype(bool)


def _build_box_void_mask(
    grid_shape: tuple[int, int, int],
    grid_origin_m: tuple[float, float, float],
    x_range_m: tuple[float, float],
    y_range_m: tuple[float, float],
    z_range_m: tuple[float, float],
) -> np.ndarray:
    """
    Returns bool array shape (nx, ny, nz).
    True where grid cell centre is inside the axis-aligned box.

    Used for canister void (simple box --- shape pending U2).
    """
    nx, ny, nz = grid_shape
    ox, oy, oz = grid_origin_m
    dx = GRID_SPACING_M

    xs = ox + np.arange(nx) * dx
    ys = oy + np.arange(ny) * dx
    zs = oz + np.arange(nz) * dx

    in_x = (xs >= x_range_m[0]) & (xs <= x_range_m[1])
    in_y = (ys >= y_range_m[0]) & (ys <= y_range_m[1])
    in_z = (zs >= z_range_m[0]) & (zs <= z_range_m[1])

    X = in_x[:, np.newaxis, np.newaxis]
    Y = in_y[np.newaxis, :, np.newaxis]
    Z = in_z[np.newaxis, np.newaxis, :]

    return (X & Y & Z).astype(bool)

def place_fixed_hardware(
    W_mm: float,
    halo_geometry: HaloGeometry,
    canister_com_mm: Optional[tuple[float, float, float]],   # ! U2: None until confirmed
    canister_box_half_size_mm: float,                         # half-size of canister void box
    wheel_axle_mass_kg: float,
    wheel_axle_com_mm: tuple[float, float, float],
    wheel_x_half_width_mm: float,                             # half-width of wheel assembly in x
    wheel_axle_z_mm: float,                                   # axle height above track in mm
    rear_wing_mass_kg: float,
    rear_wing_com_mm: Optional[tuple[float, float, float]],  # ! U5: None until confirmed
    body_grid_shape: tuple[int, int, int],
    body_grid_origin_m: tuple[float, float, float],
) -> FixedHardwareResult:
    """
    Place all fixed hardware. Validate positions. Build void masks. Construct FixedHardwareSpec.

    Args:
        W_mm: wheelbase in mm
        halo_geometry: halo dimensions (x_front_m, x_rear_m, cross_section_yz_m)
        canister_com_mm: (x, y, z) of CO2 canister centre in mm, or None (! U2)
        canister_box_half_size_mm: half-size of cubic void box around canister in mm
        wheel_axle_mass_kg: total mass of all 4 wheels + axles combined, in kg
        wheel_axle_com_mm: (x, y, z) of combined wheels+axles COM in mm
        wheel_x_half_width_mm: half-width of wheel+axle assembly in x (for forbidden zone)
        wheel_axle_z_mm: height of axle centreline above track in mm
        rear_wing_mass_kg: rear wing mass in kg
        rear_wing_com_mm: (x, y, z) of rear wing COM in mm, or None (! U5)
        body_grid_shape: (nx, ny, nz) of the main body phi grid
        body_grid_origin_m: (x0, y0, z0) of the main body grid in metres

    Returns:
        FixedHardwareResult with all void masks, forbidden cylinders, and FixedHardwareSpec
    """

    validate_W(W_mm)
    W_m = mm_to_m(W_mm)

    # ?? Front and rear forbidden cylinders ????????????????????????????????
    axle_z_m = mm_to_m(wheel_axle_z_mm)
    axle_x_half_m = mm_to_m(wheel_x_half_width_mm)
    cylinder_radius_m = R_WHEEL_M + WHEEL_CLEARANCE_M

    front_cylinder = ForbiddenCylinder(
        x_center_m     = 0.0,
        y_center_m     = 0.0,
        z_center_m     = axle_z_m,
        radius_m       = cylinder_radius_m,
        x_half_width_m = axle_x_half_m,
    )
    rear_cylinder = ForbiddenCylinder(
        x_center_m     = W_m,
        y_center_m     = 0.0,
        z_center_m     = axle_z_m,
        radius_m       = cylinder_radius_m,
        x_half_width_m = axle_x_half_m,
    )

    # ?? Canister position ?????????????????????????????????????????????????
    # ! UNRESOLVED U2: CO2 canister legal position not confirmed from competition rules.
    # Provide canister_com_mm=(x,y,z) from the official STEM Racing rule sheet.
    if canister_com_mm is None:
        raise NotImplementedError(
            "! UNRESOLVED U2: CO2 canister legal position not confirmed. "
            "Provide canister_com_mm=(x_mm, y_mm, z_mm) from competition rules. "
            "The canister is at the front of the car (small x value)."
        )
    canister_com_m = tuple(mm_to_m(v) for v in canister_com_mm)
    _assert_com_in_range("CO2 canister", canister_com_m, W_m)

    # ?? Halo position validation ??????????????????????????????????????????
    _validate_halo_position(halo_geometry, canister_com_m[0], W_m)

    # ?? Wheel+axle COM ????????????????????????????????????????????????????
    wheel_axle_com_m = tuple(mm_to_m(v) for v in wheel_axle_com_mm)
    _assert_com_in_range("Wheels+axles", wheel_axle_com_m, W_m)

    # ?? Rear wing COM ?????????????????????????????????????????????????????
    # ! UNRESOLVED U5: Rear wing fixed position coordinate not confirmed.
    if rear_wing_com_mm is None:
        raise NotImplementedError(
            "! UNRESOLVED U5: Rear wing fixed position not confirmed from competition rules. "
            "Provide rear_wing_com_mm=(x_mm, y_mm, z_mm)."
        )
    rear_wing_com_m = tuple(mm_to_m(v) for v in rear_wing_com_mm)
    _assert_com_in_range("Rear wing", rear_wing_com_m, W_m)

    # ?? Build void masks ??????????????????????????????????????????????????
    front_axle_mask = _build_cylinder_void_mask(
        body_grid_shape, body_grid_origin_m, front_cylinder
    )
    rear_axle_mask = _build_cylinder_void_mask(
        body_grid_shape, body_grid_origin_m, rear_cylinder
    )

    # Canister void: simple box around canister COM
    cs_half = mm_to_m(canister_box_half_size_mm)
    cx, cy, cz = canister_com_m
    canister_mask = _build_box_void_mask(
        body_grid_shape, body_grid_origin_m,
        x_range_m=(cx - cs_half, cx + cs_half),
        y_range_m=(cy - cs_half, cy + cs_half),
        z_range_m=(cz - cs_half, cz + cs_half),
    )

    # Halo void
    # ! UNRESOLVED U1: Halo cross-section shape (y-z polygon vertices) not provided.
    if halo_geometry.cross_section_yz_m is None:
        raise NotImplementedError(
            "! UNRESOLVED U1: Halo cross-section shape (y-z polygon vertices in mm) "
            "not provided. Measure physical halo hardware and supply "
            "HaloGeometry(cross_section_yz_m=[(y1,z1),(y2,z2),...]) in metres. "
            "The polygon defines the halo tube cross-section at each x slice."
        )
    halo_mask = _build_polygon_void_mask(
        body_grid_shape, body_grid_origin_m,
        x_min_m=halo_geometry.x_front_m,
        x_max_m=halo_geometry.x_rear_m,
        polygon_yz_m=halo_geometry.cross_section_yz_m,
    )

    # Combined mask: union of all four voids
    combined = front_axle_mask | rear_axle_mask | canister_mask | halo_mask

    # ?? Construct FixedHardwareSpec (Part 2 type) ?????????????????????????
    spec = FixedHardwareSpec(
        co2_cartridge_mass_kg = CO2_MASS_KG,          # exactly 0.023 --- Part 2 validates this
        co2_cartridge_com     = canister_com_m,        # (x, y, z) in metres
        rear_wing_mass_kg     = rear_wing_mass_kg,
        rear_wing_com         = rear_wing_com_m,
        wheels_axles_mass_kg  = wheel_axle_mass_kg,
        wheels_axles_com      = wheel_axle_com_m,
    )
    # Part 2's __post_init__ raises ValueError if co2_cartridge_mass_kg != 0.023.
    # If that raise fires, it means CO2_MASS_KG drifted from Part 2's constant --- fix S1.

    return FixedHardwareResult(
        halo_void_mask       = halo_mask,
        canister_void_mask   = canister_mask,
        front_axle_void_mask = front_axle_mask,
        rear_axle_void_mask  = rear_axle_mask,
        front_cylinder       = front_cylinder,
        rear_cylinder        = rear_cylinder,
        combined_void_mask   = combined,
        fixed_hardware_spec  = spec,
    )