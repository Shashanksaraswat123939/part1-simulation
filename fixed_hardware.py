from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
import sys, os
from pathlib import Path

# Resolve Part 2 path: env var PART2_PATH overrides default (sibling directory).
# This avoids hardcoded absolute paths that break on every non-developer machine.
# The repo folder is "part2-simulation" (hyphen); "part2_simulation" (underscore)
# is the spec/package name. Try both — guessing only the underscore form was the
# single root cause of 5 failing Part 1 test files (ModuleNotFoundError:
# mass_com_ingest), which sandbox/coarse.py had been working around by setting
# PART2_PATH itself.
_part2_env = os.environ.get("PART2_PATH")
_part2_candidates = (
    [_part2_env] if _part2_env else
    [str(Path(__file__).resolve().parent.parent / name)
     for name in ("part2-simulation", "part2_simulation")]
)
for _part2_path in _part2_candidates:
    if _part2_path and _part2_path not in sys.path:
        sys.path.insert(0, _part2_path)

from mass_com_ingest import FixedHardwareSpec   # Part 2 type
from geometry_contract import (
    CO2_MASS_KG, GRID_SPACING_M, WHEEL_CLEARANCE_M,
    HARDWARE_CLEARANCE_M, HALO_MIN_Z_M, mm_to_m,
    R_WHEEL_M, validate_W, validate_x_front,
    COM_Z_LOWER_BOUND_M, COM_Z_UPPER_BOUND_M,
    WHEEL_X_CLEARANCE_HALF_WIDTH_MM,
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


@dataclass(frozen=True)
class WheelDiscZone:
    """
    Physical clearance volume for one wheel disc.

    A wheel spins about the LATERAL (y) axis: its circular face lies in the
    x-z plane, offset out to the wheel's real lateral position -- not
    centred on the car's centreline. This is what ForbiddenCylinder above
    got wrong when (mis)used for wheels: that shape is a rod aligned along
    x, circular in y-z, centred at y=0 -- a different axis and a different
    location from any real wheel. See sandbox/README.md finding #12.

    All values in metres.
    """
    x_center_m: float   # axle position in x
    y_min_m:    float   # inner (track-contact) face
    y_max_m:    float   # outer face (y_min_m + wheel width)
    z_center_m: float   # axle height above track
    radius_m:   float   # wheel radius + clearance, in the x-z plane

    def contains_point(self, x: float, y: float, z: float) -> bool:
        if not (self.y_min_m <= y <= self.y_max_m):
            return False
        r2 = (x - self.x_center_m) ** 2 + (z - self.z_center_m) ** 2
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

    # The cartridge chamber bore, as geometry rather than a rasterised mask.
    # The bore spans the main_body/rearpod boundary, so every component grid it
    # touches has to carve it -- masks above are main-body-shaped and cannot do
    # that. Callers rasterise this onto each component grid via
    # _build_cylinder_void_mask (see phi_grid_factory).
    canister_cylinder: ForbiddenCylinder

    # Combined void mask (union of all four) --- convenience, fed to PhiGrid
    combined_void_mask: np.ndarray

    # Part 2 interface
    fixed_hardware_spec: FixedHardwareSpec


def _validate_halo_position(
    halo: HaloGeometry,
    canister_x_m: float,
    front_axle_m: float,
    rear_axle_m: float,
) -> None:
    """
    Enforce halo position rules.

    Coordinate system: x=0 at nose tip. front_axle_m = x_front_m, rear_axle_m = x_front_m + W_m.

    Rule H1 (T4.4.4, real regulation): halo bottom must be at z >= 24mm = 0.024m
             above track. Checked against the bottom of the cross-section (min z vertex).

    Sanity bound (NOT a numbered regulation -- the actual regs text has no rule
             constraining the halo's x-position relative to the front axle or
             canister; an earlier assumption to that effect was removed here
             since it wrongly rejected legal d_halo values below 16mm). We keep
             only a basic containment check: the halo must not extend past the
             rear axle, since it is a main_body-mounted part and going past the
             rear axle would place it in rearpod territory.

    canister_x_m is accepted for signature stability but no longer checked
    against halo position (see above).
    """
    del canister_x_m   # no longer checked -- see docstring

    # Sanity bound: halo must not extend past rear axle (main_body containment,
    # not a specific numbered regulation)
    if halo.x_rear_m >= rear_axle_m:
        raise ValueError(
            f"Halo rear edge (x={halo.x_rear_m:.4f} m) extends past or to "
            f"rear axle (x={rear_axle_m:.4f} m). Halo must fit within the car body."
        )

    # Rule H1 (T4.4.4): check z bottom if cross-section is known
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
    rear_axle_m: float,
) -> None:
    """
    Validate that a COM coordinate is physically plausible.

    This catches mm-vs-m units bugs before they reach Part 2's polynomial,
    which would produce race time values of 10^15 seconds (Part 2 audit finding 3.1).

    Coordinate system: x=0 at nose tip. rear_axle_m = x_front_m + W_m.
    Bound includes a 50mm margin aft of the rear axle for rearpod overhang (T9.4.2 max 40mm).

    Rules:
      x in [-0.01, rear_axle_m + 0.05]   within car length
      y in [-0.05, 0.05]                 within car width
      z in [COM_Z_LOWER_BOUND_M, COM_Z_UPPER_BOUND_M]  physical COM height
    """
    x, y, z = com_m
    x_max = rear_axle_m + 0.05
    if not (-0.01 <= x <= x_max):
        raise ValueError(f"{label}: x={x:.6f} outside [-0.01, {x_max:.6f}]")
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


def _build_wheel_disc_void_mask(
    grid_shape: tuple[int, int, int],
    grid_origin_m: tuple[float, float, float],
    zone: WheelDiscZone,
) -> np.ndarray:
    """
    Returns bool array shape (nx, ny, nz).
    True where grid cell centre is inside the wheel disc's clearance volume:
    circular in the x-z plane (centred on the axle), banded in y (the
    wheel's width, at its real lateral offset).
    """
    nx, ny, nz = grid_shape
    ox, oy, oz = grid_origin_m
    dx = GRID_SPACING_M

    xs = ox + np.arange(nx) * dx
    ys = oy + np.arange(ny) * dx
    zs = oz + np.arange(nz) * dx

    X = xs[:, np.newaxis, np.newaxis]
    Y = ys[np.newaxis, :, np.newaxis]
    Z = zs[np.newaxis, np.newaxis, :]

    in_y = (Y >= zone.y_min_m) & (Y <= zone.y_max_m)
    r2 = (X - zone.x_center_m) ** 2 + (Z - zone.z_center_m) ** 2
    in_r = r2 <= zone.radius_m ** 2

    return (in_y & in_r).astype(bool)


def _build_four_wheel_zones(
    x_front_m: float,
    rear_axle_m: float,
    axle_z_m: float,
    radius_m: float,
) -> list[WheelDiscZone]:
    """
    Build the four WheelDiscZones (front-left, front-right, rear-left,
    rear-right) at their real measured lateral offsets (geometry_contract's
    FRONT/REAR_WHEEL_INNER_Y_M + WHEEL_WIDTH_M), not the centreline.

    The y-band is widened inward by WHEEL_CLEARANCE_M (the same clearance
    already applied to the radial x-z direction via `radius_m`) so the
    wheel's own inner-face boundary isn't a bare, zero-margin edge -- caught
    live: at coarse grid spacing, real wheel-disc vertices sitting exactly
    on that boundary rounded onto the solid side of the nearest cell.
    """
    from geometry_contract import (
        FRONT_WHEEL_WIDTH_M, REAR_WHEEL_WIDTH_M,
        FRONT_WHEEL_INNER_Y_M, REAR_WHEEL_INNER_Y_M, WHEEL_CLEARANCE_M,
    )

    zones = []
    for x_center_m, inner_y_m, width_m in (
        (x_front_m, FRONT_WHEEL_INNER_Y_M, FRONT_WHEEL_WIDTH_M),
        (rear_axle_m, REAR_WHEEL_INNER_Y_M, REAR_WHEEL_WIDTH_M),
    ):
        band_min_m = inner_y_m - WHEEL_CLEARANCE_M
        band_max_m = inner_y_m + width_m + WHEEL_CLEARANCE_M
        for sign in (+1.0, -1.0):
            if sign > 0:
                y_min_m, y_max_m = band_min_m, band_max_m
            else:
                y_min_m, y_max_m = -band_max_m, -band_min_m
            zones.append(WheelDiscZone(x_center_m, y_min_m, y_max_m, axle_z_m, radius_m))
    return zones


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
    x_front_mm: float,
    halo_geometry: HaloGeometry,
    canister_com_mm: Optional[tuple[float, float, float]],   # ! U2: None until confirmed
    canister_radius_mm: float,                                # chamber bore radius (T5.1)
    canister_depth_mm: float,                                 # chamber bore depth along x (T5.3)
    canister_rear_face_x_mm: float,                           # rearmost machined face (T5.6)
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
        x_front_mm: front axle position from nose tip in mm (x=0 = nose tip)
        halo_geometry: halo dimensions (x_front_m, x_rear_m, cross_section_yz_m)
        canister_com_mm: (x, y, z) of CO2 canister centre in mm, or None (! U2)
        canister_radius_mm: bore radius of the cartridge chamber (T5.1: diameter
            18.0-18.5 mm, so 9.0-9.25 mm). This is the chamber ITSELF, not an
            inflated keep-clear box -- T5.5's 3.0 mm safety zone is a requirement
            that 3 mm of Model Block material SURROUND the bore, which is a check
            on the finished surface, not something you get by enlarging the void.
        canister_depth_mm: bore depth along x (T5.3: 45.0-58.0 mm)
        canister_rear_face_x_mm: x of the rearmost machined face of the car. The
            bore runs forward from here so the cartridge can be inserted and
            protrude >= 5 mm out the back (T5.6).
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
    validate_x_front(x_front_mm, W_mm)
    W_m = mm_to_m(W_mm)
    x_front_m = mm_to_m(x_front_mm)
    rear_axle_m = x_front_m + W_m

    # ?? Front and rear forbidden cylinders ????????????????????????????????
    # Coordinate system: x=0 at nose tip. Front axle at x_front_m, rear at x_front_m+W_m.
    axle_z_m = mm_to_m(wheel_axle_z_mm)
    axle_x_half_m = mm_to_m(wheel_x_half_width_mm)
    cylinder_radius_m = R_WHEEL_M + WHEEL_CLEARANCE_M

    front_cylinder = ForbiddenCylinder(
        x_center_m     = x_front_m,
        y_center_m     = 0.0,
        z_center_m     = axle_z_m,
        radius_m       = cylinder_radius_m,
        x_half_width_m = axle_x_half_m,
    )
    rear_cylinder = ForbiddenCylinder(
        x_center_m     = rear_axle_m,
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
    _assert_com_in_range("CO2 canister", canister_com_m, rear_axle_m)

    # ?? Halo position validation ??????????????????????????????????????????
    _validate_halo_position(halo_geometry, canister_com_m[0], x_front_m, rear_axle_m)

    # ?? Wheel+axle COM ????????????????????????????????????????????????????
    wheel_axle_com_m = tuple(mm_to_m(v) for v in wheel_axle_com_mm)
    _assert_com_in_range("Wheels+axles", wheel_axle_com_m, rear_axle_m)

    # ?? Rear wing COM ?????????????????????????????????????????????????????
    # ! UNRESOLVED U5: Rear wing fixed position coordinate not confirmed.
    if rear_wing_com_mm is None:
        raise NotImplementedError(
            "! UNRESOLVED U5: Rear wing fixed position not confirmed from competition rules. "
            "Provide rear_wing_com_mm=(x_mm, y_mm, z_mm)."
        )
    rear_wing_com_m = tuple(mm_to_m(v) for v in rear_wing_com_mm)
    _assert_com_in_range("Rear wing", rear_wing_com_m, rear_axle_m)

    # ?? Build void masks ??????????????????????????????????????????????????
    # Wheel discs: circular in x-z, banded in y at their REAL lateral offset
    # (see WheelDiscZone docstring / geometry_contract's measured constants)
    # -- not the old centreline-centred cylinder that missed the wheels
    # entirely. left | right unioned per axle so front_axle_mask/
    # rear_axle_mask keep their existing shapes/names for callers below.
    front_left, front_right, rear_left, rear_right = _build_four_wheel_zones(
        x_front_m, rear_axle_m, axle_z_m, cylinder_radius_m,
    )
    front_axle_mask = (
        _build_wheel_disc_void_mask(body_grid_shape, body_grid_origin_m, front_left)
        | _build_wheel_disc_void_mask(body_grid_shape, body_grid_origin_m, front_right)
    )
    rear_axle_mask = (
        _build_wheel_disc_void_mask(body_grid_shape, body_grid_origin_m, rear_left)
        | _build_wheel_disc_void_mask(body_grid_shape, body_grid_origin_m, rear_right)
    )

    # Canister void: the cartridge chamber is a CYLINDRICAL BORE along x that
    # opens at the rear face of the car (T5.1 diameter / T5.3 depth / T5.6
    # protrusion). It is emphatically not a cube.
    #
    # HISTORY (fixed 2026-07-20): this was a cube of half-size
    # (diameter/2 + safety_zone) = 12.125 mm centred on the canister COM,
    # which was wrong three ways at once:
    #   - depth 24.25 mm against T5.3's 45 mm minimum (barely half)
    #   - bore 24.25 mm across against T5.1's 18.0-18.5 mm
    #   - sealed: it ended ~54 mm short of the rear face, leaving solid Model
    #     Block behind it, so the cartridge could not be inserted at all and
    #     T5.6 was unsatisfiable.
    # The safety zone is NOT added to the void -- see canister_radius_mm's
    # docstring above.
    #
    # The bore is deliberately extended past the rear face by one grid cell so
    # that it always cuts cleanly through the rearmost wall rather than
    # stopping a cell short of it. Cells beyond a grid's extent simply clip.
    _, cy, cz = canister_com_m
    canister_cylinder = ForbiddenCylinder(
        x_center_m=(
            mm_to_m(canister_rear_face_x_mm) + GRID_SPACING_M
            - mm_to_m(canister_depth_mm) / 2.0
        ),
        y_center_m=cy,
        z_center_m=cz,
        radius_m=mm_to_m(canister_radius_mm),
        x_half_width_m=mm_to_m(canister_depth_mm) / 2.0,
    )
    canister_mask = _build_cylinder_void_mask(
        body_grid_shape, body_grid_origin_m, canister_cylinder
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
        canister_cylinder    = canister_cylinder,
        combined_void_mask   = combined,
        fixed_hardware_spec  = spec,
    )


# ============================================================================
# Design defaults for U1 (halo cross-section), U2 (canister position), and
# U5 (rear wing COM) -- see PLACEHOLDERS.md item 16 for full rationale.
#
# None of these are "resolved" in the sense of a confirmed measured value.
# The actual regs constrain each of these to a legal RANGE, not an exact
# number (T5.1/T5.2/T5.3 for the canister, T9.4/T9.5 for the rear wing).
# Where no exact value is given, these pick a reasonable point within the
# legal range so the pipeline can run end-to-end. Override with measured/
# confirmed values once available -- these are starting points, not final
# design decisions.
# ============================================================================

# Cartridge chamber / CO2 canister (T5.1-T5.6)
# T5.1 allows 18.0-18.5 mm. Take 18.0, the MINIMUM, because that is what the
# supplied hardware_cad/canister_safety_zone.stl is drawn for: inner r 9.000,
# outer r 12.000. At the old 18.25 midpoint the wall came out 12.000 - 9.125 =
# 2.875 mm, under T5.5's 3.0 mm minimum -- the chamber and its safety zone have
# to be sized together, and the part is the authority on both.
CANISTER_DIAMETER_MM: float = 18.0
CANISTER_DEPTH_MM: float = 50.0        # T5.3: 45.0-58.0mm
CANISTER_Z_MM: float = 35.0            # T5.2: 30.0-40.0mm, midpoint (rear-centre height)
# T5.5: min 3.0mm wall of Model Block material around the chamber.
# NOT currently consumed by anything. It used to be added to the canister void's
# half-size, which was backwards -- enlarging the hole removes more material and
# cannot guarantee a wall thickness. T5.5 is a check on the FINISHED surface:
# "is there >= 3 mm of solid everywhere around the bore?" That check does not
# exist yet and belongs in surface_extraction's rule stage, next to the other
# T5 checks. Left here as the constant that check should read.
CANISTER_SAFETY_ZONE_MM: float = 3.0   # ! UNCHECKED -- see note above

# Outer radius of the cartridge ASSEMBLY: bore + T5.5's 3 mm wall = 12.125 mm.
# hardware_cad/co2_canister.stl "already includes the minimum SAFETY ZONE around
# the cartridge" (hardware_geometry.canister_front_x_mm) and measures 12.00 mm
# about the bore axis, which is this figure to within the CAD's tessellation.
#
# NOT the void radius. The void is the BORE, because T5.1 regulates the chamber
# at 18.0-18.5 mm and carving 24.25 mm makes the chamber itself illegal -- a
# test asserts exactly that. The annulus between the bore and this radius is
# supposed to be SOLID: it is the T5.5 wall, so body material there is correct
# and is not an intersection with the cartridge. (Measured while chasing what
# looked like one: the offending vertices sat at radius 9.05-11.91 mm, i.e.
# almost entirely outside the 9.125 mm bore. They were the wall.)
#
# What it IS for: the surface the halo lofts to. Anchoring the loft's rear end
# here rather than on the bore puts the deck at z = 47.1 mm instead of 44.1 mm,
# which is the cartridge's flat outer face -- and leaves exactly the 3 mm of
# material above the bore that T5.5 asks for.
# Measured on hardware_cad/canister_safety_zone.stl: a tube, inner r 9.000,
# outer r 12.000, 48 mm long. Wall exactly 3.000 mm (T5.5) and bore exactly
# 18.0 mm (T5.1 minimum). Was derived as bore/2 + safety zone = 12.125; the
# supplied part says 12.000, so take the part.
CANISTER_CLEARANCE_RADIUS_MM: float = 12.0
CANISTER_SAFETY_ZONE_INNER_R_MM: float = 9.0
CANISTER_SAFETY_ZONE_LENGTH_MM: float = 48.0
# Wall/floor thickness of the zone, T5.5's minimum and what the supplied
# part is drawn to (12.000 - 9.000).
CANISTER_SAFETY_ZONE_WALL_MM: float = 3.0

# Rear wing (T9.4, T9.5) -- mass and COM height are not given by the regs at all;
# these are placeholders pending a real measured rear wing.
REAR_WING_MASS_KG: float = 0.005       # design placeholder, ~5g
REAR_WING_OVERHANG_MM: float = 20.0    # T9.4.2: 0-40mm aft of Ref Plane B, midpoint
REAR_WING_HEIGHT_MM: float = 50.0      # within T9.4.3 max 65mm

# Wheels + axles. MEASURED by the user 2026-08-03, both sides combined:
#   front wheels + support systems   5 g
#   rear  wheels + support systems   6 g
# Split front/rear rather than lumped, because they differ AND they sit a whole
# wheelbase apart: a single 11 g mass at the axle midpoint puts the wheel COM
# (6/11 - 1/2)*W = 5.5 mm too far forward at W=120. That does not change the
# optimisation (the COM terms are ~0.2% of the shape velocity, measured), but
# check_stability ranks on com_x and the deliverable reports it.
# ── hardware masses, from CAD volume x density x infill ─────────────────────
# The v2 parts settled the 4x conflict that the v1 STLs created. v1 were SOLID
# envelopes -- 38% of their own bounding box, giving 9.1 g per support and
# 47.70 g of hardware against a 48 g floor, which left 0.30 g for the body. v2
# are the real printed geometry, 6-7% of their bounding box and thin-walled.
#
# These are CAD ONLY. Wheels and both supports print at 100% infill, so mass is
# simply volume x density with nothing to calibrate; the halo is 20%. The bench
# figures that used to live here (5 g front pair, 6 g rear, 3 g halo) are NOT
# used -- the project owner's instruction, 2026-08-10, is to take the CAD. They
# are noted only because they agree to ~12%, which is the independent check that
# the v1 numbers never had.
#
#   part                        cm3     x rho   x infill  =  g each   n
#   front wheel v2            0.775   1.04 ABS     100%      0.806    2
#   Front Wheel Support v2    1.343   1.04 ABS     100%      1.396    2
#   rear wheel v2             0.914   1.04 ABS     100%      0.951    2
#   Rear Wheel Support v2     1.485   1.04 ABS     100%      1.545    2
#   halo_helmet               6.705   0.80 LWPLA    20%      1.073    1
#
#     front pair 4.41 g   rear pair 4.99 g   halo 1.07 g
#     hardware total 15.47 g incl. the 5 g rear-wing placeholder
#     (bench, unused: 5 / 6 / 3 g -- agrees to ~12%)
#
# Volumes taken after repair -- fix_winding + fix_normals then |volume| per
# closed body. Raw trimesh volume on these files is garbage (7.75e11 cm3 on a
# 13x28x28 mm wheel) because the winding is inconsistent as exported, and the
# rear support additionally carries a 5-face degenerate speck of zero volume
# that makes it read as two bodies.
WHEEL_AXLE_FRONT_MASS_KG: float = 0.00441   # 2 wheels + 2 supports, v2 CAD
WHEEL_AXLE_REAR_MASS_KG:  float = 0.00499   # 2 wheels + 2 supports, v2 CAD
WHEEL_AXLE_MASS_KG: float = (
    WHEEL_AXLE_FRONT_MASS_KG + WHEEL_AXLE_REAR_MASS_KG)   # 9.4 g

# Halo. MEASURED 2026-08-03 at 3 g.
#
# It had NO MASS AT ALL in the production rollup. ingest_mass_com's fixed
# components are cartridge, rear wing and wheels/axles -- the halo is modelled
# only as a void that forces phi > 0, so its geometry was respected and its
# weight was not. Stage 1's proxy path does carry one
# (bayesian_outer_search.STUB_HALO_MASS_KG = 8 g), so the two stages disagreed:
# Stage 1 ranked (W, x_front) with an 8 g halo and Stage 2 computed race times
# with none.
# ── material densities (project owner, 2026-08-07) ───────────────────────────
# Recorded because COM wants a density per component, not just a lump mass.
#
# !! THESE DISAGREE WITH THE MEASURED MASSES BY ~4x AND ARE NOT YET USED FOR
# !! MASS. Volume x density on the supplied CAD gives:
#       front pair (2 wheels + 2 supports)  20.85 g  vs WHEEL_AXLE_FRONT 5 g
#       rear  pair                          20.78 g  vs WHEEL_AXLE_REAR  6 g
#       halo                                 5.36 g  vs HALO_MASS        3 g
# The supports dominate: 8.76 cm3 each, 38% of their own bounding box, i.e. the
# STL is a solid envelope rather than a printed strut with infill. Switching the
# mass model to density x solid volume would add ~31 g to a car sitting exactly
# on the 48 g T3.6 floor, on an assumption nobody has checked -- so the measured
# masses still set the magnitudes and these set nothing yet. Resolve by either
# measuring the printed parts' real infill or exporting the true printed solid.
WHEEL_SUPPORT_DENSITY_G_CM3: float = 1.04     # ABS
HALO_DENSITY_G_CM3:          float = 0.80     # LW-PLA, foamed; varies with print
CANISTER_STEEL_DENSITY_G_CM3: float = 7.85    # steel
CANISTER_CO2_DENSITY_G_CM3:   float = 0.70    # charged CO2
# Cartridge, from hardware_cad/: canister_steel.stl is a 1.943 cm3 shell and
# canister_co2_charge.stl an 11.380 cm3 charge, giving 15.25 g + 7.97 g =
# 23.22 g. That lands the CO2 on its nominal 8 g and the total within 1% of
# geometry_contract.CO2_MASS_KG (23 g), which is three independent checks that
# the two files are the right way round -- they arrived swapped.
CANISTER_STEEL_MASS_KG: float = 0.01525
CANISTER_CO2_MASS_KG:   float = 0.00797

HALO_MASS_KG: float = 0.00107

# T3.6's competition minimum is 48 g EXCLUDING the cartridge, and it is met by
# machined body + fixed hardware. If the hardware alone approaches it there is
# nothing left for the body to be, and the optimiser will carve to nothing while
# every mass check still passes -- a legal car made of almost no car. Warn once
# at import rather than let that happen quietly.
T36_COMPETITION_FLOOR_KG: float = 0.048
_MIN_BODY_HEADROOM_KG: float = 0.005      # 5 g: below this the body is a shell


def fixed_hardware_total_kg() -> float:
    """Every non-cartridge fixed part that counts toward T3.6."""
    return (WHEEL_AXLE_FRONT_MASS_KG + WHEEL_AXLE_REAR_MASS_KG
            + HALO_MASS_KG + REAR_WING_MASS_KG)


def _warn_if_hardware_eats_the_floor() -> None:
    import warnings as _w
    hw = fixed_hardware_total_kg()
    headroom = T36_COMPETITION_FLOOR_KG - hw
    if headroom < _MIN_BODY_HEADROOM_KG:
        _w.warn(
            f"fixed hardware alone is {hw*1000:.2f} g against T3.6's "
            f"{T36_COMPETITION_FLOOR_KG*1000:.0f} g competition floor, leaving "
            f"only {headroom*1000:.2f} g for the machined body. The optimiser "
            f"will carve the body away and still pass every mass check.",
            RuntimeWarning, stacklevel=2)


_warn_if_hardware_eats_the_floor()



# Halo cross-section (U1): the real halo is a downloadable fixed CAD part
# (T4.4.1) with a curved bar profile, not a constant extruded cross-section.
# Modelling it as ONE extruded polygon (matching the existing HaloGeometry
# design) is already a simplification. This uses a conservative rectangular
# bounding profile from the visible Appendix ix dimensions (25mm pocket
# width, floor at 24mm) rather than guessing the exact arch shape. Excluding
# more than the true arch needs is safe here (same principle as the T7.9
# zones): it only costs the optimizer a little shape freedom near the halo,
# it cannot produce an illegal design.
HALO_CROSS_SECTION_HALF_WIDTH_MM: float = 12.5   # matches halo_pocket.py's 25mm width
HALO_CROSS_SECTION_TOP_MM: float = 45.0          # conservative; real arch height TBD



# T4.4.2 allows the bottom of the halo to be obstructed in the front and side
# views -- the rule diagram boxes the halo ABOVE its base fillet and dimensions
# that fillet at 4.0 mm. Only material blocking the halo above this height
# violates the front/side rule. T4.4.3 (top view) has no such relief: the whole
# plan outline must be visible, obstructed only by the helmet, which this model
# does not carry.
HALO_VISIBILITY_FILLET_MM: float = 4.0

# Top of the REAL halo at its rear face, measured on hardware_cad/halo_helmet.stl
# after placement: flat at 34.0 mm across the full +-12.5 mm width.
#
# This is NOT HALO_CROSS_SECTION_TOP_MM. That constant is the conservative
# rectangular ENVELOPE the optimiser may not fill (45 mm nominal, 43 mm once
# rasterised), deliberately taller than the part so bodywork can never intrude
# on the mount. The real halo is an arch that tapers 39.0 mm at x=45 down to
# 34.0 mm at its rear face.
#
# The loft has to use the PART, not the envelope. Anchoring it on the envelope
# started the deck at 43.0 mm -- 9 mm above the halo it is supposed to leave
# from, floating over it rather than touching it. The envelope stays where it
# is; only the loft's front anchor moves.
HALO_REAR_FACE_TOP_MM: float = 34.0



# T7.9 is implemented in wheel_visibility_zones.build_t79_forbidden_mask and has
# been all along -- applied as hard air per component from unified_phi, with the
# diagram's chamfers as proper right triangles. A second, cruder copy briefly
# lived here (2026-08-07): un-chamfered full prisms, added after reading T7.9 out
# of the PDF without checking whether the rule was already modelled. It was, and
# the existing one is more faithful.
#
# Measured on the carved car with the duplicate removed: the REAL zones contain
# 0 body cells, while the prism version flagged 1,786 -- every one of them at
# x 60-66, |y| 20-31 mm, i.e. inside the chamfer T7.9.2 explicitly releases. The
# duplicate was not stricter in a useful way, it was wrong about the rule.
#
# The commit that added it (4bf3071) also claimed WheelDiscZone was the only
# wheel model and was "short of the rule in all three axes". WheelDiscZone is the
# spinning wheel's physical clearance volume and was never meant to be T7.9.


def canister_safety_zone_solid_mask(canister_cylinder,
                                    region_origin_m, shape,
                                    d_m: float) -> "np.ndarray":
    """The T5.5 safety zone: model block that MUST remain, as a solid mask.

    T5.5: "A safety zone of STEM Racing Model Block material with a minimum
    thickness of 3.0mm must be maintained around the minimum chamber depth."
    Material, not void -- and nothing required it, so the optimiser carved it
    away. Measured on the 2026-08-06 car the bodywork's top on the centreline
    sat at 22.5 mm behind the bore while the pocket's outer surface is at
    47.0 mm: the cartridge floating in a 24.5 mm gap, which is neither
    machinable nor able to hold a cartridge.

    Geometry is the supplied hardware_cad/canister_safety_zone.stl: a tube,
    inner r 9.000, outer r 12.000, 48 mm long, coaxial with the bore. Sizing the
    chamber to T5.1's 18.0 mm minimum rather than the 18.25 midpoint is what
    makes those two agree -- at 18.25 the wall is 2.875 mm, under T5.5.

    The zone is model block, the same foam the body is milled from, so it
    carries NO separate mass: it is already inside the body's own volume and
    density (project owner, 2026-08-07). This mask only says it must be there.
    """
    import numpy as _np

    if canister_cylinder is None:
        return _np.zeros(shape, dtype=bool)
    o = _np.asarray(region_origin_m, dtype=float)
    nx, ny, nz = shape
    xs = o[0] + _np.arange(nx) * d_m
    ys = o[1] + _np.arange(ny) * d_m
    zs = o[2] + _np.arange(nz) * d_m

    x0 = canister_cylinder.x_center_m - canister_cylinder.x_half_width_m
    # From one cell BEFORE x0: the end cap below stops at x0 - d, so the cell
    # straddling x0 was in neither, and the optimiser thinned the wall there to
    # 2.5 mm (2026-09-26). The annulus spares r < inner, so the bore is intact.
    in_x = (xs >= x0 - d_m) & (xs <= x0 + mm_to_m(CANISTER_SAFETY_ZONE_LENGTH_MM))
    r = _np.sqrt((ys[:, None] - canister_cylinder.y_center_m) ** 2
                 + (zs[None, :] - canister_cylinder.z_center_m) ** 2)

    # Inner radius is the BORE's own, so the ring and the void abut exactly.
    # Ringing from the part's nominal 9.000 while the bore was 9.125 put that
    # band inside the void, and hard_solid &= ~hard_air subtracted it: measured
    # 1,250 cells punched out, leaving the zone 82% solid and the bodywork
    # still not meeting the pocket.
    inner_m = max(float(canister_cylinder.radius_m),
                  mm_to_m(CANISTER_SAFETY_ZONE_INNER_R_MM))

    # At least two cells thick, whatever the spacing. The real wall is 3.0 mm --
    # 6 cells at the 0.5 mm production spacing, but exactly 1.0 cell at 3 mm and
    # 1.5 at 2 mm, and a one-cell shell in a level set is a sliver farm.
    # Measured at 3 mm it drove marching cubes to a 0.4 deg minimum angle and
    # failed the snappyHexMesh gate outright. Widening OUTWARD keeps the bore
    # exact (T5.1) and only ever adds material (T5.5-safe); it is a no-op at
    # 1.0 mm and below, where 3.0 mm already spans three cells.
    # Plus half a cell: the extracted, Taubin-smoothed skin sits inside the last
    # solid cell, and at 1 mm the scrutineer probe found 99.4 % of the 3 mm
    # annulus solid after optimisation (2026-09-26). Only ever adds material.
    outer_m = max(mm_to_m(CANISTER_CLEARANCE_RADIUS_MM) + 0.5 * d_m, inner_m + 2.0 * d_m)
    ring = (r >= inner_m) & (r <= outer_m)
    out = in_x[:, None, None] & ring[None, :, :]

    # END CAP. The zone is "around the minimum chamber depth", and a hole has a
    # floor as well as walls -- the wall annulus alone leaves nothing behind the
    # end of the bore. Measured on the 2026-08-07 car, a void sat immediately
    # forward of the chamber: at x=162 the centreline was air from z=30 to 39,
    # ten cells of nothing between the bodywork and the back of the pocket.
    # A cartridge pushed home would bottom out against a shell.
    #
    # Same 3.0 mm as the walls, and the same two-cell floor so it survives
    # coarse spacings. Full disc, not an annulus: the bore's floor spans the
    # whole chamber cross-section.
    # Stop a FULL CELL short of the bore. The bore's analytic start rarely lands
    # on a cell boundary, so `xs < x0` still claims the cell that CONTAINS x0 --
    # measured at 2 mm spacing the cap took cell 92 (centre 184.0) while the bore
    # began at 184.1, blocking the chamber's first 2 mm of depth. The bore's own
    # void mask does not cover that cell either (its centre is forward of x0), so
    # hard_air never wins it back. T5.3 measures depth "from the opening to the
    # chamber end" and a cap eating into it makes the chamber shallower than the
    # number the model claims.
    cap_m = max(mm_to_m(CANISTER_SAFETY_ZONE_WALL_MM), 2.0 * d_m)
    in_cap_x = (xs >= x0 - cap_m - d_m) & (xs <= x0 - d_m)
    out |= in_cap_x[:, None, None] & (r <= outer_m)[None, :, :]
    return out



def canister_zone_contact_shell_mask(canister_cylinder,
                                     region_origin_m, shape,
                                     d_m: float) -> "np.ndarray":
    """Bodywork that must be SOLID immediately outside the safety zone.

    The zone itself is void in phi (its own mass component -- see
    canister_safety_zone_solid_mask). Carving it out removes the body's only
    reason to be there, though: measured the moment the zone went from forced
    solid to forced air, the bodywork's top on the centreline fell from 46.5 mm
    to 19.5 mm against a zone surface at 47.0. The pocket went straight back to
    floating in air, which is the thing that started all of this.

    So the zone is excluded from the body's VOLUME but the body is required to
    MEET it: a shell two cells thick wrapped around r = 12 mm, over the zone's
    length and its end cap. Two cells rather than a fixed thickness for the same
    reason as the zone's own wall -- a one-cell shell is a sliver farm at coarse
    spacings.

    This is the difference between "the foam is not counted twice" and "there is
    no foam there". Only the first is wanted.
    """
    import numpy as _np

    if canister_cylinder is None:
        return _np.zeros(shape, dtype=bool)
    o = _np.asarray(region_origin_m, dtype=float)
    nx, ny, nz = shape
    xs = o[0] + _np.arange(nx) * d_m
    ys = o[1] + _np.arange(ny) * d_m
    zs = o[2] + _np.arange(nz) * d_m

    x0 = canister_cylinder.x_center_m - canister_cylinder.x_half_width_m
    inner_m = max(float(canister_cylinder.radius_m),
                  mm_to_m(CANISTER_SAFETY_ZONE_INNER_R_MM))
    zone_outer_m = max(mm_to_m(CANISTER_CLEARANCE_RADIUS_MM), inner_m + 2.0 * d_m)
    cap_m = max(mm_to_m(CANISTER_SAFETY_ZONE_WALL_MM), 2.0 * d_m)

    r = _np.sqrt((ys[:, None] - canister_cylinder.y_center_m) ** 2
                 + (zs[None, :] - canister_cylinder.z_center_m) ** 2)
    shell = (r > zone_outer_m) & (r <= zone_outer_m + 2.0 * d_m)
    in_x = (xs >= x0 - cap_m) & (xs <= x0 + mm_to_m(CANISTER_SAFETY_ZONE_LENGTH_MM))
    out = in_x[:, None, None] & shell[None, :, :]

    # And a disc of the same thickness behind the end cap, so the pocket has
    # bodywork against its floor as well as its walls.
    behind = (xs >= x0 - cap_m - 2.0 * d_m) & (xs < x0 - cap_m)
    out |= behind[:, None, None] & (r <= zone_outer_m + 2.0 * d_m)[None, :, :]
    return out


def halo_visibility_air_mask(halo_mask: "np.ndarray",
                             z_origin_m: float,
                             dz_m: float) -> "np.ndarray":
    """Cells that must be AIR for T4.4.2 and T4.4.3 halo visibility.

    T4.4.2 (front and side, 10 pts): "Visibility of the Halo must not be
    physically obstructed by any other component when viewed in the front or
    side views" -- and the rule's own diagram boxes only the halo ABOVE a 4.0 mm
    base fillet, so bodywork may blend into the bottom 4 mm.

    T4.4.3 (top, 10 pts): "The Halo must not be physically obstructed in the
    plan view except by the helmet." No fillet relief; the full plan outline
    must be clear.

    The halo was modelled only as a VOID -- phi > 0 so the optimiser cannot fill
    the mount -- with nothing stopping bodywork sitting above, ahead of or
    outboard of it. Measured on a carved 79 g car before this existed: 100% of
    halo cells obstructed in top view, 100% in front, 64.7% in side.

    "Unobstructed from view V" means every ray from a protected halo cell toward
    V's viewer is clear of solid, so the constraint is that the protected halo's
    SHADOW in each direction must be air:

        top    viewer at +z    -> air ABOVE it,     full halo (T4.4.3)
        front  viewer at -x    -> air AHEAD of it,  above the fillet (T4.4.2)
        side   viewer at +/-y  -> air OUTBOARD,     above the fillet (T4.4.2)

    Returned as a mask to OR into hard_mask_air, so the constraint binds at
    every step of the descent rather than being audited afterwards.

    z_origin_m / dz_m locate the grid in space, since the fillet is a physical
    4 mm rather than a cell count.
    """
    import numpy as _np
    out = _np.zeros_like(halo_mask, dtype=bool)
    if not halo_mask.any():
        return out

    nx, ny, nz = halo_mask.shape

    # ---- TOP (T4.4.3): full halo, air everywhere above, per (x, y) column ----
    # cumsum along +z: at index k the running total counts halo cells at z <= k,
    # so >0 means "there is halo at or below me", i.e. I am above the halo and
    # would block the top view.
    #
    # The first version flipped before the cumsum, which computes "there is halo
    # at or ABOVE me" -- the exact opposite -- and left the top view 100%
    # obstructed while front and side both cleared.
    above = _np.cumsum(halo_mask, axis=2) > 0
    out |= above & ~halo_mask

    # ---- the protected sub-region for T4.4.2 -------------------------------
    k_halo = _np.flatnonzero(halo_mask.any(axis=(0, 1)))
    z_bottom_m = z_origin_m + k_halo[0] * dz_m
    z_cut_m = z_bottom_m + HALO_VISIBILITY_FILLET_MM / 1000.0
    k_cut = int(_np.ceil((z_cut_m - z_origin_m) / dz_m))
    protected = halo_mask.copy()
    protected[:, :, :k_cut] = False           # bottom fillet band is exempt
    if not protected.any():
        return out & ~halo_mask

    # ---- FRONT (T4.4.2): air ahead of the protected halo, per (y, z) --------
    # Ahead = smaller x than the first protected halo cell in that (y, z) row.
    first_x = _np.argmax(protected, axis=0)                    # (y, z)
    has = protected.any(axis=0)                                # (y, z)
    xs = _np.arange(nx)[:, None, None]
    out |= (xs < first_x[None, :, :]) & has[None, :, :]

    # ---- SIDE (T4.4.2): air outboard of it on both flanks, per (x, z) ------
    # Outboard = beyond the extreme protected halo cell in y, either direction.
    ys = _np.arange(ny)[None, :, None]
    any_y = protected.any(axis=1)                              # (x, z)
    first_y = _np.argmax(protected, axis=1)                    # (x, z)
    last_y = ny - 1 - _np.argmax(_np.flip(protected, axis=1), axis=1)
    out |= (ys < first_y[:, None, :]) & any_y[:, None, :]
    out |= (ys > last_y[:, None, :]) & any_y[:, None, :]

    return out & ~halo_mask



def _loft_profile(halo_mask, canister_cylinder, x_origin_m: float,
                  y_origin_m: float, z_origin_m: float, d_m: float):
    """The loft surface height per (x, y): (i0, i1, z_top[span, ny], valid[ny]).

    The surface INTERPOLATES THE TWO CROSS-SECTIONS IT JOINS -- the halo's top
    profile at its rear face, and the cartridge assembly's circular profile at
    the bore front -- rather than sweeping one height across a flat band.

    The first version did the latter: a single z per x applied to a rectangular
    `|y| <= half_width` strip. On a 1 mm grid that renders as exactly what it
    is, a slab with a staircase -- flat-topped panels with square shoulders and
    visible steps, which is what "almost rectangles in the loft" was pointing
    at. A deck joining a round cartridge to a rounded halo has no business
    being flat across its width.

    Blending per-y gives the deck a crown that starts as the halo's section and
    ends as the cartridge's circle, so the top surface curves in BOTH x and y
    and marching cubes has a smooth field to cut rather than a plateau edge.
    Restricted to the y range where both profiles exist, which is where a
    surface between them is defined at all.
    """
    import numpy as _np

    nx, ny, nz = halo_mask.shape
    hx = _np.flatnonzero(halo_mask.any(axis=(1, 2)))
    if hx.size == 0 or canister_cylinder is None:
        return None
    i_halo_rear = int(hx[-1])

    # Halo section the loft leaves from: the REAL part's rear face, which is
    # flat at HALO_REAR_FACE_TOP_MM across its width. The mask gives the y
    # extent and the x station; it must NOT give the height, because it is the
    # conservative envelope and sits 9 mm above the part.
    rear = halo_mask[i_halo_rear]                        # (ny, nz)
    has_halo = rear.any(axis=1)
    z_halo = _np.where(has_halo, mm_to_m(HALO_REAR_FACE_TOP_MM), _np.nan)

    # Cartridge assembly section at the bore front: a circle of the clearance
    # radius, which is the part's flat outer face (see
    # CANISTER_CLEARANCE_RADIUS_MM).
    r_m = mm_to_m(CANISTER_CLEARANCE_RADIUS_MM)
    ys = y_origin_m + _np.arange(ny) * d_m
    inside = _np.abs(ys) <= r_m
    z_can = _np.full(ny, _np.nan)
    z_can[inside] = canister_cylinder.z_center_m + _np.sqrt(
        _np.maximum(r_m * r_m - ys[inside] ** 2, 0.0))

    valid = ~_np.isnan(z_halo) & ~_np.isnan(z_can)
    if not valid.any():
        return None

    x_can_front_m = (canister_cylinder.x_center_m
                     - canister_cylinder.x_half_width_m)
    i_can_front = int(round((x_can_front_m - x_origin_m) / d_m))
    i0, i1 = i_halo_rear, min(i_can_front, nx - 1)
    if i1 <= i0:
        return None

    t = (_np.arange(i0, i1 + 1, dtype=_np.float64) - i0) / float(i1 - i0)
    t = t * t * (3.0 - 2.0 * t)                     # smoothstep, C1 at both ends
    z_top = (z_halo[None, :]
             + t[:, None] * (z_can - z_halo)[None, :])
    return i0, i1, z_top, valid


def halo_canister_loft_air_mask(halo_mask: "np.ndarray",
                                canister_cylinder,
                                x_origin_m: float,
                                y_origin_m: float,
                                z_origin_m: float,
                                d_m: float) -> "np.ndarray":
    """Cells above the halo->canister loft, which must be AIR.

    01_generative_geometry specifies "the continuous top surface running from
    the cartridge chamber at the rear, forward over the halo mount", and
    unified_phi's own header records that "nothing implemented a loft and the
    boxes forbade one". Unifying the field removed the second half of that --
    a surface CAN now cross the main_body/rearpod boundary -- but still nothing
    built the loft, so the optimiser never had a reason to.

    What it produced instead, measured on the 2026-08-05 carved car along the
    centreline:

        x =  90 mm   body top 23.5 mm   (held down by the halo visibility rule)
        x = 100 mm   body top 51.5 mm
        x = 130 mm   body top 53.5 mm

    a 28 mm vertical cliff the instant the halo's shadow ends, then a flat slab
    to the tail. The visibility mask constrains only the halo's own shadow;
    aft of the halo the field was free to fill to the envelope roof, and since
    Stage 1's proxy has no aerodynamics, filling it costs nothing it can see.

    So the loft is imposed as geometry, the same way the visibility rule is: a
    CEILING over the span between the halo's rear face and the canister's front
    face, linearly interpolating from the halo top to the canister top, with
    everything above it forced to hard air. The optimiser may still carve BELOW
    the loft -- this bounds the surface, it does not prescribe it.

    Smoothstep rather than a straight ramp: a linear ceiling meets the halo top
    and the canister top at a slope discontinuity, and a crease is both a drag
    feature and a stress raiser. smoothstep is C1 at both ends.
    """
    import numpy as _np

    prof = _loft_profile(halo_mask, canister_cylinder, x_origin_m,
                         y_origin_m, z_origin_m, d_m)
    if prof is None:
        return _np.zeros_like(halo_mask, dtype=bool)
    i0, i1, z_top, valid = prof

    nx, ny, nz = halo_mask.shape
    ks = _np.arange(nz)[None, None, :]
    k_ceil = _np.ceil((z_top - z_origin_m) / d_m)

    # THE CEILING IS THE CAR'S ROOF, so it spans the full width and runs to the
    # tail -- not just the strip where both end profiles are defined.
    #
    # Capping only |y| <= ~12 mm and only x <= the bore front left the field
    # unconstrained everywhere else, and it went straight up to the envelope:
    # measured on the 2026-08-06 car, 10,662 solid cells above z=45 mm in two
    # towers, 2,156 of them at x 100-120 outboard of the deck and 7,122 at
    # x 180-205 above and behind the cartridge. Nothing was wrong with the loft;
    # there was simply no roof anywhere else, and a body that is 47 mm tall over
    # its spine and 54 mm tall beside it is not a shape anyone intended.
    #
    # Outboard of the deck the crown is held at the deck's own height rather
    # than extrapolated: the two profiles being interpolated only exist across
    # the parts' width, so continuing the curve past them would be inventing a
    # surface. Holding it flat bounds the roof without prescribing the flanks,
    # and the optimiser still chooses everything below.
    # MAX over y, not min. The profile is an arch: highest on the centreline,
    # falling away to the part's edge. Taking the minimum picked the LOWEST
    # point of that arch and clamped the whole roof to it -- measured at
    # x=180 it forced air from z=45 up, against a centreline crown of 47.1,
    # which sheared the top off the cartridge pocket and left the loft hanging
    # 21.5 mm clear of it. The crown is the peak.
    crown = _np.where(valid[None, :], k_ceil, -_np.inf).max(axis=1)   # (span,)
    full = _np.repeat(crown[:, None], ny, axis=1)
    k_ceil = _np.where(valid[None, :], k_ceil, full)

    out = _np.zeros_like(halo_mask, dtype=bool)
    out[i0:i1 + 1] = ks > k_ceil[:, :, None]
    # Aft of the cartridge front the roof holds at the cartridge's own crown:
    # there is no further part to loft to, and the tail has no business
    # standing taller than the thing it runs back from.
    if i1 + 1 < nx:
        out[i1 + 1:] = ks > crown[-1]
    return out & ~halo_mask


# Thickness of the lofted deck. Thin: this is a skin spanning halo to canister,
# not a filled block, and every gram of it comes out of the T3.6 budget.
LOFT_SKIN_THICKNESS_MM: float = 2.0


def halo_canister_loft_solid_mask(halo_mask: "np.ndarray",
                                  canister_cylinder,
                                  x_origin_m: float,
                                  y_origin_m: float,
                                  z_origin_m: float,
                                  d_m: float) -> "np.ndarray":
    """The lofted deck itself: cells that must be SOLID.

    halo_canister_loft_air_mask alone does not produce a loft. A ceiling only
    says "no higher"; with no aerodynamic term in Stage 1's proxy, mass
    minimisation then pulls the top surface as far BELOW the ceiling as it
    likes. Measured after adding the ceiling: the cliff went (53.5 -> 38.5 mm
    aft of the halo, good) but the deck settled flat at 38.5 mm while the
    canister top is at 44.1 mm, so the surface ran past the cartridge instead
    of arriving at it -- and the bore was left open to the sky, 28.4 mm of body
    at x=160 against a bore reaching 47 mm.

    "The back of the halo lofts to the canister" is a statement about a surface
    that EXISTS, so it has to be required material, not permitted volume. This
    forces a skin of LOFT_SKIN_THICKNESS_MM immediately under the same
    smoothstep line the ceiling uses, so the two agree by construction: the
    deck lands on the ceiling and the ceiling is the loft.

    Width is limited to the halo/canister half-width rather than the full body
    -- the loft is the spine joining two centreline parts, and forcing it the
    full 65 mm would be inventing bodywork the rule does not ask for and the
    mass budget cannot afford.
    """
    import numpy as _np

    prof = _loft_profile(halo_mask, canister_cylinder, x_origin_m,
                         y_origin_m, z_origin_m, d_m)
    if prof is None:
        return _np.zeros_like(halo_mask, dtype=bool)
    i0, i1, z_top, valid = prof

    # Start one cell AFT of the halo: over the halo's own footprint T4.4.3
    # forces air, and a deck there would fight the visibility rule.
    i0 = i0 + 1
    if i1 <= i0:
        return _np.zeros_like(halo_mask, dtype=bool)
    z_top = z_top[1:]

    nz = halo_mask.shape[2]
    thick = max(int(round(LOFT_SKIN_THICKNESS_MM / (d_m * 1000.0))), 1)
    k_top = _np.floor((z_top - z_origin_m) / d_m)
    k_top = _np.where(valid[None, :], k_top, -1.0)
    ks = _np.arange(nz)[None, None, :]
    out = _np.zeros_like(halo_mask, dtype=bool)
    out[i0:i1 + 1] = ((ks <= k_top[:, :, None])
                      & (ks > (k_top - thick)[:, :, None])
                      & valid[None, :, None])
    return out & ~halo_mask


def default_halo_cross_section_yz_m() -> list[tuple[float, float]]:
    """Conservative rectangular halo cross-section, see U1 note above."""
    hw = mm_to_m(HALO_CROSS_SECTION_HALF_WIDTH_MM)
    z0 = HALO_MIN_Z_M
    z1 = mm_to_m(HALO_CROSS_SECTION_TOP_MM)
    return [(-hw, z0), (hw, z0), (hw, z1), (-hw, z1)]


def default_mass_com_positions(W_mm: float, x_front_mm: float,
                               rear_face_x_m: float,
                               halo_x_front_m: float, halo_x_rear_m: float,
                               halo_z_mid_m: float) -> dict:
    """COM positions for the MASS ROLLUP, in metres.

    Separate from compute_default_fixed_hardware_inputs because that dict is
    splatted into place_fixed_hardware, which builds void masks and takes no
    COMs -- adding keys there is a TypeError.

    Front and rear wheels are separate entries: they differ in mass (5 g vs
    6 g, measured 2026-08-03) and sit a whole wheelbase apart, so lumping them
    at the axle midpoint puts the wheel COM (6/11 - 1/2)*W = 5.5 mm too far
    forward at W=120.
    """
    from geometry_contract import mm_to_m as _mm
    xf = _mm(x_front_mm)
    return {
        "wheels_front_com": (xf, 0.0, R_WHEEL_M),
        "wheels_rear_com": (xf + _mm(W_mm), 0.0, R_WHEEL_M),
        "halo_com": (0.5 * (halo_x_front_m + halo_x_rear_m), 0.0, halo_z_mid_m),
    }


def compute_default_fixed_hardware_inputs(
    W_mm: float,
    x_front_mm: float,
    d_halo_mm: float,
    ref_plane_A_m: float,
    ref_plane_B_m: float,
    rear_face_x_m: float,
) -> dict:
    """
    Build a full set of design-default fixed hardware inputs for
    place_fixed_hardware(), given the current outer-loop scalars.

    Args:
        rear_face_x_m: x of the rearmost machined face of the car, i.e.
            bounding_volumes.rearpod.x_max_m(). The cartridge chamber is bored
            forward from here so the cartridge protrudes out the back (T5.6).
            Passed in rather than derived because only the caller holds the
            BoundingVolumes that define where the car actually ends.

    Returns a dict with keys matching place_fixed_hardware()'s parameter
    names: halo_geometry, canister_com_mm, canister_radius_mm,
    canister_depth_mm, canister_rear_face_x_mm, wheel_axle_mass_kg,
    wheel_axle_com_mm, wheel_x_half_width_mm, wheel_axle_z_mm,
    rear_wing_mass_kg, rear_wing_com_mm.
    """
    from halo_pocket import compute_halo_pocket_box_m, HALO_POCKET_LENGTH_MM

    pocket = compute_halo_pocket_box_m(ref_plane_A_m, d_halo_mm)
    halo_geometry = HaloGeometry(
        x_front_m=pocket["x_min_m"],
        x_rear_m=pocket["x_max_m"],
        cross_section_yz_m=default_halo_cross_section_yz_m(),
    )

    # Canister: the chamber is bored forward from the rearmost machined face,
    # so the cartridge protrudes out the true rear of the assembled car (T5.6).
    # It was previously anchored to Ref Plane B, which sits ~40 mm forward of
    # the actual rear face and left the bore sealed inside the bodywork.
    # The canister's own COM is taken at the bore's mid-depth.
    canister_x_m = rear_face_x_m - mm_to_m(CANISTER_DEPTH_MM / 2.0)
    canister_com_mm = (canister_x_m * 1000.0, 0.0, CANISTER_Z_MM)

    # Wheels+axles: front and rear are separate masses at their own axles.
    # The old single COM at the axle midpoint assumed equal front/rear mass;
    # measured they are 5 g and 6 g, so the true combined COM sits at
    # x_front + (6/11)*W, 5.5 mm aft of the midpoint at W=120.
    x_front_m = mm_to_m(x_front_mm)
    W_m = mm_to_m(W_mm)
    wheel_front_com_mm = (x_front_m * 1000.0, 0.0, R_WHEEL_M * 1000.0)
    wheel_rear_com_mm = ((x_front_m + W_m) * 1000.0, 0.0, R_WHEEL_M * 1000.0)
    # Kept for callers that still want the lumped value; now mass-weighted
    # rather than geometric.
    _wf, _wr = WHEEL_AXLE_FRONT_MASS_KG, WHEEL_AXLE_REAR_MASS_KG
    wheel_axle_com_mm = (
        (_wf * wheel_front_com_mm[0] + _wr * wheel_rear_com_mm[0]) / (_wf + _wr),
        0.0, R_WHEEL_M * 1000.0)

    # Halo COM: centre of the pocket it occupies in x, on centreline, at the
    # mid-height of its own cross-section.
    _hz = [z for _y, z in halo_geometry.cross_section_yz_m]
    halo_com_mm = ((0.5 * (halo_geometry.x_front_m + halo_geometry.x_rear_m)) * 1000.0,
                   0.0,
                   (0.5 * (min(_hz) + max(_hz))) * 1000.0)

    # Rear wing: aft of Ref Plane B by REAR_WING_OVERHANG_MM, on centreline.
    rear_wing_x_m = ref_plane_B_m + mm_to_m(REAR_WING_OVERHANG_MM)
    rear_wing_com_mm = (rear_wing_x_m * 1000.0, 0.0, REAR_WING_HEIGHT_MM)

    return {
        "halo_geometry": halo_geometry,
        "canister_com_mm": canister_com_mm,
        # The BORE, and it must stay the bore: T5.1 regulates the chamber at
        # 18.0-18.5 mm, so carving the void at the CAD's outer 24.25 mm makes
        # the chamber itself illegal. See CANISTER_CLEARANCE_RADIUS_MM for what
        # the wider figure is and is not for.
        "canister_radius_mm": CANISTER_DIAMETER_MM / 2.0,
        "canister_depth_mm": CANISTER_DEPTH_MM,
        "canister_rear_face_x_mm": rear_face_x_m * 1000.0,
        "wheel_axle_mass_kg": WHEEL_AXLE_MASS_KG,
        "wheel_axle_com_mm": wheel_axle_com_mm,
        "wheel_x_half_width_mm": WHEEL_X_CLEARANCE_HALF_WIDTH_MM,
        "wheel_axle_z_mm": R_WHEEL_M * 1000.0,
        "rear_wing_mass_kg": REAR_WING_MASS_KG,
        "rear_wing_com_mm": rear_wing_com_mm,
    }