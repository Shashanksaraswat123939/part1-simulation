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
    R_WHEEL_M, validate_W, validate_x_front,
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
        x_front_mm: front axle position from nose tip in mm (x=0 = nose tip)
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
CANISTER_DIAMETER_MM: float = 18.25    # T5.1: 18.0-18.5mm, midpoint
CANISTER_DEPTH_MM: float = 50.0        # T5.3: 45.0-58.0mm
CANISTER_Z_MM: float = 35.0            # T5.2: 30.0-40.0mm, midpoint (rear-centre height)
CANISTER_SAFETY_ZONE_MM: float = 3.0   # T5.5: min 3.0mm wall around chamber

# Rear wing (T9.4, T9.5) -- mass and COM height are not given by the regs at all;
# these are placeholders pending a real measured rear wing.
REAR_WING_MASS_KG: float = 0.005       # design placeholder, ~5g
REAR_WING_OVERHANG_MM: float = 20.0    # T9.4.2: 0-40mm aft of Ref Plane B, midpoint
REAR_WING_HEIGHT_MM: float = 50.0      # within T9.4.3 max 65mm

# Wheels + axles combined assembly -- mass is not given by the regs (a supplier
# part); this is a design placeholder pending real measurement.
WHEEL_AXLE_MASS_KG: float = 0.015      # design placeholder, ~15g for all 4 wheels+axles

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


def default_halo_cross_section_yz_m() -> list[tuple[float, float]]:
    """Conservative rectangular halo cross-section, see U1 note above."""
    hw = mm_to_m(HALO_CROSS_SECTION_HALF_WIDTH_MM)
    z0 = HALO_MIN_Z_M
    z1 = mm_to_m(HALO_CROSS_SECTION_TOP_MM)
    return [(-hw, z0), (hw, z0), (hw, z1), (-hw, z1)]


def compute_default_fixed_hardware_inputs(
    W_mm: float,
    x_front_mm: float,
    d_halo_mm: float,
    ref_plane_A_m: float,
    ref_plane_B_m: float,
) -> dict:
    """
    Build a full set of design-default fixed hardware inputs for
    place_fixed_hardware(), given the current outer-loop scalars.

    Returns a dict with keys matching place_fixed_hardware()'s parameter
    names: halo_geometry, canister_com_mm, canister_box_half_size_mm,
    wheel_axle_mass_kg, wheel_axle_com_mm, wheel_x_half_width_mm,
    wheel_axle_z_mm, rear_wing_mass_kg, rear_wing_com_mm.
    """
    from halo_pocket import compute_halo_pocket_box_m, HALO_POCKET_LENGTH_MM

    pocket = compute_halo_pocket_box_m(ref_plane_A_m, d_halo_mm)
    halo_geometry = HaloGeometry(
        x_front_m=pocket["x_min_m"],
        x_rear_m=pocket["x_max_m"],
        cross_section_yz_m=default_halo_cross_section_yz_m(),
    )

    # Canister: centred fore-aft near the rear of main_body's machined
    # territory (Ref Plane B), so its chamber can protrude out the true rear
    # of the assembled car (T5.6).
    canister_x_m = ref_plane_B_m - mm_to_m(CANISTER_DEPTH_MM / 2.0)
    canister_com_mm = (canister_x_m * 1000.0, 0.0, CANISTER_Z_MM)
    canister_box_half_size_mm = CANISTER_DIAMETER_MM / 2.0 + CANISTER_SAFETY_ZONE_MM

    # Wheels+axles: COM at the midpoint between front and rear axle (equal
    # front/rear contribution assumed), on centreline, at wheel-radius height.
    x_front_m = mm_to_m(x_front_mm)
    W_m = mm_to_m(W_mm)
    wheel_axle_com_mm = ((x_front_m + W_m / 2.0) * 1000.0, 0.0, R_WHEEL_M * 1000.0)

    # Rear wing: aft of Ref Plane B by REAR_WING_OVERHANG_MM, on centreline.
    rear_wing_x_m = ref_plane_B_m + mm_to_m(REAR_WING_OVERHANG_MM)
    rear_wing_com_mm = (rear_wing_x_m * 1000.0, 0.0, REAR_WING_HEIGHT_MM)

    return {
        "halo_geometry": halo_geometry,
        "canister_com_mm": canister_com_mm,
        "canister_box_half_size_mm": canister_box_half_size_mm,
        "wheel_axle_mass_kg": WHEEL_AXLE_MASS_KG,
        "wheel_axle_com_mm": wheel_axle_com_mm,
        "wheel_x_half_width_mm": 8.0,
        "wheel_axle_z_mm": R_WHEEL_M * 1000.0,
        "rear_wing_mass_kg": REAR_WING_MASS_KG,
        "rear_wing_com_mm": rear_wing_com_mm,
    }