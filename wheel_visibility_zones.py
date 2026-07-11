"""
wheel_visibility_zones.py --- T7.9 "Visibility in top and bottom views" forbidden zones.

Four keep-clear zones around each wheel, per the STEM Racing UAE 2025-26 Technical
Regulations, Article T7.9 (page 32 diagram). No car body may exist within these
zones for any z in [0, 65mm] above the track surface -- this is what lets the wheel
be seen from directly above/below despite the sidepod/body sitting right next to it.

Coordinate system: x=0 at nose tip (see geometry_contract.py).
Front wheel centre at x_front_m, rear wheel centre at x_front_m + W_m.

Shapes (right triangles / rectangles in the x-y plane, extruded through z):

  T7.9.1  In front of front wheels   -- rectangle, 5.0mm (x) x [inner track-contact edge, car outer width] (y)
  T7.9.2  Behind front wheels        -- right triangle, legs 15.0mm (x) x 30.0mm (y)
  T7.9.3  In front of rear wheels    -- right triangle, legs 5.0mm (x) x 30.0mm (y)
  T7.9.4  Behind rear wheels         -- rectangle, 5.0mm (x) x [inner track-contact edge, car outer width] (y)

Component mapping (see bounding_volumes.py coordinate layout):
  T7.9.1 falls within main_body's forward stub (between Ref Plane A and the front
         axle) -- nose stops at Ref Plane A, well short of the wheel.
  T7.9.4 falls within rearpod's territory (rearpod starts exactly at the rear axle).
  T7.9.2 and T7.9.3 fall within the sidepod corridor -- these are the two zones
         that carve into the sidepod's leading and trailing inner corners.

NOTE ON ANGLE LABELS: the source diagram labels ~60 degrees (T7.9.2) and ~45
degrees (T7.9.3) alongside the linear dimensions above. Treating both linear
dimensions as exact right-triangle legs gives an implied angle of ~63.4 degrees
(T7.9.2) and ~80.5 degrees (T7.9.3) rather than the labelled 60/45 degrees -- a
discrepancy consistent with rounding in a hand-annotated diagram. This module
uses the two EXPLICIT LINEAR DIMENSIONS as authoritative (what a caliper/feeler
gauge scrutineering check actually measures), not the angle labels. If STEM
Racing clarifies the angle is the exact spec instead, revisit these vertices.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from geometry_contract import GRID_SPACING_M, mm_to_m

# T7.2 minimum inner gaps between opposing wheels. Half of each gap is the
# distance from centerline to the wheel's inner track-contact edge, at the
# absolute legal minimum spacing. Actual gap is a free design choice >= this;
# callers may override via front_inner_gap_mm / rear_inner_gap_mm.
FRONT_INNER_GAP_MIN_MM: float = 38.0   # T7.2.1
REAR_INNER_GAP_MIN_MM: float = 30.0    # T7.2.2

# T7.9 zone dimensions (mm), read from the regulation diagram (T7.9, page 32).
T79_RECT_DEPTH_MM: float = 5.0     # T7.9.1 / T7.9.4 rectangle x-depth
T79_FRONT_LEG_X_MM: float = 15.0   # T7.9.2 horizontal leg (aft of front wheel)
T79_REAR_LEG_X_MM: float = 5.0     # T7.9.3 horizontal leg (forward of rear wheel)
T79_WEDGE_LEG_Y_MM: float = 30.0   # T7.9.2 / T7.9.3 vertical (lateral) leg
T79_ZONE_HEIGHT_MM: float = 65.0   # T7.9 zones span z: [0, 65mm] from track surface


def _axis_coords(origin_m: tuple, shape: tuple):
    ox, oy, oz = origin_m
    nx, ny, nz = shape
    dx = GRID_SPACING_M
    xs = ox + np.arange(nx) * dx
    ys = oy + np.arange(ny) * dx
    zs = oz + np.arange(nz) * dx
    return xs, ys, zs


def _rect_forbidden_mask(
    origin_m: tuple,
    shape: tuple,
    x_min_m: float,
    x_max_m: float,
    y_min_m: float,
    z_max_m: float,
) -> np.ndarray:
    """Rectangle forbidden zone: x in [x_min,x_max], y >= y_min (out to car's outer edge), z in [0,z_max]."""
    xs, ys, zs = _axis_coords(origin_m, shape)
    in_x = (xs >= x_min_m) & (xs <= x_max_m)
    in_y = ys >= y_min_m
    in_z = (zs >= 0.0) & (zs <= z_max_m)
    return in_x[:, None, None] & in_y[None, :, None] & in_z[None, None, :]


def _wedge_forbidden_mask(
    origin_m: tuple,
    shape: tuple,
    x_at_wheel_m: float,
    leg_x_m: float,
    direction: str,
    y_bottom_m: float,
    y_top_m: float,
    z_max_m: float,
) -> np.ndarray:
    """
    Right-triangle forbidden wedge.

    Right angle at (x_at_wheel_m, y_top_m). Vertical leg flush against the wheel,
    running from (x_at_wheel_m, y_top_m) down to (x_at_wheel_m, y_bottom_m) -- this
    is the edge nearest the wheel's track-contact patch. Horizontal leg runs from
    (x_at_wheel_m, y_top_m) away from the wheel by leg_x_m (direction "aft" = +x,
    for the front wheel; "fwd" = -x, for the rear wheel). The hypotenuse connects
    the two free ends; a point is inside the wedge when

        |x - x_at_wheel_m| / leg_x_m + (y_top_m - y) / (y_top_m - y_bottom_m) <= 1
    """
    xs, ys, zs = _axis_coords(origin_m, shape)
    dy = y_top_m - y_bottom_m

    if direction == "aft":
        x_far = x_at_wheel_m + leg_x_m
        in_x_band = (xs >= x_at_wheel_m) & (xs <= x_far)
        x_frac = (xs - x_at_wheel_m) / leg_x_m
    elif direction == "fwd":
        x_far = x_at_wheel_m - leg_x_m
        in_x_band = (xs <= x_at_wheel_m) & (xs >= x_far)
        x_frac = (x_at_wheel_m - xs) / leg_x_m
    else:
        raise ValueError(f"direction must be 'aft' or 'fwd', got {direction!r}")

    y_frac = (y_top_m - ys) / dy   # shape (ny,)
    in_y_band = (ys >= y_bottom_m) & (ys <= y_top_m)
    in_z = (zs >= 0.0) & (zs <= z_max_m)

    sum_frac = x_frac[:, None] + y_frac[None, :]   # shape (nx, ny)
    inside_xy = (sum_frac <= 1.0) & in_x_band[:, None] & in_y_band[None, :]

    return inside_xy[:, :, None] & in_z[None, None, :]


def build_t79_forbidden_mask(
    component: str,
    origin_m: tuple,
    shape: tuple,
    W_mm: float,
    x_front_mm: float,
    wheel_x_half_width_mm: float = 8.0,
    front_inner_gap_mm: float = FRONT_INNER_GAP_MIN_MM,
    rear_inner_gap_mm: float = REAR_INNER_GAP_MIN_MM,
) -> Optional[np.ndarray]:
    """
    Build the T7.9 forbidden-zone mask for one component's grid.

    Returns a bool array (True = forbidden, must be phi > 0 / air) of shape `shape`,
    or None if this component has no T7.9 zone in this coordinate mapping (nose --
    see module docstring).

    component:
      "main_body" -> T7.9.1 (rectangle, in front of front wheels)
      "sidepod"   -> T7.9.2 + T7.9.3 (wedges, behind front / in front of rear wheels)
      "rearpod"   -> T7.9.4 (rectangle, behind rear wheels)
      "nose"      -> None
    """
    x_front_m = mm_to_m(x_front_mm)
    W_m = mm_to_m(W_mm)
    wheel_half_m = mm_to_m(wheel_x_half_width_mm)
    z_max_m = mm_to_m(T79_ZONE_HEIGHT_MM)

    front_wheel_fwd_m = x_front_m - wheel_half_m           # leading edge of front wheel
    front_wheel_aft_m = x_front_m + wheel_half_m           # trailing edge of front wheel
    rear_wheel_fwd_m = x_front_m + W_m - wheel_half_m      # leading edge of rear wheel
    rear_wheel_aft_m = x_front_m + W_m + wheel_half_m      # trailing edge of rear wheel

    y_bottom_front_m = mm_to_m(front_inner_gap_mm) / 2.0
    y_bottom_rear_m = mm_to_m(rear_inner_gap_mm) / 2.0
    y_top_front_m = y_bottom_front_m + mm_to_m(T79_WEDGE_LEG_Y_MM)
    y_top_rear_m = y_bottom_rear_m + mm_to_m(T79_WEDGE_LEG_Y_MM)

    rect_depth_m = mm_to_m(T79_RECT_DEPTH_MM)

    if component == "main_body":
        return _rect_forbidden_mask(
            origin_m, shape,
            x_min_m=front_wheel_fwd_m - rect_depth_m,
            x_max_m=front_wheel_fwd_m,
            y_min_m=y_bottom_front_m,
            z_max_m=z_max_m,
        )

    if component == "rearpod":
        return _rect_forbidden_mask(
            origin_m, shape,
            x_min_m=rear_wheel_aft_m,
            x_max_m=rear_wheel_aft_m + rect_depth_m,
            y_min_m=y_bottom_rear_m,
            z_max_m=z_max_m,
        )

    if component == "sidepod":
        wedge_front = _wedge_forbidden_mask(
            origin_m, shape,
            x_at_wheel_m=front_wheel_aft_m,
            leg_x_m=mm_to_m(T79_FRONT_LEG_X_MM),
            direction="aft",
            y_bottom_m=y_bottom_front_m,
            y_top_m=y_top_front_m,
            z_max_m=z_max_m,
        )
        wedge_rear = _wedge_forbidden_mask(
            origin_m, shape,
            x_at_wheel_m=rear_wheel_fwd_m,
            leg_x_m=mm_to_m(T79_REAR_LEG_X_MM),
            direction="fwd",
            y_bottom_m=y_bottom_rear_m,
            y_top_m=y_top_rear_m,
            z_max_m=z_max_m,
        )
        return wedge_front | wedge_rear

    return None
