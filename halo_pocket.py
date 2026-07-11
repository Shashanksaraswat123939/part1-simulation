"""
halo_pocket.py --- T4.4.4 halo mounting pocket geometry, positioned by d_halo.

The halo sits in a shallow recess machined into the main body's top surface
(regs Appendix ix, "Halo pocket and legal ballast container dimensions"): a
50mm x 25mm footprint, 3.175mm deep, cut with a 6.35mm ball-nose tool. The
pocket FLOOR is fixed at z = HALO_MIN_Z_M = 24mm above the track (T4.4.4 --
a SAFETY regulation, not a search variable; the circular notch centre must
sit at 34mm +/-1mm above track, and the notch centre is 10mm above the
pocket floor, so the floor is pinned at exactly 24mm).

d_halo is the ONE free variable here: it sets how far aft of Ref Plane A the
pocket's front edge sits. Range: [0, calibrate_d_halo_max_mm(W_mm)] mm, i.e.
[0, min(100, W+16)] -- always exactly [0, 100] mm for W in [120, 140].

This module models the pocket as its full bounding rectangle rather than the
tapered/rounded outline shown in the diagram -- a conservative choice.
Forbidding the full rectangle can only ever exclude MORE volume than the true
rounded shape strictly needs, never less, so it can't produce an illegal
design; it just leaves a little extra volume on the table right at the
pocket's corners. Revisit with the exact polygon (from the "50.0 / 30.0 /
10.0 / 10.0 / R6.35 / R9.0 / R3.175" dimensions in the diagram) if reclaiming
that sliver of volume matters later.
"""
from __future__ import annotations
import numpy as np

from geometry_contract import GRID_SPACING_M, mm_to_m, HALO_MIN_Z_M

HALO_POCKET_LENGTH_MM: float = 50.0    # x extent (Appendix ix)
HALO_POCKET_WIDTH_MM: float = 25.0     # y extent, symmetric about centerline
HALO_POCKET_DEPTH_MM: float = 3.175    # z extent, matches the ball-nose-tool cut depth


def compute_halo_pocket_box_m(ref_plane_A_m: float, d_halo_mm: float) -> dict:
    """
    Return the halo pocket's axis-aligned bounding box in metres.

    x: [ref_plane_A_m + d_halo_m, ref_plane_A_m + d_halo_m + pocket_length_m]
    y: [-half_width_m, +half_width_m]  (symmetric about centerline)
    z: [HALO_MIN_Z_M, HALO_MIN_Z_M + pocket_depth_m]  (floor fixed at 24mm, T4.4.4)
    """
    d_halo_m = mm_to_m(d_halo_mm)
    x_front_m = ref_plane_A_m + d_halo_m
    x_rear_m = x_front_m + mm_to_m(HALO_POCKET_LENGTH_MM)
    half_width_m = mm_to_m(HALO_POCKET_WIDTH_MM) / 2.0
    z_min_m = HALO_MIN_Z_M
    z_max_m = z_min_m + mm_to_m(HALO_POCKET_DEPTH_MM)
    return {
        "x_min_m": x_front_m, "x_max_m": x_rear_m,
        "y_min_m": -half_width_m, "y_max_m": half_width_m,
        "z_min_m": z_min_m, "z_max_m": z_max_m,
    }


def build_halo_pocket_forbidden_mask(
    origin_m: tuple,
    shape: tuple,
    ref_plane_A_m: float,
    d_halo_mm: float,
) -> np.ndarray:
    """Bool mask, True = forbidden (must be phi > 0 / air) for the halo pocket recess."""
    box = compute_halo_pocket_box_m(ref_plane_A_m, d_halo_mm)

    ox, oy, oz = origin_m
    nx, ny, nz = shape
    dx = GRID_SPACING_M
    xs = ox + np.arange(nx) * dx
    ys = oy + np.arange(ny) * dx
    zs = oz + np.arange(nz) * dx

    in_x = (xs >= box["x_min_m"]) & (xs <= box["x_max_m"])
    in_y = (ys >= box["y_min_m"]) & (ys <= box["y_max_m"])
    in_z = (zs >= box["z_min_m"]) & (zs <= box["z_max_m"])

    return in_x[:, None, None] & in_y[None, :, None] & in_z[None, None, :]
