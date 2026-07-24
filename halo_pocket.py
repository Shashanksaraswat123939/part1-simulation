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
pocket's front edge sits. Range: [0, calibrate_d_halo_max_mm(W_mm)) mm =
[0, W-34) mm (strict upper; placement-derived so the 50mm pocket's rear can't
reach the rear axle). For W in [120, 140] that is [0, 86) .. [0, 106) mm. (The
old min(100, W+16) bound quoted here was stale -- geometry_contract overrode
it to W-34 and that is what validates and what the Bayesian bounds use.)

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

# Legal ballast container (Appendix ix, "Halo and Ballast container drawings").
# A mandatory EMPTY oval slot inside the halo pocket footprint, below the helmet
# aperture, for optional lead ballast. It must exist AT ALL TIMES regardless of
# what the optimiser does -- it is forced air (phi > 0), and nothing may fill it.
# Plan: a capsule 12.7mm wide (= 2 x R6.35 rounded ends), ~20mm long, biased
# toward the pocket tail (under the retardation notch). Section: cut 6.35mm below
# the pocket floor, so it opens UP through the (already-void) halo pocket to the
# top surface -- an open recess, never a sealed internal bubble.
BALLAST_SLOT_WIDTH_MM: float = 12.7            # y, = 2 x R6.35
BALLAST_SLOT_LENGTH_MM: float = 20.0           # x, total incl. the two R6.35 caps (drawing estimate)
BALLAST_DEPTH_MM: float = 6.35                 # z, below the pocket floor
BALLAST_CENTRE_FROM_POCKET_FRONT_MM: float = 34.0  # oval centre aft of pocket front edge


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


def build_ballast_container_forbidden_mask(
    origin_m: tuple,
    shape: tuple,
    ref_plane_A_m: float,
    d_halo_mm: float,
) -> np.ndarray:
    """Bool mask, True = forbidden (phi > 0 / air) for the mandatory ballast slot.

    A capsule (oval) BALLAST_SLOT_WIDTH_MM wide x BALLAST_SLOT_LENGTH_MM long,
    centred BALLAST_CENTRE_FROM_POCKET_FRONT_MM aft of the pocket front edge and
    on y=0, cut BALLAST_DEPTH_MM below the pocket floor. Membership test is
    distance-to-centreline-segment <= radius, so the R6.35 rounded ends are
    exact rather than a boxy approximation.
    """
    box = compute_halo_pocket_box_m(ref_plane_A_m, d_halo_mm)
    r = mm_to_m(BALLAST_SLOT_WIDTH_MM) / 2.0            # = R6.35
    half_straight = max(mm_to_m(BALLAST_SLOT_LENGTH_MM) / 2.0 - r, 0.0)
    cx = box["x_min_m"] + mm_to_m(BALLAST_CENTRE_FROM_POCKET_FRONT_MM)
    z_top = box["z_min_m"]                              # pocket floor (24mm)
    z_bot = z_top - mm_to_m(BALLAST_DEPTH_MM)

    ox, oy, oz = origin_m
    nx, ny, nz = shape
    dx = GRID_SPACING_M
    xs = ox + np.arange(nx) * dx
    ys = oy + np.arange(ny) * dx
    zs = oz + np.arange(nz) * dx

    # Distance from each (x,y) to the capsule's centreline segment [cx±half_straight, y=0].
    dx_to_seg = np.abs(xs - cx) - half_straight
    dx_to_seg = np.clip(dx_to_seg, 0.0, None)           # 0 within the straight run
    dist2 = dx_to_seg[:, None] ** 2 + ys[None, :] ** 2  # (nx, ny)
    in_xy = dist2 <= r * r
    in_z = (zs >= z_bot) & (zs <= z_top)
    return in_xy[:, :, None] & in_z[None, None, :]
