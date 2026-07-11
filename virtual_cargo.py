"""
virtual_cargo.py --- T4.2 virtual cargo placement (main_body).

The virtual cargo is a mandatory MINIMUM solid region -- a tapered wedge,
60mm long, 10mm tall (constant), tapering in width from 55mm at one end to
10mm at the other -- that must exist somewhere in main_body, wholly between
the front and rear axle centerlines. The regs place no other constraint on
its position beyond "clearly dimensioned" and "not overlapping the halo
pocket" (it MAY overlap the ballast container).

Unlike W / x_front / d_halo, cargo position does not touch the exterior
surface or aerodynamics -- it is a purely interior constraint. Evaluating a
candidate position is cheap (a direct mask/mass computation, no CFD), so this
module is called once per Level 2 evaluation to pick a placement that avoids
the halo pocket, rather than being a Bayesian search dimension.

Shape: at parametric position s in [0, LENGTH_MM] along the wedge (s=0 = wide
end, s=LENGTH_MM = narrow end), half-width(s) linearly interpolates from
WIDE_WIDTH_MM/2 down to NARROW_WIDTH_MM/2.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

from geometry_contract import GRID_SPACING_M, mm_to_m
from halo_pocket import compute_halo_pocket_box_m

CARGO_LENGTH_MM: float = 60.0        # T4.2
CARGO_WIDE_WIDTH_MM: float = 55.0    # T4.2, wide end
CARGO_NARROW_WIDTH_MM: float = 10.0  # T4.2, narrow end
CARGO_HEIGHT_MM: float = 10.0        # T4.2, constant along the whole length

N_CANDIDATE_POSITIONS: int = 9       # scan this many evenly-spaced start positions


def _boxes_overlap_1d(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
    return a_min < b_max and b_min < a_max


def find_cargo_placement(
    x_front_mm: float,
    W_mm: float,
    ref_plane_A_m: float,
    d_halo_mm: float,
    z_floor_m: float,
    z_margin_m: float = 0.001,
) -> dict:
    """
    Pick a legal cargo placement for this (x_front, W, d_halo) combination.

    Scans N_CANDIDATE_POSITIONS evenly-spaced x-start positions within the
    axle corridor [x_front_m, x_front_m + W_m - length_m], preferring the
    corridor centre, and skips any candidate whose x-extent overlaps the halo
    pocket's x-extent (T4.2: "not the halo pocket"). z is fixed just above the
    main_body floor -- lower COM height is never worse for the race-time
    objective, and there's no regulatory reason to place it any higher.

    Returns dict with x_start_m, z_base_m, and collided_with_default (bool,
    True if the corridor-centre default had to be shifted to avoid the halo).

    Raises ValueError if no candidate in the corridor avoids the halo pocket
    (should not happen in practice: corridor is >= 120mm, cargo is 60mm, halo
    pocket is 50mm -- but a very small W combined with a large d_halo could
    theoretically leave no room).
    """
    length_m = mm_to_m(CARGO_LENGTH_MM)
    x_front_m = mm_to_m(x_front_mm)
    W_m = mm_to_m(W_mm)

    corridor_min = x_front_m
    corridor_max = x_front_m + W_m - length_m
    if corridor_max < corridor_min:
        raise ValueError(
            f"Wheelbase W={W_mm}mm is too small to fit the {CARGO_LENGTH_MM}mm "
            f"virtual cargo between the axles at all."
        )

    halo_box = compute_halo_pocket_box_m(ref_plane_A_m, d_halo_mm)

    # Candidates, centre-corridor first (preferred default), then spreading outward.
    centre = (corridor_min + corridor_max) / 2.0
    if N_CANDIDATE_POSITIONS > 1:
        offsets = np.linspace(0.0, (corridor_max - corridor_min) / 2.0, N_CANDIDATE_POSITIONS // 2 + 1)
    else:
        offsets = np.array([0.0])
    candidates = [centre]
    for off in offsets[1:]:
        candidates.append(min(centre + off, corridor_max))
        candidates.append(max(centre - off, corridor_min))

    z_base_m = z_floor_m + z_margin_m

    for i, x_start in enumerate(candidates):
        x_end = x_start + length_m
        if not _boxes_overlap_1d(x_start, x_end, halo_box["x_min_m"], halo_box["x_max_m"]):
            return {
                "x_start_m": x_start,
                "z_base_m": z_base_m,
                "collided_with_default": i > 0,
            }

    raise ValueError(
        f"No virtual cargo placement in the axle corridor "
        f"[{corridor_min:.4f}, {corridor_max:.4f}] m avoids the halo pocket "
        f"[{halo_box['x_min_m']:.4f}, {halo_box['x_max_m']:.4f}] m. "
        f"W={W_mm}mm, d_halo={d_halo_mm}mm."
    )


def build_virtual_cargo_solid_mask(
    origin_m: tuple,
    shape: tuple,
    x_start_m: float,
    z_base_m: float,
) -> np.ndarray:
    """
    Bool mask, True = forced solid (phi < 0), for the tapered cargo wedge
    starting at x_start_m, resting on z_base_m.
    """
    ox, oy, oz = origin_m
    nx, ny, nz = shape
    dx = GRID_SPACING_M
    xs = ox + np.arange(nx) * dx
    ys = oy + np.arange(ny) * dx
    zs = oz + np.arange(nz) * dx

    length_m = mm_to_m(CARGO_LENGTH_MM)
    half_wide_m = mm_to_m(CARGO_WIDE_WIDTH_MM) / 2.0
    half_narrow_m = mm_to_m(CARGO_NARROW_WIDTH_MM) / 2.0
    height_m = mm_to_m(CARGO_HEIGHT_MM)

    x_end_m = x_start_m + length_m
    z_top_m = z_base_m + height_m

    in_x = (xs >= x_start_m) & (xs <= x_end_m)
    in_z = (zs >= z_base_m) & (zs <= z_top_m)

    # Per-x half-width, linearly tapering from half_wide_m (at x_start) to
    # half_narrow_m (at x_end). Clip s to [0,1] for x values outside the wedge
    # (irrelevant there since in_x already excludes them, but avoids negative
    # half-widths in the intermediate array).
    s = np.clip((xs - x_start_m) / length_m, 0.0, 1.0)
    half_width_at_x = half_wide_m + s * (half_narrow_m - half_wide_m)   # shape (nx,)

    # in_y[i, j] = |ys[j]| <= half_width_at_x[i]
    in_y = np.abs(ys)[None, :] <= half_width_at_x[:, None]   # shape (nx, ny)

    mask = in_x[:, None, None] & in_y[:, :, None] & in_z[None, None, :]
    return mask
