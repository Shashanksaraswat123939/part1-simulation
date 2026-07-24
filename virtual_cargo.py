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
from typing import Callable, Optional
import numpy as np

from geometry_contract import GRID_SPACING_M, mm_to_m
from halo_pocket import compute_halo_pocket_box_m

CARGO_LENGTH_MM: float = 60.0        # T4.2
CARGO_WIDE_WIDTH_MM: float = 55.0    # T4.2, wide end
CARGO_NARROW_WIDTH_MM: float = 10.0  # T4.2, narrow end
CARGO_HEIGHT_MM: float = 10.0        # T4.2, constant along the whole length

N_CANDIDATE_POSITIONS: int = 9       # scan this many evenly-spaced start positions


def cargo_mass_com(
    x_start_m: float, z_base_m: float, flip: bool, density_kg_m3: float,
) -> tuple:
    """Analytic (mass_kg, com_x_m, com_y_m, com_z_m) of the cargo wedge.

    No rasterisation -- lets the placement scorer evaluate a candidate's COM
    contribution without rebuilding the whole geometry (cargo is a fixed-shape
    body-density block; only its position/orientation move). y=0 by symmetry,
    z = z_base + H/2. x = x_start + L*(a+2b)/(3(a+b)), the first moment of a
    linearly tapering width from a (at x_start) to b (at x_end); (a,b) =
    (wide,narrow) when not flipped, (narrow,wide) when flipped. Matches the
    rasterised mask's COM to sub-cell accuracy (checked in __main__).
    """
    L = mm_to_m(CARGO_LENGTH_MM)
    H = mm_to_m(CARGO_HEIGHT_MM)
    wide = mm_to_m(CARGO_WIDE_WIDTH_MM)
    narrow = mm_to_m(CARGO_NARROW_WIDTH_MM)
    mass = L * H * (wide + narrow) / 2.0 * density_kg_m3
    a, b = (narrow, wide) if flip else (wide, narrow)
    com_x = x_start_m + L * (a + 2.0 * b) / (3.0 * (a + b))
    return mass, com_x, 0.0, z_base_m + H / 2.0


def _boxes_overlap_1d(a_min: float, a_max: float, b_min: float, b_max: float) -> bool:
    return a_min < b_max and b_min < a_max


def find_cargo_placement(
    x_front_mm: float,
    W_mm: float,
    ref_plane_A_m: float,
    d_halo_mm: float,
    z_floor_m: float,
    z_margin_m: float = 0.001,
    score_fn: Optional[Callable[[float, bool], float]] = None,
) -> dict:
    """
    Pick a legal cargo placement for this (x_front, W, d_halo) combination.

    Scans N_CANDIDATE_POSITIONS evenly-spaced x-start positions within the
    axle corridor [x_front_m, x_front_m + W_m - length_m] (T4.2: "wholly
    positioned between the front and rear wheel centre lines"), skipping any
    candidate whose x-extent overlaps the halo pocket's x-extent (T4.2: "not
    the halo pocket"; it MAY overlap the ballast container). z is fixed just
    above the main_body floor -- lower COM height is never worse for the
    race-time objective, and there's no regulatory reason to place it higher.

    Two orientations are legal per T4.2 (the regs fix only symmetry + top-face-
    normal, not which end faces front), so each x-position is tried both
    wide-forward (flip=False) and wide-rearward (flip=True).

    score_fn(x_start_m, flip) -> float (LOWER is better) lets the caller pick
    the COM-optimal legal candidate -- typically the no-CFD stage scoring
    mass/COM through the locked race objective (cargo is interior, so it has
    zero aero effect; its only lever on race time is COM). When score_fn is
    None the default is the current behaviour: corridor-centre, wide-forward.

    Returns dict with x_start_m, z_base_m, flip, and collided_with_default
    (True if the corridor-centre default had to be shifted to avoid the halo).

    Raises ValueError if no candidate in the corridor avoids the halo pocket
    (should not happen in practice: corridor is >= 120mm, cargo is 60mm, halo
    pocket is 50mm -- but a very small W combined with a large d_halo could
    theoretically leave no room).
    """
    length_m = mm_to_m(CARGO_LENGTH_MM)
    x_front_m = mm_to_m(x_front_mm)
    W_m = mm_to_m(W_mm)

    # Corridor is T4.2's rule verbatim: "wholly positioned between the front and
    # rear wheel centre lines".
    #
    # NOT narrowed by the wheel keep-clear, deliberately. The cargo TAPERS
    # (55 mm -> 10 mm), so whether an end clears the wheel column depends on
    # which end is there, not just on x: measured at W=130/x_front=46/d_halo=20,
    # flip=False (narrow end rearward) erodes 0% while flip=True (wide end
    # rearward) erodes 17.8% at the same x_start. A 1-D corridor shrink is
    # therefore both wrong and over-restrictive -- applying it leaves NO legal
    # placement at all here, because the halo pocket already covers [50,100] mm.
    #
    # The real guard is in unified_phi.build_unified_geometry, which measures
    # actual mask overlap and REFUSES to build (it used to let
    # `hard_solid &= ~hard_air` delete the overlap silently). This function still
    # only screens the halo pocket, which is all it can do without the masks.
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

    # Legal (x_start, i) candidates in preference order (centre-first).
    legal = [
        (x_start, i)
        for i, x_start in enumerate(candidates)
        if not _boxes_overlap_1d(
            x_start, x_start + length_m, halo_box["x_min_m"], halo_box["x_max_m"]
        )
    ]
    if not legal:
        raise ValueError(
            f"No virtual cargo placement in the axle corridor "
            f"[{corridor_min:.4f}, {corridor_max:.4f}] m avoids the halo pocket "
            f"[{halo_box['x_min_m']:.4f}, {halo_box['x_max_m']:.4f}] m. "
            f"W={W_mm}mm, d_halo={d_halo_mm}mm."
        )

    if score_fn is None:
        # Default: corridor-centre, wide-forward. Unchanged from before.
        x_start, i = legal[0]
        return {
            "x_start_m": x_start, "z_base_m": z_base_m,
            "flip": False, "collided_with_default": i > 0,
        }

    # Scored: try every legal position in BOTH orientations, pick the minimum.
    best = min(
        ((x_start, i, flip) for (x_start, i) in legal for flip in (False, True)),
        key=lambda c: score_fn(c[0], c[2]),
    )
    x_start, i, flip = best
    return {
        "x_start_m": x_start, "z_base_m": z_base_m,
        "flip": flip, "collided_with_default": i > 0,
    }


def build_virtual_cargo_solid_mask(
    origin_m: tuple,
    shape: tuple,
    x_start_m: float,
    z_base_m: float,
    flip: bool = False,
) -> np.ndarray:
    """
    Bool mask, True = forced solid (phi < 0), for the tapered cargo wedge
    starting at x_start_m, resting on z_base_m.

    flip selects the fore-aft orientation the regs leave free (T4.2 fixes only
    symmetry about, and the top face normal to, the vertical reference plane --
    NOT which end faces front). flip=False puts the wide (55mm) end at x_start
    (forward); flip=True puts the wide end at x_end (rearward). This shifts the
    wedge's own COM by up to ~15mm in x, which is the whole point of exposing it
    as a design DOF -- the placement search picks the orientation that lands the
    total COM where the race objective wants it.
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
    if flip:
        # Wide end at x_end (rearward): width grows from narrow to wide with s.
        half_width_at_x = half_narrow_m + s * (half_wide_m - half_narrow_m)
    else:
        # Wide end at x_start (forward): the default.
        half_width_at_x = half_wide_m + s * (half_narrow_m - half_wide_m)   # shape (nx,)

    # in_y[i, j] = |ys[j]| <= half_width_at_x[i]
    in_y = np.abs(ys)[None, :] <= half_width_at_x[:, None]   # shape (nx, ny)

    mask = in_x[:, None, None] & in_y[:, :, None] & in_z[None, None, :]
    return mask
