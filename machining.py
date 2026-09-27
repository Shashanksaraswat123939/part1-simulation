"""
machining.py -- make a body the team's mill can actually cut.

Team process (2026-09-27): 3-axis milling, 6.25 mm ball-end cutter, from the
top and from the bottom (two setups). The cartridge chamber is drilled along x.

What that means for the solid S, on the body grid:
  * the cutter is a ball of radius r = 3.125 mm moving in AIR, so the air it
    can clear is the morphological OPENING of the air by that ball; the solid
    left behind is the CLOSING of S. Every concave corner gets >= r radius.
  * from the top and the bottom only: along each vertical line the body is a
    single interval (no undercuts, no enclosed air), bar drilled features.
  * the rule masks still win: every legally-required air cell (T7.9 zones,
    wheel keep-clear, halo pocket, ballast area, ...) must end up air. Where the
    ball cannot reach into a sharp corner of such a zone, the zone is grown
    until it can -- the ONLY way a round cutter clears a square corner is to
    cut beyond it. Mandatory solids (cargo, T5.5 wall) are restored last.

    make_machinable(geom, tool_r_mm=3.125) -> report dict   (mutates geom)
"""
from __future__ import annotations

import numpy as np

TOOL_R_MM = 3.125


def _dilate(mask, r):
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~mask) <= r


def _open(mask, r):
    from scipy.ndimage import distance_transform_edt
    core = distance_transform_edt(mask) > r
    return _dilate(core, r) & mask if core.any() else np.zeros_like(mask)


def _column_fill(S):
    """Fill each (x, y) column between its lowest and highest solid cell."""
    any_ = S.any(axis=2)
    nz = S.shape[2]
    k = np.arange(nz)
    lo = np.where(any_, np.argmax(S, axis=2), nz)
    hi = np.where(any_, nz - 1 - np.argmax(S[:, :, ::-1], axis=2), -1)
    return (k[None, None, :] >= lo[..., None]) & (k[None, None, :] <= hi[..., None])


def make_machinable(geom, tool_r_mm: float = TOOL_R_MM, max_iter: int = 4) -> dict:
    import unified_phi as up
    from phi_updater import reinitialise_sdf
    phi = geom.phi
    d_mm = up.GRID_SPACING_M * 1e3
    r = tool_r_mm / d_mm
    S = phi.grid < 0
    H = phi.hard_mask_air.copy()
    drilled = np.zeros_like(H)
    fh = getattr(geom, "fixed_hardware", None)
    if fh is not None and getattr(fh, "canister_void_mask", None) is not None:
        drilled = fh.canister_void_mask.astype(bool)
    # grid border and drilled bore are not milled features
    border = np.zeros_like(H)
    border[0, :, :] = border[-1, :, :] = True
    border[:, 0, :] = border[:, -1, :] = True
    border[:, :, 0] = border[:, :, -1] = True
    Hm = H & ~drilled & ~border
    A = ~S | H
    grown = 0
    for _ in range(max_iter):
        reach = _open(A, r)
        miss = Hm & ~reach
        if not miss.any():
            break
        add = _dilate(miss, r) & ~phi.hard_mask_solid
        grown += int((add & ~A).sum())
        A |= add
    S2 = ~_open(A, r)
    S2 = _column_fill(S2)
    S2 |= phi.hard_mask_solid
    S2 &= ~H
    report = {
        "tool_r_mm": tool_r_mm,
        "cells_added": int((S2 & ~S).sum()),
        "cells_removed": int((S & ~S2).sum()),
        "zone_cells_grown_for_cutter": grown,
        "legal_air_violated": int((S2 & H).sum()),
    }
    from scipy.ndimage import distance_transform_edt
    sd = (distance_transform_edt(~S2) - distance_transform_edt(S2)) * up.GRID_SPACING_M
    phi.grid = sd.astype(np.float32)
    phi.apply_hard_constraints()
    reinitialise_sdf(phi)
    phi.apply_hard_constraints()
    return report


def undercut_cells(S) -> int:
    """Air cells enclosed in a vertical column (unreachable from top or bottom)."""
    return int((_column_fill(S) & ~S).sum())
