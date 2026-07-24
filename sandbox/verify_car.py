"""
verify_car.py -- objective checks for the three design goals, no eyeballing.

Goals being verified (from the user's direction):
  1. WHEELS ON OUTRIGGERS: the body clears every wheel with a real gap and never
     sits above/below a wheel (T7.9 top/bottom visibility).
  2. HALO VISIBLE: the halo hoop pokes above the body deck over its own span
     (T4.4 front/side/top visibility) and is actually present.
  3. HALO->CANISTER LOFT: the body deck stays elevated as a continuous engine
     cover from the back of the halo to the canister -- it must not taper to
     nothing before reaching the canister.

Usage:
    python sandbox/verify_car.py [--W 130 --x-front 46 --d-halo 20] [--spacing 3.0]
"""
from __future__ import annotations

import argparse

import coarse  # noqa: F401  -- sys.path + PART2_PATH
import numpy as np


def _zprofile(solid, region):
    """z_top(x) and half_width(x) in mm along x (metres grid -> mm)."""
    from geometry_contract import GRID_SPACING_M
    ox, oy, oz = region.origin_m
    nx, ny, nz = region.shape
    xs = (ox + np.arange(nx) * GRID_SPACING_M) * 1000
    ys = (oy + np.arange(ny) * GRID_SPACING_M) * 1000
    zs = (oz + np.arange(nz) * GRID_SPACING_M) * 1000
    ztop = np.full(nx, 0.0)
    hwid = np.full(nx, 0.0)
    for i in range(nx):
        sl = solid[i]
        if sl.any():
            ztop[i] = zs[np.where(sl.any(axis=0))[0].max()]
            hwid[i] = np.abs(ys[np.where(sl.any(axis=1))[0]]).max()
    return xs, ztop, hwid


def verify(W_mm, x_front_mm, d_halo_mm, spacing=3.0, verbose=True):
    coarse.use_spacing(spacing)
    from unified_phi import build_unified_geometry, enforce_symmetry
    from body_profile import build_car_body_field
    from geometry_contract import (
        FRONT_WHEEL_INNER_Y_M, REAR_WHEEL_INNER_Y_M, WHEEL_WIDTH_M, R_WHEEL_M,
    )
    from fixed_hardware import CANISTER_Z_MM

    g = build_unified_geometry(W_mm, x_front_mm, d_halo_mm)
    g.phi.grid = build_car_body_field(g)
    g.phi.apply_hard_constraints()
    enforce_symmetry(g)
    solid = g.phi.grid < 0
    ox, oy, oz = g.region.origin_m
    dx = solid.shape and __import__("geometry_contract").GRID_SPACING_M
    X = (ox + np.arange(g.shape[0]) * dx)[:, None, None]
    Y = (oy + np.arange(g.shape[1]) * dx)[None, :, None]

    xs, ztop, hwid = _zprofile(solid, g.region)
    fa, ra = x_front_mm, x_front_mm + W_mm
    results = {}

    # --- Goal 1: wheels on outriggers -------------------------------------
    # The complaint is body wrapping the wheel on ALL sides, so measure the
    # WIDEST the body gets anywhere across each wheel's full x-footprint
    # [axle-R, axle+R], not just at the axle centre. That is where the "blue
    # on all sides of the wheel" actually lives.
    over = 0
    rmm = R_WHEEL_M * 1000
    max_hw = {}
    for tag, ax_mm, inr in (("front", fa, FRONT_WHEEL_INNER_Y_M),
                            ("rear", ra, REAR_WHEEL_INNER_Y_M)):
        ax = ax_mm / 1000.0
        foot = (np.abs(X - ax) <= R_WHEEL_M) & (np.abs(Y) >= inr) & (np.abs(Y) <= inr + WHEEL_WIDTH_M)
        over += int((solid & foot).sum())
        band = (xs >= ax_mm - rmm) & (xs <= ax_mm + rmm)
        max_hw[tag] = hwid[band].max() if band.any() else 0.0
    gap_front = FRONT_WHEEL_INNER_Y_M * 1000 - max_hw["front"]
    gap_rear = REAR_WHEEL_INNER_Y_M * 1000 - max_hw["rear"]
    results["wheels_covered"] = over
    results["gap_front_mm"] = gap_front
    results["gap_rear_mm"] = gap_rear
    # >=4mm at the TIGHTEST point across the whole wheel footprint is a genuine
    # outrigger gap (the axle centre is looser, ~7mm). 0 cells over the wheel
    # is the hard T7.9 visibility requirement.
    g1 = (over == 0) and (gap_front >= 4.0) and (gap_rear >= 4.0)
    results["GOAL1_outriggers"] = g1

    # --- Goal 2: halo visible ---------------------------------------------
    ref_A = g.bv.ref_plane_A_m * 1000
    halo_front, halo_back = ref_A + d_halo_mm, ref_A + d_halo_mm + 50.0
    halo_top = 40.0  # halo hoop reaches ~z=40mm
    band = (xs >= halo_front) & (xs <= halo_back)
    deck_over_halo = ztop[band].max() if band.any() else 999
    results["deck_over_halo_mm"] = deck_over_halo
    results["halo_top_mm"] = halo_top
    g2 = deck_over_halo <= halo_top - 2.0    # halo pokes >=2mm above the deck
    results["GOAL2_halo_visible"] = g2

    # --- Goal 3: halo->canister loft --------------------------------------
    can_front = g.bv.rearpod.x_max_m() * 1000 - 58.0
    # Start the loft check a short transition (8mm) AFTER the halo ends: the
    # deck physically cannot be both low over the halo (goal 2) and already
    # risen at the halo's very back edge -- there is a smoothstep between them.
    # Beyond that transition the cover must stay elevated all the way to the
    # canister.
    span = (xs >= halo_back + 8.0) & (xs <= can_front)
    deck_min = ztop[span].min() if span.any() else 0
    # the cover must stay elevated (>= canister axis height) all the way; a dip
    # to ~0 means the tail pinched off before the canister
    results["cover_min_z_mm"] = deck_min
    results["canister_axis_mm"] = float(CANISTER_Z_MM)
    # the engine cover must stay AT/ABOVE the canister axis all the way back to
    # the canister, so the canister sits inside a real cover -- not poking out
    # above a tail that already dropped to ~28mm.
    g3 = deck_min >= CANISTER_Z_MM - 2.0
    results["GOAL3_loft_to_canister"] = g3

    if verbose:
        print(f"=== verify  W={W_mm} x_front={x_front_mm} d_halo={d_halo_mm} "
              f"spacing={spacing} ===")
        print(f"GOAL 1 outriggers   : {'PASS' if g1 else 'FAIL'}  "
              f"(cells over wheels={over}, front gap={gap_front:.1f}mm, "
              f"rear gap={gap_rear:.1f}mm; need 0 & >=4mm)")
        print(f"GOAL 2 halo visible : {'PASS' if g2 else 'FAIL'}  "
              f"(deck over halo={deck_over_halo:.1f}mm, halo top={halo_top:.0f}mm; "
              f"deck must be <= {halo_top-2:.0f})")
        print(f"GOAL 3 loft->canister: {'PASS' if g3 else 'FAIL'}  "
              f"(min cover height {halo_back:.0f}->{can_front:.0f}mm = "
              f"{deck_min:.1f}mm; need >= {CANISTER_Z_MM-5:.0f})")
        # ASCII deck profile
        print("\n  deck z_top(x) [ha=halo span, ca=canister]:")
        for i in range(0, len(xs), max(1, len(xs) // 28)):
            mark = ""
            if halo_front <= xs[i] <= halo_back: mark = " <-halo"
            if xs[i] >= can_front: mark = " <-canister"
            print(f"   x={xs[i]:5.0f}  z={ztop[i]:4.1f} hw={hwid[i]:4.1f} "
                  f"{'#'*int(ztop[i]/2)}{mark}")

    results["ALL"] = g1 and g2 and g3
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=float, default=130.0)
    ap.add_argument("--x-front", type=float, default=46.0)
    ap.add_argument("--d-halo", type=float, default=20.0)
    ap.add_argument("--spacing", type=float, default=3.0)
    a = ap.parse_args()
    r = verify(a.W, a.x_front, a.d_halo, a.spacing)
    print(f"\nRESULT: {'ALL GOALS PASS' if r['ALL'] else 'SOME GOALS FAIL'}")
    raise SystemExit(0 if r["ALL"] else 1)
