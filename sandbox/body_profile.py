"""
body_profile.py -- a parametric STEM Racing (F1 in Schools) dragster body.

Same contract as sandbox/loft_demo.build_loft_field: hand-author a lofted phi
field over the unified envelope and return it (float32, shape == geom.shape).

The body is a single continuous loft described by two swept silhouettes plus a
top "roundness" that pulls the deck corners in so the car reads as a rounded
hull rather than an extruded slab:

    z_top(x)   -- the deck line: a low pointed nose at Ref Plane A, rising over
                  the front axle to a raised cockpit/halo plateau, then a smooth
                  aerodynamic taper down toward the rear face.
    y_half(x)  -- the plan line: full body width at both axles, waisted between
                  them so the sidepods read as sidepods, narrowing to a pointed
                  nose forward and a slim tail aft.

phi = max(  shell(y, z ; x),  z_floor - z )  which is negative only inside the
lofted hull. shell() blends a rounded top so there are no hard deck corners.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import coarse  # noqa: F401  -- sets sys.path and PART2_PATH


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Hermite smoothstep, clamped. C1-continuous -> no creases in the loft."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smooth_curve(xs: np.ndarray, xp, fp) -> np.ndarray:
    """C1 piecewise interpolation through (xp, fp) using smoothstep segments.

    smoothstep has zero slope at both ends of every segment, so the assembled
    curve is C1 at each knot (matching the loft's no-crease requirement) while
    still passing exactly through the control points.
    """
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    out = np.full_like(xs, fp[-1], dtype=float)  # default = last value (x >= xp[-1])
    out[xs <= xp[0]] = fp[0]
    # Each segment owns [x0, x1); the closing knot xp[-1] is covered by the
    # default fill above. Including the left endpoint (>=) means points landing
    # exactly on an interior knot get that knot's value (smoothstep(0)=0), so no
    # sample is ever left uninitialised -- the previous strict-inequality version
    # left on-knot samples as np.empty garbage.
    for i in range(len(xp) - 1):
        x0, x1 = xp[i], xp[i + 1]
        m = (xs >= x0) & (xs < x1)
        if not np.any(m):
            continue
        t = _smoothstep((xs[m] - x0) / (x1 - x0))
        out[m] = fp[i] + (fp[i + 1] - fp[i]) * t
    return out


def build_car_body_field(geom) -> np.ndarray:
    from geometry_contract import GRID_SPACING_M

    nx, ny, nz = geom.shape
    ox, oy, oz = geom.region.origin_m
    dx = GRID_SPACING_M

    xs = ox + np.arange(nx) * dx
    ys = oy + np.arange(ny) * dx
    zs = oz + np.arange(nz) * dx

    ref_A = geom.landmarks["ref_plane_A_m"]        # 0.030
    front_axle = geom.landmarks["front_axle_m"]     # 0.046
    rear_axle = geom.landmarks["rear_axle_m"]       # 0.176
    rear_face = geom.landmarks["rear_face_m"]       # 0.233

    xmm = xs * 1000.0

    # ── Landmark x-positions in mm ───────────────────────────────────────────
    # ALL control knots below are placed RELATIVE to these, so the body tracks
    # any (W, x_front, d_halo). An earlier version hardcoded most knot x-values
    # (116, 142, 176, ...), which only worked for the default config: with a
    # larger d_halo the halo span slides aft, the fixed cover-rise knot landed
    # INSIDE the halo span, and the rising deck buried the halo (goal-2 fail).
    rA_mm = ref_A * 1000.0
    fa_mm = front_axle * 1000.0
    ra_mm = rear_axle * 1000.0
    rf_mm = rear_face * 1000.0
    wb_mm = ra_mm - fa_mm                                    # wheelbase
    hf_mm = rA_mm + geom.d_halo_mm                           # halo front
    hb_mm = hf_mm + 50.0                                     # halo back (pocket = 50)
    x_halo_back = hb_mm

    def _knots(pairs):
        """Sort (x,z) knots by x and enforce strictly increasing x (a later
        knot that collides with or precedes an earlier one is nudged +0.1mm),
        so extreme configs can never hand _smooth_curve a non-monotone x."""
        pairs = sorted(pairs, key=lambda p: p[0])
        xs_, zs_ = [], []
        for x, z in pairs:
            if xs_ and x <= xs_[-1]:
                x = xs_[-1] + 0.1
            xs_.append(x); zs_.append(z)
        return xs_, zs_

    # ── deck line z_top(x) [mm] ──────────────────────────────────────────────
    # The deck is deliberately staged into three regions so the car reads as a
    # real F1 dragster instead of an extruded slab:
    #   nose (x<30)          : low pointed tip, capped <=25 mm below.
    #   cockpit/halo (30..100): a LOW deck (~28..32 mm) so the separately-placed
    #                           halo hoop (which reaches z~40) pokes clearly
    #                           above the body and stays visible (T4.4).
    #   engine cover (100..) : from just behind the halo (x_halo_back~100) the
    #                           deck RISES into a headrest/engine-cover spine
    #                           peaking ~45 mm, then stays elevated (>=35 mm) and
    #                           lofts continuously BACK toward the CO2 canister
    #                           (axis z=35, x~177..235) so the canister emerges
    #                           from the tail of a proper tapered cover rather
    #                           than from a pointed tail that has vanished.
    # The final knot drops the deck BELOW the floor so the cross-section closes
    # off inside the grid (~231 mm, grid edge ~234): _repair_mesh runs 10 Taubin
    # iterations that push a blunt rear cap outward, so a body reaching the rear
    # grid edge ends up with vertices out of bounds.
    # Deck knots, all landmark-relative:
    #   nose tip -> nose top (at Ref A) -> LOW across the halo span [hf,hb] so
    #   the hoop stays visible -> steep rise just behind the halo to the
    #   engine-cover peak -> stay elevated to the rear axle -> loft down to meet
    #   the canister -> close the cap below the floor at the rear grid edge.
    z_ctrl_x, z_ctrl_z = _knots([
        (0.0, 6.0), (rA_mm, 22.0),
        (hf_mm, 29.0), (hb_mm, 31.0),          # LOW deck across the halo
        (hb_mm + 14.0, 45.0),                  # cover peak just behind the halo
        (ra_mm, 41.0),                         # still elevated at the rear axle
        (rf_mm - 45.0, 40.0), (rf_mm - 18.0, 36.0),   # loft toward the canister
        (rf_mm - 4.0, 28.0), (rf_mm, -9.0),    # taper + close the cap
    ])
    z_top = _smooth_curve(xmm, z_ctrl_x, z_ctrl_z) / 1000.0

    # ── plan line y_half(x) [mm] ─────────────────────────────────────────────
    # SLIM central pod. It pinches IN to a narrow value AT each axle station so
    # the wheels stand out on their own outriggers with a clear air gap:
    #   front axle (x=46): <=13 mm  (front wheel inner face y=19.2 -> gap>=6)
    #   rear  axle (x=176): <=10 mm (rear wheel inner face y=16.2 -> gap>=6)
    # Between the axles a modest sidepod swells to ~20 mm (max <=22). Aft of the
    # rear axle it widens gently back to a ~12..13 mm engine-cover spine that
    # wraps the ~24 mm-dia canister, then tapers to a slim tail.
    # Plan knots, all landmark-relative: pinch to a slim width AT each axle so
    # the wheels stand clear on outriggers, a modest sidepod swell between the
    # axles, then a ~12mm engine-cover spine wrapping the canister at the tail.
    y_ctrl_x, y_ctrl_y = _knots([
        (0.0, 3.0), (rA_mm, 12.0),
        (fa_mm, 11.0),                                 # pinch at front axle (<=13)
        (fa_mm + 0.35 * wb_mm, 20.0),                  # sidepod swell (<=22)
        (fa_mm + 0.68 * wb_mm, 15.0),
        (ra_mm - 16.0, 10.5), (ra_mm, 9.0), (ra_mm + 16.0, 10.5),  # pinch at rear axle (<=10)
        (rf_mm - 13.0, 12.0),                          # spine wraps the canister
        (rf_mm, 5.0),
    ])
    y_half = _smooth_curve(xmm, y_ctrl_x, y_ctrl_y) / 1000.0

    # Nose caps forward of Ref Plane A (T8.5.1): <=25 mm tall, <=15 mm half-wide.
    nose = xs < ref_A
    z_top = np.where(nose, np.minimum(z_top, 0.025), z_top)
    y_half = np.where(nose, np.minimum(y_half, 0.015), y_half)

    # ── loft assembly ────────────────────────────────────────────────────────
    # Cross-section is a super-ellipse: flat central deck, near-vertical sides,
    # rounded shoulders -- a car hull, not an extruded slab. It sits flat on the
    # floor (zrel starts at 0) so there is no need for a separate floor cut.
    zrel = (zs - oz)[None, None, :]
    absY = np.abs(ys)[None, :, None]
    ZT = np.maximum(z_top - oz, 1e-4)[:, None, None]
    YH = np.maximum(y_half, 1e-4)[:, None, None]

    n = 4.0  # super-ellipse exponent: higher = boxier, lower = more elliptical
    d = ((absY / YH) ** n + (zrel / ZT) ** n) ** (1.0 / n)
    phi = (d - 1.0) * np.minimum(YH, ZT)  # rescale to ~metric so MC is well-fed

    return phi.astype(np.float32)


def main() -> None:
    coarse.use_spacing(2.0)

    from unified_phi import (
        build_unified_geometry, enforce_symmetry, extract_unified_surface,
    )

    geom = build_unified_geometry(130.0, 46.0, 20.0)
    geom.phi.grid = build_car_body_field(geom)
    geom.phi.apply_hard_constraints()
    enforce_symmetry(geom)

    mesh, report = extract_unified_surface(geom, allow_inaccessible=True)
    lo, hi = mesh.bounds * 1000

    print(f"connected bodies : {report['connected_bodies']}")
    print(f"watertight       : {report['watertight']}")
    print(f"body L x W x H    : {hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} x {hi[2]-lo[2]:.1f} mm")

    out = Path(__file__).parent / "out" / "body_profile.stl"
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out), file_type="stl_ascii")
    print(f"wrote {out}")

    # Render a 4-view PNG next to the STL so the silhouette is inspectable.
    import subprocess
    import sys
    render = Path(__file__).parent / "render_stl.py"
    subprocess.run([sys.executable, str(render), str(out)], check=False)


if __name__ == "__main__":
    main()
