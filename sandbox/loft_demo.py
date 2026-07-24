"""
loft_demo.py -- prove the halo-canister loft is representable in the unified field.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
This hand-authors a lofted phi field and extracts it. It is a REPRESENTABILITY
proof, not an optimiser output and not a design. Nothing here is aerodynamically
motivated; the profile is a plausible STEM Racing silhouette chosen to exercise
the one thing the old four-box representation could not do: a single continuous
surface running from the cartridge chamber at the rear, forward over the halo
pocket, and down to meet the nose at Ref Plane A -- crossing what used to be the
main_body/rearpod grid boundary and the main_body/sidepod boundary.

Under the old representation this shape was not merely hard to find, it was
NOT EXPRESSIBLE: each component was marching-cubed on its own box and the four
meshes were concatenated, so no surface could cross a box boundary.

Run:
    python sandbox/loft_demo.py [--spacing 1.5] [--W 130] [--x-front 46] [--d-halo 20]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import coarse  # noqa: F401  -- sets sys.path and PART2_PATH


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """Hermite smoothstep, clamped. C1-continuous, so the loft has no creases."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def build_loft_field(geom) -> np.ndarray:
    """A swept-profile phi field: phi = signed distance-ish to a lofted solid.

    The solid is described by two silhouettes swept along x:
        z_top(x)   -- the deck line: low at Ref Plane A, rising over the halo,
                      staying high across the cartridge chamber at the rear
        y_half(x)  -- the plan line: full width at the axles, waisted between
                      them so the sidepods read as sidepods

    phi = max(z - z_top, |y| - y_half, z_floor - z), which is negative only
    inside all three, i.e. inside the lofted body.
    """
    from geometry_contract import GRID_SPACING_M

    nx, ny, nz = geom.shape
    ox, oy, oz = geom.region.origin_m
    dx = GRID_SPACING_M

    xs = ox + np.arange(nx) * dx
    ys = oy + np.arange(ny) * dx
    zs = oz + np.arange(nz) * dx

    ref_A = geom.landmarks["ref_plane_A_m"]
    front_axle = geom.landmarks["front_axle_m"]
    rear_axle = geom.landmarks["rear_axle_m"]
    rear_face = geom.landmarks["rear_face_m"]

    # ── deck line ──────────────────────────────────────────────────────────
    # Meets the nose at Ref Plane A at T8.5.1's 25 mm ceiling, ramps up over
    # the front axle, plateaus at 60 mm across the halo and canister.
    z_low, z_high = 0.026, 0.060
    ramp = _smoothstep((xs - ref_A) / max(front_axle + 0.020 - ref_A, 1e-9))
    z_top = z_low + (z_high - z_low) * ramp
    # Taper the last few mm down to the rear face so the tail isn't a cliff.
    tail = _smoothstep((rear_face - xs) / 0.018)
    z_top = np.minimum(z_top, z_low + (z_high - z_low) * tail + z_low)

    # ── plan line ──────────────────────────────────────────────────────────
    # Full body half-width at both axles, waisted to a slimmer midsection.
    y_wide, y_waist = 0.0330, 0.0250
    mid = 0.5 * (front_axle + rear_axle)
    half_span = max(0.5 * (rear_axle - front_axle), 1e-9)
    waist = 1.0 - _smoothstep(np.abs(xs - mid) / half_span)
    y_half = y_wide - (y_wide - y_waist) * waist
    # Nose region: T8.5.1 caps the half-width at 15 mm forward of Ref Plane A.
    y_half = np.where(xs < ref_A, 0.015, y_half)
    z_top = np.where(xs < ref_A, 0.025, z_top)

    X_ztop = z_top[:, None, None]
    X_yhalf = y_half[:, None, None]
    absY = np.abs(ys)[None, :, None]
    Z = zs[None, None, :]

    phi = np.maximum(Z - X_ztop, absY - X_yhalf)
    phi = np.maximum(phi, oz - Z)
    return phi.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--W", type=float, default=130.0)
    ap.add_argument("--x-front", type=float, default=46.0)
    ap.add_argument("--d-halo", type=float, default=20.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    coarse.use_spacing(args.spacing)

    from unified_phi import (
        MODEL_BLOCK_LENGTH_MM, build_unified_geometry, compute_mass_com,
        enforce_symmetry, extract_unified_surface,
    )

    geom = build_unified_geometry(args.W, args.x_front, args.d_halo)
    geom.phi.grid = build_loft_field(geom)
    geom.phi.apply_hard_constraints()
    enforce_symmetry(geom)

    print(f"=== loft demo  W={args.W} x_front={args.x_front} "
          f"d_halo={args.d_halo} spacing={args.spacing} ===")
    print(f"grid {geom.shape} = {int(np.prod(geom.shape)):,} cells")

    # Full six-stage extraction with per-label manufacturing gates.
    # allow_inaccessible mirrors explore.py's waiver: the area is REPORTED and
    # carried as a manufacturing penalty, per 01_generative_geometry.md's
    # "assign manufacturing penalty, continue" rule for large failures.
    mesh, report = extract_unified_surface(geom, allow_inaccessible=True)
    lo, hi = mesh.bounds * 1000

    print()
    print(f"  connected bodies : {report['connected_bodies']}"
          f"   {'<-- ONE PIECE' if report['connected_bodies'] == 1 else '<-- FRAGMENTED'}")
    print(f"  watertight       : {report['watertight']}")
    print(f"  winding OK       : {mesh.is_winding_consistent}")
    print(f"  body L x W x H   : {hi[0]-lo[0]:.1f} x {hi[1]-lo[1]:.1f} "
          f"x {hi[2]-lo[2]:.1f} mm  (bare machined+printed body)")

    # T3.4 and T3.5 measure the ASSEMBLED car (regs p19: "maximum assembled car
    # width/height"), not the bare body. The widest points are the front wheels;
    # the body never sets the width. Comparing body width against T3.4 is a
    # category error -- it is what made withdrawn sandbox finding 8 wrong.
    from geometry_contract import (
        FRONT_WHEEL_INNER_Y_MM, REAR_WHEEL_INNER_Y_MM, WHEEL_WIDTH_MM,
    )
    assembled_w = 2 * max(FRONT_WHEEL_INNER_Y_MM, REAR_WHEEL_INNER_Y_MM) \
        + 2 * WHEEL_WIDTH_MM
    assembled_h = max(hi[2] - lo[2], 0.0)   # halo/wing not in this mesh yet
    print(f"  assembled width  : {assembled_w:.1f} mm  T3.4 65-85  "
          f"{'ok' if 65 <= assembled_w <= 85 else 'FAIL'}   (set by the wheels)")
    print(f"  assembled height : >={assembled_h:.1f} mm  T3.5 <=65   "
          f"{'ok' if assembled_h <= 65 else 'FAIL'}"
          f"   (lower bound -- halo/wing surfaces not in this mesh)")
    ml = geom.machined_length_mm()
    print(f"  milled length    : {ml:.1f} mm  (T3.1.2 block "
          f"{MODEL_BLOCK_LENGTH_MM:.0f}) {'ok' if ml <= MODEL_BLOCK_LENGTH_MM else 'FAIL'}")

    print()
    mc = compute_mass_com(geom)
    for c in mc:
        print(f"  {c.name:<10} {c.mass_kg*1000:7.3f} g   "
              f"com=({c.com_x_m*1000:6.1f}, {c.com_y_m*1000:+.3f}, "
              f"{c.com_z_m*1000:5.1f}) mm")
    total = sum(c.mass_kg for c in mc)
    print(f"  {'MACHINED':<10} {total*1000:7.3f} g   "
          f"(T3.6 min 48 g for the WHOLE car incl. fixed hardware)")

    # Does the deck actually cross the old grid seams?
    print()
    solid = geom.phi.grid < 0
    seam_rear = geom.bv.rearpod.origin_m[0]
    from geometry_contract import GRID_SPACING_M
    i_seam = int(round((seam_rear - geom.region.origin_m[0]) / GRID_SPACING_M))
    col = solid[i_seam - 1:i_seam + 2].any(axis=(1, 2))
    print(f"  material spans the old main_body|rearpod seam at x="
          f"{seam_rear*1000:.1f} mm : {bool(col.all())}")

    print()
    print("  tool-accessibility (waived, carried as a manufacturing penalty):")
    for name, area in sorted(report["inaccessible_area_mm2"].items()):
        print(f"    {name:<10} {area:9.2f} mm^2 unreachable")
    print("    nose        (3D printed -- no tool-direction constraint)")

    out = Path(args.out) if args.out else Path(__file__).parent / "out" / "loft_car.stl"
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out), file_type="stl_ascii")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
