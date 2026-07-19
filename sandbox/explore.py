"""
explore.py -- build, inspect and visualise one (W, x_front, d_halo) candidate
without any CFD.

Nothing in here touches OpenFOAM. The whole Level 3 path (phi grids ->
marching cubes -> repair -> gates -> STL) plus mass/COM is pure
numpy/scipy/skimage/trimesh and runs in seconds at coarse spacing.

Usage
-----
    python sandbox/explore.py                          # defaults, phi plots only
    python sandbox/explore.py --W 130 --x-front 75 --d-halo 60
    python sandbox/explore.py --spacing 1.0 --gates    # also extract + STL
    python sandbox/explore.py --init random --gates

Output goes to sandbox/out/<tag>/ : phi slice PNGs, .npy snapshots, and
(with --gates) per-component and assembled STLs.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np

import coarse  # noqa: F401  -- sets sys.path and PART2_PATH as a side effect


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--W", type=float, default=130.0, help="wheelbase mm [120,140]")
    p.add_argument("--x-front", type=float, default=75.0, help="front axle mm from nose tip")
    p.add_argument("--d-halo", type=float, default=60.0, help="halo offset mm from Ref Plane A")
    p.add_argument("--spacing", type=float, default=1.5,
                   help="grid spacing mm (spec is 0.3; 1.5 is the safe explore default)")
    p.add_argument("--init", default="sphere", choices=("sphere", "slab", "random"),
                   help="initial phi field shape")
    p.add_argument("--gates", action="store_true",
                   help="also run surface extraction + quality gates + STL export")
    p.add_argument("--max-cells", type=int, default=coarse.MAX_TOTAL_CELLS,
                   help="refuse to build grids totalling more than this many cells")
    p.add_argument("--out", default=None, help="output directory (default sandbox/out/<tag>)")
    p.add_argument("--no-cargo", action="store_true",
                   help="drop the T4.2 virtual cargo requirement (NOT competition-legal)")
    p.add_argument("--allow-inaccessible", action="store_true",
                   help="report tool-accessibility violations as a penalty instead of "
                        "failing the component (NOT competition-legal)")
    p.add_argument("--with-hardware", action="store_true",
                   help="also emit halo, wheels, CO2 canister and rear wing as solids "
                        "(Part 1 models these as voids + mass only -- see hardware.py)")
    p.add_argument("--wheel-width", type=float, default=None,
                   help="wheel width mm (ASSUMPTION -- not defined anywhere in Part 1)")
    return p.parse_args()


def describe_grids(phi_grids, bv, counts) -> None:
    """Print the geometry the optimizer is actually allowed to work in."""
    print("\n== Bounding volumes ==")
    print(f"  Ref Plane A  {bv.ref_plane_A_m*1000:7.2f} mm")
    print(f"  Ref Plane B  {bv.ref_plane_B_m*1000:7.2f} mm")
    print(f"  front axle   {bv.x_front_m*1000:7.2f} mm")
    print(f"  rear axle    {(bv.x_front_m + bv.W_m)*1000:7.2f} mm")

    print(f"\n{'component':<11s} {'shape':>16s} {'cells':>10s} "
          f"{'x range mm':>16s} {'solid %':>8s} {'hard-solid':>11s} {'hard-air':>10s}")
    for name, phi in phi_grids.items():
        region = bv.get(name)
        ox, oy, oz = region.origin_m
        nx, ny, nz = region.shape
        dx = coarse.__dict__["_last_spacing_m"]
        solid_frac = float((phi.grid < 0).mean()) * 100.0
        print(f"{name:<11s} {str(tuple(region.shape)):>16s} {counts[name]:>10,d} "
              f"{ox*1000:7.1f}-{(ox + nx*dx)*1000:6.1f} "
              f"{solid_frac:7.1f}% "
              f"{int(phi.hard_mask_solid.sum()):>11,d} "
              f"{int(phi.hard_mask_air.sum()):>10,d}")


def plot_phi(phi_grids, bv, out_dir: Path, spacing_m: float) -> None:
    """Render mid-plane phi slices for each component.

    Two slices per component: a plan view (mid-z) and a profile view (mid-y).
    The phi=0 contour is the actual surface marching cubes will extract, so
    this is the most direct way to see what the level set is doing. Hard-
    constrained cells are hatched -- those are the ones the optimizer is
    forbidden to move.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, phi in phi_grids.items():
        grid = phi.grid
        nx, ny, nz = grid.shape
        origin = bv.get(name).origin_m

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        views = [
            ("plan (x-y), mid-z", grid[:, :, nz // 2],
             phi.hard_mask_solid[:, :, nz // 2], phi.hard_mask_air[:, :, nz // 2],
             (origin[0], origin[0] + nx * spacing_m, origin[1], origin[1] + ny * spacing_m),
             "x [mm]", "y [mm]"),
            ("profile (x-z), mid-y", grid[:, ny // 2, :],
             phi.hard_mask_solid[:, ny // 2, :], phi.hard_mask_air[:, ny // 2, :],
             (origin[0], origin[0] + nx * spacing_m, origin[2], origin[2] + nz * spacing_m),
             "x [mm]", "z [mm]"),
        ]

        for ax, (title, sl, hs, ha, extent, xlabel, ylabel) in zip(axes, views):
            extent_mm = [e * 1000 for e in extent]
            vmax = float(np.abs(sl).max()) or 1.0
            im = ax.imshow(sl.T, origin="lower", extent=extent_mm, aspect="auto",
                           cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            # phi = 0 is the extracted surface.
            ax.contour(sl.T, levels=[0.0], colors="k", linewidths=1.6,
                       extent=extent_mm, origin="lower")
            # Hard constraints: cells the optimizer may never change.
            for mask, colour in ((hs, "#00b050"), (ha, "#ff8c00")):
                if mask.any():
                    ax.contour(mask.T.astype(float), levels=[0.5], colors=colour,
                               linewidths=0.9, extent=extent_mm, origin="lower")
            ax.set_title(f"{name} -- {title}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            fig.colorbar(im, ax=ax, label="phi [m]")

        fig.suptitle(f"{name}: blue = solid (phi<0), red = air (phi>0), "
                     f"black = surface, green = forced solid, orange = forced air")
        fig.tight_layout()
        path = out_dir / f"phi_{name}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"  wrote {path.name}")


def report_mass_com(phi_grids) -> None:
    """Mass and COM straight from the phi grids -- no mesh, no CFD."""
    from mass_com_calculator import compute_all_machined_components

    try:
        components = compute_all_machined_components(
            phi_grids["nose"], phi_grids["sidepod"],
            phi_grids["rearpod"], phi_grids["main_body"],
        )
    except Exception as exc:
        print(f"  mass/COM failed: {type(exc).__name__}: {exc}")
        return

    print(f"\n{'component':<11s} {'mass g':>9s} {'com x mm':>10s} "
          f"{'com y mm':>10s} {'com z mm':>10s}")
    total = 0.0
    for c in components:
        total += c.mass_kg
        print(f"{c.name:<11s} {c.mass_kg*1000:9.3f} {c.com_x_m*1000:10.2f} "
              f"{c.com_y_m*1000:10.2f} {c.com_z_m*1000:10.2f}")
    print(f"{'MACHINED':<11s} {total*1000:9.3f}   (T3.6 min total car mass is 48 g, "
          f"fixed hardware not included here)")


def run_gates(phi_grids, out_dir: Path, tag: str, accessibility_penalties=None) -> None:
    """Extract each component separately, then assemble.

    Deliberately NOT calling quality_gates.run_quality_gates in one shot: it
    extracts all four components inside a single try block, so one failure
    hides the status of the rest. Doing them one at a time tells you exactly
    which component is broken and how long each takes.
    """
    from surface_extraction import extract_surface
    from stl_assembler import assemble_stl

    print("\n== Surface extraction (per component) ==")
    meshes = {}
    for name, phi in phi_grids.items():
        t0 = time.perf_counter()
        try:
            mesh = extract_surface(phi)
            meshes[name] = mesh
            mesh.export(out_dir / f"{name}.stl")
            print(f"  {name:<11s} OK    {time.perf_counter()-t0:5.1f}s  "
                  f"{len(mesh.vertices):>7,d} verts  {len(mesh.faces):>7,d} faces  "
                  f"watertight={str(mesh.is_watertight):<5s} "
                  f"vol={mesh.volume*1e6:8.2f} cm3")
        except Exception as exc:
            print(f"  {name:<11s} FAIL  {time.perf_counter()-t0:5.1f}s  "
                  f"{type(exc).__name__}: {exc}")

    if accessibility_penalties:
        print("\n  Tool-accessibility violations (waived by --allow-inaccessible,")
        print("  carried as a manufacturing penalty per the spec's large-failure rule):")
        for component, area_mm2 in accessibility_penalties:
            print(f"    {component:<11s} {area_mm2:8.2f} mm^2 unreachable")

    if len(meshes) != 4:
        print(f"\n  Skipping STL assembly: only {len(meshes)}/4 components extracted.")
        return None

    try:
        full_path, half_path = assemble_stl(meshes, tag, str(out_dir))
        print(f"\n  full car  -> {full_path}")
        print(f"  half car  -> {half_path}")
        return full_path
    except Exception as exc:
        print(f"\n  assemble_stl FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()

    return None


def export_hardware(args, bv, out_dir: Path, tag: str, body_stl_path) -> None:
    """Emit the fixed hardware as solids, and a body+hardware STL for CFD.

    Exported as separate STLs as well as a combined file: snappyHexMesh takes
    multiple surface geometries happily, and keeping them separate lets you
    assign different refinement levels to the wheels and halo (which need it)
    than to the body.
    """
    import hardware
    import trimesh

    wheel_width = args.wheel_width if args.wheel_width is not None else hardware.WHEEL_WIDTH_MM

    print("\n== Fixed hardware solids ==")
    print(f"  wheel width {wheel_width} mm  <- ASSUMPTION, not defined anywhere in Part 1")

    parts = hardware.build_all(args.W, args.x_front, args.d_halo, bv,
                               wheel_width_mm=wheel_width)
    for name, mesh in sorted(parts.items()):
        mesh.export(out_dir / f"hw_{name}.stl")
        centre = mesh.bounds.mean(axis=0) * 1000
        print(f"  {name:<20s} {len(mesh.faces):>6,d} faces  "
              f"watertight={str(mesh.is_watertight):<5s} "
              f"vol={mesh.volume*1e6:7.3f} cm3  "
              f"centre=({centre[0]:6.1f},{centre[1]:6.1f},{centre[2]:5.1f}) mm")

    print("\n  Mass Part 1 already counts for fixed hardware (surfaces were the "
          "only thing missing):")
    for name, mass_kg in hardware.hardware_mass_summary(
            args.W, args.x_front, args.d_halo, bv):
        print(f"    {name:<16s} {mass_kg*1000:6.2f} g")

    if body_stl_path:
        # Load the ASSEMBLED car, not the per-component dict: assemble_stl
        # mirrors the right-half sidepod internally, so the raw component
        # meshes are missing the left one.
        body = trimesh.load(body_stl_path)
        # Concatenated, NOT boolean-unioned: the hardware interpenetrates the
        # body (that is what the void masks carve out for), so the result is
        # multi-body and will not be watertight. That is correct for a
        # snappyHexMesh multi-surface setup and WRONG for anything that
        # demands a single closed volume -- including Part 2's current
        # run_half_car_cfd watertight assertion.
        combined = trimesh.util.concatenate([body] + list(parts.values()))
        path = out_dir / f"car_{tag}_with_hardware.stl"
        combined.export(path)
        extents = combined.extents * 1000
        print(f"\n  body + hardware -> {path.name}")
        print(f"    {len(combined.faces):,d} faces  "
              f"L={extents[0]:.1f} W={extents[1]:.1f} H={extents[2]:.1f} mm  "
              f"watertight={combined.is_watertight} (expected False -- multi-body)")


def main() -> int:
    args = parse_args()

    coarse.use_spacing(args.spacing)
    spacing_m = args.spacing / 1000.0
    coarse.__dict__["_last_spacing_m"] = spacing_m

    waivers = []
    accessibility_penalties = None
    if args.no_cargo:
        coarse.disable_virtual_cargo()
        waivers.append("T4.2 virtual cargo")
    if args.allow_inaccessible:
        accessibility_penalties = coarse.relax_accessibility()
        waivers.append("tool accessibility")

    tag = f"W{args.W:.0f}_xf{args.x_front:.0f}_dh{args.d_halo:.0f}_s{args.spacing:g}"
    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "out" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== Candidate {tag} ==")
    print(f"  W={args.W} mm  x_front={args.x_front} mm  d_halo={args.d_halo} mm")
    print(f"  spacing={args.spacing} mm (spec value is 0.3 mm -- see coarse.py caveat)")
    print(f"  init={args.init}   out={out_dir}")
    if waivers:
        print(f"  WAIVED: {', '.join(waivers)} -- output is NOT competition-legal geometry")

    from bounding_volumes import compute_bounding_volumes, default_rule_envelope
    from phi_grid_factory import _default_forbidden_cylinders
    from geometry_contract import WHEEL_X_CLEARANCE_HALF_WIDTH_MM

    # Size preflight BEFORE building anything, so an oversized request fails
    # instantly instead of during marching cubes.
    front_cyl, rear_cyl = _default_forbidden_cylinders(args.W, args.x_front, WHEEL_X_CLEARANCE_HALF_WIDTH_MM)
    try:
        bv_probe = compute_bounding_volumes(
            args.W, args.x_front, args.d_halo,
            front_cyl, rear_cyl, default_rule_envelope(),
            wheel_x_half_width_mm=WHEEL_X_CLEARANCE_HALF_WIDTH_MM,
        )
    except (ValueError, NotImplementedError) as exc:
        print(f"\nGEOMETRY REJECTED at bounding-volume stage: "
              f"{type(exc).__name__}: {exc}")
        return 1

    try:
        counts = coarse.check_size(bv_probe, args.max_cells)
    except MemoryError as exc:
        print(f"\nREFUSING TO BUILD:\n{exc}")
        return 1

    from phi_grid_factory import build_phi_grids_for_candidate

    t0 = time.perf_counter()
    try:
        phi_grids, bv = build_phi_grids_for_candidate(
            args.W, args.x_front, args.d_halo, init_mode=args.init,
        )
    except (ValueError, NotImplementedError) as exc:
        print(f"\nGEOMETRY REJECTED building phi grids: {type(exc).__name__}: {exc}")
        return 1
    print(f"\nBuilt 4 phi grids in {time.perf_counter()-t0:.1f}s "
          f"({sum(counts.values()):,d} cells total)")

    describe_grids(phi_grids, bv, counts)
    report_mass_com(phi_grids)

    print("\n== Phi field plots ==")
    plot_phi(phi_grids, bv, out_dir, spacing_m)

    # PhiGrid.save does not create its output directory -- we made it above.
    for name, phi in phi_grids.items():
        phi.save(tag, str(out_dir))

    body_stl_path = None
    if args.gates:
        body_stl_path = run_gates(phi_grids, out_dir, tag, accessibility_penalties)

    if args.with_hardware:
        export_hardware(args, bv, out_dir, tag, body_stl_path)

    print(f"\nDone. Everything in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
