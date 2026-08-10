"""
hardware_assembly.py -- place the REAL downloaded hardware CAD STLs into the
project coordinate frame and combine them with a generated body into one STL.

WHY THIS EXISTS
---------------
hardware_cad/ holds the actual competition parts (co2_canister, front/rear
wheel + support, halo_helmet). Part 1 only ever modelled these as VOID MASKS
(holes carved in the body) plus a mass/COM spec -- it never put their surfaces
in the STL. sandbox/hardware.py builds crude PRIMITIVE stand-ins (cylinders,
boxes). This module uses the real meshes instead.

The downloaded parts sit in their own export frame, which is NOT the project's:
  - x is REVERSED: front axle at x~173, rear axle at x~35 (project: nose x=0,
    front axle at small x, rear axle at large x).
  - several parts are HALF meshes split at y~0 (canister, halo) -- one side only.
  - wheels are single -y-side instances to be mirrored for the +y side.

Rather than trust that frame, every part is treated as a RIGID BODY: we read
only its SHAPE, recentre it, mirror halves into wholes, and then place it where
the project's own math says it goes (axle positions, halo pocket span, cartridge
bore). So the surfaces land exactly where the void masks already are.

LOW RAM
-------
The high-poly parts (halo 131k, canister 85k, supports 60-80k verts) are
decimated before assembly. The body is generated at coarse spacing. Pass
--full to skip decimation.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

import coarse  # noqa: F401 -- sets sys.path + PART2_PATH

_HW_DIR = Path(__file__).resolve().parent.parent / "hardware_cad"


# ── rigid-body helpers ──────────────────────────────────────────────────────

def _load(name: str):
    """Load a hardware CAD part, convert mm -> m, and flip into the project frame.

    Two frame conversions:
      - SCALE mm -> m. The STLs are in millimetres (a wheel bbox is ~28 units);
        the project works in metres. Without this a 28 mm wheel is placed as a
        28 m wheel and the assembly bbox blows up to ~90 m.
      - X-MIRROR. The CAD export frame has FRONT at high x (front wheel at
        x~173, rear at x~35); the project frame has front at LOW x (nose at 0).
        The whole frame is x-reversed, so every part must be mirrored in x to
        face the right way. This is a no-op for the x-symmetric wheels, but it
        is what makes the halo/helmet face FORWARD instead of backward, and
        orients the support brackets correctly.
    """
    import trimesh
    mesh = trimesh.load(str(_HW_DIR / name), process=True)
    mesh.apply_scale(0.001)
    mesh.apply_transform(np.diag([-1.0, 1.0, 1.0, 1.0]))
    mesh.fix_normals()
    return mesh


def _mirror_y(mesh):
    """Return mesh reflected across y=0, normals fixed."""
    m = mesh.copy()
    m.apply_transform(np.diag([1.0, -1.0, 1.0, 1.0]))
    m.fix_normals()
    return m


def _complete_half(mesh, tol_mm: float = 1.0):
    """If `mesh` is only on one side of y=0, mirror-and-union into a whole.

    The canister and halo are exported as half meshes split at the centreline.
    Detected by: (almost) all of the mesh lies on one side of y=0.
    """
    import trimesh

    lo, hi = mesh.bounds
    tol = tol_mm / 1000.0
    one_sided = hi[1] <= tol or lo[1] >= -tol
    if not one_sided:
        return mesh
    full = trimesh.util.concatenate([mesh, _mirror_y(mesh)])
    full.merge_vertices()
    return full


def _decimate(mesh, target_faces: int):
    """Reduce face count for assembly/render.

    NOT the source of the streaks on the canister and halo in assembly renders.
    Those are sliver triangles in the SUPPLIED CAD, present before any
    processing here: measured per-face on the shipped files, co2_canister has
    169 triangles with an edge over 10 mm (longest 59.8 mm, median edge
    0.339 mm) and halo_helmet has 332 (longest 38.4 mm, median 0.177 mm). Only
    26% of the halo's and none of the canister's lie on the y=0 section plane,
    so they are not just the half-mesh cap either. front_wheel by contrast has
    none. Decimation neither creates nor worsens them, and a guard against it
    was removed once measured. Cosmetic only: hardware reaches the solver as
    void masks, never as surfaces.
    """
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except Exception:
        try:  # older trimesh signature
            return mesh.simplify_quadric_decimation(target_faces)
        except Exception:
            return mesh


def _place_by_center(mesh, center_xyz_m, keep_z_floor: bool = False):
    """Translate so the mesh's bbox centre lands at center_xyz_m.

    keep_z_floor: instead of centring z, put the bbox BOTTOM at center z
    (wheels/hardware that rest on a surface).
    """
    m = mesh.copy()
    lo, hi = m.bounds
    c = (lo + hi) / 2.0
    cx, cy, cz = center_xyz_m
    dz = (cz - lo[2]) if keep_z_floor else (cz - c[2])
    m.apply_translation([cx - c[0], cy - c[1], dz])
    return m


# ── part placement in project coordinates ───────────────────────────────────


def _spin_axis_to_y(mesh):
    """Rotate a wheel so it spins about y, whatever axis it was exported on.

    A wheel is round in the x-z plane with its axle along y. The v1 parts were
    exported that way ([28.2, 17.2, 28.2]); the v2 parts are not -- they measure
    [13.25, 28.24, 28.25], i.e. round in y-z with the axle along x, a quarter
    turn out. Placed unrotated they sit like discs facing down the car.

    Detected rather than hardcoded: the THIN axis of a wheel is its width, and
    that axis must end up as y. So if the thin axis is x, turn a quarter about
    z; if it is z, turn a quarter about x; if it is already y, do nothing. A
    future re-export in either convention then lands correctly without anyone
    having to notice.
    """
    import numpy as _np
    import trimesh

    m = mesh.copy()
    ext = m.bounds[1] - m.bounds[0]
    thin = int(_np.argmin(ext))
    if thin == 1:
        return m
    if thin == 0:
        R = trimesh.transformations.rotation_matrix(_np.pi / 2, [0, 0, 1])
    else:
        R = trimesh.transformations.rotation_matrix(_np.pi / 2, [1, 0, 0])
    m.apply_transform(R)
    m.fix_normals()
    return m


def place_wheels(W_mm: float, x_front_mm: float, hw_inputs) -> dict:
    """Four real wheel meshes at the project axle positions.

    The wheel part is round in x-z with its axle along y (correct for a wheel at
    any axle x). We recentre it and drop four instances: front/rear x axle
    lines, +/-y sides. Inner-face y and wheel width come straight from the
    part's own geometry so the render matches the physical wheel, not an
    assumed width.
    """
    from geometry_contract import (
        FRONT_WHEEL_INNER_Y_M, REAR_WHEEL_INNER_Y_M,
        FRONT_WHEEL_WIDTH_M, REAR_WHEEL_WIDTH_M,
    )

    front = _spin_axis_to_y(_load("front_wheel.stl"))
    rear = _spin_axis_to_y(_load("rear_wheel.stl"))

    out = {}
    for axle, mesh, x_m, inner_y, width_m in (
        ("front", front, x_front_mm / 1000.0, FRONT_WHEEL_INNER_Y_M,
         FRONT_WHEEL_WIDTH_M),
        ("rear", rear, (x_front_mm + W_mm) / 1000.0, REAR_WHEEL_INNER_Y_M,
         REAR_WHEEL_WIDTH_M),
    ):
        y_c = inner_y + width_m / 2.0
        for side, sgn in (("right", +1.0), ("left", -1.0)):
            # centre x on the axle, y on the wheel centre, bottom on the ground
            out[f"wheel_{axle}_{side}"] = _place_by_center(
                mesh, (x_m, sgn * y_c, 0.0), keep_z_floor=True
            )
    return out


def place_supports(W_mm: float, x_front_mm: float) -> dict:
    """Wheel support brackets: real meshes at each axle, both sides.

    The support reaches from the body wall out to the wheel.

    IT WAS BEING FLIPPED END-FOR-END. The old code translated by `-lo[1]` with
    the comment "inner end (min y) -> y=0", but lo[1] is the MINIMUM y, and in
    the CAD frame that is the OUTBOARD end: front_wheel_support spans
    y[-36.5, 0] and front_wheel sits at y[-36.5, -19.2], i.e. at the min-y end.
    So the translation put the wheel-mounting end on the centreline and the
    body-mounting end out at the wheel -- the bracket inside-out, on both
    sides, in every render.

    The parts are already positioned relative to each other in the export
    frame: support and wheel share an x centre (173.25 mm front, 35.25 mm
    rear), and the support runs from the centreline out to the wheel. So the
    +y instance is a pure MIRROR of the CAD half, not a translation of it --
    that maps the centreline end to y=0 and the wheel end to +36.5 while
    keeping the bracket the right way round. Only x (onto the axle) and z (onto
    the ground) are placed.
    """
    out = {}
    for axle, name, x_m in (
        ("front", "front_wheel_support.stl", x_front_mm / 1000.0),
        ("rear", "rear_wheel_support.stl", (x_front_mm + W_mm) / 1000.0),
    ):
        mesh = _load(name)
        lo, hi = mesh.bounds
        base = mesh.copy()
        # z is NOT placed. The CAD already has the support 1.5 mm off the track
        # with its axle exactly on the wheel centre -- wheel z[0.0, 28.2],
        # centre 14.13; support z[1.5, 26.8], axle 14.15. Dropping its bbox to
        # z=0 (which is what `-lo[2]` did) shoved the axle down to 12.62 mm,
        # 1.50 mm below the wheel it is supposed to carry. The bracket does not
        # sit on the track; only the wheel does.
        base.apply_translation([
            x_m - (lo[0] + hi[0]) / 2.0,  # x centre -> axle line
            0.0,                          # y: CAD is already centreline-to-wheel
            0.0,                          # z: keep the CAD's own axle height
        ])
        # CAD half lies on -y, so that instance IS the left one; mirror for +y.
        out[f"support_{axle}_left"] = base
        out[f"support_{axle}_right"] = _mirror_y(base)
    return out


def place_halo(hw_inputs) -> dict:
    """Real halo+helmet, x-aligned to the project's halo pocket span.

    The part is already at the correct height (z 24-39; z=24 == HALO_MIN_Z) and
    is a half mesh -> completed to a whole. We only slide it in x so its span
    matches halo_geometry.x_front_m..x_rear_m, the same x the void uses.
    """
    halo = _complete_half(_load("halo_helmet.stl"))
    geo = hw_inputs["halo_geometry"]
    lo, hi = halo.bounds
    target_cx = (geo.x_front_m + geo.x_rear_m) / 2.0
    m = halo.copy()
    m.apply_translation([target_cx - (lo[0] + hi[0]) / 2.0, 0.0, 0.0])
    return {"halo_helmet": m}


def place_canister(hw_inputs, bv, fh=None, body=None) -> dict:
    """Real CO2 canister seated in its own bore, protruding out the rear.

    ANCHORED TO THE BORE, not to a body-surface probe. This used to call
    _body_rear_x_at_z(body, 35 mm): "max body x in a thin z-band about
    cartridge height, on the centreline", meant to follow a tail that tapers
    early instead of floating past it. But at cartridge height ON THE
    CENTRELINE the body is hollow BY DESIGN -- that hollow is the bore. So the
    probe returned the bore's leading edge, not the tail. Measured on the
    2026-08-05 carved car: it read 156.7 mm against a true rear of 207.2 mm,
    50.5 mm short, and seated the cartridge through solid bodywork with the
    actual bore left empty.

    fh.canister_cylinder is the bore as geometry, so use it: push the cartridge
    in until it bottoms out on the front of the bore, exactly as it is loaded
    in reality. Protrusion past the rear face then follows from the part's own
    length rather than being dialled in, and T5.6's >=5 mm is checked, not
    assumed. Falls back to the old envelope anchor only if no fh is supplied.
    """
    from fixed_hardware import CANISTER_Z_MM

    can = _complete_half(_load("co2_canister.stl"))
    lo, hi = can.bounds
    z_axis_m = CANISTER_Z_MM / 1000.0

    cyl = getattr(fh, "canister_cylinder", None) if fh is not None else None
    if cyl is not None:
        bore_front_m = cyl.x_center_m - cyl.x_half_width_m
        z_axis_m = cyl.z_center_m
        dx = bore_front_m - lo[0]              # seat it against the bore end
    else:
        dx = (bv.rearpod.x_max_m() + 0.008) - hi[0]

    m = can.copy()
    m.apply_translation([dx, -(lo[1] + hi[1]) / 2.0, z_axis_m - (lo[2] + hi[2]) / 2.0])

    if body is not None:
        protrusion_mm = (m.bounds[1][0] - float(body.vertices[:, 0].max())) * 1000.0
        if protrusion_mm < 5.0:
            warnings.warn(
                f"CO2 cartridge protrudes {protrusion_mm:.1f} mm past the body "
                f"rear; T5.6 requires at least 5 mm.", RuntimeWarning,
                stacklevel=2)
    return {"co2_canister": m}


def build_hardware(W_mm: float, x_front_mm: float, d_halo_mm: float, bv,
                   decimate_to: int | None = 12000, body=None,
                   fh=None) -> dict:
    """All real hardware parts placed in project coordinates, keyed by name.

    `body` (the extracted body mesh) lets the canister anchor to the body's
    real rear rather than the envelope rear -- pass it whenever available.
    """
    from fixed_hardware import compute_default_fixed_hardware_inputs

    hw_inputs = compute_default_fixed_hardware_inputs(
        W_mm, x_front_mm, d_halo_mm, bv.ref_plane_A_m, bv.ref_plane_B_m,
        rear_face_x_m=bv.rearpod.x_max_m(),
    )
    # fh carries canister_cylinder -- the bore as geometry. place_canister
    # seats the cartridge against it instead of probing the body surface at
    # cartridge height, where the body is hollow by design.
    #
    # It is the CALLER's job to supply it (geom.fixed_hardware). This used to
    # try `place_fixed_hardware(**hw_inputs)` inside `except Exception: fh =
    # None`, which is wrong twice over: that call is missing four required
    # arguments (W_mm, x_front_mm, body_grid_shape, body_grid_origin_m) so it
    # raised TypeError every single time, and the bare except swallowed it and
    # fell back to the envelope anchor. The cartridge then landed 11 mm forward
    # of its bore while the code reported it was seated in it. Warn instead --
    # a fallback that hides its own failure is how this went unnoticed.
    if fh is None:
        warnings.warn(
            "build_hardware got no fixed-hardware result, so the cartridge is "
            "anchored to the ENVELOPE rear rather than its bore. Pass "
            "fh=geom.fixed_hardware for the real placement.",
            RuntimeWarning, stacklevel=2)

    parts: dict = {}
    parts.update(place_wheels(W_mm, x_front_mm, hw_inputs))
    parts.update(place_supports(W_mm, x_front_mm))
    parts.update(place_halo(hw_inputs))
    parts.update(place_canister(hw_inputs, bv, fh=fh, body=body))

    if decimate_to:
        parts = {k: _decimate(v, decimate_to) for k, v in parts.items()}
    return parts


# ── CLI: assemble body + hardware into one STL ──────────────────────────────

def _part_colour(name: str):
    if name.startswith("wheel"):
        return (0.15, 0.15, 0.17)      # near-black tyres
    if name.startswith("support"):
        return (0.55, 0.55, 0.58)      # grey brackets
    if name == "halo_helmet":
        return (0.85, 0.55, 0.20)      # orange halo/helmet
    if name == "co2_canister":
        return (0.80, 0.20, 0.20)      # red canister
    return (0.45, 0.62, 0.85)          # blue body


def render_coloured(body, hardware: dict, out_path: str) -> None:
    """4-view render with each part group its own colour, so placement is
    visually verifiable. Kept here rather than in render_stl.py because it
    needs the parts SEPARATE (colour per group), not a merged mesh."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    groups = [("body", body, _part_colour("body"))]
    for name, mesh in hardware.items():
        groups.append((name, mesh, _part_colour(name)))

    all_v = np.vstack([m.vertices for _, m, _ in groups]) * 1000.0
    lo, hi = all_v.min(0), all_v.max(0)
    centre, span = (lo + hi) / 2, (hi - lo).max() / 2

    fig = plt.figure(figsize=(15, 10))
    views = [("iso", 22, -60), ("side (x-z)", 0, -90),
             ("top (x-y)", 89, -90), ("front (y-z)", 0, 0)]
    for idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        for _name, mesh, base in groups:
            tris = mesh.vertices[mesh.faces] * 1000.0
            n = mesh.face_normals
            shade = 0.4 + 0.6 * np.clip(n @ np.array([0.3, 0.4, 0.86]), 0, 1)
            cols = np.stack([shade * base[0], shade * base[1], shade * base[2],
                             np.ones_like(shade)], -1)
            ax.add_collection3d(Poly3DCollection(tris, facecolors=cols, edgecolors="none"))
        ax.set_xlim(centre[0] - span, centre[0] + span)
        ax.set_ylim(centre[1] - span, centre[1] + span)
        ax.set_zlim(centre[2] - span, centre[2] + span)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
        ax.set_title(title)
    fig.suptitle(f"integrated car   L={hi[0]-lo[0]:.1f} W={hi[1]-lo[1]:.1f} "
                 f"H={hi[2]-lo[2]:.1f} mm   (body blue · wheels black · "
                 f"halo orange · canister red · brackets grey)   nose at low x (left)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=105)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=2.5)
    ap.add_argument("--W", type=float, default=130.0)
    ap.add_argument("--x-front", type=float, default=46.0)
    ap.add_argument("--d-halo", type=float, default=20.0)
    ap.add_argument("--full", action="store_true", help="skip hardware decimation")
    ap.add_argument("--body", default="loft",
                    help="'loft' (loft_demo profile) or 'body_profile' (subagent's)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    coarse.use_spacing(args.spacing)

    import trimesh
    from unified_phi import (
        build_unified_geometry, enforce_symmetry, extract_unified_surface,
    )

    geom = build_unified_geometry(args.W, args.x_front, args.d_halo)

    if args.body == "body_profile":
        from body_profile import build_car_body_field
        geom.phi.grid = build_car_body_field(geom)
    else:
        from loft_demo import build_loft_field
        geom.phi.grid = build_loft_field(geom)
    geom.phi.apply_hard_constraints()
    enforce_symmetry(geom)

    body, report = extract_unified_surface(geom, allow_inaccessible=True)
    print(f"=== assembly  W={args.W} x_front={args.x_front} d_halo={args.d_halo} "
          f"spacing={args.spacing} ===")
    print(f"body: {report['connected_bodies']} body, watertight={report['watertight']}, "
          f"{len(body.faces)} faces")

    hardware = build_hardware(
        args.W, args.x_front, args.d_halo, geom.bv,
        decimate_to=None if args.full else 12000, body=body,
    )
    for name, mesh in hardware.items():
        lo, hi = mesh.bounds * 1000
        print(f"  {name:<20} {len(mesh.faces):>7} faces  "
              f"x[{lo[0]:6.1f},{hi[0]:6.1f}] y[{lo[1]:6.1f},{hi[1]:6.1f}] "
              f"z[{lo[2]:5.1f},{hi[2]:5.1f}]")

    body.visual.face_colors = [120, 140, 170, 255]
    all_meshes = [body] + list(hardware.values())
    assembly = trimesh.util.concatenate(all_meshes)
    lo, hi = assembly.bounds * 1000
    print(f"\nASSEMBLY  L={hi[0]-lo[0]:.1f} W={hi[1]-lo[1]:.1f} H={hi[2]-lo[2]:.1f} mm  "
          f"{len(assembly.faces)} faces total")

    out = Path(args.out) if args.out else Path(__file__).parent / "out" / "integrated_car.stl"
    out.parent.mkdir(parents=True, exist_ok=True)
    assembly.export(str(out))                       # binary, small
    assembly.export(str(out.with_suffix(".ascii.stl")), file_type="stl_ascii")
    print(f"\nwrote {out}")

    render_coloured(body, hardware, str(out.with_suffix(".png")))


if __name__ == "__main__":
    main()
