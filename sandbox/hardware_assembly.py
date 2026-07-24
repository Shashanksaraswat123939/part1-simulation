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

def place_wheels(W_mm: float, x_front_mm: float, hw_inputs) -> dict:
    """Four real wheel meshes at the project axle positions.

    The wheel part is round in x-z with its axle along y (correct for a wheel at
    any axle x). We recentre it and drop four instances: front/rear x axle
    lines, +/-y sides. Inner-face y and wheel width come straight from the
    part's own geometry so the render matches the physical wheel, not an
    assumed width.
    """
    from geometry_contract import (
        FRONT_WHEEL_INNER_Y_M, REAR_WHEEL_INNER_Y_M, WHEEL_WIDTH_M,
    )

    front = _load("front_wheel.stl")
    rear = _load("rear_wheel.stl")
    half_w = WHEEL_WIDTH_M / 2.0

    out = {}
    for axle, mesh, x_m, inner_y in (
        ("front", front, x_front_mm / 1000.0, FRONT_WHEEL_INNER_Y_M),
        ("rear", rear, (x_front_mm + W_mm) / 1000.0, REAR_WHEEL_INNER_Y_M),
    ):
        y_c = inner_y + half_w
        for side, sgn in (("right", +1.0), ("left", -1.0)):
            # centre x on the axle, y on the wheel centre, bottom on the ground
            out[f"wheel_{axle}_{side}"] = _place_by_center(
                mesh, (x_m, sgn * y_c, 0.0), keep_z_floor=True
            )
    return out


def place_supports(W_mm: float, x_front_mm: float) -> dict:
    """Wheel support brackets: real meshes at each axle, both sides.

    The support reaches from the body wall out to the wheel. We anchor its
    INNER end (nearest y=0) at the body's outer wall and let it extend outward
    to the wheel; mirror for the far side.
    """
    from geometry_contract import (
        FRONT_WHEEL_INNER_Y_M, REAR_WHEEL_INNER_Y_M, WHEEL_WIDTH_M,
    )

    out = {}
    for axle, name, x_m, inner_y in (
        ("front", "front_wheel_support.stl", x_front_mm / 1000.0, FRONT_WHEEL_INNER_Y_M),
        ("rear", "rear_wheel_support.stl", (x_front_mm + W_mm) / 1000.0, REAR_WHEEL_INNER_Y_M),
    ):
        mesh = _load(name)
        # Recentre x on the axle, drop to the ground, put its span on +y.
        lo, hi = mesh.bounds
        base = mesh.copy()
        base.apply_translation([
            x_m - (lo[0] + hi[0]) / 2.0,
            -lo[1],                       # inner end (min y) -> y=0
            -lo[2],                       # bottom -> ground
        ])
        out[f"support_{axle}_right"] = base
        out[f"support_{axle}_left"] = _mirror_y(base)
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


def _body_rear_x_at_z(body, z_m: float, band_m: float = 0.004) -> float:
    """Max body x within a thin z-band around z_m, on the centreline.

    Anchors the canister to where the body ACTUALLY ends at cartridge height,
    not to the envelope rear. The two diverge once the body tapers its tail
    early -- anchoring to the envelope leaves the canister floating in space
    past the real bodywork.
    """
    if body is None:
        return None
    v = body.vertices
    near = (np.abs(v[:, 2] - z_m) <= band_m) & (np.abs(v[:, 1]) <= 0.010)
    return float(v[near, 0].max()) if near.any() else float(v[:, 0].max())


def place_canister(hw_inputs, bv, body=None) -> dict:
    """Real CO2 canister on the centreline at z=35, protruding out the rear.

    The part is a half mesh (completed to a whole) whose axis lies along x. We
    put it on the centreline at CANISTER_Z_MM and slide it in x so its rear end
    pokes ~8 mm past the body's ACTUAL rear at cartridge height (T5.6 needs
    >=5 mm protrusion). Falls back to the envelope rear if no body is given.
    """
    from fixed_hardware import CANISTER_Z_MM

    can = _complete_half(_load("co2_canister.stl"))
    lo, hi = can.bounds
    z_axis_m = CANISTER_Z_MM / 1000.0
    rear_m = _body_rear_x_at_z(body, z_axis_m)
    if rear_m is None:
        rear_m = bv.rearpod.x_max_m()
    protrusion_m = 0.008
    dx = (rear_m + protrusion_m) - hi[0]      # rear end -> body rear + protrusion
    m = can.copy()
    m.apply_translation([dx, -(lo[1] + hi[1]) / 2.0, z_axis_m - (lo[2] + hi[2]) / 2.0])
    return {"co2_canister": m}


def build_hardware(W_mm: float, x_front_mm: float, d_halo_mm: float, bv,
                   decimate_to: int | None = 12000, body=None) -> dict:
    """All real hardware parts placed in project coordinates, keyed by name.

    `body` (the extracted body mesh) lets the canister anchor to the body's
    real rear rather than the envelope rear -- pass it whenever available.
    """
    from fixed_hardware import compute_default_fixed_hardware_inputs

    hw_inputs = compute_default_fixed_hardware_inputs(
        W_mm, x_front_mm, d_halo_mm, bv.ref_plane_A_m, bv.ref_plane_B_m,
        rear_face_x_m=bv.rearpod.x_max_m(),
    )

    parts: dict = {}
    parts.update(place_wheels(W_mm, x_front_mm, hw_inputs))
    parts.update(place_supports(W_mm, x_front_mm))
    parts.update(place_halo(hw_inputs))
    parts.update(place_canister(hw_inputs, bv, body=body))

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
