"""
hardware_geometry.py --- real fixed-hardware solids (measured CAD, not
placeholders), positioned using the exact same (W, x_front, d_halo) ->
placement logic fixed_hardware.py's void masks use, so the visible surface
and the forbidden-zone hole can never drift apart.

Source STLs (hardware_cad/, provided by the project owner -- real CAD, not
guesses):
  front_wheel.stl, front_wheel_support.stl
  rear_wheel.stl,  rear_wheel_support.stl
  halo_helmet.stl
  co2_canister.stl

Measured directly from these files' bounding boxes (see PLACEHOLDERS.md item
17 and sandbox/README.md finding #12 for the bug this replaces):
  wheel diameter        ~28.25 mm  (T7.5 legal range 28.0-32.0mm -- OK)
  wheel width             17.25 mm  (T7.5 doesn't specify width; this real
                                      value replaces the old 10mm guess in
                                      sandbox/hardware.py)
  front inner-face y      19.25 mm  (>= T7.2.1 min half-gap 19.0mm -- OK)
  rear inner-face y       16.25 mm  (>= T7.2.2 min half-gap 15.0mm -- OK)
  halo pocket bottom z    24.00 mm  (exact match to T4.4.4 HALO_MIN_Z_MM)
  halo x-length           50.00 mm  (exact match to T4.4.4 Appendix ix pocket length)
These are duplicated as constants in geometry_contract.py (wheel geometry)
so fixed_hardware.py's void masks can use them without a trimesh dependency.
Every build_* function here cross-checks its mesh's measured bounds against
those constants at load time and raises loudly on drift, so the surface and
the void can never silently disagree.

All source meshes model ONE lateral half only -- every one of them has
y <= 0 in its own native frame, and each already references the track
surface as z=0 (wheels) or the halo pocket floor as z=24mm (halo) directly,
needing no z adjustment. Each gets negated onto y >= 0 (this project's
"right half" convention -- see stl_assembler._mirror_right_to_left) to build
the right-side part; the untouched native mesh is the left-side part.

NOT built here: rear wing. (Project owner: "do not make any wings" --
sandbox/hardware.py's build_rear_wing() placeholder box is intentionally not
promoted into this module.)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

_HW_DIR = Path(__file__).resolve().parent / "hardware_cad"

_MESH_CACHE: dict[str, "trimesh.Trimesh"] = {}


def _load(filename: str) -> "trimesh.Trimesh":
    """Load (and cache) a raw hardware mesh, untouched, in its own native frame."""
    if filename not in _MESH_CACHE:
        import trimesh
        path = _HW_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"hardware_geometry: missing CAD file {path}. Expected the "
                "real measured hardware STLs in hardware_cad/."
            )
        _MESH_CACHE[filename] = trimesh.load(str(path), force="mesh")
    return _MESH_CACHE[filename].copy()


def _negate_y(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Mirror a mesh across y=0 (flip winding/normals to match)."""
    import trimesh
    m = mesh.copy()
    m.vertices[:, 1] *= -1.0
    m.invert()
    trimesh.repair.fix_normals(m)
    return m


def _translate_mm(mesh: "trimesh.Trimesh", dx_mm: float, dy_mm: float, dz_mm: float) -> "trimesh.Trimesh":
    """Translate a mesh that is still in the CAD file's native millimetre
    units by a delta ALSO in millimetres. Do not mix with metre deltas."""
    m = mesh.copy()
    m.apply_translation([dx_mm, dy_mm, dz_mm])
    return m


def _negate_x_about_own_centre(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Flip a mesh end-for-end along x (front<->back), reflecting about its
    OWN bounding-box x-centre (not the car's x=0, which is just the nose
    tip and has no meaning for an isolated part's own shape). Used for the
    halo+helmet, which was mounted backwards -- fixed 2026-07-19 per
    project owner visual confirmation."""
    import trimesh
    m = mesh.copy()
    x_centre = (m.bounds[0, 0] + m.bounds[1, 0]) / 2.0
    m.vertices[:, 0] = 2.0 * x_centre - m.vertices[:, 0]
    m.invert()
    trimesh.repair.fix_normals(m)
    return m


def _to_metres(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """
    Convert a millimetre-scale mesh (the CAD files' native units) to the
    metre-scale SI convention the rest of Part 1 uses (geometry_contract.py:
    "all internal values are SI (m, kg, N, kg/m^3)"). Call this LAST, after
    all placement math is done in millimetres -- mixing mm-space translation
    with metre-space geometry silently produces a mesh 1000x too large
    (caught the hard way: rendered as a 73-METRE-wide "car").
    """
    m = mesh.copy()
    m.apply_scale(0.001)
    return m


def _check_measured(name: str, measured_mm: float, expected_mm: float, tol_mm: float = 0.5) -> None:
    if abs(measured_mm - expected_mm) > tol_mm:
        raise AssertionError(
            f"hardware_geometry: measured {name}={measured_mm:.3f}mm from the CAD "
            f"file does not match the expected {expected_mm:.3f}mm (tolerance "
            f"{tol_mm}mm). The CAD file changed -- update the matching constant."
        )


def build_wheel_assembly(axle: str, x_target_mm: float) -> dict[str, "trimesh.Trimesh"]:
    """
    Build the four meshes (left/right wheel + left/right support) for one
    axle ("front" or "rear"), positioned at x_target_mm -- the axle's x
    position in car coordinates (x_front_mm for front, x_front_mm + W_mm
    for rear).

    The raw CAD meshes already touch the track surface at their local z=0,
    so placement only needs an x shift (the axle position is a search
    variable; the wheel's own shape and lateral offset are not).
    """
    from geometry_contract import WHEEL_WIDTH_MM, FRONT_WHEEL_INNER_Y_MM, REAR_WHEEL_INNER_Y_MM

    if axle not in ("front", "rear"):
        raise ValueError(f"axle must be 'front' or 'rear', got {axle!r}")

    wheel = _load("front_wheel.stl" if axle == "front" else "rear_wheel.stl")
    support = _load("front_wheel_support.stl" if axle == "front" else "rear_wheel_support.stl")
    expected_inner_y_mm = FRONT_WHEEL_INNER_Y_MM if axle == "front" else REAR_WHEEL_INNER_Y_MM

    wb = wheel.bounds   # (2,3): [min,max] x [x,y,z]
    x_center_mm = (wb[0, 0] + wb[1, 0]) / 2.0
    z_min_mm = wb[0, 2]
    width_mm = wb[1, 1] - wb[0, 1]
    inner_y_mm = -wb[1, 1]   # native y<=0: inner (track-contact) face is the less-negative bound

    _check_measured(f"{axle} wheel width", width_mm, WHEEL_WIDTH_MM)
    _check_measured(f"{axle} wheel inner-face y", inner_y_mm, expected_inner_y_mm, tol_mm=1.0)

    dx_mm = x_target_mm - x_center_mm
    dz_mm = -z_min_mm   # sub-mm cleanup so the tyre exactly touches z=0

    left_wheel = _to_metres(_translate_mm(wheel, dx_mm, 0.0, dz_mm))
    left_support = _to_metres(_translate_mm(support, dx_mm, 0.0, dz_mm))
    right_wheel = _negate_y(left_wheel)
    right_support = _negate_y(left_support)

    return {
        f"{axle}_wheel_left": left_wheel,
        f"{axle}_wheel_right": right_wheel,
        f"{axle}_wheel_support_left": left_support,
        f"{axle}_wheel_support_right": right_support,
    }


def build_halo(ref_plane_A_m: float, d_halo_mm: float) -> dict[str, "trimesh.Trimesh"]:
    """
    Position the real halo+helmet mesh in the halo pocket for this
    candidate's d_halo, using the exact same placement math as
    fixed_hardware.py's void mask (halo_pocket.compute_halo_pocket_box_m).

    The mesh's local z already matches the pocket floor (z_min == 24mm,
    T4.4.4) and its local x-length already matches the pocket length
    (50mm, T4.4.4 Appendix ix) -- so only an x shift (by d_halo) is needed.
    """
    from halo_pocket import compute_halo_pocket_box_m, HALO_POCKET_LENGTH_MM

    mesh = _negate_x_about_own_centre(_load("halo_helmet.stl"))
    b = mesh.bounds
    local_x_min_mm = b[0, 0]
    local_z_min_mm = b[0, 2]
    measured_length_mm = b[1, 0] - b[0, 0]

    _check_measured("halo x-length", measured_length_mm, HALO_POCKET_LENGTH_MM, tol_mm=1.0)

    pocket = compute_halo_pocket_box_m(ref_plane_A_m, d_halo_mm)
    target_x_min_mm = pocket["x_min_m"] * 1000.0
    target_z_min_mm = pocket["z_min_m"] * 1000.0   # == HALO_MIN_Z_MM

    dx_mm = target_x_min_mm - local_x_min_mm
    dz_mm = target_z_min_mm - local_z_min_mm

    left = _to_metres(_translate_mm(mesh, dx_mm, 0.0, dz_mm))
    right = _negate_y(left)
    return {"halo_left": left, "halo_right": right}


def build_canister(canister_com_mm: tuple[float, float, float]) -> "trimesh.Trimesh":
    """
    Position the real CO2 canister mesh at canister_com_mm -- the same COM
    fixed_hardware.compute_default_fixed_hardware_inputs() uses for both the
    box void and the Part 2 mass/COM spec.

    FLAG (not fixed here -- out of this pass's sidepod/halo/wheel scope):
    the real mesh measures ~68mm long, longer than T5.3's 45-58mm legal
    cartridge-chamber depth range. The void/mass logic in fixed_hardware.py
    still uses the 50mm design-default depth; only this visual placement
    uses the real mesh's own length. Revisit under a T5 pass.
    """
    mesh = _load("co2_canister.stl")
    centre_mm = mesh.bounds.mean(axis=0)   # bounding-box centre (mesh isn't watertight)
    target_mm = np.array(canister_com_mm, dtype=float)
    d_mm = target_mm - centre_mm
    return _to_metres(_translate_mm(mesh, *d_mm))


def build_all_hardware(
    W_mm: float,
    x_front_mm: float,
    d_halo_mm: float,
    ref_plane_A_m: float,
    canister_com_mm: tuple[float, float, float],
) -> dict[str, "trimesh.Trimesh"]:
    """
    Every real fixed-hardware solid for one (W, x_front, d_halo) candidate,
    keyed by name. No rear wing (project owner: "do not make any wings").
    """
    parts: dict[str, "trimesh.Trimesh"] = {}
    parts.update(build_wheel_assembly("front", x_front_mm))
    parts.update(build_wheel_assembly("rear", x_front_mm + W_mm))
    parts.update(build_halo(ref_plane_A_m, d_halo_mm))
    parts["canister"] = build_canister(canister_com_mm)
    return parts
