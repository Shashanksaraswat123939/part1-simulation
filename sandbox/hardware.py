"""
hardware.py -- build actual solid geometry for the fixed hardware.

WHY THIS DOESN'T ALREADY EXIST
------------------------------
Part 1 models fixed hardware ONLY as void masks (`FixedHardwareResult.
halo_void_mask`, `canister_void_mask`, `front/rear_axle_void_mask`) plus a
`FixedHardwareSpec` carrying mass and COM. The voids carve holes in the body
so the optimizer cannot fill the space the hardware occupies; the spec makes
the hardware count toward mass/COM. Neither produces a surface.

`stl_assembler.assemble_stl` then concatenates exactly four meshes -- nose,
sidepod, rearpod, main_body -- and nothing else. So the STL handed to Part 2's
`run_half_car_cfd` is a bare machined body: no wheels, no halo, no canister.
Part 2 doesn't add them either (`cfd_wrapper.py` and `mesh_validation.py`
contain no reference to any of them), and the Part 2 interface contract in
`spec_sections/17_*.txt` specifies only "Right-half STL ... must pass
trimesh.is_watertight, all vertices y >= -1e-6".

That matters for CFD specifically: four wheels of 30mm diameter on a 65mm-wide
car, plus a halo that T4.4.2/3 require to be VISIBLE from front/side/top, are
dominant drag sources. Optimising body shape against a drag number computed
without them is optimising against the wrong objective.

This module builds those solids from the SAME constants the void masks use, so
the surface and the hole can never drift apart.

ASSUMPTION YOU MUST CONFIRM
---------------------------
Wheel WIDTH does not exist anywhere in the project. geometry_contract has
R_WHEEL_M (15mm) and WHEEL_CLEARANCE_M; wheel_visibility_zones has the T7.2
inner gaps (front 38mm, rear 30mm, so inner faces at y = +/-19 and +/-15).
Nothing defines how wide a wheel is. WHEEL_WIDTH_MM below is an assumption,
exposed as a parameter -- override it with the real measurement.
"""

from __future__ import annotations

import numpy as np

import coarse  # noqa: F401 -- sys.path + PART2_PATH

# Wheel width is NOT defined anywhere in Part 1. See module docstring.
WHEEL_WIDTH_MM: float = 10.0


def _yz_polygon_extruded_along_x(cross_section_yz_m, x_min_m: float, x_max_m: float):
    """Extrude a (y,z) polygon along x -- the same operation _build_polygon_void_mask
    performs on the grid, but producing a surface instead of a mask."""
    import trimesh
    from shapely.geometry import Polygon

    polygon = Polygon([(y, z) for y, z in cross_section_yz_m])
    mesh = trimesh.creation.extrude_polygon(polygon, height=x_max_m - x_min_m)
    # extrude_polygon builds in the x-y plane extruded along +z. Map that to
    # our (y, z) cross-section extruded along +x: (px, py, pz) -> (pz, px, py).
    verts = mesh.vertices.copy()
    mesh.vertices = np.column_stack([verts[:, 2] + x_min_m, verts[:, 0], verts[:, 1]])
    mesh.fix_normals()
    return mesh


def build_halo(hw_inputs) -> "trimesh.Trimesh":
    """Halo bar: the cross-section polygon extruded across the pocket's x-span.

    This is the same conservative rectangular profile fixed_hardware uses for
    the void (see its HALO_CROSS_SECTION_* notes) -- the real part is a curved
    downloadable CAD arch (T4.4.1), so this is a bounding stand-in, not the
    true shape. Swap in the real STL when you have it.
    """
    halo = hw_inputs["halo_geometry"]
    return _yz_polygon_extruded_along_x(
        halo.cross_section_yz_m, halo.x_front_m, halo.x_rear_m
    )


def build_canister(hw_inputs) -> "trimesh.Trimesh":
    """CO2 canister as a cylinder on the x axis at its specified COM.

    Diameter/depth/height come from fixed_hardware's T5 design defaults
    (CANISTER_DIAMETER_MM 18.25, CANISTER_DEPTH_MM 50, CANISTER_Z_MM 35).
    """
    import trimesh
    from fixed_hardware import CANISTER_DIAMETER_MM, CANISTER_DEPTH_MM

    x_mm, y_mm, z_mm = hw_inputs["canister_com_mm"]
    mesh = trimesh.creation.cylinder(
        radius=CANISTER_DIAMETER_MM / 2000.0, height=CANISTER_DEPTH_MM / 1000.0
    )
    # cylinder() is built along z; rotate onto x, then translate to the COM.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    mesh.apply_translation([x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0])
    return mesh


def build_wheels(W_mm: float, x_front_mm: float,
                 wheel_width_mm: float = WHEEL_WIDTH_MM) -> dict:
    """Four wheels as cylinders on y axes at the front and rear axle lines.

    Radius is R_WHEEL_M (15mm, locked -- race_objective.py depends on it).
    Inner faces sit at the T7.2 minimum inner gaps (front 38mm, rear 30mm),
    which is the tightest legal track; widening is a free design choice.
    Width is an ASSUMPTION -- see module docstring.
    """
    import trimesh
    from geometry_contract import R_WHEEL_M
    from wheel_visibility_zones import FRONT_INNER_GAP_MIN_MM, REAR_INNER_GAP_MIN_MM

    width_m = wheel_width_mm / 1000.0
    wheels = {}
    axles = (
        ("front", x_front_mm / 1000.0, FRONT_INNER_GAP_MIN_MM / 2000.0),
        ("rear", (x_front_mm + W_mm) / 1000.0, REAR_INNER_GAP_MIN_MM / 2000.0),
    )
    for axle_name, x_m, inner_y_m in axles:
        for side, sign in (("right", +1.0), ("left", -1.0)):
            mesh = trimesh.creation.cylinder(radius=R_WHEEL_M, height=width_m)
            # Built along z; rotate onto the y axis (the axle direction).
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])
            )
            # Wheel centre sits half a width outboard of the inner face.
            y_centre = sign * (inner_y_m + width_m / 2.0)
            mesh.apply_translation([x_m, y_centre, R_WHEEL_M])
            wheels[f"wheel_{axle_name}_{side}"] = mesh
    return wheels


def build_rear_wing(hw_inputs) -> "trimesh.Trimesh":
    """Rear wing as a plain box at its specified COM.

    fixed_hardware gives the wing a mass and a COM but no dimensions at all,
    so this is a placeholder slab sized to the car's width. It exists so the
    wing is visible and blocks flow; it is not an aerofoil.
    """
    import trimesh

    x_mm, y_mm, z_mm = hw_inputs["rear_wing_com_mm"]
    mesh = trimesh.creation.box(extents=[0.010, 0.060, 0.003])
    mesh.apply_translation([x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0])
    return mesh


def build_all(W_mm: float, x_front_mm: float, d_halo_mm: float, bv,
              wheel_width_mm: float = WHEEL_WIDTH_MM) -> dict:
    """Every fixed-hardware solid, keyed by name.

    Uses compute_default_fixed_hardware_inputs -- the exact same call
    _level2_evaluate and phi_grid_factory make to build the void masks -- so
    these surfaces sit precisely where the holes are.
    """
    from fixed_hardware import compute_default_fixed_hardware_inputs

    hw_inputs = compute_default_fixed_hardware_inputs(
        W_mm, x_front_mm, d_halo_mm, bv.ref_plane_A_m, bv.ref_plane_B_m,
    )

    parts = {
        "halo": build_halo(hw_inputs),
        "canister": build_canister(hw_inputs),
        "rear_wing": build_rear_wing(hw_inputs),
    }
    parts.update(build_wheels(W_mm, x_front_mm, wheel_width_mm))
    return parts


def hardware_mass_summary(W_mm: float, x_front_mm: float, d_halo_mm: float, bv) -> list:
    """(name, mass_kg) for the fixed hardware Part 1 counts toward mass/COM.

    These masses are real inputs to ingest_mass_com; only the SURFACES are
    missing from the STL. explore.py's machined-only mass table understates
    the car by this much.
    """
    from fixed_hardware import (
        compute_default_fixed_hardware_inputs, REAR_WING_MASS_KG, WHEEL_AXLE_MASS_KG,
    )
    from geometry_contract import CO2_MASS_KG

    compute_default_fixed_hardware_inputs(
        W_mm, x_front_mm, d_halo_mm, bv.ref_plane_A_m, bv.ref_plane_B_m,
    )
    return [
        ("co2_cartridge", CO2_MASS_KG),
        ("wheels+axles", WHEEL_AXLE_MASS_KG),
        ("rear_wing", REAR_WING_MASS_KG),
    ]
