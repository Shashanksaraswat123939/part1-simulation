"""
test_unified_phi.py -- the single labelled level-set field.

Runs at coarse spacing (2.0 mm) so the suite stays fast; every property under
test here is structural (labels, symmetry, connectivity, which regions are
carved) and none of it depends on resolving the 3.15 mm machining radius.
Legality of the *surface* is a separate question that needs 0.3 mm -- see
sandbox/coarse.py's caveat.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "sandbox"))
sys.path.insert(0, str(_ROOT.parent / "part2-simulation"))

import numpy as np

import coarse  # noqa: E402  -- also sets PART2_PATH
coarse.use_spacing(2.0)

from unified_phi import (  # noqa: E402
    LABEL_IDS, LABEL_NONE, MILLED_COMPONENTS, MODEL_BLOCK_LENGTH_MM,
    build_unified_geometry, compute_mass_com, enforce_symmetry,
)

W, XF, DH = 130.0, 46.0, 20.0


def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)


_CACHE = {}


def _geom(**kw):
    key = tuple(sorted(kw.items()))
    if key not in _CACHE:
        _CACHE[key] = build_unified_geometry(W, XF, DH, **kw)
    return _CACHE[key]


def test_all_four_components_are_labelled():
    g = _geom()
    for name in LABEL_IDS:
        n = int(g.component_mask(name).sum())
        assert n > 0, f"{name} zone is empty -- no cell carries its label"
    _pass("test_all_four_components_are_labelled")


def test_zones_are_disjoint_and_cover_only_the_envelope():
    # Every cell has exactly one label; labels are stored in one array so
    # disjointness is structural, but the NONE region must be non-empty
    # (the corners outside every component's legal box).
    g = _geom()
    total = int(np.prod(g.shape))
    counted = sum(int(g.component_mask(n).sum()) for n in LABEL_IDS)
    none = int((g.labels == LABEL_NONE).sum())
    assert counted + none == total, f"{counted} + {none} != {total}"
    assert none > 0, "expected some cells outside every component's envelope"
    _pass("test_zones_are_disjoint_and_cover_only_the_envelope")


def test_y_axis_has_a_cell_exactly_on_the_centreline():
    # The half-car slice at y=0 and Part 2's sidepod com_y=0 convention both
    # depend on this. An even cell count would put no cell at y=0.
    from geometry_contract import GRID_SPACING_M
    g = _geom()
    ny = g.shape[1]
    assert ny % 2 == 1, f"y cell count {ny} must be odd"
    ys = g.region.origin_m[1] + np.arange(ny) * GRID_SPACING_M
    assert abs(ys[ny // 2]) < 1e-12, f"middle cell sits at y={ys[ny//2]}, not 0"
    _pass("test_y_axis_has_a_cell_exactly_on_the_centreline")


def test_masks_and_labels_are_exactly_symmetric():
    # Regression: float comparisons against cell coordinates are not exactly
    # symmetric (-0.036 + 34*0.0015 != 0.015), which cost the nose zone one of
    # its two boundary columns and gave com_y = +0.75 mm.
    g = _geom()
    mid = g.shape[1] // 2
    for name, arr in (("labels", g.labels),
                      ("hard_mask_air", g.phi.hard_mask_air),
                      ("hard_mask_solid", g.phi.hard_mask_solid)):
        assert np.array_equal(arr[:, mid:, :], arr[:, :mid + 1, :][:, ::-1, :]), \
            f"{name} is not symmetric about y=0"
    _pass("test_masks_and_labels_are_exactly_symmetric")


def test_enforce_symmetry_gives_every_component_com_y_zero():
    g = build_unified_geometry(W, XF, DH)
    enforce_symmetry(g)
    for c in compute_mass_com(g):
        assert abs(c.com_y_m) < 1e-12, f"{c.name} com_y = {c.com_y_m}, expected 0"
    _pass("test_enforce_symmetry_gives_every_component_com_y_zero")


def test_extraction_yields_one_connected_watertight_body():
    # THE point of the unified representation. The old four-box path
    # concatenated four meshes and routinely shipped disconnected lumps
    # (sandbox finding 7) that still passed the gates.
    import surface_extraction as SE
    g = _geom()
    mesh = SE._repair_mesh(SE._marching_cubes(g.phi))
    bodies = mesh.split(only_watertight=False)
    assert len(bodies) == 1, f"expected 1 connected body, got {len(bodies)}"
    assert mesh.is_watertight, "unified extraction must be watertight"
    assert mesh.is_winding_consistent, "unified extraction must have consistent winding"
    assert mesh.volume > 0, f"inverted normals: volume {mesh.volume}"
    _pass("test_extraction_yields_one_connected_watertight_body")


def test_cartridge_bore_is_carved_and_open_at_the_rear():
    # T5.6: the cartridge must protrude from the rear, so the bore has to break
    # through the rearmost face rather than dead-end inside the body.
    from geometry_contract import GRID_SPACING_M
    g = _geom()
    if g.fixed_hardware is None:
        return _pass("test_cartridge_bore_is_carved_and_open_at_the_rear (skipped: no Part 2)")
    cyl = g.fixed_hardware.canister_cylinder
    depth_mm = (cyl.x_max_m - cyl.x_min_m) * 1000
    assert 45.0 <= depth_mm <= 58.0, f"T5.3 depth {depth_mm:.1f} mm outside 45-58"
    assert 18.0 <= cyl.radius_m * 2000 <= 18.5, \
        f"T5.1 diameter {cyl.radius_m*2000:.2f} mm outside 18.0-18.5"
    assert 30.0 <= cyl.z_center_m * 1000 <= 40.0, \
        f"T5.2 height {cyl.z_center_m*1000:.1f} mm outside 30-40"
    # The bore must reach at least the rearmost face.
    assert cyl.x_max_m >= g.landmarks["rear_face_m"] - 1e-9, \
        "bore stops short of the rear face -- cartridge cannot be inserted"
    # And it must actually be air in the field, along its whole length.
    air = g.phi.grid > 0
    i0 = int(round((cyl.x_min_m - g.region.origin_m[0]) / GRID_SPACING_M))
    i1 = min(int(round((cyl.x_max_m - g.region.origin_m[0]) / GRID_SPACING_M)),
             g.shape[0] - 1)
    j = g.shape[1] // 2
    k = int(round((cyl.z_center_m - g.region.origin_m[2]) / GRID_SPACING_M))
    assert air[i0:i1, j, k].all(), "bore axis is not air along its full depth"
    _pass("test_cartridge_bore_is_carved_and_open_at_the_rear")


def test_virtual_cargo_is_forced_solid_when_requested():
    """with_cargo must be what decides the CARGO region, and only that.

    The second assertion used to be "with_cargo=False forces nothing solid",
    which stopped holding once the halo->canister loft deck became forced
    material independently of cargo. Comparing the two geometries isolates the
    cargo's own contribution, which is what this test is actually about, and
    still fails if with_cargo=False smuggles a cargo block in.
    """
    with_cargo = _geom(with_cargo=True)
    without = _geom(with_cargo=False)
    n_with = int(with_cargo.phi.hard_mask_solid.sum())
    n_without = int(without.phi.hard_mask_solid.sum())
    assert n_with > n_without, "T4.2 cargo should force a solid region"

    # Everything forced without cargo must also be forced with it: the cargo
    # ADDS, it does not relocate the loft.
    only_without = without.phi.hard_mask_solid & ~with_cargo.phi.hard_mask_solid
    assert int(only_without.sum()) == 0, (
        f"{int(only_without.sum()):,} cells are forced solid only when cargo is "
        f"OFF -- the two forced regions are interfering")
    _pass("test_virtual_cargo_is_forced_solid_when_requested")


def test_machined_length_excludes_the_printed_nose():
    # T3.1.2's 223 mm block bounds the MILLED components only -- the nose is
    # 3D printed, so it is not block material (confirmed 2026-07-20).
    g = _geom()
    ml = g.machined_length_mm()
    assert ml <= MODEL_BLOCK_LENGTH_MM, \
        f"milled span {ml:.1f} mm exceeds the {MODEL_BLOCK_LENGTH_MM} mm block"
    # It must be shorter than the full body, which starts at the nose tip.
    solid = g.phi.grid < 0
    xs = np.where(solid.any(axis=(1, 2)))[0]
    from geometry_contract import GRID_SPACING_M
    full = (xs[-1] - xs[0] + 1) * GRID_SPACING_M * 1000
    assert ml < full, f"milled span {ml:.1f} should be shorter than full body {full:.1f}"
    assert "nose" not in MILLED_COMPONENTS
    _pass("test_machined_length_excludes_the_printed_nose")


def test_no_attachment_strips_are_needed():
    """No CONNECTIVITY crutches -- specified geometry is a different thing.

    The four-box path forced a strip of cells solid at each component's
    neighbouring face to keep the assembly connected. With one field that
    crutch is gone, and it must stay gone.

    Forced-solid is now exactly two things, both SPECIFIED rather than
    structural: the T4.2 cargo, and the halo->canister loft deck (this
    module's own header: "the continuous top surface running from the
    cartridge chamber at the rear, forward over the halo mount"). This used to
    assert hard_solid was empty without cargo, which also happened to catch
    attachment strips -- but that stopped being the right test the moment a
    second legitimate forced region existed. Assert the COMPOSITION instead, so
    a genuine strip still fails while the loft does not.
    """
    import geometry_contract as gc
    from fixed_hardware import halo_canister_loft_solid_mask

    with_cargo = _geom(with_cargo=True)
    without = _geom(with_cargo=False)

    fh = without.fixed_hardware
    loft = halo_canister_loft_solid_mask(
        fh.halo_void_mask, getattr(fh, "canister_cylinder", None),
        x_origin_m=without.region.origin_m[0],
        y_origin_m=without.region.origin_m[1],
        z_origin_m=without.region.origin_m[2],
        d_m=gc.GRID_SPACING_M,
    )
    assert loft.any(), "the loft deck is empty -- it must span halo to canister"

    # Mirrored onto the left half by build_unified_geometry, so mirror here too.
    loft = loft | loft[:, ::-1, :]
    residue = without.phi.hard_mask_solid & ~loft
    assert int(residue.sum()) == 0, (
        f"{int(residue.sum()):,} cells are forced solid that are neither cargo "
        f"nor the halo-canister loft -- an attachment strip has come back")
    assert int(with_cargo.phi.hard_mask_solid.sum()) > \
        int(without.phi.hard_mask_solid.sum()), "cargo forces nothing solid"
    _pass("test_no_attachment_strips_are_needed")


def test_label_aware_extraction_runs_all_six_stages():
    # extract_unified_surface must dispatch the manufacturing gates per FACE
    # (by label) rather than per mesh. Run at 3.0 mm: this is the only test
    # here that ray-casts, and cost scales hard with face count.
    coarse.use_spacing(3.0)
    try:
        import importlib
        import unified_phi
        importlib.reload(unified_phi)
        g = unified_phi.build_unified_geometry(W, XF, DH)
        unified_phi.enforce_symmetry(g)
        mesh, report = unified_phi.extract_unified_surface(
            g, allow_inaccessible=True
        )
        assert report["connected_bodies"] == 1, \
            f"expected one body, got {report['connected_bodies']}"
        assert report["watertight"], "unified mesh must be watertight"
        assert report["volume_cm3"] > 0, "inverted normals"
        # Accessibility is reported per milled component, and the 3D-printed
        # nose is absent from it -- it has no tool-direction constraint at all.
        areas = report["inaccessible_area_mm2"]
        assert set(areas) == set(MILLED_COMPONENTS), \
            f"expected per-milled-component areas, got {sorted(areas)}"
        assert "nose" not in areas, "the nose is printed; it has no tool directions"
    finally:
        coarse.use_spacing(2.0)
    _pass("test_label_aware_extraction_runs_all_six_stages")


def test_face_labels_cover_every_face():
    # Every face must be attributable to a component, or a gate silently skips
    # geometry. nearest_labels fills the LABEL_NONE cells that surface points
    # can round into.
    import surface_extraction as SE
    import numpy as np
    from unified_phi import _points_to_labels, nearest_labels
    g = _geom()
    mesh = SE._repair_mesh(SE._marching_cubes(g.phi))
    filled = nearest_labels(g)
    assert not (filled == LABEL_NONE).any(), "nearest_labels left NONE cells"
    flabels = _points_to_labels(g, np.asarray(mesh.triangles_center), filled)
    assert len(flabels) == len(mesh.faces)
    assert (flabels != LABEL_NONE).all(), "some faces are unattributed"
    _pass("test_face_labels_cover_every_face")


def test_invalid_outer_scalars_raise():
    for args, why in (
        ((100.0, XF, DH), "W below 120"),
        ((W, 60.0, DH), "x_front above T8.2's 56 mm ceiling"),
        ((W, 30.0, DH), "x_front below 36 mm"),
        ((W, XF, 200.0), "d_halo above W-34"),
    ):
        try:
            build_unified_geometry(*args)
        except ValueError:
            continue
        _fail("test_invalid_outer_scalars_raise", f"expected ValueError for {why}")
    _pass("test_invalid_outer_scalars_raise")


if __name__ == "__main__":
    fns = [f for f in dir(sys.modules[__name__]) if f.startswith("test_")]
    passed, failed = 0, 0
    for name in sorted(fns):
        try:
            globals()[name]()
            passed += 1
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e!r}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
