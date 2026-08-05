"""
test_halo_visibility.py -- T4.4.2 / T4.4.3 halo visibility.

T4.4.2 (front and side views, 10 pts): "Visibility of the Halo must not be
physically obstructed by any other component when viewed in the front or side
views." The rule diagram boxes only the halo ABOVE a 4.0 mm base fillet, so
bodywork may blend into the bottom 4 mm.

T4.4.3 (top view, 10 pts): "The Halo must not be physically obstructed in the
plan view except by the helmet." No fillet relief -- the whole plan outline must
be clear. This model carries no helmet, so the requirement is absolute here.

Before this constraint existed the halo was modelled only as a VOID (phi > 0 so
the optimiser could not fill the mount) and nothing stopped bodywork sitting
above, ahead of or outboard of it. Measured on a carved 79 g car: 100% of halo
cells obstructed in top view, 100% in front, 64.7% in side. Both rules were
comprehensively violated and the geometry permitted it.
"""
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "sandbox"))

import numpy as np  # noqa: E402

_passed = _failed = 0


def _run(t):
    global _passed, _failed
    try:
        t()
        print("PASS " + t.__name__)
        _passed += 1
    except Exception as exc:  # noqa: BLE001
        print("FAIL %s: %s" % (t.__name__, exc))
        _failed += 1


def _obstructed(halo, solid, axis, viewer_positive):
    """Halo cells whose line of sight to the viewer passes through solid."""
    n = halo.shape[axis]
    acc = np.zeros(np.delete(np.array(halo.shape), axis), dtype=bool)
    hit = np.zeros_like(halo, dtype=bool)
    order = reversed(range(n)) if viewer_positive else range(n)
    for s in order:
        sl = [slice(None)] * 3
        sl[axis] = s
        sl = tuple(sl)
        hit[sl] = acc & halo[sl]
        acc = acc | solid[sl]
    return hit


def _carved_car(n_iters=10):
    import coarse
    coarse.use_spacing(2.0)
    import phi_updater as pu
    from unified_phi import build_unified_geometry

    geom = build_unified_geometry(120.0, 36.0, 20.0, init_mode="full",
                                  with_cargo=False)
    grads = {"dT_dD20": 0.4618, "dT_dmass": 17.042, "dT_dh_com": 0.0,
             "dT_dx_com": 0.0, "dT_dL": 0.0}

    class _MR:
        total_mass_kg, com_x_m, com_z_m = 0.1494, 0.122, 0.0296

    w = {"w_aero": 1.0, "w_mass": 1.0, "w_com": 0.0, "w_mfg": 0.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_iters):
            pu.apply_adjoint_to_unified(
                geom, np.zeros(0), SimpleNamespace(vertices=np.zeros((0, 3))),
                0.5, w, grads, _MR())
    return geom


def test_top_view_is_completely_unobstructed():
    """T4.4.3 protects the WHOLE plan outline -- no fillet relief."""
    geom = _carved_car()
    halo = geom.fixed_hardware.halo_void_mask
    blocked = _obstructed(halo, geom.phi.grid < 0, axis=2, viewer_positive=True)
    assert blocked.sum() == 0, (
        f"{int(blocked.sum()):,} of {int(halo.sum()):,} halo cells are "
        f"obstructed from above. T4.4.3 allows obstruction only by the helmet, "
        f"which this model does not carry.")


def test_front_and_side_are_unobstructed_above_the_fillet():
    """T4.4.2 protects the halo above its 4 mm base fillet."""
    import geometry_contract as gc
    from fixed_hardware import HALO_VISIBILITY_FILLET_MM

    geom = _carved_car()
    halo = geom.fixed_hardware.halo_void_mask
    solid = geom.phi.grid < 0

    ks = np.flatnonzero(halo.any(axis=(0, 1)))
    dz_mm = gc.GRID_SPACING_M * 1000.0
    k_cut = ks[0] + int(np.ceil(HALO_VISIBILITY_FILLET_MM / dz_mm))
    protected = halo.copy()
    protected[:, :, :k_cut] = False
    assert protected.any(), "the fillet exemption swallowed the whole halo"

    front = _obstructed(halo, solid, axis=0, viewer_positive=False)
    side = _obstructed(halo, solid, axis=1, viewer_positive=True)
    bad = int(((front | side) & protected).sum())
    assert bad == 0, (
        f"{bad:,} halo cells ABOVE the {HALO_VISIBILITY_FILLET_MM:.0f} mm "
        f"fillet are obstructed in the front or side view, violating T4.4.2.")


def test_the_fillet_exemption_is_actually_used():
    """The bottom band must stay available to bodywork.

    If the constraint were applied to the FULL halo in front/side view -- the
    obvious over-strict reading -- it would forbid any blend into the halo base
    and cut away material the rule explicitly permits. This checks the
    exemption is real rather than decorative, so an over-tightening shows up.
    """
    import geometry_contract as gc
    from fixed_hardware import halo_visibility_air_mask, HALO_VISIBILITY_FILLET_MM

    geom = _carved_car(n_iters=0)
    halo = geom.fixed_hardware.halo_void_mask
    mask = halo_visibility_air_mask(halo, z_origin_m=geom.region.origin_m[2],
                                    dz_m=gc.GRID_SPACING_M)

    ks = np.flatnonzero(halo.any(axis=(0, 1)))
    dz_mm = gc.GRID_SPACING_M * 1000.0
    k_cut = ks[0] + int(np.ceil(HALO_VISIBILITY_FILLET_MM / dz_mm))

    # Directly beside the halo base, below the cut, must NOT be forced air by
    # the front/side rule. Above the cut it must be.
    ii, jj, _ = np.where(halo)
    x0, y0 = int(ii.min()), int(jj.min())
    below = mask[x0, max(y0 - 1, 0), ks[0]]
    assert not below, (
        "a cell beside the halo BASE is forced to air; T4.4.2 exempts the "
        "bottom 4 mm and this over-tightens the rule")


def test_the_constraint_binds_during_the_descent_not_after():
    """It has to be hard AIR, not a post-hoc audit.

    A rule checked only at the end lets the optimiser spend its whole budget
    building something illegal. Forcing the shadow cells into hard_mask_air
    means no candidate can ever fill them.
    """
    geom = _carved_car(n_iters=0)
    halo = geom.fixed_hardware.halo_void_mask
    ks = np.flatnonzero(halo.any(axis=(0, 1)))
    ii, jj, _ = np.where(halo)
    # A cell directly above the halo must be in the hard air mask.
    above = (int(ii.min()) + 1, int(jj.min()) + 1, int(ks[-1]) + 1)
    if above[2] < halo.shape[2]:
        assert geom.phi.hard_mask_air[above], (
            "the cell directly above the halo is not hard air, so the "
            "optimiser is free to fill it and hide the halo from the top view")




def test_the_halo_lofts_to_the_canister():
    """A continuous top surface from the halo's rear to the cartridge.

    01_generative_geometry specifies "the halo-canister loft region", and
    unified_phi's header recorded that "nothing implemented a loft and the
    boxes forbade one". Unifying the field removed the second half -- a surface
    CAN cross the main_body/rearpod boundary now -- but still nothing built
    one, and Stage 1's proxy has no aerodynamic term, so filling to the
    envelope roof cost it nothing. Measured on the carved car along the
    centreline before this existed:

        x =  90 mm   top 23.5 mm     (held down by halo visibility)
        x = 100 mm   top 51.5 mm     <- 28 mm vertical cliff
        x = 130 mm   top 53.5 mm     flat slab to the tail

    A CEILING alone does not fix it: bounding the surface took the cliff out
    (53.5 -> 38.5 mm) but mass minimisation then settled the deck at 38.5 mm
    against a 44.1 mm canister top, running PAST the cartridge instead of
    arriving at it. So the deck is required material, and this checks it
    arrives -- within a cell of the loft line at both ends.
    """
    import geometry_contract as gc
    from fixed_hardware import (halo_canister_loft_solid_mask,
                                LOFT_SKIN_THICKNESS_MM)

    geom = _carved_car(n_iters=0)
    fh = geom.fixed_hardware
    cyl = getattr(fh, "canister_cylinder", None)
    assert cyl is not None, "no canister bore geometry to loft to"

    loft = halo_canister_loft_solid_mask(
        fh.halo_void_mask, cyl,
        x_origin_m=geom.region.origin_m[0],
        y_origin_m=geom.region.origin_m[1],
        z_origin_m=geom.region.origin_m[2],
        d_m=gc.GRID_SPACING_M,
    )
    assert loft.any(), "no loft deck was built at all"

    dz = gc.GRID_SPACING_M
    halo = fh.halo_void_mask
    i_halo_rear = int(np.flatnonzero(halo.any(axis=(1, 2)))[-1])
    z_halo_top = geom.region.origin_m[2] +         float(np.flatnonzero(halo.any(axis=(0, 1)))[-1]) * dz
    z_can_top = cyl.z_center_m + cyl.radius_m
    i_can_front = int(round(
        (cyl.x_center_m - cyl.x_half_width_m - geom.region.origin_m[0]) / dz))

    xs = np.flatnonzero(loft.any(axis=(1, 2)))
    assert xs[0] > i_halo_rear, (
        "the deck starts over the halo itself, where T4.4.3 forces air")
    assert xs[-1] >= i_can_front - 1, (
        f"the deck stops at x-index {xs[-1]} but the canister front is at "
        f"{i_can_front} -- it does not reach the cartridge")

    # Its top must follow the loft line at both ends, within one cell.
    def deck_top(i):
        k = np.flatnonzero(loft[i].any(axis=0))
        return geom.region.origin_m[2] + float(k[-1]) * dz

    tol = 1.5 * dz
    assert abs(deck_top(xs[0]) - z_halo_top) <= tol, (
        f"deck starts at {deck_top(xs[0])*1000:.1f} mm, halo top is "
        f"{z_halo_top*1000:.1f} mm -- it does not leave from the halo")
    assert abs(deck_top(xs[-1]) - z_can_top) <= tol, (
        f"deck ends at {deck_top(xs[-1])*1000:.1f} mm, canister top is "
        f"{z_can_top*1000:.1f} mm -- it does not arrive at the cartridge")

    # A skin, not a filled block: the mass budget cannot afford the latter.
    thick_cells = loft.sum(axis=2).max()
    max_cells = int(round(LOFT_SKIN_THICKNESS_MM / (dz * 1000.0))) + 1
    assert thick_cells <= max_cells, (
        f"the deck is {thick_cells} cells thick; it is meant to be a "
        f"{LOFT_SKIN_THICKNESS_MM} mm skin, not a solid block")


def test_the_loft_ceiling_and_deck_agree():
    """Ceiling and deck must be the same surface, not two guesses at it."""
    import geometry_contract as gc
    from fixed_hardware import (halo_canister_loft_air_mask,
                                halo_canister_loft_solid_mask)

    geom = _carved_car(n_iters=0)
    fh = geom.fixed_hardware
    kw = dict(x_origin_m=geom.region.origin_m[0],
              z_origin_m=geom.region.origin_m[2], d_m=gc.GRID_SPACING_M)
    air = halo_canister_loft_air_mask(fh.halo_void_mask,
                                      fh.canister_cylinder, **kw)
    deck = halo_canister_loft_solid_mask(
        fh.halo_void_mask, fh.canister_cylinder,
        y_origin_m=geom.region.origin_m[1], **kw)
    overlap = int((air & deck).sum())
    assert overlap == 0, (
        f"{overlap:,} cells are both forced air (above the loft) and forced "
        f"solid (the deck). hard_solid &= ~hard_air would silently delete "
        f"them and the deck would come out full of holes.")


if __name__ == "__main__":
    _mod = sys.modules[__name__]
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        _run(getattr(_mod, _n))
    print("%d passed, %d failed" % (_passed, _failed))
    sys.exit(1 if _failed else 0)
