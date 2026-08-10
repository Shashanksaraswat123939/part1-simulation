"""
test_remap_geometry.py — coarse-to-fine phi remap.

Guards the capability that makes a "big steps early, small steps later"
schedule possible at all. Surface travel per HJ iteration is CFL x spacing and
phi cost scales with spacing^-3, so carving on a coarse grid then polishing on a
fine one is ~12x fewer iterations and ~60x cheaper per iteration than doing it
all at 0.5 mm. Without a remap that is impossible: warm_start rebuilds a fresh
field, so dropping to a finer grid means starting from a brick again.

The property that matters is that the ZERO LEVEL SET survives the transfer --
not that the arrays match, which they cannot (different resolutions).
"""
import sys
from pathlib import Path

_P1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_P1))
sys.path.insert(0, str(_P1 / "sandbox"))
sys.path.insert(0, str(_P1.parent / "part2-simulation"))

import numpy as np

_passed = _failed = 0


def _run(fn):
    global _passed, _failed
    try:
        fn(); print(f"PASS {fn.__name__}"); _passed += 1
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {fn.__name__}: {exc}"); _failed += 1


def _carve(geom, n, up):
    """Erode the field a little so the shape is NOT the trivial full brick."""
    from phi_updater import hj_update, cfl_limited_dt, reinitialise_sdf
    for _ in range(n):
        vel = np.full(geom.phi.grid.shape, -1.0)   # uniform inward
        hj_update(geom.phi, vel, cfl_limited_dt(vel, 0.5))
        reinitialise_sdf(geom.phi, n_steps=3)
        up.enforce_symmetry(geom)
    return geom


def test_remap_preserves_the_shape_across_a_grid_change():
    import coarse
    coarse.use_spacing(3.0)
    import unified_phi as up

    src = up.build_unified_geometry(130.0, 46.0, 20.0, init_mode="full", seed=0)
    up.enforce_symmetry(src)
    _carve(src, 4, up)
    src_vol = up.extract_unified_surface(src, allow_inaccessible=True)[0].volume
    assert src.spacing_m == 0.003, f"source spacing not recorded: {src.spacing_m}"

    coarse.use_spacing(1.5)                      # refine
    dst = up.remap_geometry(src)

    assert dst.spacing_m == 0.0015, dst.spacing_m
    assert dst.region.shape != src.region.shape, "grid did not actually change"
    dst_vol = up.extract_unified_surface(dst, allow_inaccessible=True)[0].volume

    # Same solid, twice the resolution: volume must agree to a few percent.
    # A rebuilt-from-scratch field would come back as the FULL brick, which is
    # far bigger than a carved one -- that is the failure this catches.
    rel = abs(dst_vol - src_vol) / src_vol
    assert rel < 0.10, (
        f"volume changed {rel*100:.1f}% across the remap "
        f"({src_vol*1e9:.0f} -> {dst_vol*1e9:.0f} mm^3); the shape did not survive"
    )


def test_remap_is_not_the_same_as_rebuilding():
    """The whole point: a rebuild loses the carving, a remap keeps it."""
    import coarse
    coarse.use_spacing(3.0)
    import unified_phi as up

    src = up.build_unified_geometry(130.0, 46.0, 20.0, init_mode="full", seed=0)
    up.enforce_symmetry(src)
    _carve(src, 6, up)
    carved_vol = up.extract_unified_surface(src, allow_inaccessible=True)[0].volume

    coarse.use_spacing(1.5)
    remapped = up.remap_geometry(src)
    rebuilt = up.build_unified_geometry(130.0, 46.0, 20.0, init_mode="full", seed=0)
    up.enforce_symmetry(rebuilt)

    v_remap = up.extract_unified_surface(remapped, allow_inaccessible=True)[0].volume
    v_rebuild = up.extract_unified_surface(rebuilt, allow_inaccessible=True)[0].volume

    assert v_remap < v_rebuild, (
        f"remapped ({v_remap*1e9:.0f} mm^3) should be SMALLER than a fresh brick "
        f"({v_rebuild*1e9:.0f} mm^3) -- it carries the carving"
    )
    assert abs(v_remap - carved_vol) / carved_vol < abs(v_rebuild - carved_vol) / carved_vol, (
        "the remap must be closer to the carved shape than a rebuild is")


def test_remapped_field_is_a_usable_signed_distance_field():
    """HJ steps assume |grad phi| ~ 1; interpolation breaks that, reinit fixes it."""
    import coarse
    coarse.use_spacing(3.0)
    import unified_phi as up

    src = up.build_unified_geometry(130.0, 46.0, 20.0, init_mode="full", seed=0)
    up.enforce_symmetry(src)
    _carve(src, 3, up)

    coarse.use_spacing(1.5)
    dst = up.remap_geometry(src)

    g = dst.phi.grid.astype(np.float64)
    gx, gy, gz = np.gradient(g, dst.spacing_m)
    mag = np.sqrt(gx**2 + gy**2 + gz**2)
    band = np.abs(g) < 3 * dst.spacing_m      # only meaningful near the surface
    assert band.sum() > 1000, "no narrow band found"
    med = float(np.median(mag[band]))
    assert 0.5 < med < 1.6, f"|grad phi| median {med:.3f} near the surface, expected ~1"
    assert np.isfinite(g).all(), "non-finite values in the remapped field"




def test_warm_start_carries_a_CARVED_field_to_the_next_d_halo():
    """Stage 2 warm-starts each d_halo from the previous one's converged phi.

    Until the redistancing fix of 2026-07-30 the geometry never changed, so
    every warm start remapped an untouched envelope and this path had never run
    on a carved car -- the only shape it will ever see in production. Three
    things have to survive the remap, and a fourth has to still work after it:

      * the CARVING, not just the field. A remap that quietly reset to the
        envelope would look fine (valid geometry, plausible mass) and silently
        throw away every CFD solve spent getting there;
      * |grad phi| ~ 1, or the next update is the no-op this whole fix was
        about;
      * a watertight single-body extraction, since the next iteration meshes it;
      * one more update step, at the NEW d_halo.

    d_halo 43.72 is in the list on purpose: it is the value that killed two
    earlier runs.
    """
    import copy
    import warnings
    from types import SimpleNamespace

    import numpy as np

    # Set BEFORE the GRID_SPACING_M import below, which binds a value rather
    # than a live reference. This test used to inherit 2 mm from whichever file
    # pytest collected earlier; at the production 0.3 mm the build below takes
    # ~15 min and the cargo-erosion comment further down stops making sense.
    import coarse
    coarse.use_spacing(2.0)

    import phi_updater as pu
    from geometry_contract import GRID_SPACING_M
    from unified_phi import (build_unified_geometry, compute_mass_com,
                             extract_half_surface, extract_unified_surface,
                             remap_geometry)

    grads = {"dT_dD20": 0.4868, "dT_dmass": 17.4478, "dT_dh_com": -0.2391,
             "dT_dx_com": 0.000251, "dT_dL": -0.004868}

    class _MR:
        total_mass_kg, com_x_m, com_z_m = 0.1494, 0.122, 0.0296

    def _mass(g):
        return sum(c.mass_kg for c in compute_mass_com(g))

    def _band_grad(g):
        # g.spacing_m, not the captured global: a remap can change the spacing
        # under this helper, and an empty band makes np.median return NaN,
        # which compares False against every threshold and reads as a failure.
        gm = pu._grad_magnitude(g.phi.grid.astype(np.float64))
        band = np.abs(g.phi.grid) < 2.0 * (g.spacing_m or GRID_SPACING_M)
        assert band.any(), "no interface band found -- spacing mismatch"
        return float(np.median(gm[band]))

    # x_front 42.9 is what Stage 1 chose for the live 2026-07-29 run. 46.0
    # paired with d_halo=43.72 is infeasible -- the cargo collides with the halo
    # pocket, remap_geometry says so, and run_two_stage's _feasible_d_halo would
    # never hand that pair to the loop. Using the real pair keeps this test
    # about warm starting rather than about placement validity.
    # with_cargo=False on purpose. The cargo-erosion guard is measured in CELLS
    # and this test runs coarse, so it rejects at 2 mm placements that are legal
    # at the production 0.5 mm (measured: Stage 1's own choice for the live run,
    # x_start 42.9 mm / flip, reads 6.9% eroded at coarse and builds fine at
    # production spacing). That is a real wrinkle in the guard, but it is not
    # what this test is about -- carrying a CARVED field across a d_halo change
    # is, and cargo placement validity has its own tests in test_virtual_cargo.
    base = build_unified_geometry(120.0, 42.9, 20.0, init_mode="full",
                                  with_cargo=False)
    verts = np.asarray(extract_half_surface(base).vertices)
    rng = np.random.default_rng(3)
    sens = 3.497e3 * rng.normal(size=len(verts))
    weights = {"w_aero": 1.0, "w_mass": 1.0, "w_com": 0.0, "w_mfg": 0.0}

    def _step(g):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pu.apply_adjoint_to_unified(
                g, sens, SimpleNamespace(vertices=verts), 0.5, weights,
                grads, _MR())

    envelope_mass = _mass(base)
    carved = copy.deepcopy(base)
    for _ in range(4):
        _step(carved)
    carved_mass = _mass(carved)
    assert carved_mass < 0.95 * envelope_mass, (
        f"the carve did not remove material ({envelope_mass*1000:.2f} g -> "
        f"{carved_mass*1000:.2f} g); this test cannot say anything about warm "
        "starting a carved field if the field was never carved")

    for new_d in (16.0, 43.72):
        w = remap_geometry(carved, W_mm=120.0, x_front_mm=42.9,
                           d_halo_mm=new_d, with_cargo=False)
        assert abs(w.d_halo_mm - new_d) < 1e-9, "remap ignored the new d_halo"

        m = _mass(w)
        assert m < 0.95 * envelope_mass, (
            f"warm start to d_halo={new_d} came back at {m*1000:.2f} g against "
            f"a {envelope_mass*1000:.2f} g envelope and a {carved_mass*1000:.2f} g "
            "carved car -- the remap reset the geometry and threw away every "
            "CFD solve that produced it")

        g = _band_grad(w)
        assert abs(g - 1.0) < 0.05, (
            f"|grad phi| is {g:.3f} after warm starting to d_halo={new_d}, not "
            "~1; the first update at the new d_halo would be a no-op")

        mesh, _rep = extract_unified_surface(w, allow_inaccessible=True)
        assert mesh.is_watertight and mesh.body_count == 1, (
            f"warm start to d_halo={new_d} produced a mesh that is "
            f"watertight={mesh.is_watertight} with {mesh.body_count} bodies; "
            "the next iteration has to hand this to snappyHexMesh")

        before = _mass(w)
        _step(w)
        assert _mass(w) < before, (
            f"the first update after warm starting to d_halo={new_d} removed "
            "nothing")


if __name__ == "__main__":
    # Collected by name; a hand-written call list silently drops every test
    # appended below it, which has already hidden several tests in this repo.
    import sys as _sys
    _mod = _sys.modules[__name__]
    _passed = _failed = 0
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        try:
            getattr(_mod, _n)()
            print(f"PASS {_n}")
            _passed += 1
        except Exception as _e:  # noqa: BLE001
            print(f"FAIL {_n}: {_e}")
            _failed += 1
    print(f"{_passed} passed, {_failed} failed")
    _sys.exit(1 if _failed else 0)
