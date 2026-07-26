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


if __name__ == "__main__":
    for t in (test_remap_preserves_the_shape_across_a_grid_change,
              test_remap_is_not_the_same_as_rebuilding,
              test_remapped_field_is_a_usable_signed_distance_field):
        _run(t)
    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(1 if _failed else 0)
