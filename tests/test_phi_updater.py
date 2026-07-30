"""
Tests for phi_updater.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from phi_updater import (
    hj_update, reinitialise_sdf, combine_gradients, extend_velocity,
    _godunov_gradient, _grad_magnitude,
    _splat_vertex_sensitivity_to_grid, apply_adjoint_sensitivity_symmetric,
)
from phi_grid import PhiGrid
from bounding_volumes import BoundingRegion
from geometry_contract import GRID_SPACING_M

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

def _make_phi(nx=20, ny=20, nz=20):
    bv = BoundingRegion("main_body", (0.0, -0.003, 0.0), (nx, ny, nz))
    solid = np.zeros((nx, ny, nz), dtype=bool)
    air = np.zeros((nx, ny, nz), dtype=bool)
    air[0, :, :] = True; air[-1, :, :] = True
    air[:, 0, :] = True; air[:, -1, :] = True
    air[:, :, 0] = True; air[:, :, -1] = True
    phi = PhiGrid("main_body", bv, np.zeros((nx,ny,nz), dtype=np.float32), solid, air)
    phi.init("sphere")
    return phi

def test_godunov_gradient_returns_array():
    phi = _make_phi()
    g = _godunov_gradient(phi.grid, 0)
    assert g.shape == phi.grid.shape, f"Gradient shape {g.shape} != {phi.grid.shape}"
    _pass("test_godunov_gradient_returns_array")

def test_grad_magnitude_positive():
    phi = _make_phi()
    gm = _grad_magnitude(phi.grid)
    assert np.all(gm >= 0), "Gradient magnitude should be non-negative"
    _pass("test_grad_magnitude_positive")

def test_hj_update_changes_grid():
    phi = _make_phi()
    grid_before = phi.grid.copy()
    velocity = np.ones_like(phi.grid, dtype=np.float64) * 0.001
    hj_update(phi, velocity, dt=0.1)
    assert not np.allclose(grid_before, phi.grid), "Grid should change after HJ update"
    _pass("test_hj_update_changes_grid")

def test_hj_update_preserves_hard_constraints():
    phi = _make_phi()
    velocity = np.ones_like(phi.grid, dtype=np.float64) * 0.001
    hj_update(phi, velocity, dt=0.1)
    # Air mask cells should still have phi > 0
    assert np.all(phi.grid[phi.hard_mask_air] > 0), "Air mask should still have phi > 0"
    # Solid mask cells should still have phi < 0
    assert np.all(phi.grid[phi.hard_mask_solid] < 0), "Solid mask should still have phi < 0"
    _pass("test_hj_update_preserves_hard_constraints")

def test_reinitialise_sdf_runs():
    phi = _make_phi()
    grid_before = phi.grid.copy()
    reinitialise_sdf(phi, n_steps=3, dt_reinit=0.1)
    assert phi.grid.shape == grid_before.shape
    _pass("test_reinitialise_sdf_runs")

def test_extend_velocity_returns_same_shape():
    phi_grid = np.random.randn(10, 10, 10).astype(np.float32)
    surface_vel = np.zeros((10, 10, 10), dtype=np.float64)
    surface_vel[5, 5, 5] = 1.0
    F = extend_velocity(phi_grid, surface_vel, n_steps=3, dt_ext=0.1)
    assert F.shape == phi_grid.shape, f"Velocity field shape {F.shape} != {phi_grid.shape}"
    _pass("test_extend_velocity_returns_same_shape")

def test_combine_gradients_normalizes():
    aero = np.ones((5, 5, 5)) * 100.0
    mass = np.ones((5, 5, 5)) * 0.001
    com = np.ones((5, 5, 5)) * 50.0
    mfg = np.ones((5, 5, 5)) * 0.1
    result = combine_gradients(aero, mass, com, mfg, 1.0, 1.0, 1.0, 1.0)
    # Each gradient normalized to unit RMS. With all-ones arrays (positive),
    # all normalized gradients are identical. Weighted sum = 4 per element.
    rms = np.sqrt(np.mean(result ** 2))
    assert 0.5 < rms < 6.0, f"Combined RMS {rms} not in expected range"
    _pass("test_combine_gradients_normalizes")

def test_combine_gradients_zero_safe():
    zeros = np.zeros((5, 5, 5))
    result = combine_gradients(zeros, zeros, zeros, zeros, 1.0, 1.0, 1.0, 1.0)
    assert np.allclose(result, 0.0), "Zero gradients should give zero result"
    _pass("test_combine_gradients_zero_safe")


def test_splat_vertex_sensitivity_places_value_in_correct_cell():
    # Live value: _real_geom() rewrites geometry_contract.GRID_SPACING_M via
    # coarse.use_spacing, and the module-level import above froze the old one.
    import geometry_contract as _gc
    phi = _make_phi()
    ox, oy, oz = phi.bv.origin_m
    dx = _gc.GRID_SPACING_M
    # Single vertex placed exactly at cell (5,6,7)'s centre
    target = (ox + 5 * dx, oy + 6 * dx, oz + 7 * dx)
    vertices = np.array([target])
    sensitivity = np.array([3.5])
    grid = _splat_vertex_sensitivity_to_grid(sensitivity, vertices, phi)
    assert grid.shape == phi.bv.shape
    assert abs(grid[5, 6, 7] - 3.5) < 1e-9, f"Expected 3.5 at (5,6,7), got {grid[5,6,7]}"
    # Everywhere else should still be zero
    grid_copy = grid.copy()
    grid_copy[5, 6, 7] = 0.0
    assert np.allclose(grid_copy, 0.0), "Non-target cells should remain zero"
    _pass("test_splat_vertex_sensitivity_places_value_in_correct_cell")


def test_splat_vertex_sensitivity_averages_multiple_hits():
    import geometry_contract as _gc
    phi = _make_phi()
    ox, oy, oz = phi.bv.origin_m
    dx = _gc.GRID_SPACING_M
    target = (ox + 5 * dx, oy + 6 * dx, oz + 7 * dx)
    # Two vertices very close together, both rounding to the same cell
    vertices = np.array([target, (target[0] + 1e-6, target[1], target[2])])
    sensitivity = np.array([2.0, 4.0])
    grid = _splat_vertex_sensitivity_to_grid(sensitivity, vertices, phi)
    assert abs(grid[5, 6, 7] - 3.0) < 1e-9, f"Expected average 3.0, got {grid[5,6,7]}"
    _pass("test_splat_vertex_sensitivity_averages_multiple_hits")


def test_apply_adjoint_sensitivity_mismatched_lengths_raises():
    phi = _make_phi()

    class _FakeMesh:
        vertices = np.array([[0.0, 0.0, 0.0], [0.001, 0.001, 0.001]])

    try:
        apply_adjoint_sensitivity_symmetric(
            {"main_body": phi}, np.array([1.0]), _FakeMesh(), dt=1e-4,
            gradient_weights={},
        )
        _fail("test_apply_adjoint_sensitivity_mismatched_lengths_raises", "should have raised ValueError")
    except ValueError:
        _pass("test_apply_adjoint_sensitivity_mismatched_lengths_raises")


def test_apply_adjoint_sensitivity_raises_on_none():
    # P1-13 fix: a None sensitivity/mesh used to silently warn-and-skip,
    # which in the optimizer loop meant DeltaT=0 -> false convergence. It
    # now raises ValueError instead, so a skipped update can never masquerade
    # as a converged result.
    phi = _make_phi()
    try:
        apply_adjoint_sensitivity_symmetric(
            {"main_body": phi}, None, None, dt=1e-4, gradient_weights={},
        )
        _fail("test_apply_adjoint_sensitivity_raises_on_none", "expected ValueError")
    except ValueError:
        _pass("test_apply_adjoint_sensitivity_raises_on_none")


def test_apply_adjoint_sensitivity_updates_symmetric_component():
    """
    A symmetric component (main_body) must receive contributions mirrored
    across y=0: a vertex at y>0 also influences the corresponding y<0 cell.
    """
    nx, ny, nz = 20, 21, 20   # odd ny so y=0 sits exactly on a grid line
    bv = BoundingRegion("main_body", (0.0, -0.0031, 0.0), (nx, ny, nz))
    solid = np.zeros((nx, ny, nz), dtype=bool)
    air = np.zeros((nx, ny, nz), dtype=bool)
    air[0, :, :] = True; air[-1, :, :] = True
    air[:, 0, :] = True; air[:, -1, :] = True
    air[:, :, 0] = True; air[:, :, -1] = True
    phi = PhiGrid("main_body", bv, np.zeros((nx, ny, nz), dtype=np.float32), solid, air)
    phi.init("sphere")
    grid_before = phi.grid.copy()

    class _FakeMesh:
        # A single vertex well inside the grid, off-centre in y
        vertices = np.array([[bv.origin_m[0] + 10 * GRID_SPACING_M,
                               bv.origin_m[1] + 15 * GRID_SPACING_M,
                               bv.origin_m[2] + 10 * GRID_SPACING_M]])

    sensitivity = np.array([5.0])
    apply_adjoint_sensitivity_symmetric(
        {"main_body": phi}, sensitivity, _FakeMesh(), dt=1e-6,
        # K-3 fix: strict key names w_aero/w_mass/w_com/w_mfg (was
        # aero/mass/com/mfg, silently discarded via .get()-with-default).
        gradient_weights={"w_aero": 1.0, "w_mass": 0.0, "w_com": 0.0, "w_mfg": 0.0},
    )
    assert not np.array_equal(phi.grid, grid_before), "Symmetric component grid should change"
    _pass("test_apply_adjoint_sensitivity_updates_symmetric_component")


def _make_unified_fake(nx=24, ny=25, nz=24):
    """Minimal stand-in for UnifiedGeometry: apply_adjoint_to_unified only ever
    touches .phi, .shape and .labels."""
    from types import SimpleNamespace
    from unified_phi import LABEL_MAIN_BODY
    bv = BoundingRegion("main_body",
                        (0.0, -(ny // 2) * GRID_SPACING_M, 0.0), (nx, ny, nz))
    solid = np.zeros((nx, ny, nz), dtype=bool)
    air = np.zeros((nx, ny, nz), dtype=bool)
    air[0, :, :] = True; air[-1, :, :] = True
    air[:, 0, :] = True; air[:, -1, :] = True
    air[:, :, 0] = True; air[:, :, -1] = True
    phi = PhiGrid("main_body", bv, np.zeros((nx, ny, nz), dtype=np.float32),
                  solid, air)
    phi.init("sphere")
    labels = np.full((nx, ny, nz), LABEL_MAIN_BODY, dtype=np.int16)
    return SimpleNamespace(phi=phi, shape=(nx, ny, nz), labels=labels)


def _surface_vertices(geom, n=300):
    """Vertices scattered through the interior of the grid, y > 0."""
    ox, oy, oz = geom.phi.bv.origin_m
    nx, ny, nz = geom.shape
    rng = np.random.default_rng(0)
    i = rng.integers(2, nx - 2, n)
    j = rng.integers(ny // 2, ny - 2, n)
    k = rng.integers(2, nz - 2, n)
    return np.stack([ox + i * GRID_SPACING_M,
                     oy + j * GRID_SPACING_M,
                     oz + k * GRID_SPACING_M], axis=1)


_MASS_REPORT = {"total_mass_kg": 0.050, "com_x_m": 0.10, "com_z_m": 0.025}
_GRADS = {"dT_dmass": 4.0, "dT_dh_com": 0.5, "dT_dx_com": 0.1}


def _run_update(sens, verts, w_mass=0.0):
    from types import SimpleNamespace
    from phi_updater import apply_adjoint_to_unified
    geom = _make_unified_fake()
    apply_adjoint_to_unified(
        geom, sens, SimpleNamespace(vertices=verts), dt=1e-6,
        gradient_weights={"w_aero": 1.0, "w_mass": w_mass,
                          "w_com": 0.0, "w_mfg": 0.0},
        objective_gradients=_GRADS, mass_report=_MASS_REPORT,
    )
    return geom.phi.grid.copy()



# --------------------------------------------------------------------------
# Real-geometry probes.
#
# The _make_unified_fake toy above is a ~20-cell grid, and the update now
# redistances: reinitialise_sdf runs 50 pseudo-steps at 0.4 cells each, which
# on a 20-cell grid propagates across the WHOLE domain and reconverges to the
# exact distance function of the toy's near-spherical interface, erasing the
# local detail the step just produced. On the real 8M-cell grid those same 50
# steps reach only 20 cells, so local structure survives -- which is why the
# toy stopped being able to see aero steering while the real geometry still
# can. Tests that ask "did the shape change" therefore build a real car at a
# coarse spacing and measure the SOLID CELL COUNT (the geometry) rather than
# raw phi values (which redistancing rewrites far from the surface).
# --------------------------------------------------------------------------
_REAL_CACHE = {}


def _real_geom():
    import os
    import sys as _s
    _here = os.path.dirname(os.path.abspath(__file__))
    _sb = os.path.join(os.path.dirname(_here), "sandbox")
    if _sb not in _s.path:
        _s.path.insert(0, _sb)
    if "geom" not in _REAL_CACHE:
        import coarse
        coarse.use_spacing(2.0)
        from unified_phi import build_unified_geometry, extract_half_surface
        g = build_unified_geometry(120.0, 46.0, 20.0, init_mode="full",
                                   with_cargo=False)
        _REAL_CACHE["geom"] = g
        _REAL_CACHE["verts"] = np.asarray(extract_half_surface(g).vertices)
    return _REAL_CACHE["geom"], _REAL_CACHE["verts"]


_REAL_GRADS = {"dT_dD20": 0.4868, "dT_dmass": 17.4478, "dT_dh_com": -0.2391,
               "dT_dx_com": 0.000251, "dT_dL": -0.004868}


class _RealMassReport:
    total_mass_kg, com_x_m, com_z_m = 0.1494, 0.122, 0.0296


def _step_real(sens_scale, w_mass, sign=1.0, seed=3):
    """One production update on a real car; returns (solid cell count, geom)."""
    import copy
    import warnings as _w
    from types import SimpleNamespace
    from phi_updater import apply_adjoint_to_unified
    base, verts = _real_geom()
    rng = np.random.default_rng(seed)
    sens = sign * sens_scale * rng.normal(size=len(verts))
    g = copy.deepcopy(base)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        apply_adjoint_to_unified(
            g, sens, SimpleNamespace(vertices=verts), 0.5,
            {"w_aero": 1.0, "w_mass": w_mass, "w_com": 0.0, "w_mfg": 0.0},
            _REAL_GRADS, _RealMassReport())
    return int((g.phi.grid < 0).sum()), g


def test_aero_gradient_actually_steers_the_shape():
    """THE regression test for the 2026-07-27 silent failure.

    With the mass term off, the shape update is the aero gradient and nothing
    else, so negating the sensitivity MUST produce a materially different body.
    It did not for a full smoke run: the adjoint mesh-movement chain returned a
    field whose top 10 points carried 99.33% of the sum of squares, unit-RMS
    normalisation crushed everything else, and flipping the sign moved the
    geometry by 0.04%. Every test passed throughout, because they all fed a
    synthetic well-conditioned sensitivity to a toy grid.

    Measured on the solid cell count, not on phi: the update redistances, so
    phi far from the surface is rewritten every step and a phi-difference norm
    would mostly measure that rebuild.
    """
    pos, _ = _step_real(3.497e3, w_mass=0.0, sign=+1.0)
    neg, _ = _step_real(3.497e3, w_mass=0.0, sign=-1.0)
    assert pos != neg, (
        "negating the sensitivity left the same {:,} solid cells -- the aero "
        "gradient is not steering the shape".format(pos))


def test_gradient_balance_is_physical_not_normalised():
    """Doubling the aero gradient must change its share of the update.

    Under unit-RMS normalisation it could not: each field was rescaled to the
    same RMS before weighting, so the magnitudes the JAX objective computed
    were discarded and w_aero:w_mass set the balance instead. That is how a
    completely inert aero channel went unnoticed.

    The mass term must be ON for this to mean anything. With w_mass=0 the whole
    velocity is aero and cfl_limited_dt divides dt by max|V|, so the step is
    exactly CFL*dx whatever the sensitivity's scale and the result is
    scale-invariant BY CONSTRUCTION (measured: 1.0, 1e2 and 1e4 give
    bit-identical fields). Only against a fixed mass term does aero's magnitude
    change the balance.
    """
    single, _ = _step_real(3.497e3, w_mass=1.0)
    double, _ = _step_real(2.0 * 3.497e3, w_mass=1.0)
    assert single != double, (
        "doubling the aero sensitivity left the same {:,} solid cells -- the "
        "magnitude is being normalised away, which is the bug this "
        "replaced".format(single))


def test_weights_are_ablation_switches_not_magnitude_setters():
    """w_mass=0 must remove the mass term and nothing else."""
    geom0 = _make_unified_fake()
    verts = _surface_vertices(geom0)
    rng = np.random.default_rng(8)
    sens = rng.normal(size=len(verts))

    with_mass = _run_update(sens, verts, w_mass=1.0)
    without = _run_update(sens, verts, w_mass=0.0)
    assert not np.allclose(with_mass, without), (
        "w_mass had no effect; --aero-only would be a no-op")
    _pass("test_weights_are_ablation_switches_not_magnitude_setters")


def test_spiky_sensitivity_warns_and_still_steers():
    """A field like the real diverged one must be flagged AND survive the clip.

    Ten values at 1e55 against a unit-scale background reproduces the measured
    99.33% concentration that made the aero channel inert. The guard must warn,
    and the clipped field must still move the geometry -- a clip that flattened
    the signal would be the same silent failure by another route.

    On the real car, not the toy: the update redistances, and on a 20-cell grid
    that reconverges the whole domain to the exact distance function of the
    toy's interface, erasing what the step just did.
    """
    import warnings as _w
    base, verts = _real_geom()
    rng = np.random.default_rng(11)
    sens = 3.497e3 * rng.normal(size=len(verts))
    sens[:10] = 1.0e55

    import copy
    from types import SimpleNamespace
    from phi_updater import apply_adjoint_to_unified
    g = copy.deepcopy(base)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        apply_adjoint_to_unified(
            g, sens, SimpleNamespace(vertices=verts), 0.5,
            {"w_aero": 1.0, "w_mass": 1.0, "w_com": 0.0, "w_mfg": 0.0},
            _REAL_GRADS, _RealMassReport())
    assert any("spiky" in str(c.message).lower() for c in caught), (
        "a sensitivity with 10 points carrying ~all the energy was not "
        "flagged; that field made the aero channel inert for a whole smoke "
        "run. Warnings seen: " + "; ".join(str(c.message)[:60] for c in caught))

    before = int((base.phi.grid < 0).sum())
    after = int((g.phi.grid < 0).sum())
    assert before != after, (
        "clipped spiky sensitivity still does not steer the shape")


def test_non_finite_sensitivity_raises():
    geom0 = _make_unified_fake()
    verts = _surface_vertices(geom0)
    sens = np.ones(len(verts))
    sens[3] = np.nan
    try:
        _run_update(sens, verts)
    except ValueError as exc:
        assert "non-finite" in str(exc), f"unexpected message: {exc}"
        _pass("test_non_finite_sensitivity_raises")
        return
    _fail("test_non_finite_sensitivity_raises", "no ValueError for NaN sensitivity")



def test_the_update_actually_moves_the_surface():
    """A step must redistance, or it is a no-op.

    hj_update steps `phi - dt*V*|grad phi|`, and cfl_limited_dt sizes dt on the
    assumption that |grad phi| ~ 1. build_unified_geometry does NOT hand back a
    distance function -- measured on a freshly built car, |grad phi| in the
    interface band has median 0.000 and 87.5% of band cells sit below 0.1. So
    without redistancing the velocity is multiplied by ~zero and the geometry
    stands still however good the adjoint is.

    That is not hypothetical. The live sweep of 2026-07-29 produced ten
    candidates across five d_halo values whose total mass was identical to
    0.75 mg, while D20 wandered 3-7% -- pure remeshing noise on a shape that
    never changed. Stage 1's evolution loop had always redistanced, which is
    why the no-CFD path reached the 48 g floor and this one never left 149 g.
    """
    from phi_updater import _grad_magnitude
    from geometry_contract import GRID_SPACING_M as _dx

    def band_grad_median(g):
        gm = _grad_magnitude(g.phi.grid.astype(np.float64))
        return float(np.median(gm[np.abs(g.phi.grid) < 2.0 * _dx]))

    # The as-built field is deliberately NOT a distance function: _init_field's
    # "full" mode writes a constant -GRID_SPACING_M, and redistancing it at
    # BUILD time was tried and reverted (it turns the staircase isosurface into
    # a smooth one that decimates below the 10 deg angle gate --
    # test_remap_is_not_the_same_as_rebuilding catches that). So the update owns
    # the redistancing, and does it FIRST, before the splat and the velocity
    # extension that read the field. Both ends are asserted here: flat going in,
    # a distance function coming out, and material actually removed.
    base, _verts = _real_geom()
    assert band_grad_median(base) < 0.5, (
        f"|grad phi| is {band_grad_median(base):.3f} as built. If the builder "
        "now redistances, check test_remap_is_not_the_same_as_rebuilding still "
        "passes -- that is what made this the update's job instead")

    before = int((base.phi.grid < 0).sum())
    after, g = _step_real(0.0, w_mass=1.0)

    assert abs(band_grad_median(g) - 1.0) < 0.05, (
        "|grad phi| is {:.3f} after a step, not ~1: the update did not "
        "redistance and the next step will not move the "
        "surface".format(band_grad_median(g)))
    # The no-op regime removed 0.008 g of a 108 g body in one step. Anything
    # this small is that failure coming back.
    moved = abs(after - before)
    assert moved > 0.001 * before, (
        "one step moved {:,} of {:,} solid cells ({:.4f}%) -- the shape update "
        "is not moving material (see the redistancing note in "
        "apply_adjoint_to_unified)".format(moved, before, 100.0 * moved / before))


if __name__ == "__main__":
    # Collected by name. The hand-written call list below this line silently
    # dropped every test appended after it -- the bug that has already hidden
    # five tests in this repo, including the one guarding the no-op update.
    import sys as _sys
    _mod = _sys.modules[__name__]
    _fail = 0
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        try:
            getattr(_mod, _n)()
            print(f"PASS {_n}")
        except Exception as _e:  # noqa: BLE001
            print(f"FAIL {_n}: {_e!r}")
            _fail += 1
    print("All phi_updater tests passed." if not _fail else f"{_fail} failed")
    _sys.exit(1 if _fail else 0)
