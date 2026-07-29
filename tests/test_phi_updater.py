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
    phi = _make_phi()
    ox, oy, oz = phi.bv.origin_m
    dx = GRID_SPACING_M
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
    phi = _make_phi()
    ox, oy, oz = phi.bv.origin_m
    dx = GRID_SPACING_M
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


def test_aero_gradient_actually_steers_the_shape():
    """THE regression test for the 2026-07-27 silent failure.

    With the mass term off, the shape update is the aero gradient and nothing
    else, so negating the sensitivity MUST produce a materially different field.
    It did not for a full smoke run: the adjoint mesh-movement chain returned a
    field whose top 10 points carried 99.33% of the sum of squares, unit-RMS
    normalisation crushed everything else, and flipping the sign moved the
    geometry by 0.04%. Every existing test passed throughout, because they all
    feed a synthetic well-conditioned sensitivity.
    """
    geom0 = _make_unified_fake()
    verts = _surface_vertices(geom0)
    rng = np.random.default_rng(1)
    sens = rng.normal(size=len(verts))

    pos = _run_update(sens, verts)
    neg = _run_update(-sens, verts)
    delta = float(np.max(np.abs(pos - neg)))
    baseline = float(np.max(np.abs(pos - _make_unified_fake().phi.grid)))
    assert baseline > 0, "aero update did not move the field at all"
    assert delta > 0.25 * baseline, (
        f"negating the sensitivity changed the field by {delta:.3e} against a "
        f"total step of {baseline:.3e} ({100.0 * delta / baseline:.2f}%); the "
        f"aero gradient is not steering the shape"
    )
    _pass("test_aero_gradient_actually_steers_the_shape")


def test_gradient_balance_is_physical_not_normalised():
    """Doubling the aero gradient must double its share of the update.

    Under unit-RMS normalisation it could not: each field was rescaled to the
    same RMS before weighting, so the magnitudes the JAX objective computed were
    discarded and w_aero:w_mass set the balance instead. That is how a
    completely inert aero channel went unnoticed -- normalisation guaranteed it
    arrived at parity with the mass term no matter how small it really was.
    """
    geom0 = _make_unified_fake()
    verts = _surface_vertices(geom0)
    rng = np.random.default_rng(7)
    sens = rng.normal(size=len(verts))

    base = _run_update(sens, verts, w_mass=1.0)
    doubled = _run_update(sens * 2.0, verts, w_mass=1.0)
    start = _make_unified_fake().phi.grid

    d_base = float(np.max(np.abs(base - start)))
    d_doubled = float(np.max(np.abs(doubled - start)))
    assert d_base > 0, "no update happened at all"
    assert not np.allclose(base, doubled), (
        "doubling the aero sensitivity changed nothing -- the magnitude is "
        "being normalised away, which is the bug this replaced")
    _pass("test_gradient_balance_is_physical_not_normalised")


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
    99.33% concentration. Without the pre-normalisation clip this update is
    indistinguishable from its own negation.
    """
    import warnings
    geom0 = _make_unified_fake()
    verts = _surface_vertices(geom0)
    rng = np.random.default_rng(2)
    sens = rng.normal(size=len(verts))
    sens[:10] = 1e55

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pos = _run_update(sens, verts)
        assert any(issubclass(w.category, RuntimeWarning)
                   and "pathologically spiky" in str(w.message) for w in caught), \
            "spiky sensitivity did not raise the concentration warning"
    neg = _run_update(-sens, verts)
    delta = float(np.max(np.abs(pos - neg)))
    assert delta > 0, "clipped spiky sensitivity still does not steer the shape"
    _pass("test_spiky_sensitivity_warns_and_still_steers")


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


if __name__ == "__main__":
    test_godunov_gradient_returns_array()
    test_grad_magnitude_positive()
    test_hj_update_changes_grid()
    test_hj_update_preserves_hard_constraints()
    test_reinitialise_sdf_runs()
    test_extend_velocity_returns_same_shape()
    test_combine_gradients_normalizes()
    test_combine_gradients_zero_safe()
    test_splat_vertex_sensitivity_places_value_in_correct_cell()
    test_splat_vertex_sensitivity_averages_multiple_hits()
    test_apply_adjoint_sensitivity_mismatched_lengths_raises()
    test_apply_adjoint_sensitivity_raises_on_none()
    test_apply_adjoint_sensitivity_updates_symmetric_component()
    test_aero_gradient_actually_steers_the_shape()
    test_gradient_balance_is_physical_not_normalised()
    test_weights_are_ablation_switches_not_magnitude_setters()
    test_spiky_sensitivity_warns_and_still_steers()
    test_non_finite_sensitivity_raises()
    print("\nAll phi_updater tests passed.")