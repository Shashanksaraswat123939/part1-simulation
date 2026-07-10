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

if __name__ == "__main__":
    test_godunov_gradient_returns_array()
    test_grad_magnitude_positive()
    test_hj_update_changes_grid()
    test_hj_update_preserves_hard_constraints()
    test_reinitialise_sdf_runs()
    test_extend_velocity_returns_same_shape()
    test_combine_gradients_normalizes()
    test_combine_gradients_zero_safe()
    print("\nAll phi_updater tests passed.")