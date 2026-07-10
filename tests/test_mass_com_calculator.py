"""
Tests for mass_com_calculator.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from mass_com_calculator import (
    compute_component_mass_com, compute_sidepod_pair_mass_com,
    compute_all_machined_components, ComponentMassCOM,
)
from phi_grid import PhiGrid
from bounding_volumes import BoundingRegion
from geometry_contract import GRID_SPACING_M, DENSITY_NOSE_KGM3, DENSITY_BODY_KGM3

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

def _make_phi_with_solid(component="main_body", nx=20, ny=20, nz=20):
    bv = BoundingRegion(component, (0.0, -0.003, 0.0), (nx, ny, nz))
    solid = np.zeros((nx, ny, nz), dtype=bool)
    air = np.zeros((nx, ny, nz), dtype=bool)
    air[0, :, :] = True; air[-1, :, :] = True
    air[:, 0, :] = True; air[:, -1, :] = True
    air[:, :, 0] = True; air[:, :, -1] = True
    phi = PhiGrid(component, bv, np.zeros((nx,ny,nz), dtype=np.float32), solid, air)
    phi.init("sphere")
    return phi

def test_compute_component_mass_positive():
    phi = _make_phi_with_solid()
    result = compute_component_mass_com(phi, DENSITY_BODY_KGM3)
    assert result.mass_kg > 0, f"Mass should be positive, got {result.mass_kg}"
    _pass("test_compute_component_mass_positive")

def test_compute_component_mass_com_in_bounds():
    phi = _make_phi_with_solid("main_body", 20, 20, 20)
    result = compute_component_mass_com(phi, DENSITY_BODY_KGM3)
    ox, oy, oz = phi.bv.origin_m
    x_max = ox + 20 * GRID_SPACING_M
    y_max = oy + 20 * GRID_SPACING_M
    z_max = oz + 20 * GRID_SPACING_M
    assert ox <= result.com_x_m <= x_max, f"COM x={result.com_x_m} outside [{ox}, {x_max}]"
    assert oy <= result.com_y_m <= y_max, f"COM y={result.com_y_m} outside [{oy}, {y_max}]"
    assert oz <= result.com_z_m <= z_max, f"COM z={result.com_z_m} outside [{oz}, {z_max}]"
    _pass("test_compute_component_mass_com_in_bounds")

def test_zero_solid_cells_returns_zero_mass():
    bv = BoundingRegion("main_body", (0.0, -0.003, 0.0), (20, 20, 20))
    solid = np.zeros((20, 20, 20), dtype=bool)
    air = np.ones((20, 20, 20), dtype=bool)  # all air
    phi = PhiGrid("main_body", bv, np.ones((20,20,20), dtype=np.float32), solid, air)
    result = compute_component_mass_com(phi, DENSITY_BODY_KGM3)
    assert result.mass_kg == 0.0, f"Zero solid should give zero mass, got {result.mass_kg}"
    _pass("test_zero_solid_cells_returns_zero_mass")

def test_sidepod_pair_mass_is_double():
    phi = _make_phi_with_solid("sidepod", 20, 10, 20)
    right = compute_component_mass_com(phi, 163.0)
    pair = compute_sidepod_pair_mass_com(phi)
    assert abs(pair.mass_kg - 2.0 * right.mass_kg) < 1e-12, "Pair mass should be 2x right"
    assert abs(pair.com_y_m) < 1e-12, f"Pair COM y should be 0 by symmetry, got {pair.com_y_m}"
    _pass("test_sidepod_pair_mass_is_double")

def test_sidepod_pair_com_x_matches_right():
    phi = _make_phi_with_solid("sidepod", 20, 10, 20)
    right = compute_component_mass_com(phi, 163.0)
    pair = compute_sidepod_pair_mass_com(phi)
    assert abs(pair.com_x_m - right.com_x_m) < 1e-12
    assert abs(pair.com_z_m - right.com_z_m) < 1e-12
    _pass("test_sidepod_pair_com_x_matches_right")

def test_compute_all_returns_four_components():
    nose = _make_phi_with_solid("nose", 10, 20, 20)
    sidepod = _make_phi_with_solid("sidepod", 20, 10, 20)
    rearpod = _make_phi_with_solid("rearpod", 10, 20, 20)
    body = _make_phi_with_solid("main_body", 30, 20, 20)
    results = compute_all_machined_components(nose, sidepod, rearpod, body)
    assert len(results) == 4, f"Expected 4 components, got {len(results)}"
    names = [r.name for r in results]
    assert names == ["nose", "sidepod", "rearpod", "main_body"], f"Names: {names}"
    _pass("test_compute_all_returns_four_components")

def test_nose_density_higher_than_body():
    phi_nose = _make_phi_with_solid("nose", 10, 10, 10)
    phi_body = _make_phi_with_solid("main_body", 10, 10, 10)
    nose = compute_component_mass_com(phi_nose, DENSITY_NOSE_KGM3)
    body = compute_component_mass_com(phi_body, DENSITY_BODY_KGM3)
    assert nose.mass_kg > body.mass_kg, "Nose (1000 kg/m^3) should be heavier than body (163 kg/m^3)"
    _pass("test_nose_density_higher_than_body")

if __name__ == "__main__":
    test_compute_component_mass_positive()
    test_compute_component_mass_com_in_bounds()
    test_zero_solid_cells_returns_zero_mass()
    test_sidepod_pair_mass_is_double()
    test_sidepod_pair_com_x_matches_right()
    test_compute_all_returns_four_components()
    test_nose_density_higher_than_body()
    print("\nAll mass_com_calculator tests passed.")