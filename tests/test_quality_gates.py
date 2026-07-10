"""
Tests for quality_gates.py
"""
import sys, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from quality_gates import GateResult, run_quality_gates
from phi_grid import PhiGrid
from bounding_volumes import BoundingRegion

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

def _make_phi(component="main_body", nx=30, ny=30, nz=30):
    bv = BoundingRegion(component, (0.0, -0.0045, 0.0), (nx, ny, nz))
    solid = np.zeros((nx, ny, nz), dtype=bool)
    air = np.zeros((nx, ny, nz), dtype=bool)
    air[0, :, :] = True; air[-1, :, :] = True
    air[:, 0, :] = True; air[:, -1, :] = True
    air[:, :, 0] = True; air[:, :, -1] = True
    phi = PhiGrid(component, bv, np.zeros((nx,ny,nz), dtype=np.float32), solid, air)
    phi.init("sphere")
    return phi

def _make_all_phi():
    return {
        "nose":      _make_phi("nose", 15, 30, 30),
        "sidepod":   _make_phi("sidepod", 30, 15, 30),
        "rearpod":   _make_phi("rearpod", 15, 30, 30),
        "main_body": _make_phi("main_body", 40, 30, 30),
    }

def test_gate_result_validates_lifecycle_state():
    try:
        GateResult(
            lifecycle_state="invalid_state",
            meshes=None,
            phi_snapshot_paths={"nose": "path"},
            stl_path=None,
            stl_half_path=None,
            failure_reason="test",
        )
        _fail("test_gate_result_validates_lifecycle_state", "should have raised")
    except ValueError:
        _pass("test_gate_result_validates_lifecycle_state")

def test_gate_result_rejects_empty_phi_paths():
    try:
        GateResult(
            lifecycle_state="valid_simulated",
            meshes=None,
            phi_snapshot_paths={},
            stl_path=None,
            stl_half_path=None,
            failure_reason="test",
        )
        _fail("test_gate_result_rejects_empty_phi_paths", "should have raised")
    except ValueError:
        _pass("test_gate_result_rejects_empty_phi_paths")

def test_run_quality_gates_success():
    phi_grids = _make_all_phi()
    with tempfile.TemporaryDirectory() as d:
        result = run_quality_gates(phi_grids, "cand1", d)
        assert result.lifecycle_state in ("valid_simulated", "geometry_repaired"), \
            f"Expected success state, got {result.lifecycle_state}"
        assert result.stl_path is not None, "Should have STL path on success"
        assert result.stl_half_path is not None, "Should have half STL path on success"
        assert len(result.phi_snapshot_paths) == 4, "Should have 4 phi snapshots"
        _pass("test_run_quality_gates_success")

def test_run_quality_gates_phi_snapshots_always_saved():
    phi_grids = _make_all_phi()
    with tempfile.TemporaryDirectory() as d:
        result = run_quality_gates(phi_grids, "cand1", d)
        for name, path in result.phi_snapshot_paths.items():
            assert path, f"Phi snapshot for {name} is empty"
            assert os.path.exists(path), f"Phi snapshot for {name} not at {path}"
        _pass("test_run_quality_gates_phi_snapshots_always_saved")

def test_run_quality_gates_all_components_present():
    phi_grids = _make_all_phi()
    with tempfile.TemporaryDirectory() as d:
        result = run_quality_gates(phi_grids, "cand1", d)
        expected = {"nose", "sidepod", "rearpod", "main_body"}
        assert set(result.phi_snapshot_paths.keys()) == expected, \
            f"Expected {expected}, got {set(result.phi_snapshot_paths.keys())}"
        _pass("test_run_quality_gates_all_components_present")

if __name__ == "__main__":
    test_gate_result_validates_lifecycle_state()
    test_gate_result_rejects_empty_phi_paths()
    test_run_quality_gates_success()
    test_run_quality_gates_phi_snapshots_always_saved()
    test_run_quality_gates_all_components_present()
    print("\nAll quality_gates tests passed.")