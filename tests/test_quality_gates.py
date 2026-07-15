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

def _rounded_box_grid(shape, round_r=5.0, margin=1.0):
    """Uniformly edge-rounded box SDF (Inigo Quilez rounded-box formula).

    Used for the nose only: its 2mm wall-thickness gate needs rounded
    edges/corners to pass (sharp box corners always register as locally
    "thin" under the correct morphological-opening thickness test,
    regardless of overall box size -- verified live, 2026-07-15), and its
    curvature is small enough after the surface_extraction._repair_mesh
    Taubin-smoothing fix (2026-07-15) to still clear the mesh-quality gate.
    """
    nx, ny, nz = shape
    cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
    hx = max((nx - 1) / 2.0 - margin - round_r, 0.5)
    hy = max((ny - 1) / 2.0 - margin - round_r, 0.5)
    hz = max((nz - 1) / 2.0 - margin - round_r, 0.5)
    i, j, k = np.indices(shape).astype(np.float64)
    qx = np.abs(i - cx) - hx
    qy = np.abs(j - cy) - hy
    qz = np.abs(k - cz) - hz
    qx0, qy0, qz0 = np.maximum(qx, 0), np.maximum(qy, 0), np.maximum(qz, 0)
    outside = np.sqrt(qx0 ** 2 + qy0 ** 2 + qz0 ** 2)
    inside = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0)
    d = outside + inside - round_r
    from geometry_contract import GRID_SPACING_M
    return (d * GRID_SPACING_M).astype(np.float32)


def _make_phi(component, shape):
    """Milled components (sidepod/rearpod/main_body): a box filling the
    entire valid region (flush to every boundary) reliably clears the
    tool-accessibility gate -- every face is boundary-coincident and thus
    exempted by the boundary-face exemption (Option A), regardless of
    whether TOOL_DIRECTIONS covers that face's normal (verified live,
    2026-07-15). Nose: see _rounded_box_grid."""
    from geometry_contract import GRID_SPACING_M
    attachment_faces = {
        "nose": ["rear"], "sidepod": ["inner_y"], "rearpod": ["rear"], "main_body": [],
    }[component]
    bv = BoundingRegion(component, (0.0, -0.0045, 0.0015), shape)
    solid, air = PhiGrid.build_hard_masks(bv, [], attachment_faces)
    if component == "nose":
        grid = _rounded_box_grid(shape)
    else:
        grid = np.full(shape, -GRID_SPACING_M, dtype=np.float32)
    phi = PhiGrid(component, bv, grid, solid, air)
    phi.apply_hard_constraints()
    return phi

def _make_all_phi():
    return {
        "nose":      _make_phi("nose", (30, 30, 30)),
        "sidepod":   _make_phi("sidepod", (30, 30, 30)),
        "rearpod":   _make_phi("rearpod", (30, 30, 30)),
        "main_body": _make_phi("main_body", (40, 30, 30)),
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