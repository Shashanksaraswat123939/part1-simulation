"""
Tests for stl_assembler.py
"""
import sys, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import trimesh
from stl_assembler import assemble_stl, _mirror_right_to_left

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

def _make_box_mesh(x0, y0, z0, dx, dy, dz):
    """Create a simple box mesh at given position with given dimensions."""
    verts = np.array([
        [x0,     y0,     z0],
        [x0+dx,  y0,     z0],
        [x0+dx,  y0+dy,  z0],
        [x0,     y0+dy,  z0],
        [x0,     y0,     z0+dz],
        [x0+dx,  y0,     z0+dz],
        [x0+dx,  y0+dy,  z0+dz],
        [x0,     y0+dy,  z0+dz],
    ])
    faces = np.array([
        [0,1,2],[0,2,3],  # bottom
        [4,6,5],[4,7,6],  # top
        [0,5,1],[0,4,5],  # front
        [2,6,7],[2,7,3],  # back
        [1,5,6],[1,6,2],  # right
        [0,3,7],[0,7,4],  # left
    ])
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    return mesh

def test_mirror_flips_y():
    right = _make_box_mesh(0.01, 0.0, 0.0, 0.02, 0.02, 0.02)
    left = _mirror_right_to_left(right)
    assert np.all(left.vertices[:, 1] <= 1e-10), "Left mesh y should be <= 0"
    assert np.all(right.vertices[:, 1] >= -1e-10), "Right mesh y should be >= 0"
    _pass("test_mirror_flips_y")

def test_assemble_stl_returns_two_paths():
    meshes = {
        "nose":     _make_box_mesh(0.0, -0.03, 0.0, 0.01, 0.06, 0.04),
        "sidepod":  _make_box_mesh(0.02, 0.03, 0.0, 0.06, 0.03, 0.035),
        "rearpod":  _make_box_mesh(0.13, -0.03, 0.0, 0.03, 0.06, 0.035),
        "main_body":_make_box_mesh(0.0, -0.03, 0.0, 0.16, 0.06, 0.045),
    }
    with tempfile.TemporaryDirectory() as d:
        full, half = assemble_stl(meshes, "cand1", d)
        assert os.path.exists(full), f"Full STL not created at {full}"
        assert os.path.exists(half), f"Half STL not created at {half}"
        _pass("test_assemble_stl_returns_two_paths")

def test_assemble_half_has_y_geq_zero():
    meshes = {
        "nose":     _make_box_mesh(0.0, -0.03, 0.0, 0.01, 0.06, 0.04),
        "sidepod":  _make_box_mesh(0.02, 0.03, 0.0, 0.06, 0.03, 0.035),
        "rearpod":  _make_box_mesh(0.13, -0.03, 0.0, 0.03, 0.06, 0.035),
        "main_body":_make_box_mesh(0.0, -0.03, 0.0, 0.16, 0.06, 0.045),
    }
    with tempfile.TemporaryDirectory() as d:
        full, half = assemble_stl(meshes, "cand1", d)
        half_mesh = trimesh.load(half)
        assert np.all(half_mesh.vertices[:, 1] >= -1e-6), "Half STL vertices should have y >= 0"
        _pass("test_assemble_half_has_y_geq_zero")

def test_missing_sidepod_raises():
    meshes = {
        "nose":     _make_box_mesh(0.0, -0.03, 0.0, 0.01, 0.06, 0.04),
        "rearpod":  _make_box_mesh(0.13, -0.03, 0.0, 0.03, 0.06, 0.035),
        "main_body":_make_box_mesh(0.0, -0.03, 0.0, 0.16, 0.06, 0.045),
    }
    try:
        with tempfile.TemporaryDirectory() as d:
            assemble_stl(meshes, "cand1", d)
        _fail("test_missing_sidepod_raises", "should have raised")
    except ValueError:
        _pass("test_missing_sidepod_raises")

if __name__ == "__main__":
    test_mirror_flips_y()
    test_assemble_stl_returns_two_paths()
    test_assemble_half_has_y_geq_zero()
    test_missing_sidepod_raises()
    print("\nAll stl_assembler tests passed.")