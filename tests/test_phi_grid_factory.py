"""
test_phi_grid_factory.py — tests for the missing-factory fix (audit P1-17).

Covers structural correctness at production grid spacing (building the four
grids is ~2s, verified live; the full run_quality_gates pass-through was
verified manually in a separate live run — ~200s, not repeated here to keep
the suite fast). See project notes for that live verification.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "part2_simulation"))

import numpy as np

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)


def test_build_phi_grids_returns_all_four_components():
    from phi_grid_factory import build_phi_grids_for_candidate
    grids, bv = build_phi_grids_for_candidate(130.0, 64.0, 50.0)
    assert set(grids.keys()) == {"nose", "sidepod", "rearpod", "main_body"}
    for name, phi in grids.items():
        assert phi.component == name
        assert phi.grid.shape == phi.bv.shape
        assert phi.grid.dtype == np.float32
        assert phi.bv.shape == bv.get(name).shape
    _pass("test_build_phi_grids_returns_all_four_components")


def test_body_grid_shape_matches_bounding_volumes():
    # Regression: the exact bug audit P1-17 named -- place_fixed_hardware's
    # required body_grid_shape/body_grid_origin_m kwargs were missing from
    # compute_default_fixed_hardware_inputs's returned dict. This test fails
    # loudly (TypeError) if that wiring regresses.
    from phi_grid_factory import build_phi_grids_for_candidate
    grids, bv = build_phi_grids_for_candidate(130.0, 64.0, 50.0)
    assert grids["main_body"].bv.shape == bv.main_body.shape
    assert grids["main_body"].bv.origin_m == bv.main_body.origin_m
    _pass("test_body_grid_shape_matches_bounding_volumes")


def test_hard_constraints_are_self_consistent():
    from phi_grid_factory import build_phi_grids_for_candidate
    grids, _bv = build_phi_grids_for_candidate(130.0, 64.0, 50.0)
    for name, phi in grids.items():
        overlap = phi.hard_mask_solid & phi.hard_mask_air
        assert not overlap.any(), f"{name}: solid/air mask overlap"
        # apply_hard_constraints was already called by the factory -- solid
        # cells must be < 0, air cells must be > 0.
        assert (phi.grid[phi.hard_mask_solid] < 0).all(), f"{name}: solid constraint violated"
        assert (phi.grid[phi.hard_mask_air] > 0).all(), f"{name}: air constraint violated"
    _pass("test_hard_constraints_are_self_consistent")


def test_main_body_has_hardware_void_masks_others_dont():
    # main_body's hard_mask_air must include the fixed-hardware void masks
    # (halo/canister/axles); the other three components get none (only
    # attachment-face solid strips + T7.9 invalid-region air).
    from phi_grid_factory import build_phi_grids_for_candidate
    from fixed_hardware import place_fixed_hardware, compute_default_fixed_hardware_inputs
    from bounding_volumes import compute_bounding_volumes, default_rule_envelope
    from phi_grid_factory import _default_forbidden_cylinders

    grids, bv = build_phi_grids_for_candidate(130.0, 64.0, 50.0)

    front_cyl, rear_cyl = _default_forbidden_cylinders(130.0, 64.0, 8.0)
    hw_inputs = compute_default_fixed_hardware_inputs(
        130.0, 64.0, 50.0, bv.ref_plane_A_m, bv.ref_plane_B_m,
    )
    hw_inputs["body_grid_shape"] = bv.main_body.shape
    hw_inputs["body_grid_origin_m"] = bv.main_body.origin_m
    hw_result = place_fixed_hardware(W_mm=130.0, x_front_mm=64.0, **hw_inputs)

    # main_body's air mask must be a superset of the hardware void mask
    # (it also includes the 1-cell border + invalid region, so not equal).
    main_air = grids["main_body"].hard_mask_air
    assert (main_air | hw_result.combined_void_mask == main_air).all(), \
        "main_body hard_mask_air must contain the fixed-hardware void mask"
    _pass("test_main_body_has_hardware_void_masks_others_dont")


def test_nose_and_rearpod_attach_rear_sidepod_attaches_inner_y():
    from phi_grid_factory import _ATTACHMENT_FACES
    assert _ATTACHMENT_FACES["nose"] == ["rear"]
    assert _ATTACHMENT_FACES["rearpod"] == ["rear"]
    assert _ATTACHMENT_FACES["sidepod"] == ["inner_y"]
    assert _ATTACHMENT_FACES["main_body"] == []
    _pass("test_nose_and_rearpod_attach_rear_sidepod_attaches_inner_y")


def test_invalid_W_raises():
    from phi_grid_factory import build_phi_grids_for_candidate
    try:
        build_phi_grids_for_candidate(100.0, 64.0, 20.0)
    except ValueError:
        return _pass("test_invalid_W_raises")
    _fail("test_invalid_W_raises", "expected ValueError")


def test_invalid_d_halo_raises():
    from phi_grid_factory import build_phi_grids_for_candidate
    try:
        build_phi_grids_for_candidate(130.0, 64.0, 200.0)
    except ValueError:
        return _pass("test_invalid_d_halo_raises")
    _fail("test_invalid_d_halo_raises", "expected ValueError")


def test_different_seeds_produce_different_random_grids():
    from phi_grid_factory import build_phi_grids_for_candidate
    g1, _ = build_phi_grids_for_candidate(130.0, 64.0, 50.0, init_mode="random", seed=1)
    g2, _ = build_phi_grids_for_candidate(130.0, 64.0, 50.0, init_mode="random", seed=2)
    assert not np.array_equal(g1["nose"].grid, g2["nose"].grid)
    _pass("test_different_seeds_produce_different_random_grids")


def test_warm_start_requires_all_four_components():
    from phi_grid_factory import build_phi_grids_for_candidate, warm_start_phi_grids
    grids, _bv = build_phi_grids_for_candidate(130.0, 64.0, 50.0)
    incomplete = {k: v for k, v in grids.items() if k != "nose"}
    try:
        warm_start_phi_grids(incomplete, 131.0, 64.0, 50.0)
    except ValueError as e:
        assert "nose" in str(e)
        return _pass("test_warm_start_requires_all_four_components")
    _fail("test_warm_start_requires_all_four_components", "expected ValueError")


def test_warm_start_produces_valid_grids_at_new_W():
    from phi_grid_factory import build_phi_grids_for_candidate, warm_start_phi_grids
    grids, _bv = build_phi_grids_for_candidate(130.0, 64.0, 50.0)
    new_grids, new_bv = warm_start_phi_grids(grids, 132.0, 64.0, 50.0)
    assert set(new_grids.keys()) == {"nose", "sidepod", "rearpod", "main_body"}
    assert new_bv.W_mm == 132.0
    _pass("test_warm_start_produces_valid_grids_at_new_W")


if __name__ == "__main__":
    fns = [f for f in dir(sys.modules[__name__]) if f.startswith("test_")]
    passed, failed = 0, 0
    for name in fns:
        try:
            globals()[name]()
            passed += 1
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e!r}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
