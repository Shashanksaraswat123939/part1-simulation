"""
Tests for bayesian_outer_search.py.

These tests do NOT require BoTorch — they cover the geometry/normalisation layer
that surrounds the GP loop. The one test that actually invokes BoTorch is skipped
if botorch is not installed (it is an optional heavy dependency during development).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os, math, tempfile
import numpy as np

from bayesian_outer_search import (
    _abs_bounds, _to_unit, _from_unit, _is_valid,
    SearchConfig, SearchResult, EvaluationResult,
    BayesianOuterSearch, _level2_evaluate,
)
from bounding_volumes import RuleEnvelope, default_rule_envelope
from geometry_contract import W_MIN_MM, W_MAX_MM, X_FRONT_MIN_MM

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)

# Real confirmed/derived RuleEnvelope -- see bounding_volumes.default_rule_envelope()
STUB_RE = default_rule_envelope()


# ── Normalisation tests ────────────────────────────────────────────────────────

def test_to_unit_at_bounds():
    """Corners of the search space should map to 0 or 1."""
    (w_lo, w_hi), (xf_lo, xf_hi), (dh_lo, dh_hi) = _abs_bounds()
    u = _to_unit(w_lo, xf_lo, dh_lo)
    assert u == (0.0, 0.0, 0.0), f"lower corner → (0,0,0) but got {u}"
    u = _to_unit(w_hi, xf_hi, dh_hi)
    assert all(abs(x - 1.0) < 1e-9 for x in u), f"upper corner → (1,1,1) but got {u}"
    _pass("test_to_unit_at_bounds")


def test_round_trip_normalisation():
    """_from_unit(_to_unit(p)) ≈ p for arbitrary interior point."""
    try:
        import torch
        params = (128.0, 65.0, 40.0)
        u = _to_unit(*params)
        u_t = torch.tensor(list(u), dtype=torch.double)
        recovered = _from_unit(u_t)
        for a, b in zip(params, recovered):
            assert abs(a - b) < 1e-6, f"round-trip failed: {params} → {recovered}"
    except ImportError:
        # Verify normalisation without torch by checking the math directly
        params = (128.0, 65.0, 40.0)
        u = _to_unit(*params)
        (w_lo, w_hi), (xf_lo, xf_hi), (dh_lo, dh_hi) = _abs_bounds()
        W_rec   = w_lo  + u[0] * (w_hi  - w_lo)
        xf_rec  = xf_lo + u[1] * (xf_hi - xf_lo)
        dh_rec  = dh_lo + u[2] * (dh_hi - dh_lo)
        recovered = (W_rec, xf_rec, dh_rec)
        for a, b in zip(params, recovered):
            assert abs(a - b) < 1e-6, f"round-trip failed: {params} → {recovered}"
    _pass("test_round_trip_normalisation")


# ── Constraint validation tests ────────────────────────────────────────────────

def test_is_valid_accepts_good_params():
    assert _is_valid(130.0, 64.0, 50.0), "W=130 xf=64 dh=50 should be valid"
    _pass("test_is_valid_accepts_good_params")


def test_is_valid_rejects_W_out_of_range():
    assert not _is_valid(119.0, 64.0, 50.0), "W=119 should be invalid"
    assert not _is_valid(141.0, 64.0, 50.0), "W=141 should be invalid"
    _pass("test_is_valid_rejects_W_out_of_range")


def test_is_valid_rejects_x_front_too_small():
    assert not _is_valid(130.0, 50.0, 50.0), "x_front=50 < 61 should be invalid"
    _pass("test_is_valid_rejects_x_front_too_small")


def test_is_valid_rejects_d_halo_too_large():
    # d_halo_max = W + 16 = 130 + 16 = 146
    assert not _is_valid(130.0, 64.0, 147.0), "d_halo=147 > W+16=146 should be invalid"
    _pass("test_is_valid_rejects_d_halo_too_large")


# ── Level 2 stub tests ─────────────────────────────────────────────────────────

def test_level2_returns_evaluation_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _level2_evaluate(
            W_mm=130.0, x_front_mm=64.0, d_halo_mm=10.0,
            rule_envelope=STUB_RE, n_iters=0,
            output_dir=tmpdir, eval_id=1,
        )
    assert isinstance(result, EvaluationResult)
    assert result.W_mm == 130.0
    assert result.x_front_mm == 64.0
    assert result.d_halo_mm == 10.0
    _pass("test_level2_returns_evaluation_result")


def test_level2_race_time_is_positive():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _level2_evaluate(
            W_mm=130.0, x_front_mm=64.0, d_halo_mm=10.0,
            rule_envelope=STUB_RE, n_iters=0,
            output_dir=tmpdir, eval_id=1,
        )
    assert result.race_time > 0, f"race_time={result.race_time}"
    _pass("test_level2_race_time_is_positive")


def test_level2_mass_is_physical():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _level2_evaluate(
            W_mm=130.0, x_front_mm=64.0, d_halo_mm=10.0,
            rule_envelope=STUB_RE, n_iters=0,
            output_dir=tmpdir, eval_id=1,
        )
    # Total mass should include fixed hardware (≥ CO2 cartridge 23 g)
    assert result.mass_kg >= 0.023, f"mass_kg={result.mass_kg} — must include CO2 cartridge"
    _pass("test_level2_mass_is_physical")


def test_level2_saves_phi_snapshots():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _level2_evaluate(
            W_mm=130.0, x_front_mm=64.0, d_halo_mm=10.0,
            rule_envelope=STUB_RE, n_iters=0,
            output_dir=tmpdir, eval_id=1,
        )
        if result.lifecycle == "valid_simulated":
            for comp in ("nose", "sidepod", "rearpod", "main_body"):
                assert comp in result.phi_snapshots, f"missing snapshot for {comp}"
                assert os.path.exists(result.phi_snapshots[comp]), f"file missing: {result.phi_snapshots[comp]}"
    _pass("test_level2_saves_phi_snapshots")


def test_level2_smaller_W_gives_different_time():
    """Proxy time should vary with W (search space is non-trivial)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        r1 = _level2_evaluate(120.0, 64.0, 10.0, STUB_RE, 0, tmpdir, 1)
        r2 = _level2_evaluate(140.0, 64.0, 10.0, STUB_RE, 0, tmpdir, 2)
    assert r1.race_time != r2.race_time, "Proxy should vary with W"
    _pass("test_level2_smaller_W_gives_different_time")


def test_level2_evolution_loop_runs_without_crashing():
    """
    Regression test: n_iters > 0 previously crashed with AttributeError because
    hj_update()/reinitialise_sdf() take PhiGrid objects (mutate in place, return
    None), not raw arrays — the evolution loop was calling them with pg.grid and
    assigning the (None) return value back into pg.grid[:].
    """
    # n_iters kept small -- real grid sizes at 0.3mm spacing are large (millions
    # of cells for main_body), and this test only needs to prove the call path
    # doesn't crash, not exercise full convergence.
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _level2_evaluate(130.0, 70.0, 10.0, STUB_RE, n_iters=2, output_dir=tmpdir, eval_id=1)
    assert result.lifecycle == "valid_simulated", f"lifecycle={result.lifecycle}"
    _pass("test_level2_evolution_loop_runs_without_crashing")


# ── Warm-start tests ───────────────────────────────────────────────────────────

def test_warm_start_none_when_no_results():
    config = SearchConfig(rule_envelope=STUB_RE)
    search = BayesianOuterSearch(config)
    assert search._find_warm_start(130.0, 64.0, 50.0) is None
    _pass("test_warm_start_none_when_no_results")


def test_warm_start_found_for_nearby_point():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SearchConfig(rule_envelope=STUB_RE, output_dir=tmpdir)
        search = BayesianOuterSearch(config)
        # Inject a fake result with snapshots close to (130, 64, 50)
        r = _level2_evaluate(130.0, 64.0, 10.0, STUB_RE, 0, tmpdir, 1)
        search._results.append(r)
        # A nearby point should warm-start from this result
        warm = search._find_warm_start(130.5, 64.1, 10.2)
        if r.lifecycle == "valid_simulated":
            assert warm is not None, "Should find warm start for nearby point"
    _pass("test_warm_start_found_for_nearby_point")


def test_warm_start_none_for_distant_point():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SearchConfig(rule_envelope=STUB_RE, output_dir=tmpdir)
        search = BayesianOuterSearch(config)
        r = _level2_evaluate(120.0, 64.0, 10.0, STUB_RE, 0, tmpdir, 1)
        search._results.append(r)
        # A point far away should NOT warm-start
        warm = search._find_warm_start(140.0, 64.0, 145.0)
        assert warm is None, "Should not warm-start from a distant point"
    _pass("test_warm_start_none_for_distant_point")


# ── EvaluationResult helper tests ─────────────────────────────────────────────

def test_evaluation_result_normalised_params():
    r = EvaluationResult(
        W_mm=130.0, x_front_mm=64.0, d_halo_mm=10.0,
        race_time=1.5, mass_kg=0.05, h_com_m=0.025, x_com_m=0.08,
        lifecycle="valid_simulated",
    )
    u = r.normalised_params
    assert len(u) == 3
    assert all(0.0 <= v <= 1.0 for v in u), f"unit params out of [0,1]: {u}"
    _pass("test_evaluation_result_normalised_params")


# ── BoTorch integration test (skipped if botorch not installed) ────────────────

def test_mini_search_with_botorch():
    """Run a tiny search (3 seed + 3 BO) to verify the full BoTorch loop works."""
    try:
        import torch
        import botorch
    except ImportError:
        print("SKIP test_mini_search_with_botorch (botorch not installed)")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        config = SearchConfig(
            rule_envelope=STUB_RE,
            n_initial=3,
            n_iterations=3,
            output_dir=tmpdir,
            random_seed=0,
            level2_iters=0,
        )
        search = BayesianOuterSearch(config)
        result = search.run()

    assert isinstance(result, SearchResult)
    assert len(result.all_results) == 6       # 3 seed + 3 BO
    assert result.best_time > 0
    assert "W_mm" in result.best_params
    assert "x_front_mm" in result.best_params
    assert "d_halo_mm" in result.best_params
    _pass("test_mini_search_with_botorch")


if __name__ == "__main__":
    test_to_unit_at_bounds()
    test_round_trip_normalisation()
    test_is_valid_accepts_good_params()
    test_is_valid_rejects_W_out_of_range()
    test_is_valid_rejects_x_front_too_small()
    test_is_valid_rejects_d_halo_too_large()
    test_level2_returns_evaluation_result()
    test_level2_race_time_is_positive()
    test_level2_mass_is_physical()
    test_level2_saves_phi_snapshots()
    test_level2_smaller_W_gives_different_time()
    test_level2_evolution_loop_runs_without_crashing()
    test_warm_start_none_when_no_results()
    test_warm_start_found_for_nearby_point()
    test_warm_start_none_for_distant_point()
    test_evaluation_result_normalised_params()
    test_mini_search_with_botorch()
    print("\nAll bayesian_outer_search tests passed.")
