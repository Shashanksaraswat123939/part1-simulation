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
        params = (128.0, 46.0, 40.0)
        u = _to_unit(*params)
        u_t = torch.tensor(list(u), dtype=torch.double)
        recovered = _from_unit(u_t)
        for a, b in zip(params, recovered):
            assert abs(a - b) < 1e-6, f"round-trip failed: {params} → {recovered}"
    except ImportError:
        # Verify normalisation without torch by checking the math directly
        params = (128.0, 46.0, 40.0)
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
    assert _is_valid(130.0, 46.0, 50.0), "W=130 xf=64 dh=50 should be valid"
    _pass("test_is_valid_accepts_good_params")


def test_is_valid_rejects_W_out_of_range():
    assert not _is_valid(119.0, 46.0, 50.0), "W=119 should be invalid"
    assert not _is_valid(141.0, 46.0, 50.0), "W=141 should be invalid"
    _pass("test_is_valid_rejects_W_out_of_range")


def test_is_valid_rejects_x_front_too_small():
    assert not _is_valid(130.0, 30.0, 50.0), "x_front=30 < X_FRONT_MIN_MM=36 should be invalid"
    assert not _is_valid(130.0, 60.0, 50.0), \
        "x_front=60 exceeds 56: nose overhang would be 44mm against T8.2's 40mm max"
    _pass("test_is_valid_rejects_x_front_too_small")


def test_is_valid_rejects_d_halo_too_large():
    # d_halo_max = W + 16 = 130 + 16 = 146
    assert not _is_valid(130.0, 46.0, 147.0), "d_halo=147 > W+16=146 should be invalid"
    _pass("test_is_valid_rejects_d_halo_too_large")


# ── Level 2 stub tests ─────────────────────────────────────────────────────────

def test_level2_returns_evaluation_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _level2_evaluate(
            W_mm=130.0, x_front_mm=46.0, d_halo_mm=20.0,
            rule_envelope=STUB_RE, n_iters=0,
            output_dir=tmpdir, eval_id=1,
        )
    assert isinstance(result, EvaluationResult)
    assert result.W_mm == 130.0
    assert result.x_front_mm == 46.0
    assert result.d_halo_mm == 20.0
    _pass("test_level2_returns_evaluation_result")


def test_level2_race_time_is_positive():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _level2_evaluate(
            W_mm=130.0, x_front_mm=46.0, d_halo_mm=20.0,
            rule_envelope=STUB_RE, n_iters=0,
            output_dir=tmpdir, eval_id=1,
        )
    assert result.race_time > 0, f"race_time={result.race_time}"
    _pass("test_level2_race_time_is_positive")


def test_level2_mass_is_physical():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _level2_evaluate(
            W_mm=130.0, x_front_mm=46.0, d_halo_mm=20.0,
            rule_envelope=STUB_RE, n_iters=0,
            output_dir=tmpdir, eval_id=1,
        )
    # Total mass should include fixed hardware (≥ CO2 cartridge 23 g)
    assert result.mass_kg >= 0.023, f"mass_kg={result.mass_kg} — must include CO2 cartridge"
    _pass("test_level2_mass_is_physical")


def test_level2_saves_phi_snapshots():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _level2_evaluate(
            W_mm=130.0, x_front_mm=46.0, d_halo_mm=20.0,
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
        r1 = _level2_evaluate(120.0, 46.0, 20.0, STUB_RE, 0, tmpdir, 1)
        r2 = _level2_evaluate(140.0, 46.0, 20.0, STUB_RE, 0, tmpdir, 2)
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
        result = _level2_evaluate(130.0, 46.0, 20.0, STUB_RE, n_iters=2, output_dir=tmpdir, eval_id=1)
    assert result.lifecycle == "valid_simulated", f"lifecycle={result.lifecycle}"
    _pass("test_level2_evolution_loop_runs_without_crashing")


# ── Warm-start tests ───────────────────────────────────────────────────────────

def test_warm_start_none_when_no_results():
    config = SearchConfig(rule_envelope=STUB_RE)
    search = BayesianOuterSearch(config)
    assert search._find_warm_start(130.0, 46.0, 50.0) is None
    _pass("test_warm_start_none_when_no_results")


def test_warm_start_found_for_nearby_point():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SearchConfig(rule_envelope=STUB_RE, output_dir=tmpdir)
        search = BayesianOuterSearch(config)
        # Inject a fake result with snapshots close to (130, 64, 50)
        r = _level2_evaluate(130.0, 46.0, 20.0, STUB_RE, 0, tmpdir, 1)
        search._results.append(r)
        # A nearby point should warm-start from this result
        warm = search._find_warm_start(130.5, 46.1, 10.2)
        if r.lifecycle == "valid_simulated":
            assert warm is not None, "Should find warm start for nearby point"
    _pass("test_warm_start_found_for_nearby_point")


def test_warm_start_none_for_distant_point():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SearchConfig(rule_envelope=STUB_RE, output_dir=tmpdir)
        search = BayesianOuterSearch(config)
        r = _level2_evaluate(120.0, 46.0, 20.0, STUB_RE, 0, tmpdir, 1)
        search._results.append(r)
        # A point far away should NOT warm-start
        warm = search._find_warm_start(140.0, 46.0, 145.0)
        assert warm is None, "Should not warm-start from a distant point"
    _pass("test_warm_start_none_for_distant_point")


# ── EvaluationResult helper tests ─────────────────────────────────────────────

def test_evaluation_result_normalised_params():
    r = EvaluationResult(
        W_mm=130.0, x_front_mm=46.0, d_halo_mm=20.0,
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




def test_the_proxy_descent_rests_legal_not_just_near_the_floor():
    """A subtracted barrier rests INSIDE the illegal region, not at the floor.

    _proxy_objective_gradients used to compute
        dT/dm = w_mass/m_ref - 2*barrier_w*(m_min - m)/m_min^2
    and its docstring recorded the outcome as fine: "equilibrium lands just
    under 48 g at the default weight". It is not fine -- the resting car is
    illegal under T3.6. Subtracting lets the two terms cancel and the descent
    rests exactly where they do, which is necessarily below the floor, because
    at the floor the barrier contributes nothing while the proxy still says
    lighter is faster.

    Solving for that rest point at the shipped weights predicts 47.895 g, and
    the 2026-08-05 Stage 1 run landed at 47.940 g -- one discrete step away,
    and 0.06 g outside the rule. Part 3 had the identical defect in
    t36_mass_barrier and is fixed the same way.
    """
    import bayesian_outer_search as b
    cartridge = 0.023
    for start in (0.0469, 0.0440, 0.0300):
        m = start
        for _ in range(50000):
            m -= 1e-6 * b._proxy_objective_gradients(m + cartridge)["dT_dmass"]
        assert m >= b.PROXY_MIN_MASS_KG, (
            f"proxy descent from {start*1000:.1f} g rests at {m*1000:.3f} g "
            f"competition mass, under the {b.PROXY_MIN_MASS_KG*1000:.0f} g "
            f"floor -- the barrier is cancelling against the mass term instead "
            f"of replacing it")
    _pass("test_the_proxy_descent_rests_legal_not_just_near_the_floor")


def test_the_proxy_ranking_penalty_still_uses_the_true_floor():
    """Descent aims above the floor; RANKING must not.

    The two jobs are separate: the penalty decides which design wins, the
    gradient decides which way the shape moves. If the margin leaked into the
    penalty, a legal 48.2 g car would be scored as a rule violation.
    """
    import bayesian_outer_search as b
    cartridge = 0.023
    assert b._proxy_mass_barrier(b.PROXY_MIN_MASS_KG + cartridge) == 0.0, (
        "a car exactly at the 48 g floor is being penalised")
    assert b._proxy_mass_barrier(
        b.PROXY_MIN_MASS_KG + b.PROXY_MASS_TARGET_MARGIN_KG + cartridge) == 0.0
    assert b._proxy_mass_barrier(0.047 + cartridge) > 0.0, (
        "an underweight car carries no ranking penalty")
    _pass("test_the_proxy_ranking_penalty_still_uses_the_true_floor")


if __name__ == "__main__":
    # Collected BY NAME, not listed by hand. The hand-written list that used to
    # live here silently skipped every test added after it was written -- it
    # missed two and still printed "All bayesian_outer_search tests passed".
    # tests/test_test_suites_run_what_they_define.py guards against exactly this.
    import sys as _sys
    _mod = _sys.modules[__name__]
    _failed = 0
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        _fn = getattr(_mod, _n)
        if not callable(_fn):
            continue
        try:
            # No PASS print here -- the tests in this file announce themselves
            # via _pass(), and printing again reported every test twice.
            _fn()
        except Exception as _exc:  # noqa: BLE001
            print("FAIL %s: %s" % (_n, _exc))
            _failed += 1
    if _failed:
        print("%d bayesian_outer_search test(s) FAILED." % _failed)
        _sys.exit(1)
    print("All bayesian_outer_search tests passed.")
