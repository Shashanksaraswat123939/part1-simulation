"""
test_bayesian_real_pipeline_wiring.py --- proves Level 1 (BayesianOuterSearch)
correctly calls Part 3's real evolutionary inner loop (P1-18 wiring) without
needing real OpenFOAM/jax/pandas -- fake PipelineBindings return controlled,
deterministic outcomes so this test runs in milliseconds, the same way Part
3's own tests avoid needing a real CFD solver.

This does NOT prove the real CFD/adjoint physics works (that was verified
separately, live, against actual OpenFOAM -- see project notes). It proves
the GLUE CODE between Level 1 and Part 3 is correct: that a Bayesian-proposed
(W, x_front, d_halo) point actually reaches wheelbase_sweep.optimize_single_w,
and that its CandidateOutcome correctly becomes an EvaluationResult.
"""
import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "part3-simulation"))

import numpy as np

def _pass(n): print(f"PASS {n}")
def _fail(n, m): print(f"FAIL {n}: {m}"); sys.exit(1)


def _make_fake_bindings():
    """Fake PipelineBindings where every candidate succeeds deterministically
    with a KNOWN T_penalized, so the test can assert the real race_time
    (not the proxy formula) flows through."""
    from pipeline_interface import PipelineBindings, GateOutcome, CFDOutcome, MassReport, ObjectiveOutcome

    KNOWN_T_PENALIZED = 12.3456

    class _FakePhiGrid:
        def save(self, candidate_id, out_dir):
            return f"/fake/{candidate_id}.npy"

    def initialize_phi_fields(W_mm, x_front_mm, d_halo_mm, seed):
        return {"nose": _FakePhiGrid(), "sidepod": _FakePhiGrid(),
                "rearpod": _FakePhiGrid(), "main_body": _FakePhiGrid()}

    def warm_start_phi_fields(prev, W_mm, x_front_mm, d_halo_mm):
        return prev

    def perturb_phi_fields(phi_grids, seed, amplitude):
        return phi_grids

    def run_quality_gates(phi_grids, candidate_id, out_dir):
        return GateOutcome(
            lifecycle_state="valid_simulated",
            phi_snapshot_paths={},
            stl_path="/fake/full.stl",
            stl_half_path="/fake/half.stl",
            failure_reason=None,
            meshes=None,
        )

    def compute_mass_report(phi_grids):
        return MassReport(total_mass_kg=0.052, com_x_m=0.06, com_y_m=0.0, com_z_m=0.02)

    def run_cfd(stl_half_path):
        return CFDOutcome(D20=1.2, L=-0.3, Cm=0.01, A=0.008, converged=True, residual_final=1e-5)

    def evaluate_objective(D20, L, m_total, h_com, x_com, mu, wheel_moi):
        return ObjectiveOutcome(
            T_raw=KNOWN_T_PENALIZED - 0.01,
            T_com_penalized=KNOWN_T_PENALIZED,
            gradients={"dT_dD20": 0.001, "dT_dmass": 0.01, "dT_dh_com": 0.1, "dT_dx_com": 0.0, "dT_dL": 0.0},
        )

    def compute_adjoint_weight(D20, L, m_total, h_com, x_com, mu, wheel_moi):
        return 0.001

    def run_adjoint(stl_half_path, objective_weight):
        return np.zeros(3)

    def update_phi(phi_grids, sensitivity_field, meshes, dt, weights, objective_gradients, mass_report):
        return None

    def write_candidate_record(outcome_dict):
        return "/fake/record.json"

    return PipelineBindings(
        initialize_phi_fields=initialize_phi_fields,
        warm_start_phi_fields=warm_start_phi_fields,
        perturb_phi_fields=perturb_phi_fields,
        run_quality_gates=run_quality_gates,
        compute_mass_report=compute_mass_report,
        run_cfd=run_cfd,
        evaluate_objective=evaluate_objective,
        compute_adjoint_weight=compute_adjoint_weight,
        run_adjoint=run_adjoint,
        update_phi=update_phi,
        write_candidate_record=write_candidate_record,
    ), KNOWN_T_PENALIZED


def test_search_config_rejects_real_pipeline_without_required_fields():
    from bayesian_outer_search import SearchConfig
    try:
        SearchConfig(use_real_pipeline=True)
    except ValueError as e:
        assert "thrust_csv_path" in str(e)
        return _pass("test_search_config_rejects_real_pipeline_without_required_fields")
    _fail("test_search_config_rejects_real_pipeline_without_required_fields", "expected ValueError")


def test_search_config_rejects_real_pipeline_without_prerequisite_flags():
    from bayesian_outer_search import SearchConfig
    try:
        SearchConfig(
            use_real_pipeline=True,
            thrust_csv_path="x.csv", fixed_hardware_kwargs={}, gradient_weights=object(),
            rtc_validated_against_track_data=False,
            cfd_pipeline_validated_on_known_geometry=True,
        )
    except ValueError as e:
        assert "rtc_validated_against_track_data" in str(e)
        return _pass("test_search_config_rejects_real_pipeline_without_prerequisite_flags")
    _fail("test_search_config_rejects_real_pipeline_without_prerequisite_flags", "expected ValueError")


def test_level2_evaluate_real_calls_part3_and_returns_real_T():
    from bayesian_outer_search import _level2_evaluate, SearchConfig, _real_bindings_cache
    from optimizer_contract import GradientWeights
    from bounding_volumes import default_rule_envelope

    fake_bindings, known_T = _make_fake_bindings()
    weights = GradientWeights(w_aero=1.0, w_mass=0.3, w_com=0.3, w_mfg=0.1)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SearchConfig(
            use_real_pipeline=True,
            thrust_csv_path="unused.csv",
            fixed_hardware_kwargs={},
            gradient_weights=weights,
            rtc_validated_against_track_data=True,
            cfd_pipeline_validated_on_known_geometry=True,
            n_candidates_per_point=1,
            n_evolution_rounds=1,
            inner_iteration_budget=1,
            output_dir=tmpdir,
        )
        # Inject the fake bindings directly, bypassing real_bindings()
        # construction (which needs a real CSV + jax/pandas) -- this is the
        # same cache _get_real_bindings reads from.
        _real_bindings_cache[id(cfg)] = fake_bindings

        result = _level2_evaluate(
            130.0, 64.0, 10.0,
            rule_envelope=default_rule_envelope(),
            n_iters=0, output_dir=tmpdir, eval_id=1,
            search_config=cfg,
        )
        del _real_bindings_cache[id(cfg)]

        # The fake bindings return the exact same T_penalized every
        # iteration, so the convergence tracker correctly sees
        # |delta_T|=0 < 1ms and promotes the outcome to "converged"
        # (a real, expected success state -- see inner_loop.py's
        # "Promote the best outcome to 'converged'" logic) rather than
        # "valid_simulated". Both are SUCCESS_STATES.
        assert result.lifecycle in ("valid_simulated", "converged"), \
            f"expected a success lifecycle, got {result.lifecycle}"
        assert abs(result.race_time - known_T) < 1e-9, \
            f"expected the FAKE bindings' known T_penalized ({known_T}), got {result.race_time} " \
            "-- this means _level2_evaluate_real is not actually reading Part 3's real result"
        # Proves this did NOT fall through to the proxy formula (which would
        # give mass/W-based T, never exactly the injected constant).
        assert result.mass_kg == 0.0, "real-pipeline path documents mass_kg=0.0 (not on CandidateOutcome)"
    _pass("test_level2_evaluate_real_calls_part3_and_returns_real_T")


def test_level2_evaluate_real_rejects_invalid_geometry_before_part3():
    from bayesian_outer_search import _level2_evaluate, SearchConfig, _real_bindings_cache
    from optimizer_contract import GradientWeights
    from bounding_volumes import default_rule_envelope

    fake_bindings, _ = _make_fake_bindings()
    weights = GradientWeights(w_aero=1.0, w_mass=0.3, w_com=0.3, w_mfg=0.1)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SearchConfig(
            use_real_pipeline=True,
            thrust_csv_path="unused.csv", fixed_hardware_kwargs={}, gradient_weights=weights,
            rtc_validated_against_track_data=True, cfd_pipeline_validated_on_known_geometry=True,
            n_candidates_per_point=1, n_evolution_rounds=1, inner_iteration_budget=1,
            output_dir=tmpdir,
        )
        _real_bindings_cache[id(cfg)] = fake_bindings
        # d_halo=200mm at W=130 is far outside the legal (placement-derived)
        # range -- must be rejected before ever touching the fake bindings.
        result = _level2_evaluate(
            130.0, 64.0, 200.0,
            rule_envelope=default_rule_envelope(),
            n_iters=0, output_dir=tmpdir, eval_id=2,
            search_config=cfg,
        )
        del _real_bindings_cache[id(cfg)]
        assert result.lifecycle == "geometry_rejected", f"expected geometry_rejected, got {result.lifecycle}"
    _pass("test_level2_evaluate_real_rejects_invalid_geometry_before_part3")


def test_proxy_path_unaffected_when_search_config_omitted():
    """Regression: every pre-existing call site that doesn't pass
    search_config must keep getting the exact old proxy behavior."""
    from bayesian_outer_search import _level2_evaluate
    from bounding_volumes import default_rule_envelope

    with tempfile.TemporaryDirectory() as tmpdir:
        result = _level2_evaluate(130.0, 64.0, 10.0, default_rule_envelope(), 0, tmpdir, 1)
        assert result.lifecycle == "valid_simulated"
        assert result.mass_kg > 0.0  # proxy path always computes a real mass from phi grids
    _pass("test_proxy_path_unaffected_when_search_config_omitted")


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
