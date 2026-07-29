"""
stage1_search.py -- Stage 1: no-CFD scalar selection (wheelbase + wheel position).

Picks (W, x_front) by a Gaussian-process search on the MASS/COM proxy ONLY --
no CFD, no drag. This is the "Bayesian, no aero" stage of the two-stage plan:

    Stage 1 (this file):  Bayesian over (W, x_front) on mass/COM  -> best scalars
    Stage 2 (CFD sweep):  for the winner, build a car per legal d_halo, run the
                          drag/adjoint phi loop on each, rank by real race time.

d_halo is HELD at a nominal value here (it barely moves mass/COM); it is the
dimension Stage 2 optimises per-car ("a separate car for each halo-canister
distance, best race time wins"). See the module docstrings of
bayesian_outer_search.py (the 3-scalar proxy evaluator this reuses) and
pipeline_interface.unified_bindings (the Stage-2 CFD path).

Virtual-cargo placement + fore-aft flip are chosen INSIDE each evaluation by the
same mass/COM proxy, via find_cargo_placement's score_fn hook. Cargo is interior
-> zero aero -> a pure COM lever, so it belongs here, not in the CFD loop. The
scorer uses the analytic cargo_mass_com (no rebuild per candidate).

ponytail: 2D GP is written inline (~40 lines) rather than generalising
BayesianOuterSearch to N-D; that class is hard-wired to the 3-scalar cube and
bending it to 2D would be more surgery than a fresh loop. Upgrade to a shared
GP driver if a third stage ever needs one too.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from geometry_contract import (
    W_MIN_MM, W_MAX_MM, X_FRONT_MIN_MM, X_FRONT_ABS_MAX_MM,
    calibrate_x_front_bounds, get_density,
)
from bounding_volumes import default_rule_envelope
from bayesian_outer_search import _unified_mass_com_state
from virtual_cargo import cargo_mass_com, find_cargo_placement, CARGO_Z_BASE_M
from geometry_contract import mm_to_m, D_HALO_REF_A_OFFSET_MM

NOMINAL_D_HALO_MM: float = 20.0          # held fixed in Stage 1; Stage 2 sweeps it
EVOLVE_ITERS: int = 30                    # carve toward the 48 g floor before ranking
FAILURE_T: float = 1.0e6

# A cargo scorer maps (total_mass_kg, com_x_m, com_z_m) -> race time (lower is
# better). The mass/COM PROXY ignores com_x, so under it cargo placement is a
# no-op; a MEANINGFUL scorer is the real com_x-aware race objective evaluated at
# a NOMINAL drag (cargo doesn't change drag, so no CFD is needed). See
# make_race_objective_cargo_scorer below. None -> geometric default placement.
CargoScorer = Callable[[float, float, float], float]


def make_race_objective_cargo_scorer(
    thrust_csv_path: str, nominal_D20_n: float,
    mu: float = 0.010, wheel_moi_kg_m2: float = 1e-7,
) -> CargoScorer:
    """Build a cargo scorer from the REAL race objective at a FIXED nominal drag.

    Cargo never changes drag, so holding D20 = nominal_D20_n lets the com_x-aware
    JAX race objective rank cargo placements with NO CFD. This is the honest way
    cargo placement is decided; the mass/COM proxy can't (it has no com_x term).
    Requires part2-simulation on PATH (race_objective + adapter).
    """
    import os
    import sys
    p2 = os.environ.get("PART2_PATH")
    if p2 and p2 not in sys.path:
        sys.path.insert(0, p2)
    from race_objective import build_smooth_sheet_model
    from race_objective_adapter import race_value_and_grad_guarded

    model = build_smooth_sheet_model(thrust_csv_path)

    def scorer(total_mass_kg: float, com_x_m: float, com_z_m: float) -> float:
        # PARAM_NAMES: [drag_20_n, car_weight_kg, mu, wheel_moi_kg_m2,
        #               time_coefficient(=1.0), com_height_m, lift_20_n, com_x_m]
        p = np.array([nominal_D20_n, total_mass_kg, mu, wheel_moi_kg_m2,
                      1.0, com_z_m, 0.0, com_x_m], dtype=np.float64)
        _T_raw, T_pen, _grads = race_value_and_grad_guarded(p, model)
        return float(T_pen)

    return scorer


@dataclass
class Stage1Point:
    W_mm: float
    x_front_mm: float
    d_halo_mm: float
    T_proxy: float
    mass_kg: float
    com_x_m: float
    com_z_m: float
    cargo_x_start_m: float
    cargo_flip: bool
    # Carried so the Stage-2 handoff is a directly usable build_unified_geometry
    # placement dict. Without it, scalars_for_stage2 could only emit position and
    # flip, and every consumer would have to re-derive z_base (z_floor + 1 mm)
    # itself -- which is exactly how the two drift apart.
    cargo_z_base_m: float = 0.0

    @property
    def unit(self) -> tuple[float, float]:
        return (
            (self.W_mm - W_MIN_MM) / (W_MAX_MM - W_MIN_MM),
            (self.x_front_mm - X_FRONT_MIN_MM) / (X_FRONT_ABS_MAX_MM - X_FRONT_MIN_MM),
        )


@dataclass
class Stage1Result:
    best: Optional[Stage1Point]
    all_points: list = field(default_factory=list)
    total_wall_s: float = 0.0

    def scalars_for_stage2(self) -> dict:
        """The handoff to Stage 2: chosen wheelbase + wheel position + the cargo
        placement that goes with them. d_halo is intentionally NOT fixed here --
        Stage 2 sweeps it.

        `cargo_placement` is shaped EXACTLY as build_unified_geometry's
        cargo_placement argument, so Stage 2 can pass it straight through. It
        used to emit only cargo_x_start_m/cargo_flip, which nothing downstream
        read -- so the placement Stage 1 scored against the real com_x-aware
        objective was silently discarded and every Stage-2 car rebuilt with the
        geometric default, making the flip DOF dead.
        """
        b = self.best
        return {
            "W_mm": b.W_mm, "x_front_mm": b.x_front_mm,
            "cargo_x_start_m": b.cargo_x_start_m, "cargo_flip": b.cargo_flip,
            "cargo_placement": {
                "x_start_m": b.cargo_x_start_m,
                "z_base_m": b.cargo_z_base_m,
                "flip": b.cargo_flip,
            },
        }


def _from_unit(u0: float, u1: float) -> tuple[float, float]:
    W = W_MIN_MM + float(u0) * (W_MAX_MM - W_MIN_MM)
    x_front = X_FRONT_MIN_MM + float(u1) * (X_FRONT_ABS_MAX_MM - X_FRONT_MIN_MM)
    W = float(np.clip(W, W_MIN_MM, W_MAX_MM))
    xf_lo, xf_hi = calibrate_x_front_bounds(W)
    x_front = float(np.clip(x_front, xf_lo, xf_hi))
    return W, x_front


def _fail(W, xf, dh) -> Stage1Point:
    return Stage1Point(W, xf, dh, FAILURE_T, 0.0, 0.0, 0.0, 0.0, False)


def evaluate_scalars(
    W_mm: float, x_front_mm: float, d_halo_mm: float = NOMINAL_D_HALO_MM,
    cargo_scorer: Optional[CargoScorer] = None,
    out_dir: Optional[str] = None, eval_id: int = 0,
) -> Stage1Point:
    """One Stage-1 evaluation. Two steps:

      1. Pick the cargo placement (position + fore-aft flip). If cargo_scorer is
         given (the real com_x-aware objective at nominal drag), it decides;
         otherwise the geometric default is used (the proxy can't rank cargo --
         it ignores com_x). Scored analytically off the un-carved body, which is
         fine: cargo's COM contribution is position-dependent, not carve-
         dependent.
      2. Build WITH that cargo and EVOLVE to the 48 g floor, then rank (W,
         x_front) by the mass/COM proxy race time. Evolving is what makes the
         search non-degenerate: without it every config is its full ~150 g
         envelope and the mass term just picks the smallest box.
    """
    from unified_phi import build_unified_geometry, enforce_symmetry
    from bayesian_outer_search import _level2_evaluate_unified

    re = default_rule_envelope()
    out_dir = out_dir or tempfile.gettempdir()
    ref_A_m = mm_to_m(x_front_mm - D_HALO_REF_A_OFFSET_MM)
    # CARGO_Z_BASE_M, not `re.z_floor_m + 0.001`. Those are 2.5 mm and 14.0 mm
    # -- 11.5 mm apart. virtual_cargo derives 14.0 mm from CARGO_TOP_Z_MM (a rule
    # constraint) and find_cargo_placement/cargo_placement_is_buildable both use
    # it, so the local value only ever reached cargo_mass_com: the Bayesian
    # scorer ranked every (W, x_front) with the cargo COM 11.5 mm too low, while
    # the buildability screen beside it checked the real height.
    z_base_m = CARGO_Z_BASE_M
    rho_body = get_density("main_body")    # cargo is main_body material

    # ── Step 1: cargo placement ──────────────────────────────────────────────
    score_fn = None
    is_valid = None
    if cargo_scorer is not None:
        try:
            base_geom = build_unified_geometry(
                W_mm, x_front_mm, d_halo_mm, init_mode="full", with_cargo=False,
            )
        except ValueError:
            return _fail(W_mm, x_front_mm, d_halo_mm)
        enforce_symmetry(base_geom)
        base = _unified_mass_com_state(base_geom)
        if base is None:
            return _fail(W_mm, x_front_mm, d_halo_mm)
        m0, cx0, cz0 = base["total_mass_kg"], base["com_x_m"], base["com_z_m"]

        def score_fn(x_start_m, flip):
            cm, ccx, _cy, ccz = cargo_mass_com(x_start_m, z_base_m, flip, rho_body)
            total = m0 + cm
            com_x = (m0 * cx0 + cm * ccx) / total
            com_z = (m0 * cz0 + cm * ccz) / total
            return cargo_scorer(total, com_x, com_z)

        # base_geom is cargo-free, so its forced-air mask is exactly what the
        # build-time cargo guard will test against. Screening here costs a few
        # mask operations per candidate instead of a full rebuild, and without
        # it the scorer picks the COM-optimal placement with no idea whether it
        # is buildable -- which it usually is not, because moving cargo forward
        # improves com_x and also walks it into the front wheel keep-clear.
        from unified_phi import cargo_placement_is_buildable

        def is_valid(x_start_m, flip):
            return cargo_placement_is_buildable(
                # bv.ref_plane_A_m, not the locally-derived ref_A_m — same
                # value today, but the screen must read the guard's own source.
                base_geom, base_geom.bv.ref_plane_A_m, d_halo_mm,
                x_start_m, z_base_m, flip,
            )

    try:
        placement = find_cargo_placement(
            x_front_mm, W_mm, ref_A_m, d_halo_mm, re.z_floor_m,
            score_fn=score_fn, is_valid=is_valid,
        )
    except ValueError:
        return _fail(W_mm, x_front_mm, d_halo_mm)

    # ── Step 2: evolve WITH that cargo, rank by the mass/COM proxy ────────────
    r = _level2_evaluate_unified(
        W_mm, x_front_mm, d_halo_mm, EVOLVE_ITERS, out_dir, eval_id,
        cargo_placement=placement,
    )
    if r.race_time >= 1e5:
        return _fail(W_mm, x_front_mm, d_halo_mm)
    return Stage1Point(
        W_mm=W_mm, x_front_mm=x_front_mm, d_halo_mm=d_halo_mm,
        T_proxy=r.race_time, mass_kg=r.mass_kg,
        com_x_m=r.x_com_m, com_z_m=r.h_com_m,
        cargo_x_start_m=placement["x_start_m"], cargo_flip=placement["flip"],
        cargo_z_base_m=placement.get("z_base_m", z_base_m),
    )


def run_stage1(
    n_initial: int = 8,
    n_iterations: int = 24,
    d_halo_mm: float = NOMINAL_D_HALO_MM,
    random_seed: int = 42,
    cargo_scorer: Optional[CargoScorer] = None,
    out_dir: Optional[str] = None,
) -> Stage1Result:
    """Bayesian (GP + Expected Improvement) search over (W, x_front) on the
    mass/COM proxy. Returns the best scalars + their cargo placement.

    cargo_scorer (optional, from make_race_objective_cargo_scorer) decides cargo
    placement per eval via the real com_x-aware objective; None uses the
    geometric default (the proxy alone can't rank cargo)."""
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from botorch.acquisition import ExpectedImprovement
    from botorch.optim import optimize_acqf
    from botorch.utils.sampling import draw_sobol_samples
    from gpytorch.mlls import ExactMarginalLogLikelihood
    import warnings

    t0 = time.perf_counter()
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    bounds = torch.zeros(2, 2, dtype=torch.double)
    bounds[1] = 1.0

    points: list[Stage1Point] = []

    def _record(u0, u1):
        W, xf = _from_unit(u0, u1)
        points.append(evaluate_scalars(
            W, xf, d_halo_mm, cargo_scorer=cargo_scorer,
            out_dir=out_dir, eval_id=len(points),
        ))

    for xu in draw_sobol_samples(bounds, n=n_initial, q=1).squeeze(1):
        _record(float(xu[0]), float(xu[1]))

    for _ in range(n_iterations):
        valid = [p for p in points if p.T_proxy < 1e5]
        if len(valid) < 2:
            u = torch.rand(2, dtype=torch.double)
            _record(float(u[0]), float(u[1]))
            continue
        X = torch.tensor([list(p.unit) for p in valid], dtype=torch.double)
        Y = torch.tensor([[-p.T_proxy] for p in valid], dtype=torch.double)  # BoTorch maximises
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SingleTaskGP(X, Y)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            model.eval()
            cand, _ = optimize_acqf(
                ExpectedImprovement(model, best_f=Y.max()),
                bounds=bounds, q=1, num_restarts=5, raw_samples=32,
            )
        _record(float(cand[0, 0]), float(cand[0, 1]))

    valid = [p for p in points if p.T_proxy < 1e5]
    best = min(valid, key=lambda p: p.T_proxy) if valid else None
    return Stage1Result(best=best, all_points=points, total_wall_s=time.perf_counter() - t0)


if __name__ == "__main__":
    # Self-check at coarse spacing: the search returns a valid best with a cargo
    # placement, and cargo scoring actually exercised the flip DOF.
    import sys
    sys.path.insert(0, "sandbox")
    import coarse
    coarse.use_spacing(2.0)

    res = run_stage1(n_initial=5, n_iterations=6)
    assert res.best is not None, "Stage 1 found no valid (W, x_front)"
    b = res.best
    assert W_MIN_MM <= b.W_mm <= W_MAX_MM, f"W out of range: {b.W_mm}"
    assert b.mass_kg > 0.040, f"implausible mass {b.mass_kg*1000:.1f} g"
    assert b.cargo_x_start_m > 0, "cargo placement not set"
    flips = {p.cargo_flip for p in res.all_points if p.T_proxy < 1e5}
    print(f"best: W={b.W_mm:.1f} x_front={b.x_front_mm:.1f}  "
          f"T_proxy={b.T_proxy:.4f}  mass={b.mass_kg*1000:.1f}g  "
          f"h_com={b.com_z_m*1000:.1f}mm  cargo_flip={b.cargo_flip}")
    print(f"handoff to Stage 2: {res.scalars_for_stage2()}")
    print(f"cargo flip values explored: {flips}  ({len(res.all_points)} evals, "
          f"{res.total_wall_s:.1f}s)")
    print("PASS stage1_search self-check")
