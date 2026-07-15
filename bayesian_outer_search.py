"""
bayesian_outer_search.py --- Level 1: Bayesian outer search over (W, x_front, d_halo).

Implements the three-level optimization structure described in 01_generative_geometry.md:

  Level 1 (this file): BoTorch Gaussian-process search over the three outer scalars.
  Level 2 (level2_stub): Inner phi-field optimization loop. Currently a stub that
                          exercises the geometry pipeline and returns a proxy race time.
                          Replace with real CFD+adjoint driver when Part 2 is ready.
  Level 3 (surface_extraction.py): Marching-cubes → repair → gates. Called by Level 2.

Search space:
  W        ∈ [120, 140] mm          wheelbase (T7.3)
  x_front  ∈ [61, 207-W] mm        front axle from nose tip (W-dependent upper bound)
  d_halo   ∈ [0,  W+16]  mm        halo offset from Ref Plane A (W-dependent upper bound)

BoTorch operates in a unit [0,1]^3 normalised space. Parameters are de-normalised
per evaluation using their W-dependent bounds. Samples that violate derived constraints
after de-normalisation are rejected before the Level 2 call.

Warm-starting:
  When the next candidate is within WARM_START_THRESHOLD units of a previous evaluation
  (in normalised space), the best phi-grid snapshot from that evaluation is remapped
  into the new bounding volumes before the Level 2 loop starts. This cuts Level 2
  iterations for adjacent samples significantly.

Usage:
  from bayesian_outer_search import BayesianOuterSearch, SearchConfig
  from bounding_volumes import RuleEnvelope

  config = SearchConfig(rule_envelope=my_rule_env, n_initial=15, n_iterations=65)
  search = BayesianOuterSearch(config)
  result = search.run()
  print(result.best_params, result.best_time)
"""

from __future__ import annotations

import os
import sys
import time
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ── Part 3 path (for the real Level 2 pipeline) ────────────────────────────────
# Mirrors fixed_hardware.py's PART2_PATH pattern. Only needed when
# SearchConfig.use_real_pipeline=True; the proxy path never imports Part 3.
_part3_path = os.environ.get(
    "PART3_PATH",
    str(Path(__file__).resolve().parent.parent / "part3-simulation"),
)
if _part3_path not in sys.path:
    sys.path.insert(0, _part3_path)

# ── Geometry imports ───────────────────────────────────────────────────────────
from geometry_contract import (
    W_MIN_MM, W_MAX_MM, X_FRONT_MIN_MM, X_FRONT_ABS_MAX_MM,
    calibrate_x_front_bounds, calibrate_d_halo_max_mm,
    validate_W, validate_x_front, validate_d_halo,
    mm_to_m, CO2_MASS_KG, R_WHEEL_M, WHEEL_CLEARANCE_M,
    PHI_SNAPSHOT_COMPONENT_KEYS,
)
from bounding_volumes import RuleEnvelope, BoundingVolumes, compute_bounding_volumes, default_rule_envelope
from phi_grid import PhiGrid
from mass_com_calculator import compute_all_machined_components
from virtual_cargo import find_cargo_placement, build_virtual_cargo_solid_mask

# ── Constants ─────────────────────────────────────────────────────────────────
# Wheel half-width in x (placeholder — measure physical hardware).
# Used for ForbiddenCylinder construction. Matches test stubs in test_bounding_volumes.
WHEEL_X_HALF_WIDTH_M: float = 0.008    # 8 mm

# Axle height above track surface (placeholder).
AXLE_Z_M: float = R_WHEEL_M            # wheel centre at 15 mm

# Fixed hardware mass stubs (g/kg). Replace with real measurements.
STUB_WHEEL_AXLE_MASS_KG: float = 0.015   # ~15 g for 4 wheels + axles
STUB_HALO_MASS_KG:       float = 0.008   # ~8 g
STUB_REAR_WING_MASS_KG:  float = 0.005   # ~5 g

# Distance below which we warm-start from a previous result (normalised space).
WARM_START_THRESHOLD: float = 0.15


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SearchConfig:
    """All tunable parameters for one Bayesian outer search run."""
    rule_envelope:   RuleEnvelope = field(default_factory=default_rule_envelope)
    n_initial:       int   = 15     # quasi-random seed evaluations before GP fitting
    n_iterations:    int   = 65     # BO iterations after seed (total = n_initial + n_iterations)
    output_dir:      str   = "bayesian_results"
    random_seed:     int   = 42
    # GP fitting: restart count for acquisition optimisation
    acqf_restarts:   int   = 5
    acqf_raw_samples: int  = 32
    # Level 2 stub: number of phi-update iterations (0 = geometry-only, no evolution).
    # Only used when use_real_pipeline=False.
    level2_iters:    int   = 0

    # ── Real Level 2 (Part 3's evolutionary inner loop, real CFD+adjoint) ──
    # When True, each evaluated (W, x_front, d_halo) point is scored by
    # actually running Part 3's wheelbase_sweep.optimize_single_w -- a real
    # M-candidate evolutionary population, each candidate running the full
    # inner loop (gates -> mass/COM -> real OpenFOAM CFD -> race objective ->
    # real OpenFOAM adjoint -> phi update) -- instead of the proxy formula
    # below. This is what "wiring the Bayesian search to the evolutionary
    # loop" means architecturally (P1-18: Level 1 proposes, Part 3 executes).
    #
    # HONEST COST WARNING: a single evolutionary candidate's adjoint solve
    # alone took ~105s on a trivial coarse-mesh toy geometry in this
    # project's own live testing (see project notes, 2026-07-14); real
    # production-scale geometry will be slower (P1-14's known pure-Python
    # performance limits compound this further). A default-sized search
    # (n_initial=15, n_iterations=65 => 80 outer evaluations, each running
    # n_candidates_per_point evolutionary candidates for n_evolution_rounds
    # rounds) is a multi-hour-to-multi-day undertaking with this flag on.
    # Keep n_candidates_per_point / n_evolution_rounds / iteration_budget
    # small, and n_initial+n_iterations small, for a first real run -- this
    # is not a bug, it is what real CFD costs.
    #
    # Defaults to False (the fast geometry-only proxy) so constructing a
    # SearchConfig with no arguments never accidentally triggers hours of
    # OpenFOAM solves.
    use_real_pipeline: bool = False
    thrust_csv_path:  Optional[str] = None
    fixed_hardware_kwargs: Optional[dict] = None
    gradient_weights: "object" = None          # optimizer_contract.GradientWeights
    mu:               float = 0.4
    wheel_moi_kg_m2:  float = 1e-6
    rtc_validated_against_track_data: bool = False
    cfd_pipeline_validated_on_known_geometry: bool = False
    n_candidates_per_point: int = 3     # M, evolutionary population size per outer point
    n_evolution_rounds:     int = 2
    inner_iteration_budget: int = 5     # Part 3 OptimizerConfig.iteration_budget

    def __post_init__(self) -> None:
        if self.use_real_pipeline:
            missing = [
                name for name in ("thrust_csv_path", "fixed_hardware_kwargs", "gradient_weights")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    f"use_real_pipeline=True requires {missing} to be set. "
                    "The real Level 2 path needs a real thrust curve, fixed-"
                    "hardware mass/COM spec, and gradient weights -- there is "
                    "no safe default for any of these (see project spec's "
                    "Adjoint Objective / Gradient Combination sections)."
                )
            if not (self.rtc_validated_against_track_data and self.cfd_pipeline_validated_on_known_geometry):
                raise ValueError(
                    "use_real_pipeline=True requires BOTH "
                    "rtc_validated_against_track_data and "
                    "cfd_pipeline_validated_on_known_geometry to be True -- "
                    "this mirrors Part 3's own orchestrator.run_full_search "
                    "gate (03_optimizer_workflow spec, Search Strategy steps "
                    "1-2). Flipping these without doing the real physical/CFD "
                    "validation experiments is a lie the code cannot detect, "
                    "but it will not silently proceed as if you had."
                )


@dataclass
class EvaluationResult:
    """Outcome of one Level 2 evaluation at a specific (W, x_front, d_halo) point."""
    W_mm:       float
    x_front_mm: float
    d_halo_mm:  float
    race_time:  float           # proxy race time (lower is better)
    mass_kg:    float
    h_com_m:    float           # COM height above track
    x_com_m:    float           # COM fore-aft from nose tip
    lifecycle:  str             # ALLOWED_LIFECYCLE_STATES value
    phi_snapshots: dict[str, str] = field(default_factory=dict)  # comp → .npy path
    wall_time_s:   float = 0.0

    @property
    def normalised_params(self) -> tuple[float, float, float]:
        """Return (W, x_front, d_halo) as unit [0,1] values (for GP input)."""
        return _to_unit(self.W_mm, self.x_front_mm, self.d_halo_mm)


@dataclass
class SearchResult:
    """Final output of a completed BayesianOuterSearch.run() call."""
    best_params:    dict[str, float]   # {"W_mm": ..., "x_front_mm": ..., "d_halo_mm": ...}
    best_time:      float
    all_results:    list[EvaluationResult]
    total_wall_s:   float
    converged:      bool               # True if GP uncertainty < convergence threshold


# ── Normalisation helpers ──────────────────────────────────────────────────────

def _abs_bounds() -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Absolute (W-independent) search bounds used for unit normalisation.

    W:        [120, 140] mm  (T7.3)
    x_front:  [61,   90] mm  (X_FRONT_MIN to X_FRONT_ABS_MAX)
    d_halo:   [0,   106) mm  (0 to calibrate_d_halo_max_mm(W_MAX_MM), K-5:
              placement-derived W-34, exclusive -- was the stale W_MAX+16=156
              until this pass, which let Sobol/BO samples land in a dead zone
              (100-156mm) that geometry always rejects, wasting evaluations)
    """
    return (
        (W_MIN_MM,         W_MAX_MM),
        (X_FRONT_MIN_MM,   X_FRONT_ABS_MAX_MM),
        (0.0,              calibrate_d_halo_max_mm(W_MAX_MM)),
    )


def _to_unit(W_mm: float, x_front_mm: float, d_halo_mm: float) -> tuple[float, float, float]:
    """Map (W, x_front, d_halo) in mm to [0,1]^3 using absolute bounds."""
    (w_lo, w_hi), (xf_lo, xf_hi), (dh_lo, dh_hi) = _abs_bounds()
    return (
        (W_mm       - w_lo)  / (w_hi  - w_lo),
        (x_front_mm - xf_lo) / (xf_hi - xf_lo),
        (d_halo_mm  - dh_lo) / (dh_hi - dh_lo),
    )


def _from_unit(u: "torch.Tensor") -> tuple[float, float, float]:
    """De-normalise a [0,1]^3 tensor to (W_mm, x_front_mm, d_halo_mm)."""
    (w_lo, w_hi), (xf_lo, xf_hi), (dh_lo, dh_hi) = _abs_bounds()
    u0, u1, u2 = float(u[0]), float(u[1]), float(u[2])
    W_mm       = w_lo  + u0 * (w_hi  - w_lo)
    x_front_mm = xf_lo + u1 * (xf_hi - xf_lo)
    d_halo_mm  = dh_lo + u2 * (dh_hi - dh_lo)
    return W_mm, x_front_mm, d_halo_mm


def _is_valid(W_mm: float, x_front_mm: float, d_halo_mm: float) -> bool:
    """True if all three parameters satisfy their W-dependent constraints."""
    try:
        validate_W(W_mm)
        validate_x_front(x_front_mm, W_mm)
        validate_d_halo(d_halo_mm, W_mm)
        return True
    except ValueError:
        return False


# ── ForbiddenCylinder construction ────────────────────────────────────────────
# Imported lazily to avoid the Part-2 dependency at module level.

def _make_cylinders(x_front_mm: float, W_mm: float):
    """
    Build ForbiddenCylinder objects for front and rear wheel exclusion zones.
    Positions are in nose-tip coordinates (x=0 at nose tip).
    """
    from fixed_hardware import ForbiddenCylinder   # Part 2 dep — load lazily
    x_front_m = mm_to_m(x_front_mm)
    W_m       = mm_to_m(W_mm)
    r         = R_WHEEL_M + WHEEL_CLEARANCE_M
    front = ForbiddenCylinder(x_front_m,        0.0, AXLE_Z_M, r, WHEEL_X_HALF_WIDTH_M)
    rear  = ForbiddenCylinder(x_front_m + W_m,  0.0, AXLE_Z_M, r, WHEEL_X_HALF_WIDTH_M)
    return front, rear


# ── Level 2 stub ──────────────────────────────────────────────────────────────

def _level2_evaluate(
    W_mm:       float,
    x_front_mm: float,
    d_halo_mm:  float,
    rule_envelope: RuleEnvelope,
    n_iters:    int,
    output_dir: str,
    eval_id:    int,
    warm_phi_paths: Optional[dict[str, str]] = None,
    search_config: Optional[SearchConfig] = None,
) -> EvaluationResult:
    """
    Level 2 inner loop.

    Two modes, selected by search_config.use_real_pipeline (default False /
    search_config=None -- unchanged proxy behavior, so every existing caller
    and test keeps working exactly as before):

      use_real_pipeline=False (proxy, default): builds the phi grids for
        (W, x_front, d_halo), runs n_iters of Hamilton-Jacobi evolution
        (currently 0 by default -- pure geometry), computes mass/COM, and
        returns a PROXY race time:
            T = 0.5 * (mass / 0.055)      # mass contribution (target ~55 g)
              + 0.3 * (h_com / 0.025)     # COM height contribution (target ~25 mm)
              + 0.2 * (W / 130)           # wheelbase penalty (prefer shorter)
        All three terms normalised so a "perfect" car would score ~1.0.
        Fast, no CFD -- for wiring/architecture testing and quick sweeps.

      use_real_pipeline=True: delegates to _level2_evaluate_real, which runs
        Part 3's actual evolutionary inner loop (wheelbase_sweep.optimize_single_w)
        -- real quality gates, real OpenFOAM CFD, the real race objective,
        real OpenFOAM adjoint, real phi updates -- and returns the winning
        candidate's real T_penalized. See SearchConfig.use_real_pipeline's
        docstring for the wall-clock cost this implies.
    """
    if search_config is not None and search_config.use_real_pipeline:
        return _level2_evaluate_real(
            W_mm, x_front_mm, d_halo_mm, output_dir, eval_id,
            warm_phi_paths, search_config,
        )

    t0 = time.perf_counter()

    # Build forbidden zones in nose-tip coordinate system
    try:
        front_cyl, rear_cyl = _make_cylinders(x_front_mm, W_mm)
    except ImportError:
        # Part 2 not available — create a minimal stub cylinder that only exposes
        # x_min_m / x_max_m so bounding_volumes can compute the sidepod corridor.
        front_cyl = _StubCylinder(x_front_mm, W_mm, lead=True)
        rear_cyl  = _StubCylinder(x_front_mm, W_mm, lead=False)

    # Bounding volumes
    try:
        bv = compute_bounding_volumes(
            W_mm, x_front_mm, d_halo_mm,
            front_cyl, rear_cyl, rule_envelope,
        )
    except (ValueError, NotImplementedError) as exc:
        return EvaluationResult(
            W_mm=W_mm, x_front_mm=x_front_mm, d_halo_mm=d_halo_mm,
            race_time=1e6, mass_kg=0.0, h_com_m=0.0, x_com_m=0.0,
            lifecycle="geometry_rejected",
            wall_time_s=time.perf_counter() - t0,
        )

    # T4.2: pick a virtual cargo placement that dodges this evaluation's halo
    # pocket. Cheap (no CFD) -- see virtual_cargo.py for why this is decided
    # here rather than as a Bayesian search dimension.
    try:
        cargo_placement = find_cargo_placement(
            x_front_mm, W_mm, bv.ref_plane_A_m, d_halo_mm, rule_envelope.z_floor_m,
        )
    except ValueError:
        return EvaluationResult(
            W_mm=W_mm, x_front_mm=x_front_mm, d_halo_mm=d_halo_mm,
            race_time=1e6, mass_kg=0.0, h_com_m=0.0, x_com_m=0.0,
            lifecycle="geometry_rejected",
            wall_time_s=time.perf_counter() - t0,
        )

    # Fixed hardware (halo cross-section U1, canister U2, rear wing U5, axle
    # holes) using design defaults within legal T5/T9 bounds -- see
    # fixed_hardware.py's "Design defaults" section for exact values and
    # rationale. Falls back to no fixed-hardware voids if Part 2 (which
    # supplies the FixedHardwareSpec type) is unavailable.
    fixed_hardware_result = None
    try:
        from fixed_hardware import place_fixed_hardware, compute_default_fixed_hardware_inputs
        hw_inputs = compute_default_fixed_hardware_inputs(
            W_mm, x_front_mm, d_halo_mm, bv.ref_plane_A_m, bv.ref_plane_B_m,
        )
        fixed_hardware_result = place_fixed_hardware(
            W_mm, x_front_mm,
            body_grid_shape=bv.main_body.shape,
            body_grid_origin_m=bv.main_body.origin_m,
            **hw_inputs,
        )
    except ImportError:
        pass   # Part 2 not available -- proceed without fixed-hardware voids
    except ValueError:
        # E.g. halo pocket would extend past the rear axle for this
        # (W, x_front, d_halo) combination -- a real geometric infeasibility.
        return EvaluationResult(
            W_mm=W_mm, x_front_mm=x_front_mm, d_halo_mm=d_halo_mm,
            race_time=1e6, mass_kg=0.0, h_com_m=0.0, x_com_m=0.0,
            lifecycle="geometry_rejected",
            wall_time_s=time.perf_counter() - t0,
        )

    # Initialise phi grids (warm-start if available)
    phi_grids: dict[str, PhiGrid] = {}
    for comp in PHI_SNAPSHOT_COMPONENT_KEYS:
        region = bv.get(comp)
        solid_masks = []
        void_masks = []
        if comp == "main_body":
            solid_masks.append(build_virtual_cargo_solid_mask(
                region.origin_m, region.shape,
                cargo_placement["x_start_m"], cargo_placement["z_base_m"],
            ))
            if fixed_hardware_result is not None:
                void_masks.append(fixed_hardware_result.combined_void_mask)
        solid_mask, air_mask = PhiGrid.build_hard_masks(region, void_masks, [], solid_masks)
        grid_data = np.zeros(region.shape, dtype=np.float32)
        pg = PhiGrid(comp, region, grid_data, solid_mask, air_mask)

        if warm_phi_paths and comp in warm_phi_paths:
            try:
                loaded = PhiGrid.load(warm_phi_paths[comp])
                pg = loaded.remap(region, (solid_mask, air_mask))
            except Exception:
                pg.init("sphere")
        else:
            pg.init("sphere")

        phi_grids[comp] = pg

    # Level 2 evolution (n_iters Hamilton-Jacobi steps without real CFD sensitivity)
    if n_iters > 0:
        from phi_updater import hj_update, reinitialise_sdf
        dt = 1e-4
        for _ in range(n_iters):
            for pg in phi_grids.values():
                # Zero sensitivity → pure reinitialisation / smoothing step.
                # hj_update/reinitialise_sdf take the PhiGrid object (not the
                # raw array) and mutate it in place, returning None.
                hj_update(pg, np.zeros_like(pg.grid), dt)
            if (_ + 1) % 10 == 0:
                for pg in phi_grids.values():
                    reinitialise_sdf(pg)

    # Mass and COM from phi grids
    try:
        components = compute_all_machined_components(
            phi_grids["nose"],
            phi_grids["sidepod"],
            phi_grids["rearpod"],
            phi_grids["main_body"],
        )
    except Exception:
        return EvaluationResult(
            W_mm=W_mm, x_front_mm=x_front_mm, d_halo_mm=d_halo_mm,
            race_time=1e6, mass_kg=0.0, h_com_m=0.0, x_com_m=0.0,
            lifecycle="objective_failed",
            wall_time_s=time.perf_counter() - t0,
        )

    # Total mass and COM: use Part 2's real ingest_mass_com() (validated mass
    # sum + mass-weighted COM, with sanity-bound checking) when the fixed
    # hardware spec was built successfully above. Falls back to an ad-hoc
    # proxy sum only if Part 2 itself is unavailable.
    if fixed_hardware_result is not None:
        from mass_com_ingest import ingest_mass_com
        try:
            full_car = ingest_mass_com(components, fixed_hardware_result.fixed_hardware_spec)
        except ValueError:
            return EvaluationResult(
                W_mm=W_mm, x_front_mm=x_front_mm, d_halo_mm=d_halo_mm,
                race_time=1e6, mass_kg=0.0, h_com_m=0.0, x_com_m=0.0,
                lifecycle="objective_failed",
                wall_time_s=time.perf_counter() - t0,
            )
        total_mass = full_car.total_mass_kg
        x_com = full_car.com_x_m
        h_com = max(full_car.com_z_m, 0.001)
    else:
        machined_mass = sum(c.mass_kg for c in components)
        total_mass = machined_mass + CO2_MASS_KG + STUB_WHEEL_AXLE_MASS_KG + STUB_HALO_MASS_KG + STUB_REAR_WING_MASS_KG
        if total_mass > 0:
            x_com = sum(c.mass_kg * c.com_x_m for c in components) / total_mass
            z_com = sum(c.mass_kg * c.com_z_m for c in components) / total_mass
        else:
            x_com = mm_to_m(x_front_mm)
            z_com = 0.020
        h_com = max(z_com, 0.001)

    # Proxy race time
    T_proxy = (
        0.5 * (total_mass / 0.055)
        + 0.3 * (h_com / 0.025)
        + 0.2 * (W_mm / 130.0)
    )

    # Save phi snapshots — PhiGrid.save(candidate_id, out_dir) → absolute path.
    # save() prepends "phi_{component}_" so candidate_id needs no suffix.
    os.makedirs(output_dir, exist_ok=True)
    snapshots: dict[str, str] = {}
    candidate_id = f"eval{eval_id:04d}"
    for comp, pg in phi_grids.items():
        path = pg.save(candidate_id, output_dir)
        snapshots[comp] = path

    return EvaluationResult(
        W_mm=W_mm, x_front_mm=x_front_mm, d_halo_mm=d_halo_mm,
        race_time=T_proxy, mass_kg=total_mass, h_com_m=h_com, x_com_m=x_com,
        lifecycle="valid_simulated",
        phi_snapshots=snapshots,
        wall_time_s=time.perf_counter() - t0,
    )


# ── Level 2 real (Part 3's evolutionary inner loop) ─────────────────────────────
# Bindings are expensive to construct (parses the thrust CSV, fits the locked
# race objective's SmoothSheetModel) -- cached per SearchConfig instance so a
# whole search only pays that cost once, not once per outer evaluation.
_real_bindings_cache: dict[int, object] = {}


def _get_real_bindings(search_config: SearchConfig):
    key = id(search_config)
    if key not in _real_bindings_cache:
        from pipeline_interface import real_bindings
        _real_bindings_cache[key] = real_bindings(
            thrust_csv_path=search_config.thrust_csv_path,
            fixed_hardware_kwargs=search_config.fixed_hardware_kwargs,
            out_dir=search_config.output_dir,
        )
    return _real_bindings_cache[key]


def _level2_evaluate_real(
    W_mm:       float,
    x_front_mm: float,
    d_halo_mm:  float,
    output_dir: str,
    eval_id:    int,
    warm_phi_paths: Optional[dict[str, str]],
    search_config: SearchConfig,
) -> EvaluationResult:
    """
    Real Level 2: run Part 3's actual evolutionary inner loop at this
    (W, x_front, d_halo) point and return the winning candidate's real
    T_penalized as the race time.

    This is the P1-18 wiring: Level 1 (this file) proposes points, Part 3
    (wheelbase_sweep.optimize_single_w) executes an M-candidate evolutionary
    population at each proposed point -- real quality gates, real OpenFOAM
    CFD, the real race objective, real OpenFOAM adjoint, real phi updates.

    Known scope limitation: Level 1's own warm-start mechanism (nearest
    previous evaluation in normalised 3D space, via saved .npy snapshots) is
    NOT threaded into Part 3's call here -- Part 3's warm-starting is built
    around ADJACENT W values in a monotonic sweep (see wheelbase_sweep.py's
    docstring), which doesn't map cleanly onto Bayesian optimisation's
    arbitrary jumps around the 3D space. Every real evaluation currently
    starts from fresh phi grids (bindings.initialize_phi_fields). warm_phi_paths
    is accepted for signature symmetry with the proxy path but intentionally
    unused here; revisit if evaluation cost makes this matter in practice.

    CandidateOutcome (Part 3's per-candidate result type) does not carry
    mass_kg/h_com_m/x_com_m -- those are internal to the inner loop's
    objective computation and never surface on the record. mass_kg/h_com_m/
    x_com_m are therefore 0.0 in the returned EvaluationResult for this path
    (documented, not fabricated) -- only race_time and lifecycle are
    meaningful. Use the candidate record (record_path) for the real mass/COM
    if needed.
    """
    from wheelbase_sweep import optimize_single_w
    from optimizer_contract import OptimizerConfig, FAILURE_PENALTY_S

    t0 = time.perf_counter()

    # Cheap pre-check before touching Part 3 at all. optimize_single_w only
    # eagerly validates W and x_front (see wheelbase_sweep.py) -- an invalid
    # d_halo is NOT caught until every candidate in the evolutionary
    # population has individually failed inside run_inner_loop (found live:
    # without this check, an invalid d_halo wasted a full n_candidates_per_point
    # population attempt and was misreported as "objective_failed" instead
    # of "geometry_rejected"). _is_valid checks all three together.
    if not _is_valid(W_mm, x_front_mm, d_halo_mm):
        return EvaluationResult(
            W_mm=W_mm, x_front_mm=x_front_mm, d_halo_mm=d_halo_mm,
            race_time=FAILURE_PENALTY_S, mass_kg=0.0, h_com_m=0.0, x_com_m=0.0,
            lifecycle="geometry_rejected",
            wall_time_s=time.perf_counter() - t0,
        )

    bindings = _get_real_bindings(search_config)

    part3_config = OptimizerConfig(
        rtc_validated_against_track_data=search_config.rtc_validated_against_track_data,
        cfd_pipeline_validated_on_known_geometry=search_config.cfd_pipeline_validated_on_known_geometry,
        mu=search_config.mu,
        wheel_moi_kg_m2=search_config.wheel_moi_kg_m2,
        iteration_budget=search_config.inner_iteration_budget,
        random_seed=search_config.random_seed + eval_id,
    )

    try:
        w_result = optimize_single_w(
            bindings, part3_config, W_mm, x_front_mm, d_halo_mm,
            n_candidates=search_config.n_candidates_per_point,
            out_dir=output_dir,
            gradient_weights=search_config.gradient_weights,
            n_evolution_rounds=search_config.n_evolution_rounds,
            candidate_prefix=f"bo_eval{eval_id:04d}",
        )
    except ValueError as exc:
        # (W, x_front, d_halo) itself is geometrically invalid -- same
        # "cheap rejection before Level 2" behavior as the proxy path.
        return EvaluationResult(
            W_mm=W_mm, x_front_mm=x_front_mm, d_halo_mm=d_halo_mm,
            race_time=FAILURE_PENALTY_S, mass_kg=0.0, h_com_m=0.0, x_com_m=0.0,
            lifecycle="geometry_rejected",
            wall_time_s=time.perf_counter() - t0,
        )

    if w_result.best is None:
        if w_result.task_failures:
            print(f"  [bo_eval{eval_id:04d}] all candidates failed; last error: "
                  f"{w_result.task_failures[-1].message}")
        return EvaluationResult(
            W_mm=W_mm, x_front_mm=x_front_mm, d_halo_mm=d_halo_mm,
            race_time=FAILURE_PENALTY_S, mass_kg=0.0, h_com_m=0.0, x_com_m=0.0,
            lifecycle="objective_failed",
            wall_time_s=time.perf_counter() - t0,
        )

    snapshots: dict[str, str] = {}
    if w_result.best_phi_grids is not None:
        os.makedirs(output_dir, exist_ok=True)
        candidate_id = f"bo_eval{eval_id:04d}"
        for comp, pg in w_result.best_phi_grids.items():
            snapshots[comp] = pg.save(candidate_id, output_dir)

    return EvaluationResult(
        W_mm=W_mm, x_front_mm=x_front_mm, d_halo_mm=d_halo_mm,
        race_time=w_result.best.ranking_time_s(),
        mass_kg=0.0, h_com_m=0.0, x_com_m=0.0,  # not on CandidateOutcome -- see docstring
        lifecycle=w_result.best.lifecycle_state,
        phi_snapshots=snapshots,
        wall_time_s=time.perf_counter() - t0,
    )


class _StubCylinder:
    """Minimal cylinder stub when fixed_hardware (Part 2 dep) is unavailable."""
    def __init__(self, x_front_mm: float, W_mm: float, lead: bool):
        x_front_m = mm_to_m(x_front_mm)
        W_m       = mm_to_m(W_mm)
        cx = x_front_m if lead else x_front_m + W_m
        hw = WHEEL_X_HALF_WIDTH_M
        self.x_min_m = cx - hw
        self.x_max_m = cx + hw


# ── Main Bayesian outer search ─────────────────────────────────────────────────

class BayesianOuterSearch:
    """
    Level 1: Gaussian-process Bayesian optimisation over (W, x_front, d_halo).

    Algorithm (per 01_generative_geometry.md):
      1. Evaluate n_initial quasi-random seed points (Sobol sequence).
      2. Fit SingleTaskGP to all observations.
      3. Maximise Expected Improvement acquisition to pick next candidate.
      4. Evaluate candidate with Level 2 inner loop.
      5. Repeat steps 2-4 for n_iterations total BO steps.
    """

    def __init__(self, config: SearchConfig) -> None:
        self.cfg    = config
        self._results: list[EvaluationResult] = []
        self._eval_count = 0
        os.makedirs(config.output_dir, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(self) -> SearchResult:
        """Execute the full search and return the best result found."""
        import torch
        from botorch.utils.sampling import draw_sobol_samples

        t_start = time.perf_counter()
        torch.manual_seed(self.cfg.random_seed)
        np.random.seed(self.cfg.random_seed)

        bounds_t = torch.zeros(2, 3, dtype=torch.double)
        bounds_t[0] = 0.0
        bounds_t[1] = 1.0

        # ── Phase 1: seed evaluations (Sobol) ─────────────────────────────────
        print(f"[BO] Phase 1: {self.cfg.n_initial} seed evaluations (Sobol)")
        seed_X = draw_sobol_samples(bounds_t, n=self.cfg.n_initial, q=1).squeeze(1)

        for i, xu in enumerate(seed_X):
            W, xf, dh = _from_unit(xu)
            self._evaluate_and_record(W, xf, dh, label=f"seed-{i+1}")

        # ── Phase 2: BO iterations ─────────────────────────────────────────────
        print(f"[BO] Phase 2: {self.cfg.n_iterations} BO iterations")
        for i in range(self.cfg.n_iterations):
            candidate_u = self._propose_next()
            W, xf, dh   = _from_unit(candidate_u)
            self._evaluate_and_record(W, xf, dh, label=f"bo-{i+1}")

        total_time = time.perf_counter() - t_start

        best = min(self._results, key=lambda r: r.race_time)
        print(
            f"\n[BO] Done. Best: W={best.W_mm:.1f} x_front={best.x_front_mm:.1f} "
            f"d_halo={best.d_halo_mm:.1f}  T={best.race_time:.4f}  "
            f"mass={best.mass_kg*1000:.1f}g  h_com={best.h_com_m*1000:.1f}mm"
        )

        return SearchResult(
            best_params={"W_mm": best.W_mm, "x_front_mm": best.x_front_mm, "d_halo_mm": best.d_halo_mm},
            best_time=best.race_time,
            all_results=self._results,
            total_wall_s=total_time,
            converged=self._check_convergence(),
        )

    # ── Private helpers ────────────────────────────────────────────────────────

    def _evaluate_and_record(
        self,
        W_mm: float,
        x_front_mm: float,
        d_halo_mm:  float,
        label: str = "",
    ) -> None:
        """Validate, warm-start, run Level 2, and record the result."""
        # Clamp to W-dependent bounds, rejecting if still invalid
        W_mm       = float(np.clip(W_mm,       W_MIN_MM,    W_MAX_MM))
        xf_lo, xf_hi = calibrate_x_front_bounds(W_mm)
        x_front_mm = float(np.clip(x_front_mm, xf_lo,       xf_hi))
        # calibrate_d_halo_max_mm returns a STRICT (exclusive) upper bound
        # (K-5: placement-derived W-34, not the stale W+16). Clip to a hair
        # under it so the clamped value still passes validate_d_halo's "<".
        dh_hi      = calibrate_d_halo_max_mm(W_mm) - 1e-6
        d_halo_mm  = float(np.clip(d_halo_mm,  0.0,         dh_hi))

        warm = self._find_warm_start(W_mm, x_front_mm, d_halo_mm)

        self._eval_count += 1
        result = _level2_evaluate(
            W_mm, x_front_mm, d_halo_mm,
            rule_envelope=self.cfg.rule_envelope,
            n_iters=self.cfg.level2_iters,
            output_dir=self.cfg.output_dir,
            eval_id=self._eval_count,
            warm_phi_paths=warm,
            search_config=self.cfg,
        )
        self._results.append(result)
        n_valid = sum(1 for r in self._results if r.race_time < 1e5)
        print(
            f"  [{label:>10s}] W={W_mm:5.1f} xf={x_front_mm:5.1f} dh={d_halo_mm:5.1f}"
            f"  T={result.race_time:.4f}  {result.lifecycle:20s}"
            f"  ({n_valid} valid so far)"
        )

    def _propose_next(self) -> "torch.Tensor":
        """
        Fit the GP to all valid observations and return the next candidate
        in unit space that maximises Expected Improvement.

        Falls back to a random sample if fewer than 2 valid points exist.
        """
        import torch
        from botorch.models import SingleTaskGP
        from botorch.fit import fit_gpytorch_mll
        from botorch.acquisition import ExpectedImprovement
        from botorch.optim import optimize_acqf
        from gpytorch.mlls import ExactMarginalLogLikelihood

        valid = [r for r in self._results if r.race_time < 1e5]

        if len(valid) < 2:
            # Not enough data for GP — random sample
            return torch.rand(3, dtype=torch.double)

        # Build training tensors (unit space)
        train_X = torch.tensor(
            [list(r.normalised_params) for r in valid],
            dtype=torch.double,
        )
        train_Y = torch.tensor(
            [[-r.race_time] for r in valid],   # negate: BoTorch maximises
            dtype=torch.double,
        )

        # Fit GP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SingleTaskGP(train_X, train_Y)
            mll   = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            model.eval()

        # Optimise Expected Improvement acquisition
        bounds_t = torch.zeros(2, 3, dtype=torch.double)
        bounds_t[1] = 1.0
        ei = ExpectedImprovement(model, best_f=train_Y.max())

        candidate, _ = optimize_acqf(
            acq_function=ei,
            bounds=bounds_t,
            q=1,
            num_restarts=self.cfg.acqf_restarts,
            raw_samples=self.cfg.acqf_raw_samples,
        )
        return candidate.squeeze(0)

    def _find_warm_start(
        self,
        W_mm: float,
        x_front_mm: float,
        d_halo_mm:  float,
    ) -> Optional[dict[str, str]]:
        """
        Return phi snapshot paths from the nearest previous evaluation if it is
        within WARM_START_THRESHOLD in normalised space; else return None.
        """
        if not self._results:
            return None

        u_new = _to_unit(W_mm, x_front_mm, d_halo_mm)

        best_dist = math.inf
        best_result: Optional[EvaluationResult] = None
        for r in self._results:
            if not r.phi_snapshots:
                continue
            u_prev = r.normalised_params
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(u_new, u_prev)))
            if dist < best_dist:
                best_dist   = dist
                best_result = r

        if best_result is not None and best_dist < WARM_START_THRESHOLD:
            # Verify snapshot files still exist
            if all(os.path.exists(p) for p in best_result.phi_snapshots.values()):
                return best_result.phi_snapshots

        return None

    def _check_convergence(self) -> bool:
        """
        Declare convergence if the best 10 BO results (excluding seed) span
        less than 1% of the proxy race-time range. Rough heuristic.
        """
        bo_results = [r for r in self._results[self.cfg.n_initial:] if r.race_time < 1e5]
        if len(bo_results) < 10:
            return False
        times = sorted(r.race_time for r in bo_results[-10:])
        span  = times[-1] - times[0]
        best  = times[0]
        return best > 0 and (span / best) < 0.01


# ── Convenience entry point ────────────────────────────────────────────────────

def run_bayesian_search(
    rule_envelope:  Optional[RuleEnvelope] = None,
    n_initial:      int = 15,
    n_iterations:   int = 65,
    output_dir:     str = "bayesian_results",
    random_seed:    int = 42,
    level2_iters:   int = 0,
) -> SearchResult:
    """
    One-call entry point for the Bayesian outer search.

    Args:
        rule_envelope: UAE regulation envelope (U6). Defaults to
                       default_rule_envelope() (see bounding_volumes.py for
                       which fields are real regulation numbers vs. design
                       choices within a legal range) if not provided.
        n_initial:     Sobol seed evaluations (10-20 recommended).
        n_iterations:  BO iterations after seed. Total = n_initial + n_iterations.
        output_dir:    Directory for phi snapshots and logs.
        random_seed:   Reproducibility seed.
        level2_iters:  Inner phi-update steps per evaluation (0 = geometry only).

    Returns:
        SearchResult with best (W, x_front, d_halo) and all intermediate results.
    """
    config = SearchConfig(
        rule_envelope=rule_envelope if rule_envelope is not None else default_rule_envelope(),
        n_initial=n_initial,
        n_iterations=n_iterations,
        output_dir=output_dir,
        random_seed=random_seed,
        level2_iters=level2_iters,
    )
    search = BayesianOuterSearch(config)
    return search.run()
