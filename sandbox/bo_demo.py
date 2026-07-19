"""
bo_demo.py -- watch the Bayesian outer search explore (W, x_front, d_halo)
without CFD.

This drives bayesian_outer_search._level2_evaluate directly -- the SAME
Level 2 proxy path the real search uses when use_real_pipeline=False. So what
you see here is the actual wiring: real bounding volumes, real fixed-hardware
placement, real virtual-cargo placement, real mass/COM, and the proxy race
time

    T = 0.5*(mass/0.055) + 0.3*(h_com/0.025) + 0.2*(W/130)

No OpenFOAM, no adjoint. The proxy is a stand-in for the real objective --
it rewards light, low-COM, short-wheelbase cars and knows nothing about
aerodynamics. Use it to check that the search MOVES SENSIBLY and that
geometry gets accepted/rejected where you expect, not to pick a real design.

GP backend: uses BoTorch if installed (the real code path). Falls back to a
small numpy GP + Expected Improvement so the demo runs on a machine without
torch/botorch. The fallback is a teaching aid, not a replacement -- it uses
a fixed lengthscale instead of fitting hyperparameters by marginal likelihood.

Usage
-----
    python sandbox/bo_demo.py                       # 12 seed + 18 BO evals
    python sandbox/bo_demo.py --n-seed 8 --n-bo 12
    python sandbox/bo_demo.py --spacing 2.0         # faster, coarser
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import coarse  # sets sys.path + PART2_PATH


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-seed", type=int, default=12, help="quasi-random seed evaluations")
    p.add_argument("--n-bo", type=int, default=18, help="Bayesian iterations after seeding")
    p.add_argument("--spacing", type=float, default=2.0, help="grid spacing mm")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


# ── Minimal GP + EI fallback (used when botorch is absent) ────────────────────

def _rbf(a: np.ndarray, b: np.ndarray, lengthscale: float) -> np.ndarray:
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / lengthscale ** 2)


def _gp_posterior(X, y, Xq, lengthscale=0.3, noise=1e-4):
    """Standard GP regression posterior. X, Xq in unit space; y standardised."""
    K = _rbf(X, X, lengthscale) + noise * np.eye(len(X))
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Ks = _rbf(X, Xq, lengthscale)
    mu = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var = np.clip(1.0 - (v ** 2).sum(0), 1e-12, None)
    return mu, np.sqrt(var)


def _expected_improvement(mu, sigma, best):
    """EI for MINIMISATION of the standardised objective."""
    from scipy.stats import norm
    imp = best - mu
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def _propose_numpy(results, rng, n_candidates=4000):
    """Pick the next unit-space point by maximising EI over a random candidate set."""
    valid = [r for r in results if r.race_time < 1e5]
    if len(valid) < 3:
        return rng.random(3)

    X = np.array([r.normalised_params for r in valid])
    raw = np.array([r.race_time for r in valid])
    mu_y, sd_y = raw.mean(), raw.std() or 1.0
    y = (raw - mu_y) / sd_y

    Xq = rng.random((n_candidates, 3))
    mu, sigma = _gp_posterior(X, y, Xq)
    ei = _expected_improvement(mu, sigma, y.min())
    return Xq[int(np.argmax(ei))]


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_search(results, out_dir: Path, n_seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in results if r.race_time < 1e5]
    rejected = [r for r in results if r.race_time >= 1e5]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Convergence: best-so-far.
    ax = axes[0, 0]
    times = [r.race_time if r.race_time < 1e5 else np.nan for r in results]
    best_so_far = np.fmin.accumulate(np.array(times, dtype=float))
    ax.plot(range(1, len(results) + 1), times, "o", ms=4, alpha=0.5, label="evaluation")
    ax.plot(range(1, len(results) + 1), best_so_far, "-", lw=2, label="best so far")
    ax.axvline(n_seed + 0.5, color="k", ls="--", lw=1, label="seed -> BO")
    ax.set_xlabel("evaluation")
    ax.set_ylabel("proxy race time")
    ax.set_title(f"Convergence ({len(valid)} valid, {len(rejected)} rejected)")
    ax.legend(fontsize=8)

    # Each parameter against the objective.
    for ax, attr, label in (
        (axes[0, 1], "W_mm", "W [mm]"),
        (axes[1, 0], "x_front_mm", "x_front [mm]"),
        (axes[1, 1], "d_halo_mm", "d_halo [mm]"),
    ):
        if valid:
            xs = [getattr(r, attr) for r in valid]
            ys = [r.race_time for r in valid]
            order = np.arange(len(valid))
            sc = ax.scatter(xs, ys, c=order, cmap="viridis", s=36)
            fig.colorbar(sc, ax=ax, label="evaluation order")
            best = min(valid, key=lambda r: r.race_time)
            ax.plot(getattr(best, attr), best.race_time, "r*", ms=18, label="best")
            ax.legend(fontsize=8)
        if rejected:
            # Show rejected samples along the bottom so dead zones are visible.
            ax.plot([getattr(r, attr) for r in rejected],
                    [ax.get_ylim()[0]] * len(rejected), "rx", ms=5,
                    label="geometry rejected")
        ax.set_xlabel(label)
        ax.set_ylabel("proxy race time")
        ax.set_title(f"{label} vs objective")

    fig.suptitle("Bayesian outer search -- proxy objective, no CFD")
    fig.tight_layout()
    path = out_dir / "bo_search.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\nwrote {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()
    coarse.use_spacing(args.spacing)

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "out" / "bo_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    from bayesian_outer_search import (
        _level2_evaluate, _from_unit, _abs_bounds,
    )
    from bounding_volumes import default_rule_envelope

    try:
        import botorch  # noqa: F401
        backend = "botorch"
    except ImportError:
        backend = "numpy-gp"

    print(f"== Bayesian outer search demo ==")
    print(f"  backend    {backend}"
          f"{'  (botorch not installed -- using the fallback GP)' if backend == 'numpy-gp' else ''}")
    print(f"  spacing    {args.spacing} mm")
    print(f"  budget     {args.n_seed} seed + {args.n_bo} BO = {args.n_seed + args.n_bo} evaluations")
    (w_lo, w_hi), (xf_lo, xf_hi), (dh_lo, dh_hi) = _abs_bounds()
    print(f"  W          [{w_lo:.0f}, {w_hi:.0f}] mm")
    print(f"  x_front    [{xf_lo:.0f}, {xf_hi:.0f}] mm")
    print(f"  d_halo     [{dh_lo:.0f}, {dh_hi:.1f}] mm")

    from geometry_contract import calibrate_x_front_bounds, calibrate_d_halo_max_mm

    rng = np.random.default_rng(args.seed)
    rule_envelope = default_rule_envelope()
    results = []
    n_clipped = 0

    print(f"\n{'#':>3s} {'phase':<5s} {'W':>6s} {'x_front':>8s} {'d_halo':>7s} "
          f"{'T':>9s} {'mass g':>8s} {'h_com mm':>9s} {'lifecycle':<18s} {'s':>5s}")

    t_start = time.perf_counter()
    for i in range(args.n_seed + args.n_bo):
        if i < args.n_seed:
            phase, u = "seed", rng.random(3)
        else:
            phase = "bo"
            u = _propose_numpy(results, rng)

        W, xf, dh = _from_unit(u)
        # Mirror BayesianOuterSearch._evaluate_and_record exactly: the unit
        # cube is normalised against ABSOLUTE bounds, but x_front and d_halo
        # have W-dependent bounds, so the real search clips before evaluating.
        # Skipping this makes the demo look far more broken than the search is.
        W = float(np.clip(W, w_lo, w_hi))
        xf_l, xf_h = calibrate_x_front_bounds(W)
        xf = float(np.clip(xf, xf_l, xf_h))
        dh = float(np.clip(dh, 0.0, calibrate_d_halo_max_mm(W) - 1e-6))
        n_clipped += int(abs(xf - _from_unit(u)[1]) > 1e-9
                         or abs(dh - _from_unit(u)[2]) > 1e-9)

        t0 = time.perf_counter()
        result = _level2_evaluate(
            W, xf, dh,
            rule_envelope=rule_envelope,
            n_iters=0,
            output_dir=str(out_dir),
            eval_id=i + 1,
        )
        results.append(result)

        t_str = f"{result.race_time:9.4f}" if result.race_time < 1e5 else "  REJECTED"
        print(f"{i+1:>3d} {phase:<5s} {W:6.1f} {xf:8.1f} {dh:7.1f} {t_str} "
              f"{result.mass_kg*1000:8.2f} {result.h_com_m*1000:9.2f} "
              f"{result.lifecycle:<18s} {time.perf_counter()-t0:5.1f}")

    valid = [r for r in results if r.race_time < 1e5]
    print(f"\nTotal {time.perf_counter()-t_start:.1f}s   "
          f"{len(valid)}/{len(results)} evaluations produced valid geometry")
    print(f"{n_clipped}/{len(results)} samples were clipped onto a W-dependent "
          f"bound before evaluation -- those land on the edge of the feasible\n"
          f"box, so several distinct GP inputs can map to the same geometry.")

    if valid:
        best = min(valid, key=lambda r: r.race_time)
        print(f"\nBest: W={best.W_mm:.1f} x_front={best.x_front_mm:.1f} "
              f"d_halo={best.d_halo_mm:.1f}")
        print(f"      T={best.race_time:.4f}  mass={best.mass_kg*1000:.2f} g  "
              f"h_com={best.h_com_m*1000:.2f} mm  x_com={best.x_com_m*1000:.2f} mm")
        print(f"\nInspect it with:\n"
              f"  python sandbox/explore.py --W {best.W_mm:.0f} "
              f"--x-front {best.x_front_mm:.0f} --d-halo {best.d_halo_mm:.0f} --gates")
    else:
        print("\nNo valid geometry found -- every sample was rejected. "
              "That is a wiring/bounds problem, not an optimisation result.")

    plot_search(results, out_dir, args.n_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
