"""
why_rejected.py -- find out WHICH stage rejects each (W, x_front, d_halo).

_level2_evaluate collapses four different failure causes into the single
lifecycle string "geometry_rejected", so a search log tells you that most of
the space is dead but not why. This re-runs the same stages one at a time and
reports the first one that raises, plus the exception message.

Stages, in the order _level2_evaluate runs them:
    1. validate_W / validate_x_front / validate_d_halo   (contract bounds)
    2. compute_bounding_volumes                          (corridor collapse etc.)
    3. find_cargo_placement                              (T4.2 wedge doesn't fit)
    4. place_fixed_hardware                              (halo pocket placement)

Usage
-----
    python sandbox/why_rejected.py                  # sweep the whole space
    python sandbox/why_rejected.py --W 130 --x-front 75 --d-halo 60   # one point
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

import coarse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--W", type=float, default=None)
    p.add_argument("--x-front", type=float, default=None)
    p.add_argument("--d-halo", type=float, default=None)
    p.add_argument("--spacing", type=float, default=2.0)
    p.add_argument("--n", type=int, default=8, help="samples per axis for the sweep")
    p.add_argument("--out", default=None)
    return p.parse_args()


def diagnose(W: float, xf: float, dh: float, rule_envelope) -> tuple[str, str]:
    """Return (stage, message). stage == 'ok' if the candidate survives."""
    from geometry_contract import validate_W, validate_x_front, validate_d_halo
    from bounding_volumes import compute_bounding_volumes
    from phi_grid_factory import _default_forbidden_cylinders
    from virtual_cargo import find_cargo_placement
    from fixed_hardware import (
        place_fixed_hardware, compute_default_fixed_hardware_inputs,
    )

    try:
        validate_W(W)
        validate_x_front(xf, W)
        validate_d_halo(dh, W)
    except ValueError as e:
        return "1_contract_bounds", str(e)

    try:
        from geometry_contract import WHEEL_X_CLEARANCE_HALF_WIDTH_MM
        front, rear = _default_forbidden_cylinders(W, xf, WHEEL_X_CLEARANCE_HALF_WIDTH_MM)
        bv = compute_bounding_volumes(W, xf, dh, front, rear, rule_envelope,
                                      wheel_x_half_width_mm=WHEEL_X_CLEARANCE_HALF_WIDTH_MM)
    except (ValueError, NotImplementedError) as e:
        return "2_bounding_volumes", str(e)

    try:
        find_cargo_placement(xf, W, bv.ref_plane_A_m, dh, rule_envelope.z_floor_m)
    except ValueError as e:
        return "3_virtual_cargo", str(e)

    try:
        hw = compute_default_fixed_hardware_inputs(
            W, xf, dh, bv.ref_plane_A_m, bv.ref_plane_B_m)
        hw["body_grid_shape"] = bv.main_body.shape
        hw["body_grid_origin_m"] = bv.main_body.origin_m
        place_fixed_hardware(W_mm=W, x_front_mm=xf, **hw)
    except (ValueError, NotImplementedError) as e:
        return "4_fixed_hardware", str(e)

    return "ok", ""


def main() -> int:
    args = parse_args()
    coarse.use_spacing(args.spacing)

    from bounding_volumes import default_rule_envelope
    from geometry_contract import (
        W_MIN_MM, W_MAX_MM, calibrate_x_front_bounds, calibrate_d_halo_max_mm,
    )
    rule_envelope = default_rule_envelope()

    # Single-point mode.
    if args.W is not None and args.x_front is not None and args.d_halo is not None:
        stage, msg = diagnose(args.W, args.x_front, args.d_halo, rule_envelope)
        print(f"W={args.W} x_front={args.x_front} d_halo={args.d_halo}")
        print(f"  stage : {stage}")
        if msg:
            print(f"  reason: {msg}")
        return 0 if stage == "ok" else 1

    # Sweep mode: grid over the space, using each W's own valid sub-ranges so
    # we are testing real candidates rather than trivially out-of-bounds ones.
    n = args.n
    counter: Counter[str] = Counter()
    examples: dict[str, str] = {}
    records = []

    for W in np.linspace(W_MIN_MM, W_MAX_MM, n):
        xf_lo, xf_hi = calibrate_x_front_bounds(W)
        dh_hi = calibrate_d_halo_max_mm(W)
        for xf in np.linspace(xf_lo, xf_hi, n):
            for dh in np.linspace(0.0, dh_hi * 0.999, n):
                stage, msg = diagnose(float(W), float(xf), float(dh), rule_envelope)
                counter[stage] += 1
                records.append((float(W), float(xf), float(dh), stage))
                if stage != "ok" and stage not in examples:
                    examples[stage] = f"(W={W:.1f}, x_front={xf:.1f}, d_halo={dh:.1f}) {msg}"

    total = sum(counter.values())
    print(f"== Rejection sweep: {n}^3 = {total} candidates, spacing {args.spacing} mm ==\n")
    print(f"{'stage':<20s} {'count':>7s} {'share':>8s}")
    for stage, count in counter.most_common():
        print(f"{stage:<20s} {count:>7d} {100.0*count/total:7.1f}%")

    if examples:
        print("\nFirst failure of each kind:")
        for stage, example in sorted(examples.items()):
            print(f"\n  {stage}\n    {example}")

    # Where in the space does d_halo actually work? This is the axis the
    # search wastes the most samples on.
    ok_dh = [r[2] for r in records if r[3] == "ok"]
    if ok_dh:
        print(f"\nd_halo of accepted candidates: "
              f"min={min(ok_dh):.1f} max={max(ok_dh):.1f} mm "
              f"(search samples 0-{calibrate_d_halo_max_mm(W_MAX_MM):.1f} mm)")

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(records, out_dir)
    return 0


def plot(records, out_dir: Path) -> None:
    """Map the feasible region: x_front vs d_halo, one panel per W."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # "ok" is always green; failure stages get distinct warm colours. Zipping
    # a palette against sorted(stages) would hand green to whichever stage
    # sorts first, which is exactly backwards when a failure name sorts ahead
    # of "ok".
    stages = ["ok"] + sorted({r[3] for r in records} - {"ok"})
    failure_palette = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#1f77b4"]
    colours = {"ok": "#2ca02c"}
    for stage, colour in zip(stages[1:], failure_palette):
        colours[stage] = colour

    Ws = sorted({r[0] for r in records})
    ncols = min(4, len(Ws))
    nrows = int(np.ceil(len(Ws) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows),
                             squeeze=False)

    for ax, W in zip(axes.ravel(), Ws):
        subset = [r for r in records if r[0] == W]
        for stage in stages:
            pts = [(r[1], r[2]) for r in subset if r[3] == stage]
            if pts:
                ax.scatter(*zip(*pts), s=26, c=colours[stage],
                           label=stage if W == Ws[0] else None)
        ax.set_title(f"W = {W:.1f} mm")
        ax.set_xlabel("x_front [mm]")
        ax.set_ylabel("d_halo [mm]")

    for ax in axes.ravel()[len(Ws):]:
        ax.axis("off")

    fig.legend(loc="lower center", ncol=len(stages), fontsize=9)
    fig.suptitle("Feasible region of the outer search space (green = accepted)")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = out_dir / "feasible_region.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
