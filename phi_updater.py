"""
phi_updater.py --- Hamilton-Jacobi level-set evolution and adjoint sensitivity.

Applies adjoint surface sensitivity to phi grids via the Hamilton-Jacobi
update equation:  phi_new = phi_old - dt * F * |grad phi|

Uses Godunov upwind scheme for |grad phi|, redistances after every step, and
extends the surface velocity into the volume.

The redistancing is not optional bookkeeping: hj_update's displacement is
dt*V*|grad phi|, so a field that is not a distance function does not move.
"""
from __future__ import annotations
import warnings

import numpy as np

from geometry_contract import GRID_SPACING_M, get_density

# Share of the sensitivity's sum-of-squares the 10 largest points may carry
# before the field counts as too spiky for unit-RMS normalisation to survive.
# A healthy drag sensitivity is smooth over the patch, so 10 points out of
# ~10k should carry a few percent; the diverged mesh-movement field measured on
# 2026-07-27 carried 99.33%. 0.5 sits far above the former and far below the
# latter.
_SENS_CONCENTRATION_LIMIT = 0.5
from phi_grid import PhiGrid


# ------------------------------------------------------------------ #
#  Godunov gradient
# ------------------------------------------------------------------ #

def _godunov_gradient(phi: np.ndarray, axis: int) -> np.ndarray:
    """
    Compute Godunov upwind gradient magnitude along one axis.

    Returns array of same shape as phi with the Godunov gradient component.
    Interior cells use the Godunov scheme; boundary cells use one-sided differences.
    """
    dx = GRID_SPACING_M
    grad = np.zeros_like(phi, dtype=np.float64)

    # One-sided differences at boundaries
    if axis == 0:
        grad[0] = (phi[1] - phi[0]) / dx
        grad[-1] = (phi[-1] - phi[-2]) / dx
        D_minus = (phi[1:-1] - phi[:-2]) / dx   # shape (nx-2, ny, nz)
        D_plus = (phi[2:] - phi[1:-1]) / dx
    elif axis == 1:
        np.take(phi, 0, axis=1)
        grad_slice_0 = (phi[:, 1, :] - phi[:, 0, :]) / dx if phi.shape[1] > 1 else np.zeros_like(phi[:, 0, :])
        grad_slice_n = (phi[:, -1, :] - phi[:, -2, :]) / dx if phi.shape[1] > 1 else np.zeros_like(phi[:, -1, :])
        grad[:, 0, :] = grad_slice_0
        grad[:, -1, :] = grad_slice_n
        D_minus = (phi[:, 1:-1, :] - phi[:, :-2, :]) / dx
        D_plus = (phi[:, 2:, :] - phi[:, 1:-1, :]) / dx
    elif axis == 2:
        grad[:, :, 0] = (phi[:, :, 1] - phi[:, :, 0]) / dx if phi.shape[2] > 1 else 0.0
        grad[:, :, -1] = (phi[:, :, -1] - phi[:, :, -2]) / dx if phi.shape[2] > 1 else 0.0
        D_minus = (phi[:, :, 1:-1] - phi[:, :, :-2]) / dx
        D_plus = (phi[:, :, 2:] - phi[:, :, 1:-1]) / dx
    else:
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}")

    # Godunov scheme for interior cells
    phi_interior = None
    if axis == 0:
        phi_interior = phi[1:-1]
    elif axis == 1:
        phi_interior = phi[:, 1:-1, :]
    elif axis == 2:
        phi_interior = phi[:, :, 1:-1]

    # phi > 0: G = max(max(D-, 0)^2, min(D+, 0)^2)
    # phi <= 0: G = max(min(D-, 0)^2, max(D+, 0)^2)
    D_minus_sq = D_minus ** 2
    D_plus_sq = D_plus ** 2

    pos_mask = phi_interior > 0
    G = np.where(
        pos_mask,
        np.maximum(np.maximum(D_minus, 0.0) ** 2, np.minimum(D_plus, 0.0) ** 2),
        np.maximum(np.minimum(D_minus, 0.0) ** 2, np.maximum(D_plus, 0.0) ** 2),
    )
    grad_val = np.sqrt(np.maximum(G, 0.0))

    if axis == 0:
        grad[1:-1] = grad_val
    elif axis == 1:
        grad[:, 1:-1, :] = grad_val
    elif axis == 2:
        grad[:, :, 1:-1] = grad_val

    return grad


def _grad_magnitude(phi: np.ndarray) -> np.ndarray:
    """Compute |grad phi| using Godunov scheme on all three axes."""
    gx = _godunov_gradient(phi, 0)
    gy = _godunov_gradient(phi, 1)
    gz = _godunov_gradient(phi, 2)
    return np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)


# ------------------------------------------------------------------ #
#  Hamilton-Jacobi update
# ------------------------------------------------------------------ #

def hj_update(phi: PhiGrid, velocity: np.ndarray, dt: float) -> None:
    """
    phi_new = phi_old - dt * F * |grad phi|

    Sign convention (standard level-set):
        F > 0  →  phi decreases  →  surface moves outward (solid grows)
        F < 0  →  phi increases  →  surface moves inward  (solid shrinks)

    Computation runs in float64 to prevent float32 overflow in the Godunov
    squared-difference terms, then casts back to float32 for storage.
    After update: call phi.apply_hard_constraints().
    """
    grid_f64 = phi.grid.astype(np.float64)
    grad_mag = _grad_magnitude(grid_f64)
    vel_f64 = np.asarray(velocity, dtype=np.float64)
    updated = grid_f64 - dt * vel_f64 * grad_mag
    phi.grid = updated.astype(np.float32)
    phi.apply_hard_constraints()


# ------------------------------------------------------------------ #
#  SDF reinitialisation
# ------------------------------------------------------------------ #

def reinitialise_sdf(phi: PhiGrid, n_steps: int = 50, dt_reinit: float = None) -> None:
    """
    Reinitialise phi as a signed distance field.
    Solve dphi/dtau + sign(phi)(|grad phi| - 1) = 0 for n_steps pseudo-time steps.
    After reinit: |grad phi| should be ~1.0 everywhere.

    CFL STABILITY: The explicit pseudo-time scheme is conditionally stable.
    The stable time step is dt <= 0.5 * GRID_SPACING_M (CFL number = 0.5).
    dt_reinit=0.3 (the original value) equals ~1000 * GRID_SPACING_M — massively
    unstable; the grid explodes to 10^60 within 5 steps even in float64.

    dt_reinit defaults to 0.4 * GRID_SPACING_M (CFL number = 0.4, safely stable).
    n_steps defaults to 50 to give enough pseudo-time to propagate corrections
    across the grid from the zero level set.

    All computation runs in float64 to prevent float32 overflow in Godunov
    squared-difference terms. Result is cast back to float32 for storage.
    """
    if dt_reinit is None:
        dt_reinit = 0.4 * GRID_SPACING_M  # CFL-stable: 0.4 * dx

    # Work entirely in float64
    grid_f64 = phi.grid.astype(np.float64)

    # SMOOTHED sign, not np.sign. This is what stops redistancing from eating
    # the car.
    #
    # np.sign is discontinuous at the interface, so the cells that straddle
    # phi=0 -- exactly the ones that define where the surface IS -- get driven
    # by a full-magnitude +1 or -1 depending on which side of zero they land.
    # The zero level set drifts, and it drifts inward. Measured before this
    # change, at 1 mm spacing on a real car:
    #     reinit on the as-built field      -8.43 g
    #     reinit AGAIN on the result        -8.35 g   <- not converging
    #     reinit on a post-hj field         -2.38 g
    # A correct redistancing is idempotent: applied to a field that is already
    # a distance function it should change nothing. This one removed another
    # 8.35 g every time it was called.
    #
    # It mattered enormously, because apply_adjoint_to_unified redistances every
    # iteration. Of one full production step's mass loss, the redistancing
    # accounted for 68% at 1 mm and 90% at 2 mm -- so most of what looked like
    # the optimiser carving the car was numerical dissipation, and the T3.6 mass
    # barrier could never hold the floor because erosion outran its restoring
    # force (it settled 7.4 g under, at 40.5 g against an equilibrium of 47.9).
    #
    # phi / sqrt(phi^2 + dx^2) is the standard smoothing (Peng et al. 1999): it
    # tends to +/-1 away from the interface, where the sign is unambiguous, and
    # to 0 AT the interface, where it should not push at all. |grad phi| -> 1
    # just as before; the surface simply stops moving while it happens.
    eps = GRID_SPACING_M
    for _ in range(n_steps):
        grad_mag = _grad_magnitude(grid_f64)
        sign_phi = grid_f64 / np.sqrt(grid_f64 * grid_f64 + eps * eps)
        grid_f64 = grid_f64 - dt_reinit * sign_phi * (grad_mag - 1.0)

        # Clamp to prevent runaway in cells far from the zero level set.
        # SDF values should never exceed the grid diagonal in metres.
        max_sdf = 2.0 * max(phi.bv.shape) * GRID_SPACING_M
        grid_f64 = np.clip(grid_f64, -max_sdf, max_sdf)

        # Re-enforce hard constraints in float64 space each step.
        grid_f64[phi.hard_mask_solid] = np.minimum(
            grid_f64[phi.hard_mask_solid], -GRID_SPACING_M
        )
        grid_f64[phi.hard_mask_air] = np.maximum(
            grid_f64[phi.hard_mask_air], +GRID_SPACING_M
        )

    phi.grid = grid_f64.astype(np.float32)
    phi.apply_hard_constraints()


# ------------------------------------------------------------------ #
#  Velocity extension
# ------------------------------------------------------------------ #

def extend_velocity(
    phi_grid: np.ndarray,
    surface_velocity: np.ndarray,
    n_steps: int = 10,
    dt_ext: float = None,
) -> np.ndarray:
    """
    Propagate surface velocity into the volume by solving the extension PDE:

        dF/dtau + sign(phi) * (grad_phi / |grad_phi|) · grad_F = 0

    The upwind direction is determined by sign(phi): characteristics propagate
    away from the zero level set (outward into both + and - regions).

    The dot product is with the UNIT NORMAL n = grad_phi / |grad_phi|, not with
    grad_phi directly. Using raw grad_phi would weight the extension by |grad_phi|,
    which distorts the velocity field in regions where phi is not an exact SDF.

    For each axis, the correct upwind finite difference of F is selected based on
    the sign of the corresponding component of n (not the sign of F itself).

    CFL condition: dt_ext <= 0.4 * GRID_SPACING_M (same as reinitialise_sdf).

    Args:
        phi_grid: signed distance field (nx, ny, nz), float32 or float64
        surface_velocity: velocity defined near phi=0 surface (nx, ny, nz)
        n_steps: number of pseudo-time steps
        dt_ext: pseudo-time step; defaults to 0.4 * GRID_SPACING_M (CFL-stable)

    Returns:
        F: velocity field extended into the full volume, shape (nx, ny, nz), float64
    """
    if dt_ext is None:
        dt_ext = 0.4 * GRID_SPACING_M

    phi_f64 = np.asarray(phi_grid, dtype=np.float64)
    F = np.asarray(surface_velocity, dtype=np.float64).copy()
    dx = GRID_SPACING_M

    for _ in range(n_steps):
        # Compute unit normal n = grad_phi / |grad_phi|
        dphi_x = _godunov_gradient(phi_f64, 0)
        dphi_y = _godunov_gradient(phi_f64, 1)
        dphi_z = _godunov_gradient(phi_f64, 2)
        grad_phi_mag = np.sqrt(dphi_x**2 + dphi_y**2 + dphi_z**2)
        eps = 1e-12
        nx_ = dphi_x / (grad_phi_mag + eps)
        ny_ = dphi_y / (grad_phi_mag + eps)
        nz_ = dphi_z / (grad_phi_mag + eps)

        sign_phi = np.sign(phi_f64)

        # Upwind differences of F: direction chosen by sign of normal component
        # dF/dx upwind: use backward diff where n_x * sign_phi > 0 (char moves right)
        #               use forward  diff where n_x * sign_phi < 0 (char moves left)
        def _upwind_diff(arr: np.ndarray, axis: int, n_comp: np.ndarray) -> np.ndarray:
            """First-order upwind difference of arr along axis, direction from n_comp."""
            diff = np.zeros_like(arr)
            n = arr.shape[axis]

            # Slices for interior (indices 1..n-2), forward pair, backward pair
            slc_int  = [slice(None)] * 3; slc_int[axis]  = slice(1, n - 1)
            slc_fwd0 = [slice(None)] * 3; slc_fwd0[axis] = slice(1, n - 1)   # i
            slc_fwd1 = [slice(None)] * 3; slc_fwd1[axis] = slice(2, n)        # i+1
            slc_bwd0 = [slice(None)] * 3; slc_bwd0[axis] = slice(0, n - 2)   # i-1
            slc_bwd1 = [slice(None)] * 3; slc_bwd1[axis] = slice(1, n - 1)   # i

            fwd = (arr[tuple(slc_fwd1)] - arr[tuple(slc_fwd0)]) / dx
            bwd = (arr[tuple(slc_bwd1)] - arr[tuple(slc_bwd0)]) / dx

            char_int = (sign_phi * n_comp)[tuple(slc_int)]
            diff[tuple(slc_int)] = np.where(char_int > 0, bwd, fwd)

            # Boundaries: one-sided
            slc_lo  = [slice(None)] * 3; slc_lo[axis]  = slice(0, 1)
            slc_lo1 = [slice(None)] * 3; slc_lo1[axis] = slice(1, 2)
            slc_hi  = [slice(None)] * 3; slc_hi[axis]  = slice(-1, None)
            slc_hi1 = [slice(None)] * 3; slc_hi1[axis] = slice(-2, -1)
            diff[tuple(slc_lo)] = (arr[tuple(slc_lo1)] - arr[tuple(slc_lo)]) / dx
            diff[tuple(slc_hi)] = (arr[tuple(slc_hi)] - arr[tuple(slc_hi1)]) / dx

            return diff

        dF_x = _upwind_diff(F, 0, nx_)
        dF_y = _upwind_diff(F, 1, ny_)
        dF_z = _upwind_diff(F, 2, nz_)

        # dot product: n · grad_F
        dot = nx_ * dF_x + ny_ * dF_y + nz_ * dF_z

        F = F - dt_ext * sign_phi * dot

    return F


# ------------------------------------------------------------------ #
#  Gradient combination
# ------------------------------------------------------------------ #

def combine_gradients(
    aero_gradient: np.ndarray,
    mass_gradient: np.ndarray,
    com_gradient: np.ndarray,
    mfg_gradient: np.ndarray,
    w_aero: float,
    w_mass: float,
    w_com: float,
    w_mfg: float,
) -> np.ndarray:
    """Normalize each gradient to unit RMS, then weighted sum."""
    def _normalize(g: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(g ** 2))
        return g / rms if rms > 1e-12 else np.zeros_like(g)

    return (
        _normalize(aero_gradient) * w_aero
        + _normalize(mass_gradient) * w_mass
        + _normalize(com_gradient) * w_com
        + _normalize(mfg_gradient) * w_mfg
    )


# ------------------------------------------------------------------ #
#  Adjoint sensitivity application
# ------------------------------------------------------------------ #

def _splat_vertex_sensitivity_to_grid(
    vertex_sensitivity: np.ndarray,   # (n_vertices,) scalar sensitivity per vertex
    vertices: np.ndarray,             # (n_vertices, 3) world coords of mesh vertices
    phi: PhiGrid,
) -> np.ndarray:
    """
    Map per-vertex scalar sensitivity from a surface mesh onto the φ grid using
    nearest-cell assignment (splatting). Each mesh vertex deposits its sensitivity
    value into the grid cell whose centre is closest to the vertex position.

    When multiple vertices map to the same cell, their sensitivities are averaged.
    Cells with no nearby vertices get zero velocity.

    Returns float64 array of shape (nx, ny, nz).
    """
    nx, ny, nz = phi.bv.shape
    ox, oy, oz = phi.bv.origin_m
    dx = GRID_SPACING_M

    velocity = np.zeros((nx, ny, nz), dtype=np.float64)
    counts   = np.zeros((nx, ny, nz), dtype=np.int32)

    for vi in range(len(vertices)):
        x, y, z = vertices[vi]
        ci = int(round((x - ox) / dx))
        cj = int(round((y - oy) / dx))
        ck = int(round((z - oz) / dx))
        if 0 <= ci < nx and 0 <= cj < ny and 0 <= ck < nz:
            velocity[ci, cj, ck] += float(vertex_sensitivity[vi])
            counts[ci, cj, ck] += 1

    # Average where multiple vertices hit the same cell
    nonzero = counts > 0
    velocity[nonzero] /= counts[nonzero]

    return velocity


CFL_NUMBER: float = 0.3


def cfl_limited_dt(
    velocity: np.ndarray,
    dt_requested: float,
    cfl: float = CFL_NUMBER,
) -> float:
    """Clamp dt so the zero level set moves at most `cfl` cells per step.

    The Hamilton-Jacobi update is `phi <- phi - dt*V*|grad phi|`. With phi a
    signed distance (|grad phi| ~ 1), the surface displacement per step is
    dt*|V| metres, i.e. dt*|V|/dx CELLS. Upwind HJ schemes are only stable while
    that stays below ~1, and the narrow band around the surface is only a few
    cells wide, so overshooting does not merely lose accuracy -- it teleports
    the surface out of the band and the field stops being a distance function.

    Both callers were far outside that limit and neither had ever been run to
    the point of noticing (fixed 2026-07-20):

      proxy path   dt=1e-4, |V| ~ dT_dmass*rho ~ 1.5e3  ->  ~15 cells/step at
                   2 mm spacing. The car was carved away to nothing in the
                   first few steps and mass/COM then failed outright.
      adjoint path dt=0.5 (optimizer_contract.hj_dt) with combine_gradients
                   RMS-normalising |V| to ~1  ->  ~1667 cells/step at the
                   0.3 mm spec spacing. This one had never executed at all
                   because the phi update crashed upstream on a dict/Trimesh
                   mismatch, so its dt was never exercised.

    A fixed dt cannot be correct for both: the stable value depends on the grid
    spacing and on the velocity magnitude, and the latter changes every
    iteration. Deriving it here is the only way it stays right.

    Returns 0.0 for a velocity field that is everywhere ~zero (nothing to do).
    """
    vmax = float(np.max(np.abs(velocity))) if velocity.size else 0.0
    if not np.isfinite(vmax) or vmax < 1e-30:
        return 0.0
    return float(min(dt_requested, cfl * GRID_SPACING_M / vmax))


def _mass_report_values(mass_report) -> tuple[float, float, float]:
    """(total_mass_kg, com_x_m, com_z_m) from a MassReport-like object or dict.

    Duck-typed on purpose: Part 1 must not import Part 3's MassReport, and the
    proxy path in bayesian_outer_search has only a plain dict.
    """
    if mass_report is None:
        raise ValueError("mass_report is required to build mass/COM gradients")
    if isinstance(mass_report, dict):
        return (
            float(mass_report["total_mass_kg"]),
            float(mass_report["com_x_m"]),
            float(mass_report["com_z_m"]),
        )
    return (
        float(mass_report.total_mass_kg),
        float(mass_report.com_x_m),
        float(mass_report.com_z_m),
    )


def scalar_objective_velocity(
    phi: PhiGrid,
    density_kgm3: float,
    objective_gradients: dict,
    mass_report,
) -> np.ndarray:
    """Descent velocity field from the objective's SCALAR gradients.

    This is the term that was missing. The race objective supplies dT/dmass,
    dT/dh_com and dT/dx_com as scalars, but the level-set update needs a
    velocity defined over the grid. The bridge is the shape derivative: moving
    the surface outward by delta at a point adds delta*dA of material there, so

        dm/dS      = rho                        [kg/m^3]
        dh_com/dS  = rho * (z - h_com) / M      [1/m^2 * m = 1/m]
        dx_com/dS  = rho * (x - x_com) / M

    Chain rule gives dT/dS, and the DESCENT direction is its negative.

    Sign check, for the mass term: dT_dmass > 0 (a heavier car is slower), so
    the velocity is -dT_dmass*rho < 0, the surface moves inward, mass falls.
    For the COM-height term above the COM (z > h_com) the velocity is likewise
    negative, carving material off the top and lowering the COM. Both behave
    the way the physics demands.

    Neither term existed before: combine_gradients was fed
    `np.zeros_like(velocity_volume)` for mass, com and mfg, so w_mass/w_com/
    w_mfg were multiplying zeros and the update was aero-only no matter how
    those weights were calibrated.
    """
    nx, ny, nz = phi.bv.shape
    ox, _oy, oz = phi.bv.origin_m
    dx = GRID_SPACING_M

    M, x_com, h_com = _mass_report_values(mass_report)
    if M <= 0:
        raise ValueError(f"total_mass_kg must be positive, got {M}")

    dT_dmass = float(objective_gradients.get("dT_dmass", 0.0))
    dT_dh_com = float(objective_gradients.get("dT_dh_com", 0.0))
    dT_dx_com = float(objective_gradients.get("dT_dx_com", 0.0))

    xs = (ox + np.arange(nx) * dx)[:, None, None]
    zs = (oz + np.arange(nz) * dx)[None, None, :]

    dT_dS = (
        dT_dmass * density_kgm3
        + dT_dh_com * density_kgm3 * (zs - h_com) / M
        + dT_dx_com * density_kgm3 * (xs - x_com) / M
    )
    return -np.broadcast_to(dT_dS, (nx, ny, nz)).astype(np.float64)


def apply_adjoint_sensitivity_symmetric(
    phi_grids: dict[str, PhiGrid],
    right_half_sensitivity: np.ndarray,
    right_half_mesh,  # trimesh.Trimesh — surface mesh from the right-half CFD run
    dt: float,
    gradient_weights: dict[str, float],
    objective_gradients: dict | None = None,
    mass_report=None,
) -> None:
    """
    Apply adjoint surface sensitivity to all phi grids via mesh-to-grid splatting.

    CFD is confirmed to run on the RIGHT-HALF mesh only (symmetry plane at y=0).
    The adjoint sensitivity array has shape (n_surface_vertices,) — one scalar
    per vertex of the right-half mesh.

    For symmetric components (nose, rearpod, main_body):
        The right-half mesh covers x=[0,W], y=[0,y_max], z=[0,z_max].
        We splat the sensitivity onto the full grid. Cells on the left half (y<0)
        receive the mirrored sensitivity from their y>0 counterpart.

    For sidepod (right half only):
        The sensitivity applies directly to the right sidepod grid.

    Splatting: each mesh vertex maps to the nearest grid cell. Cells with no
    nearby vertices get zero velocity (no update there this iteration).
    The velocity field is then extended into the volume via extend_velocity()
    before the HJ update, ensuring a well-defined velocity everywhere.

    If right_half_mesh or right_half_sensitivity is None (e.g. during testing),
    the function logs a warning and returns without updating — it does NOT
    silently skip as the previous implementation did.
    """
    if right_half_mesh is None or right_half_sensitivity is None:
        raise ValueError(
            "apply_adjoint_sensitivity_symmetric: right_half_mesh and right_half_sensitivity "
            "must both be provided. Passing None silently skips the φ update, which causes "
            "ΔT=0 and false convergence in the optimizer (audit finding P1-13)."
        )

    vertices = np.asarray(right_half_mesh.vertices, dtype=np.float64)
    sensitivity = np.asarray(right_half_sensitivity, dtype=np.float64)

    if len(sensitivity) != len(vertices):
        raise ValueError(
            f"right_half_sensitivity has {len(sensitivity)} values but "
            f"right_half_mesh has {len(vertices)} vertices. They must match."
        )

    _REQUIRED_WEIGHT_KEYS = frozenset({"w_aero", "w_mass", "w_com", "w_mfg"})
    _unknown = set(gradient_weights) - _REQUIRED_WEIGHT_KEYS
    if _unknown:
        raise KeyError(
            f"gradient_weights has unknown keys {sorted(_unknown)}. "
            f"Required keys: {sorted(_REQUIRED_WEIGHT_KEYS)}. "
            "Using .get() with defaults silently discards calibrated weights — "
            "see audit finding K-3."
        )
    _missing = _REQUIRED_WEIGHT_KEYS - set(gradient_weights)
    if _missing:
        raise KeyError(
            f"gradient_weights is missing required keys {sorted(_missing)}."
        )
    w_aero = gradient_weights["w_aero"]
    w_mass = gradient_weights["w_mass"]
    w_com  = gradient_weights["w_com"]
    w_mfg  = gradient_weights["w_mfg"]

    for name, phi in phi_grids.items():
        nx, ny, nz = phi.bv.shape

        # Splat right-half sensitivity onto grid
        vel_right = _splat_vertex_sensitivity_to_grid(sensitivity, vertices, phi)

        if name == "sidepod":
            # Sidepod grid covers right half only — use vel_right directly
            surface_vel = vel_right
        else:
            # Symmetric component: mirror sensitivity to left half (y < 0)
            # Left half vertices have y_left = -y_right, same sensitivity magnitude
            # Build mirrored vertices
            verts_left = vertices.copy()
            verts_left[:, 1] *= -1.0
            vel_left = _splat_vertex_sensitivity_to_grid(sensitivity, verts_left, phi)
            # Combine: average of right and left contributions
            surface_vel = (vel_right + vel_left) * 0.5

        # Extend surface velocity into volume
        velocity_volume = extend_velocity(phi.grid.astype(np.float64), surface_vel)

        # Mass / COM terms from the objective's scalar gradients. These used to
        # be hard zeros here with a comment calling them "Part 3's
        # responsibility"; Part 3 does compute them and did pass them, but its
        # binding dropped them before the call, so neither side delivered and
        # w_mass/w_com silently multiplied nothing. See scalar_objective_velocity.
        if objective_gradients is not None and mass_report is not None:
            scalar_vel = scalar_objective_velocity(
                phi, get_density(name), objective_gradients, mass_report
            )
            mass_grad = np.full_like(
                velocity_volume,
                -float(objective_gradients.get("dT_dmass", 0.0)) * get_density(name),
            )
            com_grad = scalar_vel - mass_grad
        else:
            # Aero-only. This is the pre-2026-07-20 behaviour, kept so callers
            # that genuinely have no objective gradients (unit tests) still run,
            # but it is NOT what the pipeline should do -- w_mass and w_com are
            # inert in this branch.
            mass_grad = np.zeros_like(velocity_volume)
            com_grad = np.zeros_like(velocity_volume)

        combined = combine_gradients(
            aero_gradient=velocity_volume,
            mass_gradient=mass_grad,
            com_gradient=com_grad,
            mfg_gradient=np.zeros_like(velocity_volume),
            w_aero=w_aero,
            w_mass=w_mass,
            w_com=w_com,
            w_mfg=w_mfg,
        )

        # dt arrives from the caller (Part 3's config.hj_dt) but is clamped to
        # the CFL limit for THIS iteration's velocity magnitude -- see
        # cfl_limited_dt for why a fixed dt cannot be safe.
        hj_update(phi, combined, cfl_limited_dt(combined, dt))


# K-2: SPEC.txt §22 names the φ-update entry point `update_phi`. Part 1 uses
# `apply_adjoint_sensitivity_symmetric`. Expose both names so Part 3's
# `from phi_updater import update_phi` succeeds without renaming the function.
update_phi = apply_adjoint_sensitivity_symmetric


def apply_adjoint_to_unified(
    geom,
    right_half_sensitivity: np.ndarray,
    right_half_mesh,
    dt: float,
    gradient_weights: dict,
    objective_gradients: dict,
    mass_report,
) -> None:
    """Evolve the SINGLE unified field with the CFD adjoint + real objective.

    The four-grid `apply_adjoint_sensitivity_symmetric` splats onto four
    separate grids. This does the same on the one labelled field:

      * the adjoint surface sensitivity (dObjective/dSurface on the right-half
        mesh, from OpenFOAM's adjoint solve) is splatted onto the field and
        mirrored to the left half -> the AERO/drag velocity;
      * the objective's scalar mass/COM gradients (from the locked JAX race
        objective) become a volume velocity via scalar_objective_velocity using
        the per-cell density field;
      * the two are unit-RMS combined (combine_gradients) and applied with a
        CFL-limited step, then symmetry is re-imposed.

    Mutates geom in place. The single connected field means the drag gradient
    can move material across what used to be component boundaries.

    THE BALANCE BETWEEN DRAG AND MASS IS THE PHYSICS', not a tuned constant.
    Both contributions are dT/dSurface as a per-unit-area density in s/m^3, so
    they are summed directly and the objective's own gradients decide how much
    each matters.

    This changed on 2026-07-28. Previously combine_gradients normalised each
    field to unit RMS before weighting, which discarded the magnitudes the JAX
    objective had just computed and let `w_aero`/`w_mass` set the balance
    instead -- so the shape update followed whatever ratio those constants
    happened to hold, and the old docstring here said as much. Two things had to
    be fixed first: the adjoint was returning a force COEFFICIENT derivative
    (see openfoam_adjoint's Aref = 2/UInf^2 note), and it had to be confirmed
    from the v2412 source that sensitivitySurfacePoints divides by point area,
    making it a per-area density comparable to the mass term.

    w_aero / w_mass remain as pure multipliers defaulting to 1.0, for ABLATION
    (--aero-only sets w_mass=0). They no longer set magnitudes, and
    gradient_combiner.calibrate_gradient_weights is not needed on this path.
    """
    from unified_phi import (density_field, enforce_machinability,
                             enforce_symmetry)

    phi = geom.phi
    verts = np.asarray(right_half_mesh.vertices, dtype=np.float64)
    sens = np.asarray(right_half_sensitivity, dtype=np.float64)
    if len(sens) != len(verts):
        raise ValueError(
            f"sensitivity has {len(sens)} values but mesh has {len(verts)} "
            f"vertices; they must be index-aligned."
        )

    w_aero = gradient_weights.get("w_aero", 1.0)
    w_mass = gradient_weights.get("w_mass", 1.0)
    # w_com / w_mfg are accepted by the signature but NOT used: there is no
    # separate COM velocity field to weight. scalar_objective_velocity already
    # folds dT_dh_com and dT_dx_com into the single mass/COM volume velocity
    # below, so the COM contribution rides on w_mass. Callers that set
    # GradientWeights(w_com=1.0) expecting an independent COM channel are
    # getting nothing from it — say so rather than let a dead knob look live.
    if gradient_weights.get("w_com") or gradient_weights.get("w_mfg"):
        warnings.warn(
            "apply_adjoint_to_unified: w_com/w_mfg are inert on the unified "
            "path — the COM gradient is folded into the mass/COM velocity and "
            "scales with w_mass; there is no manufacturing gradient field. "
            "Set them to 0.0, or tune w_mass, until a separate COM channel exists.",
            RuntimeWarning, stacklevel=2,
        )

    # DESCENT SIGN. Derived from the two conventions, NOT measured:
    #   * `sens` is dT/dSurface, positive where pushing the surface OUTWARD
    #     increases race time; minimising T means moving along -dT/dSurface;
    #   * hj_update is `phi <- phi - dt*F*|grad phi|` with phi < 0 solid, so
    #     F > 0 lowers phi, grows the solid, and moves the surface OUTWARD.
    # Hence F = -sens.
    #
    # DIRECTION VERIFIED 2026-07-27, by the only test that can settle it: an
    # AERO-ONLY run (w_mass=0, so the shape update is this gradient and nothing
    # else), two iterations against real OpenFOAM.
    #
    #     D20   0.737271 -> 0.697207 N   (-5.43%)
    #     mass  158.598  -> 158.598 g    (unchanged, confirming w_mass=0)
    #     T_raw 3.217887 -> 3.198395 s
    #
    # Drag fell with mass held exactly constant, so the drop is attributable to
    # this gradient alone and the descent direction is right. That also settles
    # which way OpenFOAM's pointSensNormal points, which no amount of reasoning
    # about conventions could. Reproduce with
    # `run_two_stage.py --smoke --aero-only`.
    #
    # ⚠ The DIRECTION is verified; the MAGNITUDE is not. Drag on this coarse
    # mesh is not reproducible to better than several percent: the same nominal
    # starting geometry gave D20 = 0.6847 N before the symmetry-plane snap fix
    # and 0.7373 N after -- a 7.7% swing from a sub-micron geometry change,
    # i.e. remeshing noise. The -5.43% above is therefore smaller than the
    # run-to-run spread and must NOT be read as a measured improvement. It
    # establishes sign, not size. A mesh-convergence study is what would make
    # drag deltas of this size meaningful.
    #
    # An earlier version of this comment claimed a 2026-07-27 A/B run proved
    # the sign was inverted (D20_half 0.342365 -> 0.374725 N over two
    # iterations). That inference was wrong twice over: the aero gradient was
    # inert at the time (the adjoint was diverging -- see the outlier guard
    # below), so the drag rise came from the mass gradient shrinking the body;
    # and w_mass was live, so nothing in that run was attributable to the
    # adjoint at all. Negating the whole gradient moved the geometry by 0.369
    # mm^3 of a 932 mm^3 step -- 0.04%. A sign that moves nothing cannot be
    # validated by what moves.
    sens = -sens

    # OUTLIER GUARD. combine_gradients normalises each gradient to unit RMS, so
    # a heavy-tailed sensitivity does not merely add noise -- it deletes the
    # signal. Measured on the 2026-07-27 smoke run, with the adjoint mesh
    # movement chain still enabled: the raw field spanned -1.13e+57..4.74e+55
    # and its TOP TEN POINTS carried 99.33% of the sum of squares. Those ten
    # absorbed the entire norm, the real per-point signal was rescaled to ~0.7%
    # of it, and the p99.9 clip further down then trimmed the spikes as well.
    # Net effect: the aero term contributed 0.04% of the shape update and the
    # optimiser silently ran on the mass gradient alone for every iteration.
    #
    # The root cause is fixed upstream (openfoam_adjoint.py sets
    # includeMeshMovement false). This is the detector that should have caught
    # it, kept because "the aero term quietly stopped steering" is exactly the
    # failure mode that survives a green test suite -- every unit test drives
    # this function with a synthetic, well-conditioned sensitivity.
    finite = np.isfinite(sens)
    if not finite.all():
        raise ValueError(
            f"adjoint sensitivity has {int((~finite).sum())} non-finite values; "
            "the adjoint solve did not produce a usable gradient."
        )
    sq = sens ** 2
    total_sq = float(sq.sum())
    if total_sq > 0:
        top = np.sort(sq)[::-1][:10]
        concentration = float(top.sum()) / total_sq
        if concentration > _SENS_CONCENTRATION_LIMIT:
            warnings.warn(
                f"adjoint sensitivity is pathologically spiky: the top 10 of "
                f"{len(sens):,} values carry {100.0 * concentration:.2f}% of the "
                f"sum of squares (limit {100.0 * _SENS_CONCENTRATION_LIMIT:.0f}%). "
                f"range [{sens.min():.3e}, {sens.max():.3e}]. Unit-RMS "
                f"normalisation will crush the real signal and the aero term "
                f"will not steer the shape. Clipping to the p99.9 magnitude so "
                f"the run stays interpretable, but the adjoint setup is wrong — "
                f"check includeMeshMovement and the adjoint residuals.",
                RuntimeWarning, stacklevel=2,
            )
        # Clip BEFORE the RMS normalisation, which is the only place it helps.
        # The existing clip in this function acts on the already-normalised sum,
        # by which point the outliers have already set the scale.
        nz = np.abs(sens)[np.abs(sens) > 0]
        if nz.size:
            cap = float(np.percentile(nz, 99.9))
            if cap > 0:
                sens = np.clip(sens, -cap, cap)

    # REDISTANCE FIRST. Without this the Stage-2 adjoint loop does not move the
    # car at all.
    #
    # hj_update steps `phi - dt*V*|grad phi|`, and cfl_limited_dt sizes dt on the
    # stated assumption that "phi is a signed distance ... so the surface
    # displacement per step is dt*|V| metres". build_unified_geometry does not
    # produce a distance function -- _init_field's "full" mode writes a CONSTANT
    # -GRID_SPACING_M everywhere, with the comment "reinitialise_sdf will
    # redistance it on the first update", and nothing ever did. Measured on a
    # freshly built car, |grad phi| in the interface band has MEDIAN 0.000 with
    # 87.5% of band cells below 0.1, so the update was multiplying the velocity
    # by ~zero.
    #
    # The live 2026-07-29 sweep is what that looks like from outside: ten
    # candidates over five d_halo values, twenty CFD+adjoint solves, and total
    # mass identical to 0.75 mg across every one of them -- 0.149424865 kg on
    # every iteration 1 and 0.149424115 on every iteration 2, the same to nine
    # decimals from d_halo 16 through 45.72. D20 wandered 3-7% and race time got
    # worse in 5 of 5 pairs, which is remeshing noise on a shape that never
    # changed, not a search. It also explains the aero share alternating 3% / 20%
    # by iteration index in all six pairs: iteration 1 splatted onto an as-built
    # field with no gradient.
    #
    # A/B over four steps at 1 mm, mass/COM term only:
    #     without reinit  -0.008, -3.619, +0.000, -0.127 g   |grad phi| 0.000
    #     with reinit     -8.915, -8.669, -3.119, -3.659 g   |grad phi| 1.000
    # 1100x the first step, and monotone instead of stalling.
    #
    # At the START of the step, not the end, and not at build time:
    #   * the splat and extend_velocity below both READ the field, and
    #     extend_velocity's upwind direction is sign(phi)*grad_phi/|grad_phi|,
    #     which is meaningless on a constant -- so the first update of every
    #     candidate needs the field fixed before they run, not after;
    #   * redistancing in build_unified_geometry instead was tried and reverted:
    #     it turns the as-built staircase isosurface into a smooth one that
    #     decimates to 1.2 deg minimum angle, under the 10 deg snappyHexMesh gate
    #     (test_remap_is_not_the_same_as_rebuilding catches it).
    #
    # Stage 1's evolution loop has always redistanced (bayesian_outer_search, in
    # both variants), which is why the no-CFD path reaches the 48 g floor and
    # this one never left 149 g. Cost is ~50 Godunov pseudo-steps against a
    # 15-minute CFD solve on the same iteration, so no cadence.
    reinitialise_sdf(phi)

    # Aero velocity: splat right-half sensitivity + its y-mirror onto the field.
    vel_r = _splat_vertex_sensitivity_to_grid(sens, verts, phi)
    verts_l = verts.copy()
    verts_l[:, 1] *= -1.0
    vel_l = _splat_vertex_sensitivity_to_grid(sens, verts_l, phi)
    surface_vel = (vel_r + vel_l) * 0.5
    aero_v = extend_velocity(phi.grid.astype(np.float64), surface_vel)

    # Mass/COM velocity from the real objective's scalar gradients.
    rho = density_field(geom)
    masscom_v = scalar_objective_velocity(phi, rho, objective_gradients, mass_report)

    # PHYSICAL SUM. Both fields are dT/dSurface as a per-unit-area density in
    # s/m^3, so they add directly and the objective's own gradients set the
    # balance between drag and mass. No normalisation, no tuned constants.
    #
    #   aero:  dT/dD20 (s/N) x dD20/dSurface (N/m^3)   <- adjoint, now a FORCE
    #                                                     derivative, see
    #                                                     openfoam_adjoint's
    #                                                     Aref = 2/UInf^2 note
    #   mass:  dT/dmass (s/kg) x rho (kg/m^3)  + the COM terms
    #                                                  <- scalar_objective_velocity
    #
    # This replaces combine_gradients, which normalised EACH field to unit RMS
    # before weighting. That threw away the magnitudes JAX had just computed and
    # substituted hand-set weights, so the shape update's drag/mass balance was
    # whatever w_aero:w_mass happened to be rather than what the physics says.
    # The module docstring in gradient_combiner admits the same thing from the
    # other side: unit-RMS AMPLIFIES a weak placeholder term to parity with a
    # real one, which is exactly wrong for the fabricated com_x penalty.
    #
    # Two prerequisites had to be true before this was safe, and both were
    # checked in the v2412 source rather than assumed:
    #   * sensitivitySurfacePoints divides by accumulated point area
    #     (includeSurfaceArea defaults false), so the adjoint field is a
    #     per-AREA density like the mass field, not a per-point lumped value;
    #   * the objective is now the drag force rather than a coefficient, so
    #     dT/dD20 in s/N is the right multiplier.
    #
    # w_aero / w_mass survive as pure multipliers, defaulting to 1.0. They are
    # for ABLATION now (--aero-only sets w_mass=0), not for setting magnitudes.
    combined = w_aero * aero_v + w_mass * masscom_v

    # Report the balance the physics actually chose. It used to be invisible:
    # normalisation guaranteed the two terms arrived at parity whatever their
    # real sizes, which is how a completely inert aero channel went unnoticed
    # for the entire project.
    # Measured AT THE INTERFACE, not over the whole grid.
    #
    # A whole-grid RMS flatters the mass term for the wrong reason: it has
    # support in every body cell, while the aero term lives in a band around the
    # surface, so the comparison would partly measure support rather than
    # strength. Only the velocity where phi ~ 0 moves the level set, so that is
    # the band worth comparing. Measured 2026-07-28, the two differ enough to
    # matter: whole-grid gave aero 19.7% / mass 80.3%.
    # Reported TWICE: over the band, and over the cells where the adjoint
    # actually has support. One number cannot answer both questions -- the band
    # figure says what the update does, the support figure says whether the
    # adjoint is healthy where it acts.
    #
    # The two used to differ wildly (19.7% vs 2.9% on one step) for a reason
    # that turned out to be the redistancing bug, not the diagnostic: on the
    # un-redistanced field phi was two-valued at exactly +/-GRID_SPACING_M, so
    # EVERY cell satisfied |phi| < 2*dx and the "band" was the whole grid --
    # 7,990,840 cells at 0.5 mm. The aero term was being divided by the entire
    # volume. With redistancing the band is a real band (830,754 cells on the
    # same geometry) and the two figures sit close together, 5.0% and 7.4%.
    #
    # Which also means any aero/mass share quoted from a run before 2026-07-30
    # was measured on a degenerate field and is not comparable to these.
    #   band  -> what the update actually does, aero diluted by empty cells
    #   where aero acts -> whether the adjoint is healthy where it has support
    _band = np.abs(phi.grid) < (2.0 * GRID_SPACING_M)
    if _band.any():
        def _share(mask, label):
            a = float(np.sqrt(np.mean((w_aero * aero_v)[mask] ** 2)))
            m = float(np.sqrt(np.mean((w_mass * masscom_v)[mask] ** 2)))
            if a + m <= 0:
                return
            print(f"[phi_updater] gradient balance {label}: "
                  f"aero {100.0 * a / (a + m):5.1f}%  "
                  f"mass/COM {100.0 * m / (a + m):5.1f}%   "
                  f"(rms {a:.3e} vs {m:.3e} s/m^3, {int(mask.sum()):,} cells)")

        _share(_band, "over the interface band")
        _support = _band & (aero_v != 0.0)
        if _support.any():
            _share(_support, "where the adjoint has support")
        else:
            warnings.warn(
                "the aero sensitivity is identically zero everywhere in the "
                "interface band: the adjoint contributes NOTHING to this shape "
                "update. Check the adjoint solve converged and that the "
                "sensitivity mapped onto the right mesh vertices.",
                RuntimeWarning, stacklevel=2)
    # Splatting the surface sensitivity and extending it leaves a few localised
    # SPIKES (max >> rms). The CFL limiter, correctly, throttles the timestep to
    # the fastest-moving cell -- so a handful of artifact spikes would freeze the
    # whole surface (measured: bulk moving 0.017 cells/step while a spike moves
    # 0.3). Clip to a high percentile so the smooth descent that carries the
    # real mass/drag/COM signal actually advances.
    absc = np.abs(combined)
    nz = absc[absc > 0]
    if nz.size:
        cap = float(np.percentile(nz, 99.9))   # trim only the wildest artifacts
        if cap > 0:
            combined = np.clip(combined, -cap, cap)
    hj_update(phi, combined, cfl_limited_dt(combined, dt))
    enforce_symmetry(geom)

    # Void no tool can reach is not void. Stage 1 has applied this since it was
    # written; STAGE 2 NEVER HAS, so the only loop that actually runs CFD was
    # also the only one free to carve sealed cavities -- and it is the one whose
    # output gets manufactured. Measured on the live 0.5 mm run: 4,942 mm^2 of
    # surface the cutter cannot reach, 7.7% of the car, now priced at ~44 ms by
    # machinability_penalty. Better to make the state unrepresentable than to
    # charge for it.
    #
    # Every step, not Stage 1's every-10 cadence. That cadence is justified there
    # by "the projection only has to hold at the states that get measured"; in
    # Stage 2 EVERY iteration is measured -- each one meshes, solves and writes a
    # record. Cost is four cumsums over the grid against a ~15 minute CFD solve
    # on the same iteration, the same argument reinitialise_sdf above makes.
    #
    # AFTER enforce_symmetry, not before: _clear_run ORs +y and -y, so the
    # reachable set of a y-symmetric field is itself y-symmetric and this cannot
    # break the symmetry the line above just imposed.
    enforce_machinability(geom)