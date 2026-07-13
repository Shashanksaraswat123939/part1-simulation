"""
phi_updater.py --- Hamilton-Jacobi level-set evolution and adjoint sensitivity.

Applies adjoint surface sensitivity to phi grids via the Hamilton-Jacobi
update equation:  phi_new = phi_old - dt * F * |grad phi|

Uses Godunov upwind scheme for |grad phi|. Includes SDF reinitialisation
and velocity extension from surface to volume.
"""
from __future__ import annotations
import numpy as np

from geometry_contract import GRID_SPACING_M
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

    for _ in range(n_steps):
        grad_mag = _grad_magnitude(grid_f64)
        sign_phi = np.sign(grid_f64)
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


def apply_adjoint_sensitivity_symmetric(
    phi_grids: dict[str, PhiGrid],
    right_half_sensitivity: np.ndarray,
    right_half_mesh,  # trimesh.Trimesh — surface mesh from the right-half CFD run
    dt: float,
    gradient_weights: dict[str, float],
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

        # Combine with trivial mass/com/mfg gradients (zero for now — these come
        # from the analytical gradient computation which is Part 3's responsibility)
        zero = np.zeros_like(velocity_volume)
        combined = combine_gradients(
            aero_gradient=velocity_volume,
            mass_gradient=zero,
            com_gradient=zero,
            mfg_gradient=zero,
            w_aero=w_aero,
            w_mass=w_mass,
            w_com=w_com,
            w_mfg=w_mfg,
        )

        hj_update(phi, combined, dt)


# K-2: SPEC.txt §22 names the φ-update entry point `update_phi`. Part 1 uses
# `apply_adjoint_sensitivity_symmetric`. Expose both names so Part 3's
# `from phi_updater import update_phi` succeeds without renaming the function.
update_phi = apply_adjoint_sensitivity_symmetric