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

    After update: call phi.apply_hard_constraints().
    """
    grad_mag = _grad_magnitude(phi.grid)
    phi.grid = (phi.grid - dt * velocity * grad_mag).astype(np.float32)
    phi.apply_hard_constraints()


# ------------------------------------------------------------------ #
#  SDF reinitialisation
# ------------------------------------------------------------------ #

def reinitialise_sdf(phi: PhiGrid, n_steps: int = 20, dt_reinit: float = 0.3) -> None:
    """
    Reinitialise phi as a signed distance field.
    Solve dphi/dtau + sign(phi)(|grad phi| - 1) = 0 for n_steps.
    After reinit: |grad phi| should be ~1.0 everywhere.
    """
    for _ in range(n_steps):
        grad_mag = _grad_magnitude(phi.grid)
        sign_phi = np.sign(phi.grid)
        phi.grid = (phi.grid - dt_reinit * sign_phi * (grad_mag - 1.0)).astype(np.float32)
        phi.apply_hard_constraints()


# ------------------------------------------------------------------ #
#  Velocity extension
# ------------------------------------------------------------------ #

def extend_velocity(
    phi_grid: np.ndarray,
    surface_velocity: np.ndarray,
    n_steps: int = 10,
    dt_ext: float = 0.1,
) -> np.ndarray:
    """
    Propagate surface velocity into the volume.
    Solve: dF/dtau + sign(phi) * grad phi . grad F = 0
    Returns F: velocity field defined everywhere in the volume.
    """
    F = surface_velocity.copy().astype(np.float64)
    for _ in range(n_steps):
        sign_phi = np.sign(phi_grid)
        gx = _godunov_gradient(F, 0) * np.sign(phi_grid)
        gy = _godunov_gradient(F, 1) * np.sign(phi_grid)
        gz = _godunov_gradient(F, 2) * np.sign(phi_grid)
        # Use upwind direction based on sign(phi)
        # Simple first-order extension
        F = F - dt_ext * sign_phi * (gx + gy + gz) / 3.0
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

def apply_adjoint_sensitivity_symmetric(
    phi_grids: dict[str, PhiGrid],
    right_half_sensitivity: np.ndarray,
    right_half_mesh,  # trimesh.Trimesh
    dt: float,
    gradient_weights: dict[str, float],
) -> None:
    """
    Apply adjoint sensitivity to all phi grids.

    For symmetric components (nose, rearpod, main_body):
        Sensitivity applies to the full grid (both y sides).

    For sidepod (right half only):
        Sensitivity applies to right sidepod grid directly.
    """
    # PLACEHOLDER: Surface sensitivity -> volume velocity mapping requires
    # nearest-cell or trilinear splatting from mesh vertices to grid cells.
    # For now, use the sensitivity array directly if it matches grid shape,
    # otherwise skip (no-op).
    for name, phi in phi_grids.items():
        if right_half_sensitivity.shape == phi.grid.shape:
            velocity = right_half_sensitivity.astype(np.float64)
        else:
            # PLACEHOLDER: proper mesh-to-grid interpolation not yet implemented.
            # Skip this component if shapes don't match.
            continue
        hj_update(phi, velocity, dt)