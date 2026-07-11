"""
phi_grid.py --- Level-set phi grid for each machined component.

Each component (nose, sidepod, rearpod, main_body) has its own PhiGrid.
The phi field is a signed distance function: phi < 0 inside solid, phi > 0 outside.
Hard constraints enforce attachment faces (phi < 0) and void regions (phi > 0).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from geometry_contract import (
    GRID_SPACING_M, GRID_SPACING_MM, ATTACHMENT_STRIP_MM, mm_to_m,
)
from bounding_volumes import BoundingRegion


def _attachment_strip_cells() -> int:
    """Number of grid cells for an attachment strip (ceil(ATTACHMENT_STRIP_MM / GRID_SPACING_MM))."""
    import math
    return max(1, math.ceil(ATTACHMENT_STRIP_MM / GRID_SPACING_MM))


@dataclass
class PhiGrid:
    """Level-set phi grid for one component."""

    component: str                  # "nose"|"sidepod"|"rearpod"|"main_body"
    bv: BoundingRegion              # from S2 --- defines shape and valid region
    grid: np.ndarray                # float32, shape (nx, ny, nz)
    hard_mask_solid: np.ndarray     # bool, shape (nx,ny,nz) --- forced phi < 0
    hard_mask_air: np.ndarray       # bool, shape (nx,ny,nz) --- forced phi > 0
    # hard_mask_air includes: bbox walls + forbidden zones + hardware voids + invalid region

    def __post_init__(self) -> None:
        if self.grid.shape != self.bv.shape:
            raise ValueError(
                f"{self.component}: grid shape {self.grid.shape} != "
                f"bv shape {self.bv.shape}"
            )
        if self.hard_mask_solid.shape != self.bv.shape:
            raise ValueError(
                f"{self.component}: hard_mask_solid shape {self.hard_mask_solid.shape} != "
                f"bv shape {self.bv.shape}"
            )
        if self.hard_mask_air.shape != self.bv.shape:
            raise ValueError(
                f"{self.component}: hard_mask_air shape {self.hard_mask_air.shape} != "
                f"bv shape {self.bv.shape}"
            )
        if self.grid.dtype != np.float32:
            self.grid = self.grid.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def init(self, mode: str = "sphere", seed: int = 42) -> None:
        """
        Initialize phi field.

        mode="sphere":
            Signed distance to sphere inscribed in bounding volume.
            Centre at (nx/2, ny/2, nz/2) in grid index space.
            Radius = 0.7 * min(nx, ny, nz) / 2 * GRID_SPACING_M
            phi[i,j,k] = dist - radius

        mode="slab":
            phi < 0 in lower half of z, phi > 0 in upper half.

        mode="random":
            sphere field + smooth noise at 0.5 * GRID_SPACING_M amplitude.
        """
        nx, ny, nz = self.bv.shape
        dx = GRID_SPACING_M

        if mode == "sphere":
            cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
            radius = 0.7 * min(nx, ny, nz) / 2.0 * dx
            i, j, k = np.indices((nx, ny, nz))
            dist = np.sqrt((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2) * dx
            self.grid = (dist - radius).astype(np.float32)

        elif mode == "slab":
            k_half = nz // 2
            i, j, k = np.indices((nx, ny, nz))
            self.grid = np.where(k < k_half, -dx, dx).astype(np.float32)

        elif mode == "random":
            cx, cy, cz = (nx - 1) / 2.0, (ny - 1) / 2.0, (nz - 1) / 2.0
            radius = 0.7 * min(nx, ny, nz) / 2.0 * dx
            i, j, k = np.indices((nx, ny, nz))
            dist = np.sqrt((i - cx) ** 2 + (j - cy) ** 2 + (k - cz) ** 2) * dx
            sphere = dist - radius
            rng = np.random.default_rng(seed)
            noise = rng.normal(0.0, 0.5 * dx, size=(nx, ny, nz))
            self.grid = (sphere + noise).astype(np.float32)

        else:
            raise ValueError(f"Unknown init mode '{mode}'. Valid: 'sphere', 'slab', 'random'.")

        self.apply_hard_constraints()

    # ------------------------------------------------------------------ #
    #  Hard constraints
    # ------------------------------------------------------------------ #

    def apply_hard_constraints(self) -> None:
        """Enforce hard_mask_solid (phi < 0) and hard_mask_air (phi > 0).
        Idempotent: cells already satisfying the constraint are not modified."""
        # For solid: ensure phi < 0. Set to -dx if phi is currently >= 0.
        solid_violated = self.hard_mask_solid & (self.grid >= 0)
        self.grid[solid_violated] = -GRID_SPACING_M
        # For air: ensure phi > 0. Set to +dx if phi is currently <= 0.
        air_violated = self.hard_mask_air & (self.grid <= 0)
        self.grid[air_violated] = GRID_SPACING_M

    # ------------------------------------------------------------------ #
    #  Mask building
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_hard_masks(
        bv: BoundingRegion,
        void_masks: list[np.ndarray],
        attachment_faces: list[str],
        solid_masks: list[np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (hard_mask_solid, hard_mask_air).

        hard_mask_solid: attachment face strips + solid_masks (e.g. virtual cargo,
                          T4.2) forced phi < 0.
        hard_mask_air:   1-cell bbox border + void masks + invalid region cells forced phi > 0.

        solid_masks: interior regions (not just named faces) that must remain solid.
                      Same shape as bv.shape. If a solid_masks region overlaps a void
                      region, void wins (see overlap resolution below) -- this should
                      never happen in practice if callers keep solid placements clear
                      of void placements (e.g. virtual cargo dodging the halo pocket).
        """
        nx, ny, nz = bv.shape
        strip = _attachment_strip_cells()

        # --- hard_mask_solid: attachment strips + interior solid_masks ---
        solid = np.zeros((nx, ny, nz), dtype=bool)
        for sm in (solid_masks or []):
            if sm.shape != bv.shape:
                raise ValueError(
                    f"solid mask shape {sm.shape} != bv shape {bv.shape}"
                )
            solid |= sm.astype(bool)
        for face in attachment_faces:
            if face == "rear":
                solid[max(0, nx - strip):, :, :] = True
            elif face == "front":
                solid[:min(strip, nx), :, :] = True
            elif face == "inner_y":
                solid[:, :min(strip, ny), :] = True
            else:
                raise ValueError(
                    f"Unknown attachment face '{face}'. Valid: 'rear', 'front', 'inner_y'."
                )

        # --- hard_mask_air: bbox walls + voids + invalid region ---
        air = np.zeros((nx, ny, nz), dtype=bool)

        # 1-cell border (skip faces that have attachment strips)
        has_front = "front" in attachment_faces
        has_rear = "rear" in attachment_faces
        has_inner_y = "inner_y" in attachment_faces

        if not has_front:
            air[0, :, :] = True
        if not has_rear:
            air[-1, :, :] = True
        air[:, 0, :] = True
        air[:, -1, :] = True
        air[:, :, 0] = True
        air[:, :, -1] = True

        # Void masks
        for vm in void_masks:
            if vm.shape != bv.shape:
                raise ValueError(
                    f"Void mask shape {vm.shape} != bv shape {bv.shape}"
                )
            air |= vm.astype(bool)

        # Invalid region (outside bv.valid_mask())
        air |= ~bv.valid_mask()

        # --- Resolve overlaps ---
        # Air (void masks, hardware) takes priority over solid (attachment strips).
        overlap = solid & air
        n_overlap = int(overlap.sum())
        if n_overlap > 0:
            solid = solid & ~air

        return solid, air

    def _validate_masks(self) -> None:
        """Raise ValueError if any cell is in both hard_mask_solid and hard_mask_air."""
        overlap = self.hard_mask_solid & self.hard_mask_air
        n_overlap = int(overlap.sum())
        if n_overlap > 0:
            raise ValueError(
                f"{self.component}: {n_overlap} cells are in BOTH solid and air masks. "
                f"This is a constraint setup error."
            )

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #

    def save(self, candidate_id: str, out_dir: str) -> str:
        """Save grid as .npy file. Returns absolute path string."""
        path = Path(out_dir) / f"phi_{self.component}_{candidate_id}.npy"
        np.save(str(path), self.grid)
        return str(path.resolve())

    def load(self, path: str) -> None:
        """Load grid from .npy file. Re-enforce hard constraints after load."""
        loaded = np.load(path).astype(np.float32)
        if loaded.shape != self.bv.shape:
            raise ValueError(
                f"{self.component}: loaded grid shape {loaded.shape} != "
                f"expected {self.bv.shape}. Wrong file or wrong bounding volume."
            )
        self.grid = loaded
        self.apply_hard_constraints()

    # ------------------------------------------------------------------ #
    #  Remapping
    # ------------------------------------------------------------------ #

    def remap(
        self,
        new_bv: BoundingRegion,
        new_hard_masks: tuple[np.ndarray, np.ndarray],
    ) -> "PhiGrid":
        """Trilinear interpolation into a new bounding volume. Returns a NEW PhiGrid."""
        from scipy.ndimage import map_coordinates

        nx_old, ny_old, nz_old = self.bv.shape
        nx_new, ny_new, nz_new = new_bv.shape
        dx = GRID_SPACING_M

        # World coordinate of new grid cell (i,j,k): new_bv.origin + (i,j,k)*dx
        # Map to old grid index: (world - old_origin) / dx
        ox_old, oy_old, oz_old = self.bv.origin_m
        ox_new, oy_new, oz_new = new_bv.origin_m

        i_new, j_new, k_new = np.indices((nx_new, ny_new, nz_new))
        x_world = ox_new + i_new * dx
        y_world = oy_new + j_new * dx
        z_world = oz_new + k_new * dx

        # Old grid indices
        i_old = (x_world - ox_old) / dx
        j_old = (y_world - oy_old) / dx
        k_old = (z_world - oz_old) / dx

        coords = np.stack([i_old, j_old, k_old])
        interpolated = map_coordinates(self.grid, coords, order=1, mode='nearest').astype(np.float32)

        new_solid, new_air = new_hard_masks
        new_phi = PhiGrid(
            component=self.component,
            bv=new_bv,
            grid=interpolated,
            hard_mask_solid=new_solid,
            hard_mask_air=new_air,
        )
        new_phi.apply_hard_constraints()
        return new_phi