"""
bounding_volumes.py --- Legal bounding regions for all phi-grid components.

Contains:
  - BoundingRegion: arbitrary 3D shape support (box / polygon / voxel modes)
  - _point_in_polygon_vectorised: ray-casting polygon test (vectorised)
  - BoundingVolumes: all four component regions for one (W, d_halo) config
  - RuleEnvelope: absolute car envelope dimensions (⚠ UNRESOLVED U6)
  - compute_bounding_volumes: main entry point
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import numpy as np

from geometry_contract import (
    validate_W, validate_d_halo, mm_to_m, grid_cells,
    WHEEL_CLEARANCE_M, GRID_SPACING_M,
)

if TYPE_CHECKING:
    from fixed_hardware import ForbiddenCylinder

@dataclass
class BoundingRegion:
    """
    A 3D region definition for one phi grid component.

    Supports three modes:
      1. BOX mode: axis-aligned bounding box (simplest)
      2. POLYGON mode: convex polygon cross-section in y-z plane, extruded along x
      3. VOXEL mode: arbitrary boolean mask (most general)

    Only one mode is active per instance. The mode is determined by which
    constructor arguments are non-None.

    In all modes:
      - origin_m: (x0, y0, z0) coordinates of grid index [0,0,0] in metres
      - shape: (nx, ny, nz) number of grid cells in each axis
      - All coordinates are in the car coordinate system (x=front-to-rear, y=CL-to-right, z=up)

    Cell centre of grid index (i, j, k) is at:
      x = origin_m[0] + i * GRID_SPACING_M
      y = origin_m[1] + j * GRID_SPACING_M
      z = origin_m[2] + k * GRID_SPACING_M
    """

    component: str
    origin_m: tuple[float, float, float]   # (x0, y0, z0) of cell [0,0,0] centre
    shape: tuple[int, int, int]            # (nx, ny, nz) --- number of cells

    # BOX mode: None means use full grid extent (same as axis-aligned box)
    # If all three polygon_* and voxel_mask are None, this is BOX mode.

    # POLYGON mode: convex polygon in y-z plane, constant along x
    # polygon_yz_m: list of (y, z) vertices in metres, counterclockwise order
    # The polygon is the same cross-section at every x slice.
    polygon_yz_m: Optional[list[tuple[float, float]]] = None

    # VOXEL mode: explicit boolean mask, shape must equal self.shape
    # True = valid (inside region), False = forbidden (outside region)
    # If provided, polygon_yz_m is ignored.
    voxel_mask: Optional[np.ndarray] = None

    # Cached valid mask (computed on first access, then cached)
    _cached_valid_mask: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def nx(self) -> int: return self.shape[0]
    @property
    def ny(self) -> int: return self.shape[1]
    @property
    def nz(self) -> int: return self.shape[2]

    @property
    def mode(self) -> str:
        """Returns 'voxel', 'polygon', or 'box'."""
        if self.voxel_mask is not None:
            return "voxel"
        if self.polygon_yz_m is not None:
            return "polygon"
        return "box"

    def valid_mask(self) -> np.ndarray:
        """
        Return boolean array of shape (nx, ny, nz).
        True = this cell is inside the valid region.
        False = this cell is outside the valid region (should be treated as phi > 0).

        In BOX mode: all True (entire grid is valid).
        In POLYGON mode: True where cell centre (y, z) is inside the polygon.
        In VOXEL mode: directly returns the provided voxel_mask.

        Result is cached after first computation.
        """
        if self._cached_valid_mask is not None:
            return self._cached_valid_mask

        if self.mode == "voxel":
            if self.voxel_mask.shape != self.shape:
                raise ValueError(
                    f"{self.component}: voxel_mask shape {self.voxel_mask.shape} "
                    f"!= grid shape {self.shape}."
                )
            self._cached_valid_mask = self.voxel_mask.astype(bool)

        elif self.mode == "polygon":
            self._cached_valid_mask = self._compute_polygon_mask()

        else:  # BOX mode
            self._cached_valid_mask = np.ones(self.shape, dtype=bool)

        return self._cached_valid_mask

    def _compute_polygon_mask(self) -> np.ndarray:
        """
        Compute boolean mask for polygon cross-section.

        Algorithm: for each grid cell, compute (y, z) of its centre.
        Test whether (y, z) is inside the polygon using the ray-casting method.
        Result is the same at every x slice (polygon is extruded along x).

        The polygon_yz_m must be a convex or non-convex simple polygon.
        Vertices can be in any order --- the code handles both CW and CCW.
        """
        from geometry_contract import GRID_SPACING_M

        nx, ny, nz = self.shape
        ox, oy, oz = self.origin_m
        dx = GRID_SPACING_M

        # Build y and z coordinate arrays for all grid cells
        ys = oy + np.arange(ny) * dx    # shape (ny,)
        zs = oz + np.arange(nz) * dx    # shape (nz,)
        # Y[j, k] = y coordinate of cell (*, j, k)
        # Z[j, k] = z coordinate of cell (*, j, k)
        Y, Z = np.meshgrid(ys, zs, indexing="ij")   # shape (ny, nz)

        inside_yz = _point_in_polygon_vectorised(
            Y.ravel(), Z.ravel(), self.polygon_yz_m
        ).reshape(ny, nz)    # shape (ny, nz)

        # Broadcast to all x slices: shape (nx, ny, nz)
        mask = np.broadcast_to(inside_yz[np.newaxis, :, :], (nx, ny, nz)).copy()
        return mask

    def x_max_m(self) -> float:
        return self.origin_m[0] + self.nx * GRID_SPACING_M

    def y_max_m(self) -> float:
        return self.origin_m[1] + self.ny * GRID_SPACING_M

    def z_max_m(self) -> float:
        return self.origin_m[2] + self.nz * GRID_SPACING_M


def _point_in_polygon_vectorised(
    ys: np.ndarray,
    zs: np.ndarray,
    polygon_yz: list[tuple[float, float]],
) -> np.ndarray:
    """
    Ray-casting polygon test for N points.

    Returns boolean array of shape (N,).
    True = point is inside the polygon.

    Algorithm: cast a ray in the +y direction from each test point.
    Count how many polygon edges it crosses. Odd = inside, even = outside.
    This handles both convex and non-convex polygons correctly.

    polygon_yz: list of (y, z) vertex coordinates.
    """
    n_pts = len(ys)
    inside = np.zeros(n_pts, dtype=bool)

    verts = list(polygon_yz)
    n_verts = len(verts)

    for i in range(n_verts):
        y1, z1 = verts[i]
        y2, z2 = verts[(i + 1) % n_verts]

        # Edge crosses z-coordinate of test point (z1 to z2 straddles test z)
        cond_z = ((z1 <= zs) & (zs < z2)) | ((z2 <= zs) & (zs < z1))

        # x-intercept of edge at test point's z coordinate
        # (if z1 == z2 this is a horizontal edge; cond_z handles it correctly)
        dz = z2 - z1
        safe = np.where(np.abs(dz) > 1e-14, dz, 1.0)
        y_intercept = y1 + (zs - z1) * (y2 - y1) / safe

        # Ray in +y direction: does intercept lie to the right of (or at) test point?
        cond_y = ys < y_intercept

        inside ^= (cond_z & cond_y)

    return inside

@dataclass(frozen=True)
class BoundingVolumes:
    """
    All four bounding regions for one (W, d_halo) configuration.

    Each region defines the legal 3D space for one component's phi grid.
    The region knows its own shape (nx, ny, nz) and origin in metres.

    sidepod_x_min_m and sidepod_x_max_m are stored separately for use
    by PhiGrid's hard constraint setup and phi_updater's symmetry logic.
    """
    nose:      BoundingRegion
    sidepod:   BoundingRegion   # RIGHT half only (y >= 0 side)
    rearpod:   BoundingRegion
    main_body: BoundingRegion

    # Sidepod corridor limits --- stored for external use
    sidepod_x_min_m: float
    sidepod_x_max_m: float

    # The W and d_halo that produced this configuration
    W_mm: float
    d_halo_mm: float

    @property
    def sidepod_length_m(self) -> float:
        return self.sidepod_x_max_m - self.sidepod_x_min_m

    @property
    def W_m(self) -> float:
        return mm_to_m(self.W_mm)

    def get(self, component: str) -> BoundingRegion:
        """Look up a region by component name."""
        mapping = {
            "nose": self.nose,
            "sidepod": self.sidepod,
            "rearpod": self.rearpod,
            "main_body": self.main_body,
        }
        if component not in mapping:
            raise ValueError(f"Unknown component '{component}'. Valid: {list(mapping)}")
        return mapping[component]


def compute_bounding_volumes(
    W_mm: float,
    d_halo_mm: float,
    front_cylinder: ForbiddenCylinder,
    rear_cylinder: ForbiddenCylinder,
    rule_envelope: "RuleEnvelope",
) -> BoundingVolumes:
    """
    Compute all four bounding volumes given outer loop scalars and wheel geometry.

    Args:
        W_mm: wheelbase in mm. Must be in [120, 140].
        d_halo_mm: halo-canister distance in mm. Must be in [0, W+16].
        front_cylinder: from S6, defines front wheel forbidden zone
        rear_cylinder: from S6, defines rear wheel forbidden zone
        rule_envelope: RuleEnvelope containing absolute car dimensions.
                       ! UNRESOLVED U6 until UAE rule dimensions are confirmed.

    Returns:
        BoundingVolumes with all four regions.
    """
    validate_W(W_mm)
    validate_d_halo(d_halo_mm, W_mm)

    # ! UNRESOLVED U6: Absolute envelope dimensions not yet provided.
    # The RuleEnvelope must supply:
    #   y_body_half_m: half-width of car body in y (both sides)
    #   y_sidepod_inner_m: inner wall of sidepod in y (outer edge of body)
    #   y_sidepod_outer_m: outer edge of sidepod in y
    #   z_floor_m: bottom of all volumes (should be 0.0 or close to track surface)
    #   z_nose_top_m: maximum z of nose volume
    #   z_sidepod_top_m: maximum z of sidepod volume
    #   z_rearpod_top_m: maximum z of rearpod volume
    #   z_body_top_m: maximum z of main body volume
    #   rearpod_max_length_m: maximum x-length of rearpod from rear axle
    if rule_envelope is None:
        raise NotImplementedError(
            "! UNRESOLVED U6: UAE competition regulation envelope dimensions not provided. "
            "Create a RuleEnvelope object with y_body_half_m, y_sidepod_inner_m, "
            "y_sidepod_outer_m, z_floor_m, z_nose_top_m, z_sidepod_top_m, "
            "z_rearpod_top_m, z_body_top_m, rearpod_max_length_m --- all in metres."
        )

    W_m = mm_to_m(W_mm)
    d_halo_m = mm_to_m(d_halo_mm)

    # ?? Sidepod corridor (THE W-DEPENDENT COMPUTATION) ????????????????????
    sidepod_x_min = front_cylinder.x_max_m + WHEEL_CLEARANCE_M
    sidepod_x_max = rear_cylinder.x_min_m  - WHEEL_CLEARANCE_M
    sidepod_length = sidepod_x_max - sidepod_x_min

    if sidepod_length <= 0.0:
        raise ValueError(
            f"Sidepod corridor has zero or negative length at W={W_mm} mm: "
            f"x_min={sidepod_x_min:.5f} m, x_max={sidepod_x_max:.5f} m, "
            f"length={sidepod_length*1000:.2f} mm. "
            f"Check wheel clearance ({WHEEL_CLEARANCE_M*1000:.1f} mm) and "
            f"wheel half-width settings. The rules require a positive sidepod corridor."
        )

    re = rule_envelope   # shorthand

    # ?? Nose volume ????????????????????????????????????????????????????????
    # x: from 0 (front axle) to d_halo_m
    # y: from -y_body_half to +y_body_half (full width, symmetric)
    # z: from z_floor to z_nose_top
    nose_nx = grid_cells(d_halo_mm)   # might be 1 if d_halo=0
    nose_ny = grid_cells((re.y_body_half_m * 2) * 1000)
    nose_nz = grid_cells((re.z_nose_top_m - re.z_floor_m) * 1000)
    nose = BoundingRegion(
        component = "nose",
        origin_m  = (0.0, -re.y_body_half_m, re.z_floor_m),
        shape     = (nose_nx, nose_ny, nose_nz),
        # No polygon or voxel override --- nose uses full box
        polygon_yz_m = None,
        voxel_mask   = None,
    )

    # ?? Sidepod volume (right half only, y >= 0) ???????????????????????????
    # x: from sidepod_x_min to sidepod_x_max (W-dependent!)
    # y: from y_sidepod_inner to y_sidepod_outer (right side only, y > 0)
    # z: from z_floor to z_sidepod_top
    # The polygon_yz_m can be set to a non-rectangular cross-section if needed.
    # For now, use box mode. Once UAE envelope gives a non-rectangular profile,
    # replace with polygon_yz_m=[(y1,z1), ...].
    sp_x_length_mm = (sidepod_x_max - sidepod_x_min) * 1000
    sp_y_length_mm = (re.y_sidepod_outer_m - re.y_sidepod_inner_m) * 1000
    sp_z_length_mm = (re.z_sidepod_top_m - re.z_floor_m) * 1000
    sidepod = BoundingRegion(
        component = "sidepod",
        origin_m  = (sidepod_x_min, re.y_sidepod_inner_m, re.z_floor_m),
        shape     = (
            grid_cells(sp_x_length_mm),
            grid_cells(sp_y_length_mm),
            grid_cells(sp_z_length_mm),
        ),
        polygon_yz_m = None,   # Switch to polygon for non-rectangular cross-sections
        voxel_mask   = None,
    )

    # ?? Rearpod volume ?????????????????????????????????????????????????????
    # x: from W_m (rear axle) to W_m + rearpod_max_length_m
    # y: from -y_body_half to +y_body_half
    # z: from z_floor to z_rearpod_top
    rp_x_mm = re.rearpod_max_length_m * 1000
    rp_y_mm = re.y_body_half_m * 2 * 1000
    rp_z_mm = (re.z_rearpod_top_m - re.z_floor_m) * 1000
    rearpod = BoundingRegion(
        component = "rearpod",
        origin_m  = (W_m, -re.y_body_half_m, re.z_floor_m),
        shape     = (
            grid_cells(rp_x_mm),
            grid_cells(rp_y_mm),
            grid_cells(rp_z_mm),
        ),
        polygon_yz_m = None,
        voxel_mask   = None,
    )

    # ?? Main body volume ???????????????????????????????????????????????????
    # x: from 0 (front axle) to W_m + rearpod_max_length_m
    # y: from -y_body_half to +y_body_half
    # z: from z_floor to z_body_top
    # Note: this x range overlaps with nose and rearpod.
    # That is intentional --- the main body grid covers the full car length.
    # Hard constraints in S3 handle the attachment at those interfaces.
    body_x_mm = (W_m + re.rearpod_max_length_m) * 1000
    body_y_mm = re.y_body_half_m * 2 * 1000
    body_z_mm = (re.z_body_top_m - re.z_floor_m) * 1000
    main_body = BoundingRegion(
        component = "main_body",
        origin_m  = (0.0, -re.y_body_half_m, re.z_floor_m),
        shape     = (
            grid_cells(body_x_mm),
            grid_cells(body_y_mm),
            grid_cells(body_z_mm),
        ),
        polygon_yz_m = None,
        voxel_mask   = None,
    )

    return BoundingVolumes(
        nose             = nose,
        sidepod          = sidepod,
        rearpod          = rearpod,
        main_body        = main_body,
        sidepod_x_min_m  = sidepod_x_min,
        sidepod_x_max_m  = sidepod_x_max,
        W_mm             = W_mm,
        d_halo_mm        = d_halo_mm,
    )


@dataclass
class RuleEnvelope:
    """
    ! UNRESOLVED U6: Absolute car envelope dimensions from UAE competition rules.

    All values in metres. Must be provided before bounding volumes can be computed.
    Fill these in once UAE rules are confirmed.

    Until then, any call to compute_bounding_volumes() raises NotImplementedError.
    """
    y_body_half_m:       float   # Half-width of car body in y (e.g. 0.030 m = 30 mm)
    y_sidepod_inner_m:   float   # Inner y-edge of sidepod corridor (outer edge of body)
    y_sidepod_outer_m:   float   # Outer y-edge of sidepod
    z_floor_m:           float   # Bottom of all components (usually 0.0 m = track surface)
    z_nose_top_m:        float   # Maximum z of nose volume
    z_sidepod_top_m:     float   # Maximum z of sidepod volume
    z_rearpod_top_m:     float   # Maximum z of rearpod volume
    z_body_top_m:        float   # Maximum z of main body volume
    rearpod_max_length_m: float  # Max x-extent of rearpod from rear axle