"""
bounding_volumes.py --- Legal bounding regions for all phi-grid components.

Coordinate system (see geometry_contract.py):
  x=0 = nose tip, car extends in +x direction.
  Key x landmarks (all derived from outer scalars W_mm and x_front_mm):
    Ref Plane A  = x_front_m - 0.016   (16 mm ahead of front axle, T1.17)
    front axle   = x_front_m
    rear axle    = x_front_m + W_m
    Ref Plane B  = x_front_m + W_m + 0.016

Contains:
  - BoundingRegion: arbitrary 3D shape support (box / polygon / voxel modes)
  - _point_in_polygon_vectorised: ray-casting polygon test (vectorised)
  - BoundingVolumes: all four component regions for one (W, x_front, d_halo) config
  - RuleEnvelope: absolute car envelope dimensions (⚠ UNRESOLVED U6)
  - compute_bounding_volumes: main entry point
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import numpy as np

from geometry_contract import (
    validate_W, validate_x_front, validate_d_halo, mm_to_m, grid_cells,
    WHEEL_CLEARANCE_M, GRID_SPACING_M,
)
from wheel_visibility_zones import build_t79_forbidden_mask
from halo_pocket import build_halo_pocket_forbidden_mask

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
    All four bounding regions for one (W, x_front, d_halo) configuration.

    Each region defines the legal 3D space for one component's phi grid.
    The region knows its own shape (nx, ny, nz) and origin in metres.

    Coordinate system: x=0 at nose tip; see geometry_contract.py.

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

    # The outer-loop scalars that produced this configuration
    W_mm:       float
    x_front_mm: float
    d_halo_mm:  float

    @property
    def sidepod_length_m(self) -> float:
        return self.sidepod_x_max_m - self.sidepod_x_min_m

    @property
    def W_m(self) -> float:
        return mm_to_m(self.W_mm)

    @property
    def x_front_m(self) -> float:
        return mm_to_m(self.x_front_mm)

    @property
    def ref_plane_A_m(self) -> float:
        """Ref Plane A = 16 mm ahead of front axle (T1.17)."""
        return self.x_front_m - 0.016

    @property
    def ref_plane_B_m(self) -> float:
        """Ref Plane B = 16 mm behind rear axle (T1.17)."""
        return self.x_front_m + self.W_m + 0.016

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
    x_front_mm: float,
    d_halo_mm: float,
    front_cylinder: "ForbiddenCylinder",
    rear_cylinder: "ForbiddenCylinder",
    rule_envelope: "RuleEnvelope",
    wheel_x_half_width_mm: float = 8.0,
) -> BoundingVolumes:
    """
    Compute all four bounding volumes given outer loop scalars and wheel geometry.

    Coordinate system: x=0 at nose tip (see geometry_contract.py).
    The caller is responsible for creating ForbiddenCylinders at the correct
    x positions in this coordinate system (front at x_front_m, rear at x_front_m+W_m).

    Args:
        W_mm:        wheelbase in mm. Must be in [120, 140].
        x_front_mm:  front axle position from nose tip in mm. W-dependent bounds.
        d_halo_mm:   halo offset from Ref Plane A in mm. Must be in [0, W+16].
        front_cylinder: from S6, defines front wheel forbidden zone
        rear_cylinder:  from S6, defines rear wheel forbidden zone
        rule_envelope:  RuleEnvelope containing absolute car dimensions.
                        ! UNRESOLVED U6 until UAE rule dimensions are confirmed.
        wheel_x_half_width_mm: half-width of wheel+axle assembly in x. Used to
                        derive wheel leading/trailing edges for T7.9 zones.

    Returns:
        BoundingVolumes with all four regions. main_body, sidepod, and rearpod
        carry a voxel_mask that excludes the T7.9 "visibility in top/bottom
        views" keep-clear zones around each wheel (see wheel_visibility_zones.py).
    """
    validate_W(W_mm)
    validate_x_front(x_front_mm, W_mm)
    validate_d_halo(d_halo_mm, W_mm)

    # ! UNRESOLVED U6: Absolute envelope dimensions not yet provided.
    if rule_envelope is None:
        raise NotImplementedError(
            "! UNRESOLVED U6: UAE competition regulation envelope dimensions not provided. "
            "Create a RuleEnvelope object with y_body_half_m, y_sidepod_inner_m, "
            "y_sidepod_outer_m, z_floor_m, z_nose_top_m, z_sidepod_top_m, "
            "z_rearpod_top_m, z_body_top_m, rearpod_max_length_m --- all in metres."
        )

    W_m = mm_to_m(W_mm)
    x_front_m = mm_to_m(x_front_mm)
    ref_A_m = x_front_m - 0.016   # Ref Plane A: 16 mm ahead of front axle (T1.17)
    rear_axle_m = x_front_m + W_m

    # ?? Sidepod corridor (W- and x_front-DEPENDENT) ??????????????????????????????????
    # Caller places cylinders at x_front_m and x_front_m+W_m (nose-tip coords).
    sidepod_x_min = front_cylinder.x_max_m + WHEEL_CLEARANCE_M
    sidepod_x_max = rear_cylinder.x_min_m  - WHEEL_CLEARANCE_M
    sidepod_length = sidepod_x_max - sidepod_x_min

    if sidepod_length <= 0.0:
        raise ValueError(
            f"Sidepod corridor has zero or negative length at W={W_mm} mm, "
            f"x_front={x_front_mm} mm: "
            f"x_min={sidepod_x_min:.5f} m, x_max={sidepod_x_max:.5f} m, "
            f"length={sidepod_length*1000:.2f} mm. "
            f"Check wheel clearance ({WHEEL_CLEARANCE_M*1000:.1f} mm) and "
            f"wheel half-width settings."
        )

    re = rule_envelope   # shorthand

    # ?? Nose volume ????????????????????????????????????????????????????????
    # x: from 0 (nose tip) to Ref Plane A (x_front - 16 mm)
    # y: from -y_body_half to +y_body_half
    # z: from z_floor to z_nose_top
    nose_length_mm = max(0.3, (x_front_mm - 16.0))   # Ref Plane A from nose tip
    nose_nx = grid_cells(nose_length_mm)              # at least 1
    nose_ny = grid_cells((re.y_body_half_m * 2) * 1000)
    nose_nz = grid_cells((re.z_nose_top_m - re.z_floor_m) * 1000)
    nose = BoundingRegion(
        component    = "nose",
        origin_m     = (0.0, -re.y_body_half_m, re.z_floor_m),
        shape        = (nose_nx, nose_ny, nose_nz),
        polygon_yz_m = None,
        voxel_mask   = None,
    )

    # ?? Sidepod volume (right half only, y >= 0) ???????????????????????????
    # x: from sidepod_x_min to sidepod_x_max (both W- and x_front-dependent)
    # y: from y_sidepod_inner to y_sidepod_outer (right side only, y > 0)
    # z: from z_floor to z_sidepod_top
    sp_x_length_mm = (sidepod_x_max - sidepod_x_min) * 1000
    sp_y_length_mm = (re.y_sidepod_outer_m - re.y_sidepod_inner_m) * 1000
    sp_z_length_mm = (re.z_sidepod_top_m - re.z_floor_m) * 1000
    sidepod_origin_m = (sidepod_x_min, re.y_sidepod_inner_m, re.z_floor_m)
    sidepod_shape = (
        grid_cells(sp_x_length_mm),
        grid_cells(sp_y_length_mm),
        grid_cells(sp_z_length_mm),
    )
    # T7.9.2/T7.9.3: wedge-shaped keep-clear zones at the sidepod's front and
    # rear inner corners so each wheel stays visible from top/bottom views.
    sidepod_t79_forbidden = build_t79_forbidden_mask(
        "sidepod", sidepod_origin_m, sidepod_shape, W_mm, x_front_mm, wheel_x_half_width_mm,
    )
    sidepod = BoundingRegion(
        component    = "sidepod",
        origin_m     = sidepod_origin_m,
        shape        = sidepod_shape,
        polygon_yz_m = None,
        voxel_mask   = ~sidepod_t79_forbidden,
    )

    # ?? Rearpod volume ?????????????????????????????????????????????????????
    # x: from rear axle (x_front + W) to rear axle + rearpod_max_length
    # y: from -y_body_half to +y_body_half
    # z: from z_floor to z_rearpod_top
    rp_x_mm = re.rearpod_max_length_m * 1000
    rp_y_mm = re.y_body_half_m * 2 * 1000
    rp_z_mm = (re.z_rearpod_top_m - re.z_floor_m) * 1000
    rearpod_origin_m = (rear_axle_m, -re.y_body_half_m, re.z_floor_m)
    rearpod_shape = (
        grid_cells(rp_x_mm),
        grid_cells(rp_y_mm),
        grid_cells(rp_z_mm),
    )
    # T7.9.4: keep-clear rectangle immediately aft of the rear wheel.
    rearpod_t79_forbidden = build_t79_forbidden_mask(
        "rearpod", rearpod_origin_m, rearpod_shape, W_mm, x_front_mm, wheel_x_half_width_mm,
    )
    rearpod = BoundingRegion(
        component    = "rearpod",
        origin_m     = rearpod_origin_m,
        shape        = rearpod_shape,
        polygon_yz_m = None,
        voxel_mask   = ~rearpod_t79_forbidden,
    )

    # ?? Main body volume ???????????????????????????????????????????????????
    # x: from Ref Plane A (x_front - 16 mm) to rear axle + rearpod_max
    # Intentionally overlaps with rearpod region so the main body grid covers
    # the full machined section. Hard constraints handle attachment interfaces.
    body_x_mm = (W_m + re.rearpod_max_length_m) * 1000
    body_y_mm = re.y_body_half_m * 2 * 1000
    body_z_mm = (re.z_body_top_m - re.z_floor_m) * 1000
    main_body_origin_m = (ref_A_m, -re.y_body_half_m, re.z_floor_m)
    main_body_shape = (
        grid_cells(body_x_mm),
        grid_cells(body_y_mm),
        grid_cells(body_z_mm),
    )
    # T7.9.1: keep-clear rectangle immediately forward of the front wheel.
    main_body_t79_forbidden = build_t79_forbidden_mask(
        "main_body", main_body_origin_m, main_body_shape, W_mm, x_front_mm, wheel_x_half_width_mm,
    )
    # T4.4.4: halo mounting pocket, positioned by d_halo aft of Ref Plane A.
    main_body_halo_forbidden = build_halo_pocket_forbidden_mask(
        main_body_origin_m, main_body_shape, ref_A_m, d_halo_mm,
    )
    main_body = BoundingRegion(
        component    = "main_body",
        origin_m     = main_body_origin_m,
        shape        = main_body_shape,
        polygon_yz_m = None,
        voxel_mask   = ~(main_body_t79_forbidden | main_body_halo_forbidden),
    )

    return BoundingVolumes(
        nose            = nose,
        sidepod         = sidepod,
        rearpod         = rearpod,
        main_body       = main_body,
        sidepod_x_min_m = sidepod_x_min,
        sidepod_x_max_m = sidepod_x_max,
        W_mm            = W_mm,
        x_front_mm      = x_front_mm,
        d_halo_mm       = d_halo_mm,
    )


@dataclass
class RuleEnvelope:
    """
    Absolute car envelope dimensions. All values in metres.

    U6 partially resolved -- see default_rule_envelope() below for the
    confirmed/derived values and which fields are real regulation numbers
    versus design choices made within a legal range (no exact value given).
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


def default_rule_envelope() -> "RuleEnvelope":
    """
    Confirmed/derived RuleEnvelope for the UAE 2025-26 Professional Class regs.

    Real regulation numbers (exact, not a range the team chooses within):
      z_floor_m            = 0.0015 m  -- T3.7 track clearance, absolute min 1.5mm.
                              (Earlier placeholder value of 0.0 was a real bug:
                              it let the optimizer place material AT track level,
                              which is illegal.)
      z_nose_top_m          = 0.025 m  -- T8.5.1 nose/wing support max height 25mm.
      rearpod_max_length_m  = 0.040 m  -- T9.4.2 rear overhang max 40mm from Ref Plane B.
      z_rearpod_top_m       = 0.065 m  -- T9.4.3 rear overhang max height 65mm
                              (equals T3.5's overall car height ceiling; no
                              tighter rearpod-specific bound exists).
      z_sidepod_top_m       = 0.065 m  -- no sidepod-specific height rule found;
                              defaults to T3.5's overall 65mm ceiling so the
                              optimizer isn't denied shape freedom the regs
                              don't actually restrict.
      z_body_top_m          = 0.065 m  -- same reasoning, T3.5 ceiling.

    Design choices within a legal range (regs give a range, not an exact
    number -- these are starting points, override if you have a specific
    target):
      y_sidepod_outer_m = 0.0325 m  -- confirmed design target: build to
                          T3.4's legal MINIMUM half-width (32.5mm) rather
                          than its max (42.5mm), for reduced frontal area.
      y_body_half_m     = 0.028 m   -- NOT itself a regulation number. Set to
                          the minimum that still comfortably contains the
                          T4.2 virtual cargo's wide end (55mm width = 27.5mm
                          half-width) with a small margin.
      y_sidepod_inner_m = 0.028 m   -- matches y_body_half_m (sidepod attaches
                          directly to the body's outer wall, no gap).

    KNOWN TENSION worth flagging explicitly: y_sidepod_outer_m (32.5mm) minus
    y_sidepod_inner_m (28mm) leaves only a 4.5mm-wide sidepod corridor. This
    is a direct consequence of choosing the legal-minimum overall car width
    (32.5mm half-width) while the virtual cargo requirement alone needs
    27.5mm of that just for its own half-width. If a wider sidepod is wanted,
    y_sidepod_outer_m must move toward T3.4's legal max (42.5mm half-width)
    instead of its min.
    """
    return RuleEnvelope(
        y_body_half_m=0.028,
        y_sidepod_inner_m=0.028,
        y_sidepod_outer_m=0.0325,
        z_floor_m=0.0015,
        z_nose_top_m=0.025,
        z_sidepod_top_m=0.065,
        z_rearpod_top_m=0.065,
        z_body_top_m=0.065,
        rearpod_max_length_m=0.040,
    )