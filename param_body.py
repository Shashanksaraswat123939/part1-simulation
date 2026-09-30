"""
param_body.py -- a parametric car body: lofted sections, sidepods and a floor.

The level-set carve (Stage 1) shapes the body for MASS only and the drag adjoint
cannot steer it (every drag-only step raised D20, 12 of 12, 2026-09-26), so the
body a designer would draw -- a smooth fuselage, sidepods around the cargo, a
floor -- was out of reach. This builds that body from ~20 numbers, then hands it
to the SAME rule machinery as every other body: build_unified_geometry's hard
masks (T4.2 cargo, T5.5 cartridge wall, halo pocket, ballast slot, T7.9 wheel
zones, the model-block envelope) are applied on top, so whatever the numbers
say, the result is as legal as a Stage-1 carve. A CFD search (part5
body_search.py) picks the numbers.

Shape, all in mm, car coordinates (x nose->tail, z up from the track):

  fuselage   superellipse sections |y/b|^p + |(z-zc)/h|^p <= 1 lofted through
             eight stations from Ref A to the rear face (half-width, top and
             bottom at each, monotone-cubic between them). The body ends at
             Ref A; the nose cone ahead of it is a printed part (Part 4).
  sidepods   a second, wide and low superellipse loft over [s_x0, s_x1] with
             `s_taper` mm ease at each end: what encloses the 52 mm T4.2 cargo.
  modes      32 smooth bumps (+) or dents (-) on the fuselage, mm.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import numpy as np


@dataclass(frozen=True)
class BodyParams:
    # fuselage: half-width, top and bottom at STATIONS_U (the start car, from
    # the 2026-09-27 three-station loft n 10/20/6, max 18/32/5 at 0.35, rear 14/48/20)
    st_b: tuple = (10.0, 13.239184, 16.530612, 18.0, 17.786982, 17.147929, 15.908571, 14.0)
    st_zt: tuple = (20.0, 24.858776, 29.795918, 32.0, 32.852071, 35.408284, 40.365444, 48.0)
    st_zb: tuple = (6.0, 5.595102, 5.183673, 5.0, 5.798817, 8.195266, 12.842604, 20.0)
    p: float = 2.6                # section squareness: 2 ellipse, larger boxier
    # sidepods
    s_b: float = 28.0
    s_zt: float = 27.0
    s_zb: float = 12.0
    s_x0: float = 70.0
    s_x1: float = 142.0
    s_taper: float = 12.0
    s_p: float = 3.0
    # smooth blend radius where fuselage and sidepods meet (mm)
    blend_mm: float = 3.0
    # sculpt modes: strengths (mm, + grows) of smooth bumps at MODE_U x
    # MODE_ANG_DEG around the section, mirrored left/right.
    modes: tuple = (0.0,) * 32

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BodyParams":
        d = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items()
             if k not in _RETIRED}          # older search files still carry these
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown body parameters {sorted(unknown)}")
        return cls(**d)


# parameters of the retired three-station loft, nose length and floor plate
# (2026-09-30): ignored when read, so older search records still load
_RETIRED = frozenset(("nose_len", "n_b", "n_zt", "n_zb", "x_max_frac", "m_b", "m_zt", "m_zb",
                      "r_b", "r_zt", "r_zb", "f_b", "f_z", "f_t"))

STATIONS_U = (0.0, 0.08, 0.2, 0.35, 0.5, 0.65, 0.82, 1.0)
MODE_U = (0.06, 0.18, 0.3, 0.42, 0.54, 0.66, 0.78, 0.9)
MODE_ANG_DEG = (15.0, 60.0, 120.0, 165.0)      # from the top, down the side
N_MODES = len(MODE_U) * len(MODE_ANG_DEG)      # 32
MODE_SIGMA_MM = 7.0


def _smin(a, b, k):
    """Polynomial smooth minimum (k = blend radius, mm); k = 0 is min()."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1 - h)


# Search bounds (mm) of the scalar parameters, from the legal envelope
# (build_unified_geometry: y <= 20 ahead of x~80, <= 32 around the cargo) and
# the mandatory solids (cargo y +-26, z 16-24, x 78-136).
BOUNDS = {
    "p": (2.0, 4.0),
    "s_b": (26.5, 31.5), "s_zt": (24.5, 34.0), "s_zb": (4.0, 15.5),
    "s_x0": (55.0, 77.0), "s_x1": (137.0, 150.0), "s_taper": (4.0, 25.0), "s_p": (2.0, 5.0),
}


def _superellipse_phi(y, z, b, zc, h, p):
    """Approximate signed distance (mm) to |y/b|^p + |(z-zc)/h|^p = 1."""
    b = np.maximum(b, 1e-3)
    h = np.maximum(h, 1e-3)
    r = (np.abs(y) / b) ** p + (np.abs(z - zc) / h) ** p
    return (r ** (1.0 / p) - 1.0) * np.minimum(b, h)


def phi_mm(bp: BodyParams, X, Y, Z, x_ref_a: float, x_rear: float):
    """Signed distance field (mm, negative inside) on grids X, Y, Z (mm).

    The mandatory solids (T5.5 cartridge wall, halo deck) are NOT drawn here:
    build() adds them through the hard masks and the machining closing rounds
    them into the loft (on the start car the wall housing rises from the loft
    top 41.7 mm at x 150 to 47.8 mm at x 174, measured 2026-09-30). A loft
    that leaves one as a separate island fails T4.1 in the audit."""
    b, zt, zb = _fuselage(bp, X, x_ref_a, x_rear)
    zc, h = 0.5 * (zt + zb), 0.5 * (zt - zb)
    phi_f = _superellipse_phi(Y, Z, b, zc, h, bp.p)
    # The machined body ends at Ref A: the nose cone is a separate printed part
    # (Part 4 nose.py, team spec 2026-09-27). nose_len only rounds the body's
    # front face now; nothing is left ahead of Ref A.
    phi_f = np.maximum(phi_f, np.maximum(x_ref_a - X, X - x_rear))
    # sidepods
    ts = max(bp.s_taper, 1e-3)
    w = np.minimum(np.clip((X - bp.s_x0) / ts, 0, 1), np.clip((bp.s_x1 - X) / ts, 0, 1))
    w = np.sin(0.5 * np.pi * w)                      # smooth ends
    sb = bp.s_b * w
    sh = 0.5 * (bp.s_zt - bp.s_zb) * np.maximum(w, 0.35)
    phi_s = _superellipse_phi(Y, Z, sb, 0.5 * (bp.s_zt + bp.s_zb), sh, bp.s_p)
    phi_s = np.where(w > 0, phi_s, 1e3)
    phi = _smin(phi_f, phi_s, bp.blend_mm)
    if any(bp.modes):
        phi = phi - mode_field(bp, X, Y, Z, x_ref_a, x_rear)
    return phi


def _fuselage(bp, X, x_ref_a, x_rear):
    """(half-width, top, bottom) of the fuselage along X (arrays like X)."""
    from scipy.interpolate import PchipInterpolator
    u = np.clip((X - x_ref_a) / (x_rear - x_ref_a), 0.0, 1.0)
    b, zt, zb = (PchipInterpolator(STATIONS_U, v)(u) for v in (bp.st_b, bp.st_zt, bp.st_zb))
    return b, zt, np.minimum(zb, zt - 4.0)           # always a section, whatever the numbers


def mode_centres(bp, x_ref_a: float, x_rear: float) -> np.ndarray:
    """(N_MODES, 3) mode centres, mm, on the right half of the fuselage."""
    xs = np.array([x_ref_a + u * (x_rear - x_ref_a) for u in MODE_U])
    b, zt, zb = (np.asarray(a, float).ravel() for a in _fuselage(bp, xs, x_ref_a, x_rear))
    zc, h = 0.5 * (zt + zb), 0.5 * (zt - zb)
    out = []
    for i, x in enumerate(xs):
        for ang in MODE_ANG_DEG:
            t = np.radians(ang)
            y = b[i] * abs(np.sin(t)) ** (2 / bp.p)
            z = zc[i] + h[i] * np.sign(np.cos(t)) * abs(np.cos(t)) ** (2 / bp.p)
            out.append((x, y, z))
    return np.array(out)


def mode_field(bp, X, Y, Z, x_ref_a, x_rear):
    """sum_k a_k exp(-|P - c_k|^2 / 2 sigma^2) with |y| (mirrored). It is
    subtracted from phi, so a positive a_k grows the body ~a_k mm at c_k."""
    C = mode_centres(bp, x_ref_a, x_rear)
    Ya = np.abs(Y)
    out = np.zeros(np.shape(X), dtype=float)
    s2 = 2 * MODE_SIGMA_MM ** 2
    for a, (cx, cy, cz) in zip(bp.modes, C):
        if a:
            out += a * np.exp(-((X - cx) ** 2 + (Ya - cy) ** 2 + (Z - cz) ** 2) / s2)
    return out


def build(W_mm: float, x_front_mm: float, d_halo_mm: float, bp: BodyParams,
          target_body_kg: float | None = None, machinable: bool = True,
          skin_offset_mm: float | None = None):
    """UnifiedGeometry whose body is `bp`, with every rule mask applied.

    machinable: the team's 3-axis, 6.25 mm ball-end, top-and-bottom process
      (machining.make_machinable) is applied to the drawn shape.
    target_body_kg: no ballast, so the body itself carries the mass the T3.6
      floor needs. The drawn shape is offset by a uniform skin (grown or
      thinned, the design intent kept) until the machined body weighs this.
    """
    import unified_phi as up
    from phi_updater import reinitialise_sdf
    import machining
    base = up.build_unified_geometry(W_mm, x_front_mm, d_halo_mm, init_mode="full")
    # Read at call time: sandbox/coarse.use_spacing rewrites the module globals,
    # and an import-time copy drew the body at 0.3 mm on a 1 mm grid.
    r, d = base.region, up.GRID_SPACING_M
    xs = (r.origin_m[0] + np.arange(r.shape[0]) * d) * 1e3
    ys = (r.origin_m[1] + np.arange(r.shape[1]) * d) * 1e3
    zs = (r.origin_m[2] + np.arange(r.shape[2]) * d) * 1e3
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    phi0 = phi_mm(bp, X, Y, Z, base.landmarks["ref_plane_A_m"] * 1e3,
                  base.landmarks["rear_face_m"] * 1e3)

    def make(offset_mm):
        import copy
        g = copy.deepcopy(base)
        g.phi.grid = ((phi0 - offset_mm) / 1e3).astype(np.float32)
        g.phi.apply_hard_constraints()
        up.enforce_symmetry(g)
        reinitialise_sdf(g.phi)
        g.phi.apply_hard_constraints()
        rep = machining.make_machinable(g) if machinable else {}
        up.enforce_symmetry(g)
        rep["islands_removed_mm3"] = _drop_islands(g) * (d * 1e3) ** 3
        g.spacing_m = d
        g.build_report = dict(rep, skin_offset_mm=offset_mm)
        return g

    def mass(g):
        return sum(c.mass_kg for c in up.compute_mass_com(g))

    if skin_offset_mm is not None:           # fixed skin: gradient checks, no mass sizing
        return make(skin_offset_mm)
    if target_body_kg is None:
        return make(0.0)
    # Secant on the skin offset; mass is monotone in it. 0.05 g is well inside
    # the voxel-count noise of the mass itself (~0.3 g at 1 mm).
    a, b = -2.0, 4.0
    ga, gb = make(a), make(b)
    fa, fb = mass(ga) - target_body_kg, mass(gb) - target_body_kg
    a0, fa0, ga0, b0, fb0, gb0 = a, fa, ga, b, fb, gb
    tried = []
    best = min(((abs(fa), ga), (abs(fb), gb)), key=lambda t: t[0])
    for _ in range(6):
        if abs(fb - fa) < 1e-9:
            break
        c = float(np.clip(b - fb * (b - a) / (fb - fa), -8.0, 15.0))
        gc = make(c)
        fc = mass(gc) - target_body_kg
        if abs(fc) < best[0]:
            best = (abs(fc), gc)
        if abs(fc) < 5e-5:
            break
        a, fa, b, fb = b, fb, c, fc
        tried.append((c, fc, gc))
    # Bisection fallback: a skin that grows the body into a rule zone changes
    # the mass in steps, and the secant can stall there (0.21 g short with the
    # v2 CAD supports, 2026-09-28). Bisect between the closest evaluated skins
    # either side of the target.
    tried += [(a0, fa0, ga0), (b0, fb0, gb0)]
    for _ in range(8):
        if best[0] < 5e-5:
            break
        lo = max((t for t in tried if t[1] < 0), key=lambda t: t[1], default=None)
        hi = min((t for t in tried if t[1] > 0), key=lambda t: t[1], default=None)
        if lo is None or hi is None:
            break
        c = 0.5 * (lo[0] + hi[0])
        gc = make(c)
        fc = mass(gc) - target_body_kg
        tried.append((c, fc, gc))
        if abs(fc) < best[0]:
            best = (abs(fc), gc)
    g = best[1]
    g.build_report["mass_error_g"] = (mass(g) - target_body_kg) * 1e3
    return g


def _drop_islands(geom) -> int:
    """Remove solid pieces cut off from the main body that hold no mandatory
    solid (a swollen skin can poke through a rule zone and leave a sliver
    behind it, T4.1). Returns the number of cells removed."""
    from scipy.ndimage import label
    S = geom.phi.grid < 0
    lab, n = label(S)
    if n <= 1:
        return 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    main = sizes.argmax()
    keep = np.zeros(n + 1, bool)
    keep[main] = True
    keep[np.unique(lab[geom.phi.hard_mask_solid & S])] = True
    keep[0] = True
    drop = ~keep[lab]
    geom.phi.grid[drop] = abs(geom.phi.grid[drop]) + 1e-4
    return int(drop.sum())
