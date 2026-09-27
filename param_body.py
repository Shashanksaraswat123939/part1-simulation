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
             three stations -- Ref A, the maximum section at x_max, the rear
             face -- easing out of the first and into the last, so the loft is
             smooth with zero slope at the maximum. The body ends at Ref A;
             the nose cone ahead of it is a printed part (Part 4 nose.py).
  sidepods   a second, wide and low superellipse loft over [s_x0, s_x1] with
             `s_taper` mm ease at each end: what encloses the 52 mm T4.2 cargo.
  floor      an optional flat plate (f_b > 0) of thickness f_t at height f_z,
             from Ref A to the rear face, with the fuselage's bottom brought
             down onto it (a flat underside); the wheel zones cut it where T7.9
             requires.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
import numpy as np


@dataclass(frozen=True)
class BodyParams:
    # fuselage
    nose_len: float = 8.0         # unused: the nose is a Part 4 printed part now
    n_b: float = 10.0
    n_zt: float = 20.0
    n_zb: float = 6.0
    x_max_frac: float = 0.35
    m_b: float = 18.0
    m_zt: float = 32.0
    m_zb: float = 5.0
    r_b: float = 14.0
    r_zt: float = 48.0
    r_zb: float = 20.0
    p: float = 2.6
    # sidepods
    s_b: float = 28.0
    s_zt: float = 27.0
    s_zb: float = 12.0
    s_x0: float = 70.0
    s_x1: float = 142.0
    s_taper: float = 12.0
    s_p: float = 3.0
    # floor
    f_b: float = 0.0
    f_z: float = 3.5
    f_t: float = 2.0

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BodyParams":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown body parameters {sorted(unknown)}")
        return cls(**d)


# Search bounds (mm). Chosen from the legal envelope (build_unified_geometry:
# y <= 20 ahead of x~80, <= 32 around the cargo, z 4..48) and the mandatory
# solids (cargo y +-26, z 16-24, x 78-136; cartridge wall to z 47, y +-12).
BOUNDS = {
    "n_b": (6.0, 14.5), "n_zt": (14.0, 24.0), "n_zb": (4.0, 12.0),
    "x_max_frac": (0.2, 0.6), "m_b": (12.0, 20.0), "m_zt": (22.0, 42.0), "m_zb": (4.0, 12.0),
    "r_b": (12.5, 20.0), "r_zt": (47.0, 48.0), "r_zb": (4.0, 23.0), "p": (2.0, 4.0),
    "s_b": (26.5, 31.5), "s_zt": (24.5, 34.0), "s_zb": (4.0, 15.5),
    "s_x0": (55.0, 77.0), "s_x1": (137.0, 150.0), "s_taper": (4.0, 25.0), "s_p": (2.0, 5.0),
    "f_b": (0.0, 30.0), "f_z": (3.5, 6.0), "f_t": (1.5, 3.0),
}


def _ease(v0, v1, u, into: bool):
    """v0 -> v1 over u in [0,1]; zero slope at the v1 end (into=False: ease-out
    of v0 toward a maximum) or at the v0 end (into=True)."""
    u = np.clip(u, 0.0, 1.0)
    s = u * u if into else 1.0 - (1.0 - u) ** 2
    return v0 + (v1 - v0) * s


def _superellipse_phi(y, z, b, zc, h, p):
    """Approximate signed distance (mm) to |y/b|^p + |(z-zc)/h|^p = 1."""
    b = np.maximum(b, 1e-3)
    h = np.maximum(h, 1e-3)
    r = (np.abs(y) / b) ** p + (np.abs(z - zc) / h) ** p
    return (r ** (1.0 / p) - 1.0) * np.minimum(b, h)


def phi_mm(bp: BodyParams, X, Y, Z, x_ref_a: float, x_rear: float, must=None):
    """Signed distance field (mm, negative inside) on grids X, Y, Z (mm).

    `must`: optional per-x (half-width, z_top, z_bottom) the fuselage has to
    cover (the mandatory solids near the centreline); sections grow to it so
    no mandatory solid is left as a separate island (T4.1)."""
    L = x_rear - x_ref_a
    xm = x_ref_a + bp.x_max_frac * L
    # fuselage stations
    u1 = (X - x_ref_a) / max(xm - x_ref_a, 1e-6)
    u2 = (X - xm) / max(x_rear - xm, 1e-6)
    front = X <= xm
    b = np.where(front, _ease(bp.n_b, bp.m_b, u1, False), _ease(bp.m_b, bp.r_b, u2, True))
    zt = np.where(front, _ease(bp.n_zt, bp.m_zt, u1, False), _ease(bp.m_zt, bp.r_zt, u2, True))
    zb = np.where(front, _ease(bp.n_zb, bp.m_zb, u1, False), _ease(bp.m_zb, bp.r_zb, u2, True))
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
    phi = np.minimum(phi_f, phi_s)
    # floor plate
    if bp.f_b > 0:
        pf = np.maximum.reduce([np.abs(Y) - bp.f_b, bp.f_z - Z, Z - (bp.f_z + bp.f_t),
                                x_ref_a - X, X - x_rear])
        phi = np.minimum(phi, pf)
    return phi


def build(W_mm: float, x_front_mm: float, d_halo_mm: float, bp: BodyParams,
          target_body_kg: float | None = None, machinable: bool = True):
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
                  base.landmarks["rear_face_m"] * 1e3, _must_cover(base, ys, zs))

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
        g.spacing_m = d
        g.build_report = dict(rep, skin_offset_mm=offset_mm)
        return g

    def mass(g):
        return sum(c.mass_kg for c in up.compute_mass_com(g))

    if target_body_kg is None:
        return make(0.0)
    # Secant on the skin offset; mass is monotone in it. 0.05 g is well inside
    # the voxel-count noise of the mass itself (~0.3 g at 1 mm).
    a, b = -2.0, 4.0
    ga, gb = make(a), make(b)
    fa, fb = mass(ga) - target_body_kg, mass(gb) - target_body_kg
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
    g = best[1]
    g.build_report["mass_error_g"] = (mass(g) - target_body_kg) * 1e3
    return g


def _must_cover(geom, ys, zs, y_max_mm: float = 15.0, pad_mm: float = 1.5):
    """Per-x envelope (b, z_top, z_bot) of the mandatory solids within
    |y| <= y_max (the cartridge wall and halo deck; the 26 mm cargo is the
    sidepods' job), padded and smoothed along x so the loft stays smooth."""
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    hs = geom.phi.hard_mask_solid & (np.abs(ys)[None, :, None] <= y_max_mm)
    nx = hs.shape[0]
    b = np.zeros(nx)
    t = np.full(nx, -1e3)
    z0 = np.full(nx, 1e3)
    for i in np.nonzero(hs.any(axis=(1, 2)))[0]:
        jj, kk = np.nonzero(hs[i])
        b[i] = np.abs(ys[jj]).max() + pad_mm
        t[i] = zs[kk].max() + pad_mm
        z0[i] = zs[kk].min() - pad_mm
    w = 5
    return (maximum_filter1d(b, 2 * w + 1), maximum_filter1d(t, 2 * w + 1),
            minimum_filter1d(z0, 2 * w + 1))


def sample(rng, n: int) -> list:
    """n Latin-hypercube parameter sets inside BOUNDS."""
    keys = list(BOUNDS)
    u = (rng.permuted(np.tile(np.arange(n), (len(keys), 1)), axis=1).T
         + rng.random((n, len(keys)))) / n
    out = []
    for row in u:
        d = {k: BOUNDS[k][0] + q * (BOUNDS[k][1] - BOUNDS[k][0]) for k, q in zip(keys, row)}
        out.append(d)
    return out
