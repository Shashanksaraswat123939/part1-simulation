"""Checks for the 2026-09-25 upgrade (ballast, block, T5.5 wall, wheels, step)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sandbox"))


def test_ballast_capsule_and_regimes():
    import ballast as b
    import halo_pocket as hp
    assert (b.SLOT_WIDTH_MM, b.SLOT_LENGTH_MM, b.SLOT_DEPTH_MM) == (
        hp.BALLAST_SLOT_WIDTH_MM, hp.BALLAST_SLOT_LENGTH_MM, hp.BALLAST_DEPTH_MM)
    v = b.capsule_volume_m3() * 1e6
    assert 1.38 < v < 1.40, v                       # cm3
    assert 15.5 < b.capacity_kg("lead") * 1e3 < 16.0
    cart = b.CARTRIDGE_KG
    heavy = 0.052 + cart
    assert b.ballast_kg(heavy) == 0.0
    assert b.shape_dT_dmass(17.0, heavy) == 17.0              # physics applies
    mid = 0.040 + cart
    assert abs(b.ballast_kg(mid) * 1e3 - 8.2) < 1e-6
    assert b.shape_dT_dmass(17.0, mid) == 0.0                 # ballast absorbs
    light = 0.020 + cart
    assert b.shape_dT_dmass(17.0, light) < 0.0                # container full
    d = b.describe(mid)
    assert d["legal_T36"] and d["regime"] == "absorbing"


def test_stage1_proxy_with_ballast_reaches_a_legal_competition_mass():
    import ballast as bl
    import bayesian_outer_search as b
    cart = 0.023
    for start in (0.0469, 0.0440, 0.0300, 0.0600):
        m = start
        for _ in range(50000):
            m -= 1e-6 * b._proxy_objective_gradients(m + cart)["dT_dmass"]
        comp = m + bl.ballast_kg(m + cart)
        assert comp >= b.PROXY_MIN_MASS_KG - 1e-6, (start, m, comp)


def test_block_envelope():
    from bounding_volumes import default_rule_envelope
    from geometry_contract import MODEL_BLOCK_HEIGHT_MM, MODEL_BLOCK_WIDTH_MM
    re = default_rule_envelope()
    assert 2 * re.y_sidepod_outer_m * 1e3 <= MODEL_BLOCK_WIDTH_MM + 1e-9
    assert (re.z_body_top_m - re.z_floor_m) * 1e3 <= MODEL_BLOCK_HEIGHT_MM + 1e-9


def test_t55_wall_is_at_least_3mm_all_round():
    from coarse import use_spacing
    use_spacing(1.0)
    try:
        from scipy.ndimage import distance_transform_edt
        import unified_phi as up
        g = up.build_unified_geometry(120.3, 46.0, 43.72)
        cyl = g.fixed_hardware.canister_cylinder
        o = np.asarray(g.region.origin_m)
        nx, ny, nz = g.shape
        xs = o[0] + np.arange(nx) * 1e-3
        ys = o[1] + np.arange(ny) * 1e-3
        zs = o[2] + np.arange(nz) * 1e-3
        solid = g.phi.hard_mask_solid
        x0 = cyl.x_center_m - cyl.x_half_width_m
        ix = np.where((xs > x0 + 3e-3) & (xs < x0 + 45e-3))[0]
        r = np.sqrt((ys[:, None] - cyl.y_center_m) ** 2 + (zs[None, :] - cyl.z_center_m) ** 2)
        ring = (r > cyl.radius_m + 0.5e-3) & (r < cyl.radius_m + 2.5e-3)
        for i in ix[::5]:
            assert solid[i][ring].all(), f"T5.5 wall missing at x={xs[i]*1e3:.1f} mm"
    finally:
        use_spacing(0.3)


def test_wheels_are_placed_in_car_coordinates():
    import hardware_geometry as hg
    from geometry_contract import FRONT_WHEEL_INNER_Y_MM, R_WHEEL_M
    d = hg.build_wheel_assembly("front", 46.0)
    b = d["front_wheel_right"].bounds * 1e3
    assert abs(b[0, 1] - FRONT_WHEEL_INNER_Y_MM) < 0.05
    assert abs(b[0, 2]) < 0.05 and abs((b[0, 0] + b[1, 0]) / 2 - 46.0) < 0.05
    assert abs(b[1, 2] - 2 * R_WHEEL_M * 1e3) < 0.3


def test_vectorised_splat_matches_a_loop():
    from coarse import use_spacing
    use_spacing(2.0)
    try:
        import phi_updater as pu
        from bounding_volumes import BoundingRegion
        from phi_grid import PhiGrid
        reg = BoundingRegion("car", (0.0, -0.01, 0.0), (10, 11, 12))
        z = np.zeros(reg.shape, np.float32)
        phi = PhiGrid("car", reg, z, np.zeros(reg.shape, bool), np.zeros(reg.shape, bool))
        rng = np.random.default_rng(1)
        v = rng.uniform([0, -0.01, 0], [0.02, 0.012, 0.024], (500, 3))
        s = rng.normal(size=500)
        got = pu._splat_vertex_sensitivity_to_grid(s, v, phi)
        ref = np.zeros(reg.shape); cnt = np.zeros(reg.shape)
        for (x, y, zz), val in zip(v, s):
            i, j, k = (int(round((x - 0) / 2e-3)), int(round((y + 0.01) / 2e-3)),
                       int(round(zz / 2e-3)))
            if 0 <= i < 10 and 0 <= j < 11 and 0 <= k < 12:
                ref[i, j, k] += val; cnt[i, j, k] += 1
        ref[cnt > 0] /= cnt[cnt > 0]
        assert np.allclose(got, ref)
    finally:
        use_spacing(0.3)

if __name__ == "__main__":
    _mod = sys.modules[__name__]
    _fails = 0
    for _n in sorted(n for n in dir(_mod) if n.startswith("test_")):
        try:
            getattr(_mod, _n)(); print("PASS", _n)
        except Exception as e:  # noqa: BLE001
            _fails += 1; print("FAIL", _n, "->", repr(e))
    print(f"{_fails} failed")
    sys.exit(1 if _fails else 0)
