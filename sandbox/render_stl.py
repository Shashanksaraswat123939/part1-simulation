"""
render_stl.py -- render an STL to a PNG without needing a GUI or OpenGL.

trimesh's built-in viewer wants pyglet + a display; this uses matplotlib's
3D triangle plotting instead, so it works headless.

Usage:
    python sandbox/render_stl.py path/to/car.stl
    python sandbox/render_stl.py path/to/car.stl --out preview.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("stl")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh

    mesh = trimesh.load(args.stl)
    verts_mm = mesh.vertices * 1000.0
    tris = verts_mm[mesh.faces]

    fig = plt.figure(figsize=(15, 10))
    views = [("iso", 22, -60), ("side (x-z)", 0, -90),
             ("top (x-y)", 89, -90), ("front (y-z)", 0, 0)]

    for idx, (title, elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        # Shade by face normal against a fixed light so form is readable.
        normals = mesh.face_normals
        shade = 0.35 + 0.65 * np.clip(normals @ np.array([0.3, 0.4, 0.86]), 0, 1)
        colours = np.stack([shade * 0.45, shade * 0.62, shade * 0.85, np.ones_like(shade)], -1)

        coll = Poly3DCollection(tris, facecolors=colours, edgecolors="none")
        ax.add_collection3d(coll)

        # Equal aspect: matplotlib 3D distorts badly otherwise.
        lo, hi = verts_mm.min(0), verts_mm.max(0)
        centre, span = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(centre[0] - span, centre[0] + span)
        ax.set_ylim(centre[1] - span, centre[1] + span)
        ax.set_zlim(centre[2] - span, centre[2] + span)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_zlabel("z [mm]")
        ax.set_title(title)

    extents = mesh.extents * 1000
    fig.suptitle(f"{Path(args.stl).name}   "
                 f"L={extents[0]:.1f} W={extents[1]:.1f} H={extents[2]:.1f} mm   "
                 f"vol={mesh.volume*1e6:.2f} cm3   watertight={mesh.is_watertight}")
    fig.tight_layout()

    out = args.out or str(Path(args.stl).with_suffix(".png"))
    fig.savefig(out, dpi=105)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
