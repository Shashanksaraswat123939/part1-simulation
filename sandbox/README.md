# sandbox/ — run the geometry designer without OpenFOAM

Part 1 never needs CFD to produce geometry. The whole Level 3 path (phi grids →
marching cubes → repair → gates → STL) and the Level 1 proxy objective are pure
numpy/scipy/skimage/trimesh. OpenFOAM is only reached via
`SearchConfig.use_real_pipeline=True`, which defaults to `False`.

These three scripts exercise that path and make it visible.

## Setup

Nothing to install beyond `requirements.txt` (botorch is optional — `bo_demo.py`
falls back to a small numpy GP without it). The scripts set `PART2_PATH`
themselves, because `fixed_hardware.py` guesses `part2_simulation` while the
directory is actually named `part2-simulation`.

## Read this before running anything

The spec grid spacing is **0.3 mm**, which makes `main_body` a **22.5-million-cell**
field. Marching cubes plus trimesh curvature/ray-casting on grids that size
allocates several GB per component and will take a machine down. Every script
here defaults to a coarse spacing (1.5–2.0 mm) and `coarse.check_size()` refuses
to build grids over 3M cells total.

The tradeoff is honest and important: at 1.5 mm you cannot resolve the 3.15 mm
minimum machining radius or the 2.0 mm nose wall thickness. **Coarse runs tell
you the pipeline is wired correctly and roughly what shape comes out. They do
not tell you the geometry is legal.** Re-run a chosen candidate at `--spacing 0.3`
one component at a time to answer that.

## Scripts

### `explore.py` — inspect one candidate

```bash
python sandbox/explore.py                                    # phi plots + mass/COM, ~1s
python sandbox/explore.py --W 130 --x-front 75 --d-halo 60
python sandbox/explore.py --gates                            # + extraction + STL, ~6s
python sandbox/explore.py --init random --gates
```

Prints bounding volumes, grid shapes, hard-constraint cell counts and per-component
mass/COM; writes phi slice PNGs (plan and profile, with the phi=0 contour and the
forced-solid/forced-air regions outlined), `.npy` snapshots, and with `--gates`
per-component STLs.

It extracts each component **separately** rather than calling
`quality_gates.run_quality_gates`, which wraps all four in one try block and so
hides the status of the other three when one fails.

### `why_rejected.py` — map the feasible region

```bash
python sandbox/why_rejected.py --n 9                          # sweep
python sandbox/why_rejected.py --W 130 --x-front 75 --d-halo 60   # single point
```

`_level2_evaluate` collapses four different failure causes into the single string
`"geometry_rejected"`. This re-runs the stages individually and reports which one
raised, plus the message. Writes `feasible_region.png`.

### `bo_demo.py` — watch the outer search

```bash
python sandbox/bo_demo.py --n-seed 10 --n-bo 15
```

Drives the real `_level2_evaluate` proxy path, mirroring
`BayesianOuterSearch._evaluate_and_record`'s clipping of samples onto W-dependent
bounds. Writes `bo_search.png` (convergence + each parameter against the objective).

The proxy objective is
`T = 0.5*(mass/0.055) + 0.3*(h_com/0.025) + 0.2*(W/130)` — it rewards light,
low-COM, short-wheelbase cars and knows nothing about aerodynamics. Use it to
check that the search *moves sensibly*, not to pick a design.

## Getting an STL out

```bash
python sandbox/explore.py --gates --no-cargo --allow-inaccessible --init slab
python sandbox/render_stl.py out/<tag>/car_<tag>_full.stl     # headless PNG preview
```

Both waivers produce geometry that is **not competition-legal** — `--no-cargo`
drops the mandatory T4.2 solid region, `--allow-inaccessible` ships surfaces the
CNC tool cannot reach (it reports the unreachable area rather than hiding it).
They exist to unblock the pipeline so you can see what it makes.

Note `--no-cargo` does not affect `explore.py`'s own output: `phi_grid_factory`
never applied cargo in the first place (it is scoped out in that module's
docstring). Cargo is only enforced in `_level2_evaluate`, so the flag matters for
`bo_demo.py` / `why_rejected.py`, where it removes every rejection.

## What these runs found

1. **The feasible region is a d_halo band, and x_front is irrelevant to it.**
   37% of the space fails, always at `find_cargo_placement`: the halo pocket lands
   mid-corridor and splits the axle corridor into two segments, neither long enough
   for the 60 mm T4.2 cargo wedge. The dead band is roughly `d_halo` 30–75 mm at
   W=120, narrowing to 50–67 mm at W=140. `PLACEHOLDERS.md` item 15 predicted
   exactly this; the sweep shows how much of the space it costs.

2. **The GP never learns where the dead band is.**
   `bayesian_outer_search.py:758` trains only on `race_time < 1e5`, so rejected
   points are dropped from the training set entirely. The GP therefore sees no data
   in the infeasible region, reports high uncertainty there, and Expected
   Improvement keeps proposing into it. In a 25-evaluation demo run, **all 15 BO
   iterations were rejected**, clustered at W≈139, d_halo≈63 — inside the dead band.
   The search can stall permanently. A feasibility classifier or constrained EI is
   the fix; feeding rejects in as `1e6` would instead wreck the GP's length scale.

3. **The small-accessibility retry is a no-op.**
   `quality_gates.py:102` does `continue` on a non-large `AccessibilityFailure`, but
   the next attempt re-extracts from the *same unmodified phi grid* — unlike the
   radius path, which calls `_smooth_phi_neighbourhood` first. So it re-fails
   identically, three times, then reports `geometry_rejected`.
   `01_generative_geometry.md` specifies "smooth/fill region, retry" for small and
   "assign manufacturing penalty, continue" for large; the code hard-fails on both.
   At default settings `rearpod` and `main_body` both hit this.

4. **`PhiGrid.save()` doesn't create its output directory**, and
   `quality_gates.run_quality_gates` catches the resulting error and substitutes an
   empty string — so `GateResult.phi_snapshot_paths` silently comes back as
   `{"nose": "", ...}` whenever `out_dir` doesn't already exist.

5. **Warm-starting never happens.** `bayesian_outer_search.py:399` calls
   `PhiGrid.load(path)` as if it were a classmethod, but it's an instance method
   that mutates in place and returns `None`. The `except Exception` around it falls
   back to `init("sphere")` every time, so the warm-start path is dead code.

6. **The `sphere` init is a poor starting field.** In the plan/profile plots the
   initial sphere is cut in two by the halo pocket void, leaving two disconnected
   crescents — `main_body` starts at ~2.5 g of a ~48 g minimum car. Worth
   considering a `slab` or envelope-filling init instead.

7. **`init="sphere"` does not produce a car — it produces four disconnected blobs.**
   Each component seeds a sphere of radius `0.7*min(nx,ny,nz)/2` inscribed in its
   own bounding box. For the elongated boxes here that sphere is tiny, sits in the
   middle, and never touches the attachment faces, so nothing connects. The
   assembled STL is watertight and passes the gates while being four separate
   lumps totalling 8.5 g against a 48 g minimum. `--init slab` gives a connected
   67.9 g body with recognisable pockets and fins, and is the far better starting
   point: topology optimisation should start full and carve away.

8. **T3.4's minimum width is structurally unreachable.**
   `y_sidepod_outer_m = 0.0325` m puts the bounding wall at exactly 32.5 mm, i.e.
   65.0 mm total — precisely T3.4's *minimum*. But `build_hard_masks` forces the
   outer y cell to air, so the phi=0 surface can never reach the wall. Measured
   width came out 63.8 mm at 1.5 mm spacing and would be ~64.4 mm at 0.3 mm.
   **Every car this pipeline can produce is too narrow to be legal.**
   `y_sidepod_outer_m` has to exceed 32.5 mm (T3.4 allows up to 42.5 mm) to leave
   room for the border. `PLACEHOLDERS.md` item 1 records 32.5 mm as a deliberate
   minimum-frontal-area choice; it can't be built at exactly the limit.

9. **`slab` init leaves nose and rearpod with inverted normals** — both come back
   `watertight=False` with *negative* volume (-16.2 cm³ and -744.1 cm³), so
   `_repair_mesh`'s winding/normal fix is not holding for those two components.
   Assembly still succeeds, which means a broken-normal component can reach CFD.

10. **Assembled length exceeds the model block.** The slab car measures 243.1 mm
    long against the 223 mm block of T3.1.2. Bounding volumes allow nose from
    x=0 and rearpod out to Ref Plane B + 40 mm, which at x_front=75, W=130 is
    261 mm of allowable span. Worth confirming whether the block limit is meant
    to bound the machined body only.

11. **The STL contains no hardware at all — halo, wheels, canister and rear wing
    are voids and masses only.** `FixedHardwareResult` carries four void masks
    plus a `FixedHardwareSpec` (mass/COM); no surface is ever generated.
    `assemble_stl` concatenates exactly nose/sidepod/rearpod/main_body. Part 2
    doesn't add them either — `cfd_wrapper.py` and `mesh_validation.py` contain no
    reference to any of them, and the Part 2 interface contract asks only for a
    watertight right-half STL. So **CFD drag is being computed on a bare body with
    no wheels and no halo**, when four 30 mm wheels on a 65 mm car and a halo that
    T4.4.2/3 require to be *visible* from front/side/top are dominant drag sources.
    Mass/COM is fine — only the surfaces are missing. `hardware.py` builds them
    from the same constants the voids use; `--with-hardware` exports them.

12. **The wheel/axle exclusion cylinder is oriented along the wrong axis, and
    doesn't cover where the wheels are.** `ForbiddenCylinder`'s docstring says
    "aligned with the x-axis ... circular cross-section in y-z plane" — that is a
    rod pointing fore-aft. A wheel spins about **y** (lateral). Consequences,
    measured:
    - Fore-aft: `WHEEL_X_HALF_WIDTH_M = 0.008` carves a 16 mm slot, but a 30 mm
      diameter wheel needs ≥30 mm. Short by 14 mm. That constant treats the
      wheel's x-extent as its axial width; for a disc, x-extent *is* the diameter.
    - Lateral: the void is centred at y=0 with radius 17 mm, covering y ≤ 17 mm.
      The wheels sit at y = 19–29 mm (front) per the T7.2.1 gap. **There is no
      void anywhere near the wheels.**

    Result: 50% of the front wheel and 58% of the rear wheel lie *inside* solid
    bodywork. The body is never carved to make room for them, so no legal car can
    come out of this until the exclusion geometry is fixed.

13. **The sidepod corridor is 2.5 mm wide** (`y_sidepod_inner_m=0.030` vs
   `y_sidepod_outer_m=0.0325`), which is 4 cells at 1.5 mm and collapses to a
   near-empty grid (0.1% solid, 0.013 g). `PLACEHOLDERS.md` item 1 flags this as
   needing both values updated together.
