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

8. **WITHDRAWN 2026-07-20 — this finding was wrong.** It claimed
   `y_sidepod_outer_m = 0.0325` made "every car this pipeline can produce too
   narrow to be legal", by measuring the bare machined body against T3.4.
   T3.4 does not measure the body. Regs p19: *"Total width is the maximum
   **assembled car** width"*. The assembled car's widest points are the front
   wheels, whose outer faces sit at y = ±36.5 mm
   (`FRONT_WHEEL_INNER_Y_MM 19.25 + WHEEL_WIDTH_MM 17.25`), giving **73.0 mm —
   comfortably inside T3.4's 65–85 mm**. Body width is irrelevant to T3.4, and
   the 1-cell air border on the outer y wall costs nothing legally.

   `y_sidepod_outer_m` was still changed to 0.0355 m, but on the strength of
   finding 13 (the corridor was unmachinable) alone — not on legality grounds.
   The original 0.0325 minimum-frontal-area choice was not illegal.

   Lesson worth keeping: check whether a dimensional reg measures the *machined
   body* or the *assembled car* before treating a body measurement as a
   violation. T3.5 (height) has the same "assembled car" wording.

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

13. ~~**The sidepod corridor is 2.5 mm wide**~~ **FIXED 2026-07-20.** It was
   4.5 mm at the time of measurement (`y_sidepod_inner_m=0.028` vs
   `y_sidepod_outer_m=0.0325`) and collapsed to a near-empty grid (0.1% solid,
   0.013 g). It was also narrower than one 3.15 mm-radius tool (6.3 mm
   diameter), so no cutter could enter it. `y_sidepod_outer_m` is now 0.0355,
   giving a 7.5 mm corridor and a 3.8 g sidepod. `y_sidepod_inner_m` is pinned
   at 28 mm by the T4.2 cargo, so the corridor can only widen from outside.

## Findings from the 2026-07-20 pass

14. ~~**Every car violated T8.2's nose overhang limit.**~~ **FIXED.**
    `X_FRONT_MIN_MM` was 61.0, derived from "the nose must fit the CO2 cartridge
    depth (T5.3: 45 mm) forward of Ref Plane A". That premise is false — the
    cartridge chamber is *rear* of Ref Plane A (regs p22, and
    `compute_default_fixed_hardware_inputs` had always placed it at the rear, so
    the two disagreed). The consequence was not cosmetic: the nose spans
    `[0, x_front-16]`, so a 61 mm floor forced a **>= 45 mm nose overhang against
    T8.2's 40 mm maximum** (regs p35). The old range `[61, min(90, 207-W)]` does
    not intersect the legal range `[36, 56]` **at any point** — every candidate
    the search could propose was illegal, by 19 mm at the commonly-used
    `x_front=75`. It also forced the long thin front spike in the renders.
    Bounds are now `[36, 56]`; Part 3's duplicate copy in `optimizer_contract.py`
    was updated in lockstep (it had silently drifted-by-copy).

15. ~~**The cartridge chamber was a sealed cube.**~~ **FIXED.** The canister void
    was a box of half-size `diameter/2 + safety_zone` = 12.125 mm in *all three
    axes*, centred on the canister COM. Wrong three ways at once:
    24.25 mm deep against T5.3's 45 mm minimum; 24.25 mm across against T5.1's
    18.0–18.5 mm; and it ended ~54 mm short of the rear face, leaving solid Model
    Block behind it — **the cartridge could not be inserted and T5.6 was
    unsatisfiable**. It is now a proper cylindrical bore (18.25 mm dia, 50 mm
    deep, axis at z=35 mm) anchored to the car's rearmost machined face so it
    breaks through. Note the bore spans the main_body/rearpod boundary, so
    `FixedHardwareResult` now carries `canister_cylinder` as geometry and
    `phi_grid_factory` rasterises it onto *both* grids — the main-body-shaped
    masks alone could not carve it.

    T5.5's 3 mm safety zone is no longer added to the void (enlarging a hole
    cannot guarantee a wall). **It is now unchecked anywhere** — it needs a real
    check on the finished surface in `surface_extraction`'s rule stage.

17. ~~**The optimisation loop had never run.**~~ **FIXED 2026-07-20.** Both
    paths were dead, in different ways, and both looked converged from outside.

    *Proxy path*: `_level2_evaluate`'s evolution loop called
    `hj_update(pg, np.zeros_like(pg.grid), dt)` -- a literal zeros velocity.
    `hj_update` computes `phi - dt*V*|grad phi|`, so phi was mathematically
    unchanged; only `reinitialise_sdf`'s redistancing every 10th step moved
    anything. With the default `level2_iters=0` it did not even do that.
    **Every shape this project has produced was an initialisation field with
    hard constraints applied.**

    *Real path*: `inner_loop` passed `gate.meshes` (a `dict[str, Trimesh]`) as
    the adjoint's `right_half_mesh`. A dict has no `.vertices`, so the update
    raised `AttributeError` on the first iteration, the loop's broad `except`
    caught it, and the candidate was downgraded to `objective_failed` with phi
    untouched. Now fixed by `AdjointOutcome`, which makes the sensitivity and
    the mesh that indexes it travel together.

    The proxy objective's gradients are analytic, so the loop now optimises
    with no CFD at all: T 1.3932 -> 1.0084 monotonically, mass 101.7 g ->
    51.7 g, converged by ~120 iterations.

18. ~~**Mass and COM gradients were never delivered by either side.**~~
    **FIXED.** `phi_updater` fed `combine_gradients` `np.zeros_like(...)` for
    the mass, COM and manufacturing channels, with a comment calling them
    "Part 3's responsibility". Part 3 *did* compute them and *did* pass them --
    and `pipeline_interface.update_phi` accepted `objective_gradients` and
    `mass_report` as parameters and then dropped them on the floor. Each side
    assumed the other owned it, so `w_mass` and `w_com` multiplied zeros no
    matter how carefully they were calibrated.

    `phi_updater.scalar_objective_velocity` now derives the field from the
    shape derivative: `dm/dS = rho`, `dh_com/dS = rho*(z - h_com)/M`,
    `dx_com/dS = rho*(x - x_com)/M`.

19. ~~**Both HJ timesteps violated CFL by orders of magnitude.**~~ **FIXED.**
    A fixed dt cannot be right: stability depends on grid spacing and on a
    velocity magnitude that changes every iteration.
    - proxy: dt=1e-4 with |V| ~ dT_dmass*rho ~ 1.5e3 -> **~15 cells/step**
    - adjoint: `optimizer_contract.hj_dt = 0.5` with `combine_gradients`
      RMS-normalising |V| to ~1 -> **~1667 cells/step** at 0.3 mm spacing
    `phi_updater.cfl_limited_dt` now derives the step so the surface moves at
    most 0.3 cells. The adjoint value had never been exercised because the
    update crashed upstream (finding 17).

20. ~~**`_level2_evaluate` built its grids with no attachment faces.**~~
    **FIXED.** It called `PhiGrid.build_hard_masks(region, void_masks, [],
    solid_masks)` -- attachment faces hard-coded to `[]` -- while
    `phi_grid_factory` passed `_ATTACHMENT_FACES` for the same four components.
    Invisible while phi was frozen; the moment a real update started moving the
    field, mass descent carved nose and sidepod to **zero solid cells** and
    Part 2's `ingest_mass_com` raised "component has non-positive mass".
    Also switched this path's init from `sphere` to `slab` (finding 7) and gave
    the rearpod the cartridge bore that `phi_grid_factory` already had.

    Related, separate bug in `PhiGrid.build_hard_masks`: `air[:, 0, :] = True`
    was set unconditionally, but the sidepod's `inner_y` attachment strip lives
    at j=0, and overlap resolution then deleted it. The strip is
    `ceil(1.0mm / spacing)` cells, so at the 0.3 mm spec spacing 3 of 4 cells
    survived (a 25% thinning nobody would notice) while at any spacing
    >= 1.0 mm the attachment **vanished entirely**.

21. ~~**The GP could never learn where it was not allowed to go.**~~
    **IMPROVED, not solved.** `bayesian_outer_search.py` trains only on
    `race_time < 1e5`, so rejections are dropped entirely, the GP has no data
    in the infeasible region, reports high uncertainty there, and EI is drawn
    to exactly the points that cannot be evaluated. Feeding rejections in at
    their 1e6 sentinel is not the fix -- that destroys the fitted length scales.

    Now uses constrained EI (Gardner et al. 2014): a second GP over a 0/1
    feasibility indicator trained on ALL results, multiplied into the
    acquisition. The objective GP stays clean; the acquisition learns the dead
    band. Measured over four seeds, BO iterations rejected:

    | seed | plain EI | constrained EI |
    |---|---|---|
    | 0 | 13/15 | 4/15 |
    | 1 | 15/15 | 12/15 |
    | 2 | 15/15 | 6/15 |
    | 3 | 14/15 | 11/15 |

    A large, consistent improvement, but **seeds 1 and 3 still waste most of
    their budget**. A GP regressor on a 0/1 target is a crude classifier, and
    the dead band is ~37% of the space. A proper classifier (or fixing the
    cargo/halo-pocket conflict at its source, finding 1) is still wanted.

22. **Still open: assembled length exceeds the model block.** 233.2 mm against
    T3.1.2's 223 mm (down from 262.2 before the x_front fix, but still over).
    This is the same question finding 10 raises and it is *not* a bug to patch
    blindly: the block is 223 x 65 x 50 mm, yet T3.5 allows a 65 mm tall car, so
    the block plainly bounds the machined body rather than the assembled car.
    Someone has to decide which components the 223 mm applies to before the
    bounding volumes can be constrained correctly.
