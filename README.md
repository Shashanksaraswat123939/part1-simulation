# Part 1 Geometry

This repository contains the first-stage geometry layer for the CFD race-car workflow. It turns outer-loop design scalars into bounded component volumes, level-set grids, mass and centre-of-mass estimates, repaired surfaces, and assembled STL outputs that feed the Part 2 simulation layer.

The implementation is intentionally defensive: invalid wheelbase values, impossible halo placement, forbidden hardware overlap, non-manufacturable regions, and broken mesh outputs are surfaced explicitly instead of being silently accepted.

## Current status

- 10 automated test files pass.
- The Part 1 geometry contracts, bounding regions, fixed-hardware handling, phi-grid generation, mass/COM calculation, phi updates, surface extraction, STL assembly, and quality gates are implemented.
- The Part 1 to Part 2 interface is checked by an integration test.
- Placeholder items still remain where real measured geometry or manufacturing decisions are missing.

This is a development-stage engineering model. Passing tests establish software behavior; they do not validate the underlying car geometry against physical measurements or manufacturing trials.

## Pipeline

1. **Geometry contract** - defines units, bounds, densities, lifecycle states, and shared constants.
2. **Bounding volumes** - creates valid spatial envelopes for each machined component from the outer-loop inputs.
3. **Fixed hardware** - places wheel, axle, canister, and halo exclusion geometry that the optimizer must respect.
4. **Phi grid** - builds the signed-distance style optimization grids and hard masks.
5. **Mass and COM calculation** - estimates component mass and full-car center of mass from the geometry state.
6. **Phi updates** - applies optimizer-side phi changes while enforcing constraints.
7. **Surface extraction** - converts phi grids into repaired meshes and evaluates extraction success.
8. **STL assembly** - packages component surfaces into full-car and right-half STL outputs for downstream CFD use.
9. **Quality gates** - rejects invalid or non-manufacturable candidates before they reach Part 2.

## Repository layout

| Path | Easy explanation | What it does technically |
| --- | --- | --- |
| `geometry_contract.py` | The rulebook for Part 1. | Defines bounds, constants, density values, unit helpers, lifecycle states, and validation rules that the rest of the geometry pipeline depends on. |
| `bounding_volumes.py` | Builds allowed design spaces for each part. | Generates component-specific spatial envelopes and geometric support data from the outer-loop parameters while respecting the project constraints. |
| `fixed_hardware.py` | Describes the car parts the optimizer is not allowed to collide with. | Builds forbidden wheel, axle, canister, and halo regions and exposes masks and geometry helpers that constrain downstream grid generation. |
| `phi_grid.py` | Creates the editable geometry grids. | Stores component phi grids, builds hard masks, initializes candidate geometry, persists `.npy` snapshots, and remaps grids between bounding volumes. |
| `mass_com_calculator.py` | Computes the weight and balance of machined parts. | Calculates component masses and centres of mass from geometry outputs and packages them into Part 2-compatible data structures. |
| `phi_updater.py` | Safely applies geometry edits. | Updates phi fields while preserving hard constraints, shape expectations, and consistency checks needed by the optimization loop. |
| `surface_extraction.py` | Turns grids into surfaces. | Extracts meshes from phi volumes, performs basic repair/cleanup, and evaluates whether the result is valid enough to continue. |
| `stl_assembler.py` | Packs final geometry into exportable models. | Combines component meshes into full-car and right-half STL artifacts suitable for manufacturing checks and CFD handoff. |
| `quality_gates.py` | Final gate before Part 2. | Runs geometry, manufacturability, and extraction checks and assigns lifecycle states when a candidate must be rejected. |
| `tests/` | Checks whether each stage behaves correctly. | Contains standalone test scripts for contracts, hardware, grids, mass/COM, STL assembly, extraction, quality gates, and the Part 1 to Part 2 integration handshake. |
| `run_all_tests.py` | Simple one-shot validation. | Runs every Part 1 test file with the current Python interpreter and reports the total pass/fail count. |
| `BUILD_REPORT.md` | Engineering build log. | Records what was implemented, what failed during audit, what was fixed, and which placeholders remain. |
| `PLACEHOLDERS.md` | Explicit unresolved items list. | Tracks the measured values, manufacturing details, and other open questions that still need real project data. |
| `SPEC.txt` / `SPEC_ASCII.md` | Original source specification. | Captures the full Part 1 build specification used to derive the implementation and tests. |
| `spec_sections/` | Spec split into smaller chunks. | Stores the same specification in sectioned files for easier inspection and audit review. |

## Requirements

- Python 3.10 or newer
- NumPy
- SciPy

Install the Python dependencies:

```powershell
python -m pip install numpy scipy
```

## Run the test suite

From the repository root:

```powershell
python run_all_tests.py
```

Expected result:

```text
TOTAL: 10 passed, 0 failed (10 test files)
```

Individual test files can also be run directly from `tests/`.
