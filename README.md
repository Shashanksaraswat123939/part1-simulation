# Part 1: Generative Geometry Designer

Level-set phi field geometry pipeline for STEM Racing car optimisation.

> 📐 **Whole-project architecture:** see [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
> (two-stage optimiser, physics, fixed features) and
> [`../SESSION_CHANGES_2026-07-24.md`](../SESSION_CHANGES_2026-07-24.md) for the
> latest changes (Stage-1 search, ballast void, cargo flip).

## Install

```bash
pip install -r requirements.txt
```

## Run tests

```bash
python run_all_tests.py
```

Set `PART2_PATH` environment variable to the `part2_simulation` directory
before running integration tests:

```bash
export PART2_PATH=/path/to/part2_simulation
python run_all_tests.py
```

## Open items (⚠ UNRESOLVED)

| ID | Blocking | What is needed |
|----|----------|---------------|
| U1 | Yes | Halo cross-section polygon (y,z vertices in mm) |
| U2 | Yes | CO2 canister legal position (x,y,z in mm) |
| U4 | No  | UAE competition rule envelope dimensions |
| U5 | Yes | Rear wing fixed position (x,y,z in mm) |
| U6 | No  | Absolute car envelope (y-extents, z-extents, rearpod length) |

See `PLACEHOLDERS.md` for full details.
