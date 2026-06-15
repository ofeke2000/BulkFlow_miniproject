# CLAUDE.md

Research code for measuring **cosmic bulk flows** in the MDPL2 N-body simulation under
observationally motivated selection effects (CF4-like masks), and comparing to ΛCDM predictions.

Read these for context before non-trivial work:

- [docs/summary.md](docs/summary.md) — full science context, assumptions, pipeline description (the best overview)
- [docs/architecture.md](docs/architecture.md) — per-file responsibilities
- [docs/research_notes.md](docs/research_notes.md) — **current research goals**; what the user is working on right now

## Running the code

```bash
source venv/bin/activate
python main.py        # run from the repo root; config.yaml is loaded by relative path
```

- **Ask before running the full pipeline.** It is long-running, and step 3
  (`save_catalog_checkpoint`) **overwrites the Rockstar catalog CSV in `data/`** with added
  derived columns. Small isolated snippets / imports are fine to run freely.
- All tunable parameters live in `config.yaml` (at the repo root), loaded into the dataclasses
  in `src/config/`.
- `minimain.py` is a personal scratch script for ad-hoc analyses — not a stable entry point;
  don't treat its contents as load-bearing. **Do not read or modify `minimain.py` unless the
  user explicitly asks.**

## Code layout

- `main.py` — orchestrates the pipeline: preprocessing → environmental
  analysis → origin selection → bulk flow computation → postprocessing/plots
- `scripts/` — one module per pipeline stage
- `src/` — library code, grouped into sub-packages:
  - `src/physics/` — bulk-flow estimators, theoretical predictions, overdensity, near-Virgo, masks, PBC primitives
  - `src/io/` — catalog loaders and general utilities
  - `src/viz/` — plotting
  - `src/config/` — config dataclasses mirroring config.yaml
  - `src/data/` — result data structures (`Vector3D`, `BulkFlowDataset`/`BulkFlowResult`)
- `data/` — input catalogs (Rockstar halos, CF4); large CSVs, don't load fully unless needed
- `output/` — results (HDF5 per run) and plots
- `docs/` — science summary, architecture, data-structure notes, research notes, test plan

## Coding conventions

- **Prefer classes.** When adding new logic, prefer encapsulating it in a class rather than standalone functions or module-level code.
- **No bare numbers in pipeline code.** Any numeric literal introduced anywhere in the pipeline must live either in a class attribute or in `config.yaml` — never inline. For every new number, ask the user whether it belongs in a class or in the config before writing any code.

## Hard rules

1. **Periodic boundary conditions everywhere.** The simulation box is 1000 h⁻¹ Mpc with PBC.
   Every spatial calculation — distances, KDTree queries, radial velocities, masks — must
   handle periodic wrapping. Never write a plain Euclidean distance on box coordinates.
2. **Cache-and-skip pattern for derived columns.** Derived quantities (`delta_{radius}`,
   `near_virgo`, `bulkflow_{radius}`) are computed once, saved back to the catalog, and
   checked for existence before recomputation. Any new derived quantity must follow this
   pattern: check column exists → skip if present → compute → save.
3. **Keep docs in sync.** When a change affects structure or behavior described in
   `docs/architecture.md` or `docs/summary.md`, update those docs in the same task.

## Units & conventions

- Positions: h⁻¹ Mpc (periodic box coordinates)
- Velocities: km/s; masses: `mvir` in h⁻¹ M☉
- Bulk flow = average velocity vector of halos inside radius R around an observer
- Small-scale velocity noise: σ* ≈ 250 km/s (configurable)
- Two bulk flow estimators: `chi2` (ML / χ² with LU solve, cumulative in radius) and `mean`
