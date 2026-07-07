# BulkFlow

Research code for measuring **cosmic bulk flows** in the MDPL2 N-body simulation under
observationally motivated selection effects (CF4-like masks), and comparing the result to
ΛCDM predictions.

## Quick start

```bash
source venv/bin/activate
pip install -r requirements.txt          # add requirements-dev.txt to run the tests
python main.py                            # run from the repo root
```

`config.yaml` (repo root) holds all tunable parameters. The pipeline is long-running and
step 3 **overwrites the Rockstar catalog CSV in `data/`** — see [CLAUDE.md](CLAUDE.md).

## Layout

```
main.py            pipeline orchestration (preprocessing → analysis → bulk flow → plots)
config.yaml        all tunable parameters
scripts/
  pipeline/        one module per pipeline stage
  analyses/        standalone analyses, each driven by a root run_*.py entry point
src/
  physics/         bulk-flow estimators, theory, overdensity, near-Virgo, masks, PBC primitives
  io/              catalog loaders + general utilities
  viz/             plotting
  config/          config dataclasses mirroring config.yaml
  data/            result data structures (Vector3D, BulkFlowDataset)
tests/             pytest suite (run `pytest` from the repo root)
data/   output/    inputs and results (gitignored, regenerated)
docs/              science + engineering documentation
```

## Documentation

- [docs/summary.md](docs/summary.md) — full science context, assumptions, pipeline description
- [docs/architecture.md](docs/architecture.md) — per-file responsibilities
- [docs/data_structure.md](docs/data_structure.md) — how bulk-flow results are stored (xarray + netCDF)
- [docs/research_notes.md](docs/research_notes.md) — current research goals
- [docs/tests.md](docs/tests.md) — test plan and status
- [CLAUDE.md](CLAUDE.md) — conventions and hard rules for working in this repo
