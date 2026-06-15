# Architecture

Here I will describe the purpose of the different parts of the code.

## Project Structure

### config.yaml

Configuration file containing all user-tunable pipeline parameters, file paths, and settings. This includes:
- Data paths (Rockstar catalog, CF4 catalog, output directories)
- Environmental analysis parameters (overdensity radius, bulkflow radius)
- Origin selection criteria (mass cuts, overdensity ranges, bulkflow limits, `virgo_test.enabled`)
- Bulkflow computation settings (radii ranges, masks mode, calculation methods)
- Visualization options and postprocessing settings

### src/config/

Config dataclasses mirroring `config.yaml`. Each section of the yaml has a corresponding dataclass:

| Dataclass | Purpose |
|-----------|---------|
| `MDPL2Config` | Fixed simulation box parameters (box_size, HubbleParameter) |
| `CosmologyConfig` | Fixed MDPL2 cosmology (H0, Om0, growth_index, bulk_flow_amplitude_factor) — not user-edited |
| `TheoryConfig` | Colossus integration settings (z, k_min, k_max, k_limit) |
| `VirgoTestConfig` | Virgo proximity test parameters (mass_threshold, r_min, r_max) |
| `BulkFlowConfig` | Bulk flow estimator settings (radii, masks, error model) |
| `OriginConfig` | Origin selection cuts (overdensity, bulkflow, Virgo, mass) |
| `PostprocessingConfig` | Postprocessing band edges, plot ranges, and alpha values |
| `VisualizationConfig` | Plot styling (figsize, dpi, alpha, linewidth, bins) via nested `PlotStyleConfig` |
| `PathsConfig` | All file/directory paths |
| `AppConfig` | Root config holding all of the above; loaded via `AppConfig.from_dict(yaml)` |

### main.py

Main orchestration script that runs the entire bulk flow analysis pipeline. It imports and calls the modular scripts in sequence:
1. Data preprocessing
2. Environmental analysis
3. Origin selection
4. Bulkflow computation
5. Postprocessing and visualization

### scripts/

Directory containing modular pipeline stage scripts, each handling a specific phase of the analysis.

#### data_preprocessing.py
Loads the Rockstar halo catalog, applies mass cuts, builds a KDTree for spatial queries, and optionally loads CF4 catalog data. Returns processed dataframes and spatial index.

#### environment_analysis.py
Computes environmental properties for each halo:
- Local overdensity within specified radius
- Near-Virgo cluster proximity test
- Local bulk flow velocity

#### origin_selection.py
Selects origin points (halos) for bulk flow computation based on environmental criteria. Supports filtering by overdensity, mass, and local bulk flow, with options for lowest overdensity selection or random sampling.

#### bulkflow_computation.py
Computes bulk flow time series for selected origins. Supports configurable masking modes:
- "full": Full halo catalog
- "cf4": CF4-based mask
- "uniform": Uniform density mask
- "all": Compute all mask types

Writes all results to a single self-describing netCDF file via `BulkFlowDataset`.

#### postprocessing.py
Reads the netCDF output via `BulkFlowDataset.open()`, bands origins by the run's
`selection_variable`, computes per-band mean/std of U_tot, and writes a unified CSV
for plotting. Generates final comparison plots.

### src/

Directory containing reusable utility functions and core computational modules.

#### __init__.py
Package initializer to make src/ importable as a module.

#### bulkflow.py
Functions for calculating bulk flow velocities from halo catalogs, including series computation over radial ranges.

Two estimators:
- `bulk_flow_chi2_cumulative`: ML / χ² estimator (weighted least-squares with LU decomposition). Returns columns `[radius, u_x, u_y, u_z, U_total, U_debiased, sigma_U, n_used]`. `U_debiased = sqrt(max(U_total^2 - Tr(A^{-1}), 0))` removes the noise bias from the magnitude.
- `bulk_flow_mean_cumulative`: unweighted mean of the true 3D velocities `⟨vx, vy, vz⟩` of halos within each radius. Returns the same columns; `U_debiased` and `sigma_U` are NaN (no analytic covariance).

Both estimators return identical column schemas so downstream code can treat them uniformly.

#### classes.py
Re-exports `AppConfig`, `Vector3D`, `BulkFlowResult`, and `BulkFlowDataset` as a
single convenience namespace.

#### data/ (sub-package)

| Module | Purpose |
|---|---|
| `vector3d.py` | Immutable 3-D vector with PBC helpers |
| `bulkflow_dataset.py` | `BulkFlowResult` dataclass (one origin/mask/method series) and `BulkFlowDataset` (4-D xarray accumulator, netCDF writer/reader) |
| `bulkflow_result.py` | Legacy file — superseded by `bulkflow_dataset.py` |

`BulkFlowDataset` dimensions: **origin × radius × mask × method**.
Dataset variables: `u_x`, `u_y`, `u_z`, `U_tot` (biased bulk-flow magnitude),
`U_deb` (noise-debiased magnitude: `sqrt(max(U_tot^2 - Tr(A^{-1}), 0))`; NaN for
the mean estimator), `sigma_U`, `n_used`.
Per-origin coordinates (extensible without schema migration): `origin_x/y/z`,
`overdensity`, `local_bulkflow`, `mvir`, `near_virgo`.
Dataset attrs carry full provenance: `selection_variable`, all run parameters, `git_commit`, `timestamp`.

#### data_loader.py
Functions for loading and preprocessing data from Rockstar and CF4 catalog files.

#### masks.py
Functions for creating masked halo catalogs:
- CF4-based masks using cluster finder data
- Uniform density masks

#### near_virgo.py
Implements the near-Virgo cluster environmental test.

#### overdensity.py
Calculates local overdensity around halos using spatial queries.

#### specific_utils.py
Specialized utility functions for calculations, including periodic distance calculations.

#### theoretical_bulkflow.py
Theoretical models and predictions for bulk flow behavior.

#### utils.py
General utility functions:
- Logger setup
- Timing functions
- Directory creation
- Dataframe saving

#### visualize.py
Visualization utilities. Bulk-flow plotting is consolidated in two classes:

- `FacetSet` — classifies each facet (mask / method / estimator / sigma\_star / N / selection\_variable) as CONSTANT or VARYING across the curves in a figure. Constant facets go to the output filename and a corner annotation box; varying categorical facets go to legend labels; a varying selection-variable band goes to a colorbar.
- `BulkFlowPlotter` — reads directly from a `BulkFlowDataset` netCDF. Supports mean-over-origins mode, all-curves mode, and banded mode (`band_bins` triggers `groupby_bins` by the dataset's `selection_variable`). Assigns colors from the default prop cycle for categorical facets, or from a colormap when the selection-variable band varies. Four layout constants (`_TEXTBOX_X/Y`, `_TEXTBOX_FONTSIZE`, `_TEXTBOX_ALPHA`, `_COLORBAR_PAD`) live as class attributes.

Module-level helpers retained: `plot_histogram`, `plot_simulation_slice_heatmap`.

## data/

Directory containing source data files:
- Rockstar halo catalogs (CSV format)
- CF4 cluster finder data (CSV format)

## output/

Directory for all generated output files:
- Processed data checkpoints
- HDF5 files with bulk flow results
- Plots and visualizations
- Log files

## Pipeline Flow

1. **Configuration**: Load settings from config.yaml
2. **Data Preprocessing**: Load catalogs, apply cuts, build spatial index
3. **Environmental Analysis**: Compute overdensity, Virgo proximity, local bulk flow for all halos
4. **Origin Selection**: Choose halos with desired environmental properties
5. **Bulk Flow Computation**: Calculate bulk flow series for selected origins with chosen masks
6. **Postprocessing**: Aggregate results and generate plots

The pipeline is designed to be modular and configurable, allowing easy switching between different analysis modes (e.g., CF4 vs uniform masks) and parameter sets.
