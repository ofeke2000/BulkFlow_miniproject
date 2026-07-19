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
| `CosmologyConfig` | Fixed MDPL2 cosmology (H0, Om0, growth_index, bulk_flow_amplitude_factor, hubble_velocity_per_hinv_mpc); also carries CF4's own calibrated Hubble constant (H0_CF4, `cf4_hubble_velocity_per_hinv_mpc` = H0_CF4/h) for the CF4-facing velocity analyses — not user-edited |
| `PhysicalConstants` | Fixed universal physical constants (SPEED_OF_LIGHT_KM_S), independent of cosmology — not user-edited |
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

Split into two sub-packages by kind:

- `scripts/pipeline/` — the modular pipeline stages run in sequence by `main.py`, each handling a specific phase of the analysis.
- `scripts/analyses/` — standalone analyses outside the main pipeline, each driven by its own `run_*.py` entry point at the repo root.

#### pipeline/data_preprocessing.py
Loads the Rockstar halo catalog, applies mass cuts, builds a KDTree for spatial queries, and optionally loads CF4 catalog data. Returns processed dataframes and spatial index.

#### pipeline/environment_analysis.py
Computes environmental properties for each halo:
- Local overdensity within specified radius
- Near-Virgo cluster proximity test
- Local bulk flow velocity

#### pipeline/origin_selection.py
Selects origin points (halos) for bulk flow computation based on environmental criteria. Supports filtering by overdensity, mass, and local bulk flow, with options for lowest overdensity selection or random sampling.

#### pipeline/bulkflow_computation.py
Computes bulk flow time series for selected origins. Supports configurable masking modes:
- "full": Full halo catalog
- "cf4": CF4-based mask
- "uniform": Uniform density mask
- "all": Compute all mask types

Writes all results to a single self-describing netCDF file via `BulkFlowDataset`.

#### pipeline/postprocessing.py
Reads the netCDF output via `BulkFlowDataset.open()`, bands origins by the run's
`selection_variable`, computes per-band mean/std of U_tot, and writes a unified CSV
for plotting. Generates final comparison plots.

#### analyses/methods_comparison.py
Standalone estimator-comparison analysis (`MethodsComparison` class), driven by
the root `run_methods_comparison.py` entry point. Computes **both** the `chi2`
and `mean` estimators for the *same* set of origins (full mask) and stores them
in one netCDF (`method` is a dataset dimension), then renders two plots:
(1) chi2 vs mean `U_tot` against ΛCDM theory, and (2) chi2 `U_tot` vs the
noise-debiased `U_deb`. Origins are the N lowest-|delta| halos (no other cuts).
Run-specific parameters (radial grid, N origins, estimators) are class attributes
on `MethodsComparison`; physics/paths/cosmology come from the config. Both the
netCDF and the two PNGs get the run's defining constants (N, mask, methods,
sigma_star, selection_variable) appended to their filenames via the shared
`FacetSet.filename_suffix()` logic, so different runs (e.g. N=50 vs N=200) never
overwrite each other and the dataset sorts alongside its figures. Unlike
`main.py`, it does **not** run the environmental analysis and does **not**
overwrite the catalog checkpoint — it relies on the derived columns already
present in the catalog CSV.

#### analyses/velocity_comparison.py
Standalone observational analysis (`VelocityComparison` class), driven by the
root `run_velocity_comparison.py` entry point. Computes the bulk flow
**directly on the CF4 "All Group Velocities" catalogue** (no MDPL2 halos, no
PBC) for two of its radial-peculiar-velocity columns — `Vpds` and `Vpwf` —
each run through **both** the `chi2` and `mean` estimators, and overlays them
against ΛCDM theory. The observer is fixed at the
supergalactic origin (us); positions come from RA/Dec/D (converted to h⁻¹ Mpc
by `load_cf4_catalogue`), and the catalogue's own line-of-sight velocities are
fed straight into `bulk_flow_chi2_cumulative` (bypassing the simulation-only
`radial_velocity_and_error_pbc` projection). Per-object uncertainties are
propagated from the distance-modulus error column `eDM`, floored by a
distance-proportional term:
`sigma_v = max((ln10/5)·(H0_CF4/h)·r·eDM, sigma_min_fraction·(H0_CF4/h)·r_scheme)`,
using CF4's own calibrated Hubble constant
(`cfg.cosmology.cf4_hubble_velocity_per_hinv_mpc`, Tully et al. 2023) rather
than the MDPL2 `hubble_velocity_per_hinv_mpc`. The eDM term always uses the
*measured* distance `r`; the floor (`cfg.bulkflow.sigma_min_fraction`, default
0.10) uses each group's *scheme* radius `r_scheme` — it replaced the previous
flat `sigma_min` floor, which the MDPL2 pipeline still uses unchanged (the
`sigma_min` field stays in `BulkFlowConfig` with its default).

Each group is additionally placed at a radial coordinate via one of **two
distance-placement schemes** (`DISTANCE_SCHEMES = ("d", "cz")`), sharing the
same `r_hat` line-of-sight unit vectors and the same eDM sigma term (a
property of the measurement, not of placement), while the
`sigma_min_fraction` floor — and hence `sigma_v` itself — is per-scheme (a
property of placement; both are built by the `_scheme_radii`/`_sigma_model`
helpers, shared with `plot_sigma_terms`). The radius used for the validity
cut/sorting/cumulative binning also differs: `"d"` is the original **measured
distance** `r = |x, y, z|` (from `D`/distance modulus, `r > 0` cut); `"cz"` is
the **redshift distance** `r_z = vcmb / (H0_CF4/h)` (from the catalogue's
`vcmb` column and `cfg.cosmology.cf4_hubble_velocity_per_hinv_mpc`, CF4's
calibrated Hubble constant over the MDPL2 h, `vcmb > 0` cut). The
resulting 8 (column, estimator, scheme) series are stored on the dataset
`method` dimension in a single netCDF — the `"d"` scheme keeps the original,
unsuffixed label shape `f"{col}_{estimator}"` (e.g. `Vpds_chi2`) for backward
compatibility, while `"cz"` appends a `"_cz"` suffix (e.g. `Vpds_chi2_cz`).
Labels are built/parsed via the `_make_label`/`_parse_label` helper pair
(rather than blind string splitting) so a 3rd, scheme-suffix segment doesn't
break label parsing.

From that one netCDF, **four** PNGs are rendered into
`output/velocity comparison/`: a measured-distance chi2-only plot
(`CHI2_PLOT_FILE`, just `Vpds_chi2`/`Vpwf_chi2`), a measured-distance
both-methods plot (`BOTH_METHODS_PLOT_FILE`, all four measured-distance
curves), and a redshift-distance chi2-only plot (`CHI2_CZ_PLOT_FILE`, just
`Vpds_chi2_cz`/`Vpwf_chi2_cz`) — the `"cz"` counterpart of `CHI2_PLOT_FILE`,
both columns together rather than split per column — plus a **sigma
diagnostic** (`SIGMA_PLOT_FILE`, via `plot_sigma_terms`): one panel per
distance scheme (shared y-axis), scattering the shared eDM sigma term against
that scheme's radius, with the per-scheme `sigma_min_fraction` floor as a
solid line through the origin, the old flat `sigma_min` as a dotted gray
reference, and the floor-domination percentage in each panel title. In the
three comparison plots
**color encodes the velocity column** (`Vpds` blue, `Vpwf` orange).
`CHI2_PLOT_FILE`/`BOTH_METHODS_PLOT_FILE` use the default **linestyle encodes
the estimator** (`chi2` solid, `mean` dashed) styling; `CHI2_CZ_PLOT_FILE`
instead explicitly passes `linestyle_by=_LINESTYLE_BY_SCHEME`, so its curves
render dashed (via `SCHEME_LINESTYLES["cz"] = "--"`) and are visually
distinct from `CHI2_PLOT_FILE`'s solid curves even though both plots contain
only chi2-estimator curves. All three are supplied to `BulkFlowPlotter` via
its optional `method_colors`/`method_linestyles` override maps, built by
`_style_maps` (parameterized by `linestyle_by`) from the
`COLUMN_COLORS`/`METHOD_LINESTYLES`/`SCHEME_LINESTYLES` class attributes.
Run-specific parameters (radial grid, velocity columns, distance schemes, the
`ln10/5` factor, color/linestyle maps, plot filenames and sigma-diagnostic
styling) are `VelocityComparison`
class attributes; `sigma_star`, `sigma_min_fraction`, cosmology, theory, and
style come from the config. Like `methods_comparison.py`, it does **not** run the
environmental analysis and does **not** overwrite the catalog checkpoint. It also
emits a single overlaid histogram (object count vs velocity) comparing the raw
`Vpds` and `Vpwf` columns via `plot_velocity_histograms` (output name/xlabel from
the `HISTOGRAM_FILE`/`HISTOGRAM_XLABEL` class attributes), written alongside the
comparison plot. The overlay uses the shared `plot_overlaid_histogram` helper,
which draws each column as a step histogram on common bins.

#### analyses/velocity_redshift_binning.py
Standalone descriptive-statistics analysis (`VelocityRedshiftBinning` class),
driven by the root `run_velocity_redshift_binning.py` entry point. Unlike
`velocity_comparison.py`, it does **not** run the chi^2 bulk-flow estimator —
it studies the raw `Vpds`/`Vpwf` columns against each of the x-axes declared
in the `X_AXES` class attribute (a tuple of axis specs: dataframe column,
title name, axis label, filename tag, scatter fractions):

- `z` — redshift from the CMB-frame velocity, `z = Vcmb / c`;
- `D` — the catalogue's measured distance (Mpc, from the distance modulus).
  **Not** equivalent to binning by `z`: `D` carries the large distance errors
  that also drive the `Vpds`/`Vpwf` values, so binning by measured `D`
  reorders points and exposes the correlated-error (Malmquist-like) trend;
- `d_cz` — Hubble distance `cz/H0` (Mpc, `H0_KM_S_MPC`, sourced in `__init__`
  from `CosmologyConfig.H0_CF4` = 74.6, CF4's own calibration) — a pure
  rescaling of the `z` axis for unit-matched comparison with the `D` axis.

Per axis it produces two views:

- **Binned means**: groups are binned in the axis variable (equal-width or
  quantile, CLI-selectable), and per bin it computes both the signed mean and
  the mean of the absolute value of each velocity column, each with its
  standard error of the mean (bins with fewer than `min_n_per_bin` groups are
  skipped and logged). Rendered as two figures (signed and `|v|`) via the
  shared `_plot_binned` helper, plus a CSV table with bin
  x-mean/N/mean/SEM/mean-abs/SEM-abs.
- **Raw scatter**: every group's velocity vs the axis variable at the axis's
  point-retention fractions (100%/50%/10% for `z`; 10% only for the two
  distance axes), one PNG per fraction, via `plot_scatter`. All fractions are
  `df.sample(...)` from the same full dataframe with the same
  `SCATTER_RANDOM_SEED`, so the 10% points are a subset of the 50% points.

Filenames are built from templates keyed on the axis tag
(`velocity_vs_{tag}_binned.png`, `velocity_vs_{tag}_scatter_{pct}pct.png`, …),
so one run writes the complete set for every axis side by side. It has no
dependency on `AppConfig`/`config.yaml`: input path, bin count, binning mode,
and the minimum-N threshold are `VelocityRedshiftBinning` class attributes,
overridable via CLI flags (`--input`, `--bins`, `--mode`, `--min-n-per-bin`).
Per-column labels/markers are looked up from the `LABELS`/`MARKERS`
class-attribute dicts, so `_draw_series` (and `plot_scatter`) loop generically
over `VELOCITY_COLUMNS` rather than hard-coding two series. Reads the raw
catalogue via the shared `read_cf4_csv` helper (renamed from the
module-private `_read_cf4_csv` in `src/io/data_loader.py`, now also used by
`load_cf4_catalogue`). All output is written alongside the
`velocity_comparison.py` outputs, into `output/velocity comparison/`.
`bin_and_aggregate` takes an optional explicit `edges` argument (defaulting to
`None`, which preserves the original per-dataframe auto-binning behaviour) so
`VelocityDistanceMock` can reuse it with a *shared* bin-edge set across
several dataframes.

#### analyses/velocity_distance_mock.py
Mock-observation analysis (`VelocityDistanceMock` class, subclasses
`VelocityRedshiftBinning` to reuse its binning/table/figure machinery),
driven by the root `run_velocity_distance_mock.py` entry point. Builds mock
CF4 "peculiar velocity vs measured distance" catalogues from MDPL2 Rockstar
halos, to test whether the trends seen in the real CF4 plots
(`velocity_redshift_binning.py`) are consistent with LambdaCDM once
measurement errors and the CF4 selection function are included. Per observer
(a Local-Universe-like halo selected via `origin_selection.select_origin_points`
if the catalog's cached `delta_*`/`bulkflow_*`/`near_virgo` columns are
present, else a seeded-random halo -- the path taken is logged):

1. A CF4-like mock catalogue is built around the observer with
   `MaskMaker.make_cf4_mask` (CF4 sky geometry shifted to the observer, mod
   box, matched halo velocities; positions are treated as true positions).
2. The true PBC distance `r_true` and true line-of-sight velocity `v_r_true`
   come from `radial_velocity_and_error_pbc`.
3. The truth redshift is exact FLRW at the MDPL2 cosmology:
   `z_cos_true(r_true)` inverts colossus's exact comoving-distance relation
   (interpolated on a fine z grid; colossus's native distance unit is
   h^-1 Mpc, matching `r_true`), and the peculiar velocity composes
   multiplicatively, `(1 + z_obs) = (1 + z_cos_true)(1 + v_r_true/c)`, so
   `cz_obs = c z_obs` (box frame stands in for the CMB frame; the observer's
   own velocity is not subtracted). Blueshifted objects (`cz_obs <= 0`,
   undefined under Vpwf's log) are dropped and logged.
4. `D_meas = r_true * 10**(delta_mu/5)`, `delta_mu ~ N(0, eDM_av)`, where
   `eDM_av` is the matched CF4 group's own distance-modulus error, carried
   through the mask via `MaskMaker.make_cf4_mask`'s new `cf4_carry_columns`
   argument (deterministic per-object pairing, not resampled from the eDM
   distribution). The RNG is seeded `seed + observer_id` for reproducibility.
5. `Vpds`/`Vpwf` analogs are re-derived from `(cz_obs, D_meas)` using the same
   second-order cosmographic correction (Davis & Scrimgeour 2014 Eq. 14;
   Watkins & Feldman 2015), parameterized by `q0 = Om0/2 - Ode0` and `j0 = 1`
   (flat LambdaCDM). The true `v_r_true` is kept as a zero-error reference
   series (`VELOCITY_COLUMNS = (Vpds_mock, Vpwf_mock, v_r_true)`).

Three views are rendered, reusing the parent's `bin_and_aggregate`/
`save_table`/`plot`/`plot_abs`/`plot_scatter`/`_draw_series`/`_save_figure`:
per-observer binned means + scatter (one subfolder per observer, mirroring
the CF4 script's own figures); pooled binned means over all observers with
each observer's own curve overlaid (shared bin edges via
`bin_and_aggregate(..., edges=...)`, new `plot_pooled_with_observers`); and a
CF4-overlay figure (`plot_cf4_overlay`) comparing the mock binned means +/-
SEM *across observers* against the real CF4 binned means from
`VelocityRedshiftBinning` on the same axes (the key deliverable) -- CF4's
Mpc-based `D`/`d_cz` axes are converted to h^-1 Mpc with the MDPL2 `h` for
the comparison. All distances within the mock stay in h^-1 Mpc. Output goes
to `output/velocity comparison mock/`. Read-only on the Rockstar catalog: no
checkpoint is overwritten, and no environmental analysis is (re-)run.

Convention note (documented in the module docstring): the truth redshift is
exact FLRW (multiplicative peculiar-velocity composition), deliberately *not*
generated with the estimators' own second-order cosmographic bracket, so each
estimator exhibits its honest intrinsic residual bias rather than being zeroed
by construction. `Vpds_mock`'s cosmographic inversion tracks exact FLRW very
closely at these redshifts (residual ~0 km/s), while `Vpwf_mock` carries a
genuine positive residual growing with distance (~ +20 km/s at r = 50,
~ +190 km/s at r = 150 h^-1 Mpc, at zero true velocity and zero eDM) -- a real
feature of the estimator that the mock is designed to expose.

### src/

Library code, grouped into themed sub-packages. The module headings below are prefixed
with their sub-package:

- `src/physics/` — bulk-flow estimators, theoretical predictions, overdensity, near-Virgo
  test, catalog masks, and PBC primitives
- `src/io/` — catalog loaders and general utilities
- `src/viz/` — plotting
- `src/config/` — config dataclasses mirroring config.yaml (see above)
- `src/data/` — result data structures (`Vector3D`, `BulkFlowDataset`/`BulkFlowResult`)
- `src/classes.py` — top-level convenience namespace re-exporting the public types

#### __init__.py
Package initializer to make src/ importable as a module.

#### physics/bulkflow.py
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

`BulkFlowDataset` dimensions: **origin × radius × mask × method**.
Dataset variables: `u_x`, `u_y`, `u_z`, `U_tot` (biased bulk-flow magnitude),
`U_deb` (noise-debiased magnitude: `sqrt(max(U_tot^2 - Tr(A^{-1}), 0))`; NaN for
the mean estimator), `sigma_U`, `n_used`.
Per-origin coordinates (extensible without schema migration): `origin_x/y/z`,
`overdensity`, `local_bulkflow`, `mvir`, `near_virgo`.
Dataset attrs carry full provenance: `selection_variable`, all run parameters, `git_commit`, `timestamp`.

#### io/data_loader.py
Functions for loading and preprocessing data from Rockstar and CF4 catalog files.

#### physics/masks.py
Functions for creating masked halo catalogs:
- CF4-based masks using cluster finder data
- Uniform density masks

#### physics/near_virgo.py
Implements the near-Virgo cluster environmental test.

#### physics/overdensity.py
Calculates local overdensity around halos using spatial queries.

#### physics/specific_utils.py
Specialized utility functions for calculations, including periodic distance calculations.

#### physics/theoretical_bulkflow.py
Theoretical models and predictions for bulk flow behavior.

#### io/utils.py
General utility functions:
- Timing functions
- Directory creation
- Dataframe saving

#### viz/visualize.py
Visualization utilities. Bulk-flow plotting is consolidated in two classes:

- `FacetSet` — classifies each facet (mask / method / estimator / sigma\_star / N / selection\_variable) as CONSTANT or VARYING across the curves in a figure. Constant facets go to the output filename and a corner annotation box; varying categorical facets go to legend labels; a varying selection-variable band goes to a colorbar.
- `BulkFlowPlotter` — reads directly from a `BulkFlowDataset` netCDF. Supports mean-over-origins mode, all-curves mode, and banded mode (`band_bins` triggers `groupby_bins` by the dataset's `selection_variable`). Assigns colors from the default prop cycle for categorical facets, or from a colormap when the selection-variable band varies. Two optional constructor args, `method_colors`/`method_linestyles` (both default `None`, keyed by the curve's `method` label), override the auto-assigned color and total-vs-debiased linestyle per method in the non-colorbar branch only; when unset (all existing callers, incl. the main pipeline) behavior is unchanged. All behavioral flags/options (`plot_theory`, `use_mean_amplitude`, `plot_variance_band`, `variance_alpha`, `plot_all_curves`, `plot_debiased`, `show_markers`, `append_facets_to_filename`) are bundled in the `BulkFlowPlotConfig` dataclass (`src/config/visualization_config.py`), passed via the `plot_cfg` argument; call sites override only the fields they need. The `plot_debiased` flag (default `True`) controls whether the chi2 `U_deb` curve is drawn alongside `U_tot`; set it `False` for a clean estimator-vs-theory comparison. The `append_facets_to_filename` flag (default `True`) appends the constant-facet summary (the same key/value pairs shown in the corner textbox — N, mask, method/estimator, sigma_star, selection_variable) to the output filename, keeping each PNG self-describing and ensuring runs with different constants never overwrite each other. Four layout constants (`_TEXTBOX_X/Y`, `_TEXTBOX_FONTSIZE`, `_TEXTBOX_ALPHA`, `_COLORBAR_PAD`) live as class attributes.

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
