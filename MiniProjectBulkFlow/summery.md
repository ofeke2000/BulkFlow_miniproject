# Bulk Flow Mini-Project — Context & Handoff

## 1. Project Overview (What this project does)

### Overview

This project investigates **cosmic bulk flows** in large-scale structure by combining  
**cosmological N-body simulations** with **observationally motivated selection functions**.

The main scientific goal is to understand **how selection effects, local environment, and observer choice influence measured bulk flows**, and how those measurements compare to **ΛCDM theoretical expectations**.

Specifically, the project:

- Identifies **observer-like halos** in a large cosmological simulation
- Applies multiple **masking strategies** that mimic real peculiar-velocity surveys
- Computes **bulk flow as a function of radius**
- Quantifies **variance, bias, and selection effects**
- Compares simulation results to **ΛCDM predictions** using Colossus

The pipeline is designed to be:

- **Modular** (each physical test is independent)
- **Restart-safe** (derived quantities are cached and reused)
- **Physically interpretable** (all assumptions are explicit)

This project is intended both as:

- A **research-grade analysis framework**
- A **testbed for understanding observational systematics** in bulk flow studies

---

### Scientific Motivation

Bulk flows probe:

- Large-scale matter distribution
- Growth of structure
- Consistency with ΛCDM cosmology

However, real-world measurements:

- Are incomplete
- Are geometry-dependent
- Are observer-dependent

This project explicitly studies:

- How “Local Universe–like” observers differ from random observers
- How CF4-like selection affects inferred bulk flows
- Whether observed high-amplitude flows can arise naturally from ΛCDM once selection effects are included

---

## 2. Data Sources

### Rockstar Halo Catalog (Simulation)

* Source: MDPL2 Rockstar output
* Columns used:

  * Positions: `x, y, z` (periodic box, h⁻¹ Mpc)
  * Velocities: `vx, vy, vz` (km/s)
  * Mass: `mvir`
  * Halo ID: `rockstarid`
* Box size: **1000 h⁻¹ Mpc**
* Periodic boundary conditions (PBC) used everywhere

### CF4 Catalogue (Observational Proxy)

* Used to generate **CF4-like selection masks**
* Mimics real peculiar velocity survey geometry
* Used to:

  * Match number density
  * Impose radial / angular incompleteness

---

## 3. Key Decisions & Assumptions

### Physical Assumptions

* Flat ΛCDM cosmology
* Bulk flow defined as **average velocity vector** inside radius R
* Small-scale velocities modeled via `σ* ≈ 250 km/s`
* Periodic boundary conditions apply to:

  * distances
  * KDTree searches
  * radial velocity calculation

### Technical Decisions

* **Single-threaded** by default (stability > speed)
* Heavy computations cached to disk (CSV / HDF5)
* All derived columns are **added once and reused**
* Columns are checked before recomputation

### Selection Philosophy

* Observers are chosen to resemble the **Local Universe**
* Selection is based on:

  * Near-zero overdensity
  * Virgo-like massive neighbor
  * Local bulk flow constraints

---

## 4. Main Pipeline Steps

### (1) Load & Prepare Catalogs

* Load Rockstar halos
* Apply periodic wrapping to positions
* Load CF4 catalogue
* Optionally apply mass cuts

---

### (2) Build KDTree

* `scipy.spatial.cKDTree`
* Used for:

  * Overdensity calculation
  * Neighbor searches
  * Virgo test
  * Local bulk flow

---

### (3) Compute Derived Quantities (Cached)

#### 4.1 Local Overdensity

* Computes overdensity within a fixed radius
* Adds column:

```python
delta_{radius}
```

#### 4.2 Virgo Test

* Checks if halo has a neighbor:

  * `mvir > 1e14`
  * distance ∈ [10, 20] Mpc
* Uses internally-built KDTree of massive halos
* Adds column:

```python
near_virgo
```

#### 4.3 Local Bulk Flow

* For each halo:

  * Query neighbors within radius
  * Compute ⟨vx, vy, vz⟩
  * Compute ⟨|v|⟩
* Adds column:

```python
bulkflow_{radius}
```

All 3 tests:

* Checked for existence
* Skipped if already present

---

### (4) Observer Selection

* Apply **optional filters**:

  * mass cut
  * overdensity range
  * bulk flow range
  * Virgo presence
* Filtering logic is configurable
* Final observers chosen by:

  * Sorting by `|delta|`
  * OR random sampling from candidates

Produces:

```python
selected_points
selected_df
```

---

### (5) Mask Construction

For each observer:

#### CF4 Mask

* Uses CF4 catalogue geometry
* Iterative radius matching
* Produces CF4-like incompleteness

#### Uniform Mask

* Random sampling to match CF4 size
* Inside radius R

#### Full Mask

* All halos within `r_max`
* No selection function

---

### (6) Bulk Flow Calculation

For each mask and observer:

#### ML / χ² Estimator (Cumulative)

* Solves:

```
A u = b
```

* Uses **LU decomposition**
* Incremental accumulation by radius
* Propagates scalar uncertainty: `sigma_U = sqrt(uᵀ A⁻¹ u) / |u|`
* Debiased magnitude: `U_debiased = sqrt(max(U_total² - Tr(A⁻¹), 0))` — removes the noise bias E[|Û|²] = |B_true|² + Tr(A⁻¹)
* Also records `n_used` (cumulative halo count at each radius)
* Debugging on failure: determinant, eigenvalues, condition number

Outputs per mask:

```python
radius, u_x, u_y, u_z, U_total, U_debiased, sigma_U, n_used
```

#### Mean Estimator (Cumulative)

* Computes unweighted average of true 3D velocities: `u = ⟨vx, vy, vz⟩` for halos within each radius R
* No weighting by measurement uncertainty
* `sigma_U` and `U_debiased` are NaN (no analytic covariance)
* Both estimators return identical column schemas

Saved to a single **netCDF** file (via `BulkFlowDataset`) with dimensions:

```
origin × radius × mask × method
```

Dataset variables persisted: `u_x`, `u_y`, `u_z`, `U_tot` (biased magnitude),
`U_deb` (noise-debiased magnitude; NaN for the mean estimator), `sigma_U`, `n_used`.
`U_deb` is the quantity to compare against ΛCDM theory because it removes the positive
noise bias: `E[|Û|²] = |B_true|² + Tr(A⁻¹)`.

Per-origin coordinates stored alongside: `overdensity`, `local_bulkflow`, `mvir`, `near_virgo`, position.
Dataset attrs carry full provenance (`selection_variable`, all run parameters, `git_commit`, `timestamp`).

---

### (7) Visualization

#### Bulk Flow vs Radius

Options:

* Mean curves
* Individual curves (no averaging)
* ±1σ shaded variance bands
* Optional theory overlay (Colossus)
* Optional markers on/off

#### Histograms

* Distance / overdensity / mass
* Linear or log axis

#### Heatmaps

* Simulation slices (e.g. 400 < z < 500)
* Hexbin density plots
* Black background, magma colormap

---

## 5. Important Functions Written

### Core Physics

* `radial_velocity_and_error_pbc`
* `bulk_flow_chi2_cumulative`
* `calculate_bulk_flow_series`
* `calculate_local_bulkflow`

### Environment & Selection

* `compute_overdensity`
* `check_near_virgo`
* `add_periodic_distance`
* `make_cf4_mask`
* `make_uniform_mask`

### Plotting

* `plot_bulkflow_from_hdf5`
* `plot_histogram`
* `plot_distance_histogram`
* `plot_simulation_slice_heatmap`

---

## 6. Current Open Questions / TODOs

### Scientific

* How sensitive are results to:

  * σ*
  * CF4 selection details
* Comparison between:

  * ML bulk flow vs local average
* Bias induced by Virgo selection

### Technical

* Should covariance of bulk flow be stored?
* Add bootstrap resampling?
* Improve conditioning diagnostics
* Add observer-to-observer scatter plots

### Visualization

* Animate bulk flow growth vs radius
* Plot vector directions on sky
* Compare CF4 vs Full per observer

---
