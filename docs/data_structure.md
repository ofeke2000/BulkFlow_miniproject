# Bulk-Flow Result Data Structure

How bulk-flow results are stored: a single self-describing **`xarray.Dataset`**
written to **one netCDF file**. This replaces the older approach (per-origin
`DataFrame.to_hdf(append=True)` into per-band `.h5` files).

## Why xarray + netCDF

The data is naturally a *labeled N-dimensional array*, not a flat table:
for every **origin**, every **radius**, every **mask**, and every **calculation
method**, we measure a bulk-flow vector. xarray lets us keep names on every axis
and attach the origin's position and environment as coordinates, so postprocessing
into different graphs is a one-liner instead of a `groupby`.

- **netCDF** — a stable, self-describing array format (metadata travels *with* the
  data). Readable from Python, the command line (`ncdump`, `ncview`, `nco`/`cdo`),
  and GUI viewers.
- **xarray** — the standard Python library for labeled arrays. Interops with pandas
  (`.to_dataframe()`), scales with dask, and plots via matplotlib.

## The structure

```
Dimensions:   origin · radius · mask · method

Data variables  (each has dims: origin × radius × mask × method)
  u_x, u_y, u_z   bulk-flow vector components      [km/s]
  U_tot           bulk-flow magnitude               [km/s]
  sigma_U         uncertainty on U_tot              [km/s]   (chi2; NaN for mean)
  n_used          # halos inside radius R           [count]

Coordinates
  radius          radial bins (50…250)              [h^-1 Mpc]
  mask            ["full", "cf4", "uniform"]
  method          ["chi2", "mean"]

  # --- per-origin coordinates (along the `origin` axis) ---
  #     this block is EXTENSIBLE: add a new selection variable
  #     here in the future without changing anything else.
  origin_id       rockstar id  (the origin index)
  origin_x/y/z    origin position                   [h^-1 Mpc]
  overdensity     local delta_{R}                   (selection property)
  local_bulkflow  local bulkflow_{R}                [km/s]
  mvir            halo mass                          [h^-1 M_sun]
  near_virgo      proximity flag                    [bool]

Attributes (provenance + self-description)
  selection_variable = "overdensity"   # NAME of the active "changing variable"
  sigma_star, sigma_min, error_fraction, calculation_method,
  box_size, min_radius, max_radius, radii_step,
  cf4_match_radius, uniform_radius, selection_mass_min/max,
  number_of_origins, git_commit, timestamp, units…
```

### The "changing variable"

The variable we select/bin origins by (overdensity today, but it could be local
velocity, mass, …) is **one number per origin**, so it is stored as a *per-origin
coordinate*, and its name is recorded in the `selection_variable` attribute.

This keeps the raw file **re-binnable**: bands are formed at plot time, not baked
into a dimension. To band by a *different* variable later, just store it as another
per-origin coordinate — no schema change.

## How the data is built (short example)

```python
import numpy as np
import xarray as xr

# --- inputs for one run (shapes for illustration) ---
origin_ids = np.array([101, 102, 103])          # (n_origin,)
radius     = np.arange(50.0, 255.0, 5.0)        # (n_radius,)
masks      = ["full", "cf4", "uniform"]         # (n_mask,)
methods    = ["chi2", "mean"]                   # (n_method,)

shape = (origin_ids.size, radius.size, len(masks), len(methods))
dims  = ("origin", "radius", "mask", "method")

ds = xr.Dataset(
    data_vars=dict(
        u_x    =(dims, np.full(shape, np.nan)),
        u_y    =(dims, np.full(shape, np.nan)),
        u_z    =(dims, np.full(shape, np.nan)),
        U_tot  =(dims, np.full(shape, np.nan)),
        sigma_U=(dims, np.full(shape, np.nan)),
        n_used =(dims, np.zeros(shape, dtype=int)),
    ),
    coords=dict(
        origin = origin_ids,
        radius = radius,
        mask   = masks,
        method = methods,
        # per-origin coordinates (along `origin`) — the extensible block
        origin_x      =("origin", np.array([512.3, 88.1, 300.0])),
        origin_y      =("origin", np.array([110.7, 640.2, 12.5])),
        origin_z      =("origin", np.array([301.0, 25.9, 980.4])),
        overdensity   =("origin", np.array([-0.18, 0.02, 0.31])),
        local_bulkflow=("origin", np.array([240.0, 95.0, 410.0])),
        mvir          =("origin", np.array([3.1e14, 8.0e14, 1.2e15])),
        near_virgo    =("origin", np.array([False, True, False])),
    ),
    attrs=dict(
        selection_variable="overdensity",
        sigma_star=250.0,
        calculation_method="chi2",
        box_size=1000.0,
        units_U="km/s",
        units_radius="h^-1 Mpc",
        # … git_commit, timestamp, remaining run params …
    ),
)

# Fill measurements per (origin, mask, method) from the estimator, e.g.:
# ds["U_tot"].loc[dict(origin=101, mask="full", method="chi2")] = U_series

ds.to_netcdf("output/bulkflow_results.nc")      # one combined file
```

In the pipeline this is wrapped in classes (see `src/data/bulkflow_dataset.py`):
a `BulkFlowResult` holds one `(origin, mask, method)` radial series, and a
`BulkFlowDataset` accumulates them and writes the netCDF once at the end of the
origin loop.

## Reading it back / making graphs

Because the axes are named, most plots are one line:

```python
ds = xr.open_dataset("output/bulkflow_results.nc")

# mean bulk flow vs radius (the classic curve) — no groupby needed
ds.U_tot.sel(mask="full", method="chi2").mean("origin").plot()

# compare masks on one axis
ds.U_tot.sel(method="chi2").mean("origin").plot.line(x="radius", hue="mask")

# band by the active "changing variable" (re-binnable!)
var  = ds.attrs["selection_variable"]            # "overdensity"
bins = [-0.3, -0.1, 0.1, 0.3]
ds.U_tot.sel(mask="full", method="chi2") \
  .groupby_bins(ds[var], bins).mean() \
  .plot.line(x="radius", hue=f"{var}_bins")

# scatter at fixed radius, colored by environment
sl = ds.sel(radius=150, mask="full", method="chi2")
plt.scatter(sl.overdensity, sl.U_tot)

# error band from the stored uncertainty
m = ds.U_tot.sel(mask="full", method="chi2")
plt.fill_between(ds.radius, (m - ds.sigma_U).mean("origin"),
                           (m + ds.sigma_U).mean("origin"), alpha=0.3)

# drop straight into pandas if you prefer tabular work
df = ds.to_dataframe().reset_index()
```

## Notes

- **Size**: ~1000 origins × 40 radii × 3 masks × 2 methods × a handful of float
  variables ≈ tens of MB — comfortably in RAM. If origins ever grow by orders of
  magnitude, `xr.open_dataset(..., chunks=...)` enables lazy/dask reads with no
  code change.
- **Provenance**: the `selection_variable` name, run parameters, git commit, and
  units live in `attrs`, so a results file is self-documenting — no reliance on
  directory/filename conventions.
- **Periodic boundary conditions** still apply to all spatial computations upstream
  (box = 1000 h^-1 Mpc); this document only covers how results are *stored*.
```
