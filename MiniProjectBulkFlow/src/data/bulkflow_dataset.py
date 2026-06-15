"""
bulkflow_dataset.py
-------------------
Self-describing xarray Dataset for bulk flow results.

Classes
-------
BulkFlowResult
    One (origin, mask, method) radial series with per-origin environment properties.
BulkFlowDataset
    In-memory accumulator, netCDF writer, and reader for a full bulk flow run.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from .vector3d import Vector3D


# ---------------------------------------------------------------------------
# Per-variable unit strings (class-level constant, not a numeric literal)
# ---------------------------------------------------------------------------
_UNITS: dict[str, str] = {
    "u_x": "km/s",
    "u_y": "km/s",
    "u_z": "km/s",
    "U_tot": "km/s",
    "U_deb": "km/s",
    "sigma_U": "km/s",
    "n_used": "count",
    "radius": "h^-1 Mpc",
    "origin_x": "h^-1 Mpc",
    "origin_y": "h^-1 Mpc",
    "origin_z": "h^-1 Mpc",
    "overdensity": "dimensionless",
    "local_bulkflow": "km/s",
    "mvir": "h^-1 M_sun",
}


@dataclass
class BulkFlowResult:
    """
    Radial bulk-flow series for a single (origin, mask, method) combination.

    Replaces the old bulkflow_result.BulkFlowResult.  New fields: method,
    overdensity, local_bulkflow, mvir, near_virgo, sigma_U, n_used.
    Field U_tot replaces the old U.
    """

    origin_id: int
    origin: Vector3D
    mask: str
    method: str
    overdensity: float
    local_bulkflow: float
    mvir: float
    near_virgo: bool
    radii: np.ndarray
    u_x: np.ndarray
    u_y: np.ndarray
    u_z: np.ndarray
    U_tot: np.ndarray
    U_deb: np.ndarray
    sigma_U: np.ndarray
    n_used: np.ndarray

    def __post_init__(self) -> None:
        self.radii = np.asarray(self.radii, dtype=float)
        self.u_x = np.asarray(self.u_x, dtype=float)
        self.u_y = np.asarray(self.u_y, dtype=float)
        self.u_z = np.asarray(self.u_z, dtype=float)
        self.U_tot = np.asarray(self.U_tot, dtype=float)
        self.U_deb = np.asarray(self.U_deb, dtype=float)
        self.sigma_U = np.asarray(self.sigma_U, dtype=float)
        self.n_used = np.asarray(self.n_used, dtype=float)

        if self.origin is None:
            raise ValueError("BulkFlowResult requires a valid origin")
        arrays = (self.radii, self.u_x, self.u_y, self.u_z,
                  self.U_tot, self.U_deb, self.sigma_U, self.n_used)
        shapes = {a.shape for a in arrays}
        if len(shapes) != 1:
            raise ValueError(
                "BulkFlowResult: all arrays (radii, u_x, u_y, u_z, "
                "U_tot, U_deb, sigma_U, n_used) must have the same shape"
            )


class BulkFlowDataset:
    """
    Accumulator, writer, and reader for bulk flow results stored as a
    self-describing xarray.Dataset in netCDF format.

    Dimensions: origin × radius × mask × method

    Typical write workflow
    ----------------------
    >>> dataset = BulkFlowDataset()
    >>> for origin, mask, method in ...:
    ...     result = BulkFlowResult(...)
    ...     dataset.add(result)
    >>> dataset.set_attrs(attrs_dict)
    >>> dataset.write("output/results.nc")

    Typical read / postprocessing workflow
    ---------------------------------------
    >>> bfd = BulkFlowDataset.open("output/results.nc")
    >>> ds  = bfd.dataset                             # xr.Dataset
    >>> bfd.mean_over_origins("full", "chi2")         # xr.Dataset
    >>> bfd.band_by_selection_variable(bin_edges)     # DatasetGroupBy
    """

    def __init__(self) -> None:
        self._results: list[BulkFlowResult] = []
        self._attrs: dict = {}
        self._ds: xr.Dataset | None = None

    # ------------------------------------------------------------------
    # Accumulation API
    # ------------------------------------------------------------------

    def add(self, result: BulkFlowResult) -> None:
        """Append one (origin, mask, method) radial series."""
        self._results.append(result)

    def set_attrs(self, attrs: dict) -> None:
        """Store provenance and configuration attributes for the Dataset."""
        self._attrs = dict(attrs)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def to_dataset(self) -> xr.Dataset:
        """
        Assemble a 4-D xarray.Dataset from the accumulated BulkFlowResult objects.

        Dimensions: origin × radius × mask × method
        Missing (origin, mask, method) combinations are filled with NaN.
        """
        if not self._results:
            raise ValueError("No results accumulated; cannot build Dataset.")

        # -------------------------------------------------------
        # Collect unique axis values (insertion-order preserved)
        # -------------------------------------------------------
        seen_origins: dict[int, None] = {}
        seen_masks: dict[str, None] = {}
        seen_methods: dict[str, None] = {}
        for r in self._results:
            seen_origins[r.origin_id] = None
            seen_masks[r.mask] = None
            seen_methods[r.method] = None

        origin_ids = list(seen_origins)
        masks = list(seen_masks)
        methods = list(seen_methods)

        # Validate shared radii grid
        ref_radii = self._results[0].radii
        for r in self._results[1:]:
            if not np.array_equal(r.radii, ref_radii):
                raise ValueError(
                    "All BulkFlowResult objects must share the same radii array; "
                    f"origin {r.origin_id} / mask '{r.mask}' has a different grid."
                )
        radii = ref_radii

        n_origins = len(origin_ids)
        n_radii = len(radii)
        n_masks = len(masks)
        n_methods = len(methods)

        oid_idx = {oid: i for i, oid in enumerate(origin_ids)}
        mask_idx = {m: i for i, m in enumerate(masks)}
        method_idx = {m: i for i, m in enumerate(methods)}

        # -------------------------------------------------------
        # Allocate NaN-filled data arrays  shape: (O, R, M, Me)
        # -------------------------------------------------------
        shape = (n_origins, n_radii, n_masks, n_methods)
        u_x = np.full(shape, np.nan)
        u_y = np.full(shape, np.nan)
        u_z = np.full(shape, np.nan)
        U_tot = np.full(shape, np.nan)
        U_deb = np.full(shape, np.nan)
        sigma_U = np.full(shape, np.nan)
        n_used = np.full(shape, np.nan)

        # Per-origin coordinate arrays
        origin_x = np.full(n_origins, np.nan)
        origin_y = np.full(n_origins, np.nan)
        origin_z = np.full(n_origins, np.nan)
        overdensity = np.full(n_origins, np.nan)
        local_bulkflow = np.full(n_origins, np.nan)
        mvir = np.full(n_origins, np.nan)
        near_virgo = np.zeros(n_origins, dtype=np.int8)

        # -------------------------------------------------------
        # Fill arrays from results
        # -------------------------------------------------------
        for res in self._results:
            oi = oid_idx[res.origin_id]
            mi = mask_idx[res.mask]
            met_i = method_idx[res.method]

            u_x[oi, :, mi, met_i] = res.u_x
            u_y[oi, :, mi, met_i] = res.u_y
            u_z[oi, :, mi, met_i] = res.u_z
            U_tot[oi, :, mi, met_i] = res.U_tot
            U_deb[oi, :, mi, met_i] = res.U_deb
            sigma_U[oi, :, mi, met_i] = res.sigma_U
            n_used[oi, :, mi, met_i] = res.n_used

            origin_x[oi] = res.origin.x
            origin_y[oi] = res.origin.y
            origin_z[oi] = res.origin.z
            overdensity[oi] = res.overdensity
            local_bulkflow[oi] = res.local_bulkflow
            mvir[oi] = res.mvir
            near_virgo[oi] = int(res.near_virgo)

        # -------------------------------------------------------
        # Assemble Dataset
        # -------------------------------------------------------
        dims = ("origin", "radius", "mask", "method")

        ds = xr.Dataset(
            data_vars={
                "u_x": (dims, u_x, {"units": _UNITS["u_x"]}),
                "u_y": (dims, u_y, {"units": _UNITS["u_y"]}),
                "u_z": (dims, u_z, {"units": _UNITS["u_z"]}),
                "U_tot": (dims, U_tot, {"units": _UNITS["U_tot"]}),
                "U_deb": (dims, U_deb, {"units": _UNITS["U_deb"],
                          "long_name": "noise-debiased bulk-flow magnitude",
                          "description": (
                              "sqrt(max(U_tot^2 - Tr(A^{-1}), 0)); NaN for "
                              "the mean estimator (no analytic covariance)"
                          )}),
                "sigma_U": (dims, sigma_U, {"units": _UNITS["sigma_U"]}),
                "n_used": (dims, n_used, {"units": _UNITS["n_used"]}),
            },
            coords={
                "origin": origin_ids,
                "radius": (["radius"], radii, {"units": _UNITS["radius"]}),
                "mask": masks,
                "method": methods,
                "origin_x": ("origin", origin_x, {"units": _UNITS["origin_x"]}),
                "origin_y": ("origin", origin_y, {"units": _UNITS["origin_y"]}),
                "origin_z": ("origin", origin_z, {"units": _UNITS["origin_z"]}),
                "overdensity": ("origin", overdensity, {"units": _UNITS["overdensity"]}),
                "local_bulkflow": ("origin", local_bulkflow, {"units": _UNITS["local_bulkflow"]}),
                "mvir": ("origin", mvir, {"units": _UNITS["mvir"]}),
                "near_virgo": ("origin", near_virgo),
            },
            attrs=self._attrs,
        )

        return ds

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def write(self, path: str | Path) -> None:
        """Assemble and write the Dataset to a netCDF file."""
        ds = self.to_dataset()
        ds.to_netcdf(str(path))

    @classmethod
    def open(cls, path: str | Path) -> "BulkFlowDataset":
        """Load a netCDF file written by BulkFlowDataset.write()."""
        instance = cls()
        instance._ds = xr.open_dataset(str(path))
        return instance

    @property
    def dataset(self) -> xr.Dataset:
        """The underlying xarray.Dataset (loaded from file or assembled from results)."""
        if self._ds is None:
            self._ds = self.to_dataset()
        return self._ds

    # ------------------------------------------------------------------
    # Postprocessing convenience accessors
    # ------------------------------------------------------------------

    def mean_over_origins(self, mask: str, method: str) -> xr.Dataset:
        """
        Mean bulk-flow across all origins for a given mask and method.

        Returns an xr.Dataset with the `origin` dimension reduced.
        """
        return self.dataset.sel(mask=mask, method=method).mean(dim="origin")

    def band_by_selection_variable(
        self,
        bins: Sequence[float],
    ) -> xr.core.groupby.DatasetGroupBy:
        """
        Group origins by the selection variable declared in dataset attrs.

        Parameters
        ----------
        bins : sequence of float
            Bin edges for the selection variable.

        Returns
        -------
        DatasetGroupBy
            Call .mean("origin") or .std("origin") on the result to obtain
            per-band statistics ready for matplotlib.

        Example
        -------
        >>> bfd = BulkFlowDataset.open("results.nc")
        >>> bands = bfd.band_by_selection_variable(np.arange(0, 1600, 100))
        >>> mean_per_band = bands["U_tot"].mean("origin")
        """
        sel_var = self.dataset.attrs["selection_variable"]
        coord = self.dataset[sel_var]
        return self.dataset.groupby_bins(coord, bins=bins)

    def to_tidy_dataframe(self, method: str | None = None) -> pd.DataFrame:
        """
        Convert to a flat (tidy) DataFrame with one row per (origin, mask, radius).

        Columns: [origin_id, mask, radius, u_x, u_y, u_z, U_total, U_debiased]

        U_debiased is the noise-debiased bulk-flow magnitude (NaN for the mean
        estimator).  Useful for feeding plot_bulkflow_from_hdf5-style functions.
        If *method* is None, the first available method is used.
        """
        ds = self.dataset

        # Resolve method dimension
        if "method" in ds.dims:
            if method is not None:
                ds = ds.sel(method=method)
            else:
                ds = ds.isel(method=0)

        masks_available = list(ds.coords["mask"].values) if "mask" in ds.dims else []
        origin_ids = ds.coords["origin"].values
        radii = ds.coords["radius"].values

        frames: list[pd.DataFrame] = []
        for mask_val in (masks_available if masks_available else [None]):
            ds_m = ds.sel(mask=mask_val) if mask_val is not None else ds

            u_x_arr = ds_m["u_x"].values   # (n_origins, n_radii)
            u_y_arr = ds_m["u_y"].values
            u_z_arr = ds_m["u_z"].values
            U_arr = ds_m["U_tot"].values
            U_deb_arr = ds_m["U_deb"].values

            oid_rep = np.repeat(origin_ids, len(radii))
            rad_rep = np.tile(radii, len(origin_ids))

            frames.append(pd.DataFrame({
                "origin_id": oid_rep,
                "mask": str(mask_val) if mask_val is not None else "",
                "radius": rad_rep,
                "u_x": u_x_arr.ravel(),
                "u_y": u_y_arr.ravel(),
                "u_z": u_z_arr.ravel(),
                "U_total": U_arr.ravel(),
                "U_debiased": U_deb_arr.ravel(),
            }))

        return pd.concat(frames, ignore_index=True)
