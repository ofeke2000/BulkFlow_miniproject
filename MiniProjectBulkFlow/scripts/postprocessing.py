"""
postprocessing.py
-----------------
Aggregate bulk flow results from the netCDF output and create final plots.

One-liner banded-plot recipe (matplotlib-ready):
    ds = xr.open_dataset(nc_path)
    bin_edges = np.arange(pp.band_min, pp.band_max + pp.band_step, pp.band_step)
    bands = ds["U_tot"].groupby_bins(ds[ds.attrs["selection_variable"]], bins=bin_edges)
    mean_per_band = bands.mean("origin")  # shape: (band, radius, mask, method)
"""

import os
import numpy as np
import pandas as pd
import logging

from src.data.bulkflow_dataset import BulkFlowDataset
from src.theoretical_bulkflow import theoretical_bulkflow_colossus
from src.visualize import BulkFlowPlotter
from src.classes import AppConfig


def aggregate_results(cfg: AppConfig) -> None:
    """
    Read the netCDF bulk flow output, band origins by the run's selection_variable,
    compute per-band mean and std of U_tot, and write a unified CSV for plotting.

    The mask used for the comparison curve is 'full' when masks='all', otherwise
    whatever mask was configured. The method is taken from the dataset's own attrs
    so the CSV reflects what was actually computed.
    """
    nc_file = cfg.paths.output_file
    if not os.path.exists(nc_file):
        logging.error(f"netCDF file not found: {nc_file}")
        return

    bfd = BulkFlowDataset.open(nc_file)
    ds = bfd.dataset

    sel_var = ds.attrs["selection_variable"]
    pp = cfg.postprocessing

    # Determine which mask / method to aggregate for the comparison CSV
    masks_in_ds = list(ds.coords["mask"].values)
    methods_in_ds = list(ds.coords["method"].values)

    mask_mode = cfg.bulkflow.masks or "full"
    plot_mask = "full" if (mask_mode == "all" or "full" in masks_in_ds) else masks_in_ds[0]
    if plot_mask not in masks_in_ds:
        plot_mask = masks_in_ds[0]

    plot_method = ds.attrs.get("calculation_method", cfg.bulkflow.calculation_method)
    if plot_method not in methods_in_ds:
        plot_method = methods_in_ds[0]

    # (origin, radius) DataArray of U_tot for the chosen mask/method
    U_tot_sel = ds["U_tot"].sel(mask=plot_mask, method=plot_method)
    sel_coord = ds[sel_var]

    radii = ds["radius"].values
    bin_edges = np.arange(pp.band_min, pp.band_max + pp.band_step, pp.band_step)

    # Group by selection_variable bins and compute mean + std
    bin_dim = f"{sel_var}_bins"
    grouped_mean = U_tot_sel.groupby_bins(sel_coord, bins=bin_edges).mean("origin")
    grouped_std = U_tot_sel.groupby_bins(sel_coord, bins=bin_edges).std("origin")

    # Build unified CSV (per-band mean/std + theory) for external analysis
    csv_data: dict = {"radius": radii}

    intervals = grouped_mean.coords[bin_dim].values
    for i_bin, interval in enumerate(intervals):
        lo = interval.left
        hi = interval.right
        col_base = f"V_band_{lo:.0f}_to_{hi:.0f}"
        csv_data[f"{col_base}_mean"] = grouped_mean.isel(**{bin_dim: i_bin}).values
        csv_data[f"{col_base}_std"] = grouped_std.isel(**{bin_dim: i_bin}).values

    # Theory curve
    sigma_v = theoretical_bulkflow_colossus(
        radii=radii,
        cosmology_cfg=cfg.cosmology,
        theory_cfg=cfg.theory,
    )
    csv_data["U_mean_theory"] = cfg.cosmology.bulk_flow_amplitude_factor * sigma_v

    csv_file = os.path.join(cfg.paths.output_folder, "unified_results.csv")
    pd.DataFrame(csv_data).to_csv(csv_file, index=False)
    logging.info(f"Aggregated results written to {csv_file}")


def create_final_plots(cfg: AppConfig) -> None:
    """
    Create final comparison plots directly from the netCDF output.

    Origins are banded by the dataset's selection_variable using the bin edges
    from PostprocessingConfig, and encoded as a colorbar.
    """
    nc_file = cfg.paths.output_file
    base_dir = cfg.paths.output_folder
    pp = cfg.postprocessing

    if not os.path.exists(nc_file):
        logging.error(f"netCDF file not found: {nc_file}")
        return

    band_bins = np.arange(pp.band_min, pp.band_max + pp.band_step, pp.band_step)

    BulkFlowPlotter(
        nc_file=nc_file,
        output_folder=base_dir,
        output_file="bulkflow_comparison.png",
        band_bins=band_bins,
        plot_theory=True,
        show_markers=False,
        cosmology_cfg=cfg.cosmology,
        theory_cfg=cfg.theory,
        style=cfg.visualization.style,
    ).plot()
