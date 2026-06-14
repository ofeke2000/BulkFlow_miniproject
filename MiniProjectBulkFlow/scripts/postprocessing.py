"""
postprocessing.py
-----------------
Aggregate bulk flow results and create final visualizations.
"""

import os
import numpy as np
import logging

from src.specific_utils import save_average_bulkflow_to_csv
from src.visualize import plot_bulkflow_from_csv
from src.classes import AppConfig


def aggregate_results(cfg: AppConfig):
    """
    Aggregate bulk flow results from multiple HDF5 files into a unified CSV.

    Parameters:
        cfg: AppConfig instance
    """
    base_dir = cfg.paths.output_folder
    csv_file = os.path.join(base_dir, "unified_results.csv")
    pp = cfg.postprocessing

    band_edges = np.arange(pp.band_min, pp.band_max, pp.band_step)

    for low, high in zip(band_edges[:-1], band_edges[1:]):
        def clean_float(x: float) -> float:
            return 0.0 if np.isclose(x, 0.0) else x

        band_lo = clean_float(low)
        band_hi = clean_float(high)

        band_dir = f"{band_lo:.0f}-{band_hi:.0f}"
        band_str = f"{band_lo:.0f}_to_{band_hi:.0f}"

        hdf_file = os.path.join(base_dir, band_dir, f"bulkflow_cut_{band_str}.h5")
        column_name = f"V_band_{band_lo:.0f}_to_{band_hi:.0f}"

        if os.path.exists(hdf_file):
            logging.info(f"Processing band {band_lo:.0f} → {band_hi:.0f} km/s")
            save_average_bulkflow_to_csv(
                hdf_file=hdf_file,
                csv_file=csv_file,
                column_name=column_name,
                mask_type="full",
                key="bulkflow",
                cosmology_cfg=cfg.cosmology,
                theory_cfg=cfg.theory,
            )
        else:
            logging.warning(f"HDF5 file not found: {hdf_file}")


def create_final_plots(cfg: AppConfig):
    """
    Create final comparison plots from aggregated results.

    Parameters:
        cfg: AppConfig instance
    """
    base_dir = cfg.paths.output_folder
    csv_file = os.path.join(base_dir, "unified_results.csv")
    pp = cfg.postprocessing

    if not os.path.exists(csv_file):
        logging.error(f"Unified CSV file not found: {csv_file}")
        return

    plot_bulkflow_from_csv(
        csv_file=csv_file,
        output_folder=base_dir,
        variable_name="local_bulkflow",
        var_min=pp.var_min,
        var_max=pp.var_max,
        var_step=pp.var_step,
        output_file="bulkflow_comparison.png",
        plot_theory=True,
        plot_errors=False,
        error_alpha=pp.error_alpha,
        show_markers=False,
        cosmology_cfg=cfg.cosmology,
        style=cfg.visualization.style,
    )
