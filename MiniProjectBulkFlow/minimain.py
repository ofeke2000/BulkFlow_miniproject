import yaml
import numpy as np
import pandas as pd
import logging
import time
import os
from pathlib import Path

from scipy.spatial import cKDTree

# --- Local modules ---
from src.data_loader import load_rockstar_catalog, load_cf4_catalogue
from src.overdensity import compute_overdensity
from src.masks import make_cf4_mask, make_uniform_mask
from src.bulkflow import calculate_bulk_flow_series, calculate_local_bulkflow
from src.specific_utils import append_bulkflow_results, save_average_bulkflow_to_csv
from src.visualize import plot_bulkflow_from_csv, plot_bulkflow_from_hdf5, plot_histogram, plot_simulation_slice_heatmap
from src.near_virgo import near_virgo

def main():
    base_dir = os.path.expanduser(
        "~/BulkFlow_miniproject/BulkFlow_miniproject/output/"
        "delta_cuts_rmax250_rmin10_jumps5_mcut12"
    )

    csv_file = os.path.join(base_dir, "unified_results.csv")

    # Define overdensity band edges
    band_edges = np.arange(-0.5, 0.6, 0.1)  # [-0.5, ..., 0.5]

    for low, high in zip(band_edges[:-1], band_edges[1:]):

        def clean_float(x: float) -> float:
            return 0.0 if np.isclose(x, 0.0) else x
        
        band_lo = low
        band_hi = high

        band_lo = clean_float(band_lo)
        band_hi = clean_float(band_hi)

        band_dir = f"{band_lo:.1f} to {band_hi:.1f}"
        band_str = f"{band_lo:.1f}_to_{band_hi:.1f}"


        hdf_file = os.path.join(
            base_dir,
            band_dir,
            f"delta_from_{band_str}_origins_1000.h5"
        )

        column_name = f"V_band_{band_lo:.1f}_to_{band_hi:.1f}"

        print(f"Processing band {band_lo:.1f} → {band_hi:.1f}")
        print(f"  HDF5: {hdf_file}")
        print(f"  Column prefix: {column_name}")

        save_average_bulkflow_to_csv(
            hdf_file=hdf_file,
            csv_file=csv_file,
            column_name=column_name,
            mask_type="full",
            key="bulkflow"
        )
    
    plot_bulkflow_from_csv(
        csv_file=csv_file,
        output_folder=base_dir,
        output_file="bulkflow_comparison.png",
        show_markers=False,
        plot_errors=False
    )




# ------------------------------------------------------
# Entry point
# ------------------------------------------------------
if __name__ == "__main__":
    main()