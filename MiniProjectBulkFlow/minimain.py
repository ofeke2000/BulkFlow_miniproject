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

def plot():

    rockstar_path = os.path.expanduser(
        "/home/ofeke2000/BulkFlow_miniproject/BulkFlow_miniproject/data/mdpl2_rockstar_125_pid-1_mvir12.csv"
    )

    halos_df = load_rockstar_catalog(rockstar_path)

    box_size = 1000.0  # Mpc/h

    # Overdensity cuts
    overdensity_lower_cut = 100
    overdensity_upper_cut = 200
    delta_column = "delta_3"

    #Mass cuts
    mass_upper_cut = 16                # h^-1 Msun
    mass_lower_cut = 12                # h^-1 Msun
    halos_df['mvir'] = np.log10(halos_df['mvir'].astype(float))

    #velocity cuts
    bulkflow_upper_cut = 3500.0  # km/s
    bulkflow_lower_cut = 2500.0   # km/s
    bulkflow_column = "bulkflow_3"

    # Virgo criteria
    virgo_column = "near_virgo"
    

    # Fit Halos into box [0, box_size)
    # halos_df[['x','y','z']] %= box_size

    mask = (
        # (halos_df["mvir"].between(mass_lower_cut, mass_upper_cut)) #&
        # (halos_df[delta_column].between(overdensity_lower_cut, overdensity_upper_cut)) #&
        (halos_df[bulkflow_column].between(bulkflow_lower_cut, bulkflow_upper_cut)) #&
        # (halos_df[virgo_column] > 0)
    )

    candidates = halos_df.loc[mask]
    candidates_df = halos_df.loc[candidates.index].copy()
    print(f"Number of candidate origins after mass cut: {len(candidates_df)}")

    plot_histogram(
        data_df=candidates_df, 
        output_folder= "/home/ofeke2000/BulkFlow_miniproject/BulkFlow_miniproject/output/local_bulkflow_histograms/", 
        output_file=f"local_bulkflow_histogram_[{bulkflow_lower_cut:.0f},{bulkflow_upper_cut:.0f}]_with_{len(candidates_df)}_remaining_of_{len(halos_df)}_total.png", 
        key=bulkflow_column,
        origin=(0,0,0),
        bins=50,
        log_axis="none"
        )


# ------------------------------------------------------
# Entry point
# ------------------------------------------------------
if __name__ == "__main__":

    if do_main := False:
        main()

    if do_plot := True:
        plot()