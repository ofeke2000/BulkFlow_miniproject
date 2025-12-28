import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

from .data_loader import load_cf4_catalogue


logging.basicConfig(level=logging.INFO, format="%(message)s")

# Path to your HDF5 file
hdf_file = "/home/ofeke2000/BulkFlow_miniproject/BulkFlow_miniproject/output/bulkflow_results_max250_min_50_jump25_points5.h5"
cf4_catalog = "/home/ofeke2000/BulkFlow_miniproject/BulkFlow_miniproject/data/CF4_Groups (edited).csv"

show_bulkflow_plot = True
show_CF4_Histogram = False


if show_bulkflow_plot == True:


    # Load the HDF5 results
    df = pd.read_hdf(hdf_file, key="bulkflow")

    # Separate masks
    cf4_df = df[df["mask"] == "cf4"]
    uniform_df = df[df["mask"] == "uniform"]

    # Log the number of points
    logging.info(f"Number of points for CF4 mask: {len(cf4_df)}")
    logging.info(f"Number of points for uniform mask: {len(uniform_df)}")

    # Log U_total values
    logging.info(f"CF4 U_total values: {cf4_df['U_total'].values}")
    logging.info(f"Uniform U_total values: {uniform_df['U_total'].values}")


    # Plotting
    plt.figure(figsize=(8,5))
    plt.plot(cf4_df["radius"], cf4_df["U_total"], marker='o', label='CF4 Mask')
    plt.plot(uniform_df["radius"], uniform_df["U_total"], marker='s', label='Uniform Mask')
    plt.xlabel("Radius [h⁻¹ Mpc]")
    plt.ylabel("Average U_total [km/s]")
    plt.title("Average Bulk Flow vs Radius")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("bulkflow_vs_radius.png", dpi=150)
    plt.close()


if show_CF4_Histogram == True:

    cf4_df = load_cf4_catalogue(cf4_catalog)

    plt.figure()
    plt.hist(cf4_df["distance"], bins=50)
    plt.xlabel("Distance")
    plt.ylabel("Number of CF4 groups")
    plt.title("CF4 Catalogue Distance Distribution")
    plt.grid(True)
    plt.savefig("cf4_histogram_lin.png", dpi=150)
    plt.close()