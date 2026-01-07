import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

#reference line "from .data_loader import load_cf4_catalogue"
from .specific_utils import add_periodic_distance


logging.basicConfig(level=logging.INFO, format="%(message)s")

# Path to your HDF5 file
hdf_file = "/home/ofeke2000/BulkFlow_miniproject/BulkFlow_miniproject/output/bulkflow_results_max250_min_50_jump25_points5.h5"
cf4_catalog = "/home/ofeke2000/BulkFlow_miniproject/BulkFlow_miniproject/data/CF4_Groups (edited).csv"

show_bulkflow_plot = True
show_CF4_Histogram = False


def plot_bulkflow_from_hdf5(
    hdf_file: str,
    output_folder: str,
    key: str = "bulkflow",
    output_file: str = "bulkflow_vs_radius_mean.png"
) -> None:
    """
    Load bulk flow results from an HDF5 file and plot mean U_total vs radius
    for CF4 and theoretical (uniform) masks.

    Parameters
    ----------
    hdf_file : str
        Path to the HDF5 file.
    output_folder : str
        Folder to save the output plot.
    key : str
        HDF5 key containing the bulk flow DataFrame.
    output_file : str
        Output filename for the plot.
    """

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df = pd.read_hdf(hdf_file, key=key)

    # Separate masks
    cf4_df = df[df["mask"] == "cf4"]
    uniform_df = df[df["mask"] == "uniform"]

    # --------------------------------------------------
    # Average over identical radii
    # --------------------------------------------------
    cf4_mean = (
        cf4_df
        .groupby("radius", as_index=False)
        .agg(U_total_mean=("U_total", "mean"))
        .sort_values("radius")
    )

    uniform_mean = (
        uniform_df
        .groupby("radius", as_index=False)
        .agg(U_total_mean=("U_total", "mean"))
        .sort_values("radius")
    )

    # --------------------------------------------------
    # Logging (important sanity check)
    # --------------------------------------------------
    logging.info(
        f"CF4 radii: {len(cf4_mean)} unique values "
        f"(from {len(cf4_df)} total rows)"
    )
    logging.info(
        f"Uniform radii: {len(uniform_mean)} unique values "
        f"(from {len(uniform_df)} total rows)"
    )

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    plt.figure(figsize=(8, 5))

    plt.plot(
        cf4_mean["radius"],
        cf4_mean["U_total_mean"],
        marker="o",
        label="CF4 (mean)"
    )

    plt.plot(
        uniform_mean["radius"],
        uniform_mean["U_total_mean"],
        marker="s",
        label="Uniform (mean)"
    )

    plt.xlabel("Radius [h⁻¹ Mpc]")
    plt.ylabel("⟨U_total⟩ [km/s]")
    plt.title("Mean Bulk Flow vs Radius")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_file)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_distance_histogram(
        data_df, 
        output_folder="plots", 
        output_file="cf4_histogram_lin.png", 
        origin=(0,0,0),
        bins=50
        ):

    #Check for 'distance' column; calculate if missing
    if "distance" not in data_df.columns:
        if all(col in data_df.columns for col in ["x", "y", "z"]):
            # Distance formula: sqrt(x^2 + y^2 + z^2)
            data_df = add_periodic_distance(
                            df=data_df,
                            origin=origin,
                            box_size=1000.0,
                            distance_col="distance"
                        )
        else:
            raise KeyError("The dataframe is missing 'distance' and cannot find 'x, y, z' to calculate it.")

    #Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(data_df["distance"], bins=bins)
    
    plt.xlabel("Distance")
    plt.ylabel("Number of Objects")
    plt.title(output_file.replace("_", " ").replace(".png", ""))
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ensure output folder exists and save plot there
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_file)
    plt.savefig(output_path, dpi=150)
    plt.close()