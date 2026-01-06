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
    output_file: str = "bulkflow_vs_radius.png"
) -> None:
    """
    Load bulk flow results from an HDF5 file and plot U_total vs radius
    for CF4 and uniform masks.

    Parameters
    ----------
    hdf_file : str
        Path to the HDF5 file.
    key : str
        HDF5 key containing the bulk flow DataFrame.
    output_file : str
        Output filename for the plot.
    show_bulkflow_plot : bool
        Whether to generate the plot.
    """

    # Load the HDF5 results
    df = pd.read_hdf(hdf_file, key=key)

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
    plt.figure(figsize=(8, 5))
    plt.plot(cf4_df["radius"], cf4_df["U_total"],
             marker='o', label='CF4 Mask')
    plt.plot(uniform_df["radius"], uniform_df["U_total"],
             marker='s', label='Uniform Mask')

    plt.xlabel("Radius [h⁻¹ Mpc]")
    plt.ylabel("Average U_total [km/s]")
    plt.title("Average Bulk Flow vs Radius")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Ensure output folder exists and save plot there
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