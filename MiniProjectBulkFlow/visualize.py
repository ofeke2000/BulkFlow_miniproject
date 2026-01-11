import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

import yaml

#reference line "from .data_loader import load_cf4_catalogue"
from .src.specific_utils import add_periodic_distance
from .src.theoretical_bulkflow import theoretical_bulkflow_colossus


logging.basicConfig(level=logging.INFO, format="%(message)s")

# ------------------------------------------------------
# Load YAML configuration
# ------------------------------------------------------
def load_config(path: str = "config.yaml") -> dict:
    logging.info(f"Loading config from {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def visualize (plot_bulkflow: bool = False,
               ):

    # ===========================================
    # 1. Load configuration
    # ===========================================
    cfg = load_config("config.yaml")

    output_folder = cfg["paths"]["output_folder"]
    output_file = cfg["paths"]["output_file"]
    n_origins = cfg["origin_configs"]["number_of_origins"]

    # ===========================================
    # 2. Plot bulk flow results
    # ===========================================

    if plot_bulkflow:
        plot_bulkflow_from_hdf5(
            hdf_file=output_file,
            output_folder=output_folder,
            key="bulkflow",
            output_file=f"bulkflow_vs_radius_{n_origins}_points.png",
            plot_theory=True,
            use_mean_amplitude=True,
            plot_variance_band=True,
            variance_alpha=0.25
        )





def plot_bulkflow_from_hdf5(
    hdf_file: str,
    output_folder: str,
    key: str = "bulkflow",
    output_file: str = "bulkflow_vs_radius.png",
    plot_theory: bool = True,
    use_mean_amplitude: bool = True,
    plot_variance_band: bool = False,
    variance_alpha: float = 0.25,
    plot_all_curves: bool = False,
    show_markers: bool = True,
) -> None:
    """
    Plot bulk flow results from an HDF5 file.

    Options
    -------
    plot_all_curves : bool
        If True, plot all individual bulk-flow curves (no averaging).
    show_markers : bool
        If False, suppress markers (dots) on curves.
    """

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df = pd.read_hdf(hdf_file, key=key)

    cf4_df = df[df["mask"] == "cf4"]
    uniform_df = df[df["mask"] == "uniform"]

    marker_cf4 = "o" if show_markers else None
    marker_uniform = "s" if show_markers else None

    plt.figure(figsize=(8, 5))

    # ==================================================
    # OPTION 1: plot ALL curves (no averaging)
    # ==================================================
    if plot_all_curves:

        for origin_id, d in cf4_df.groupby("origin_id"):
            plt.plot(
                d["radius"],
                d["U_total"],
                color="tab:blue",
                alpha=0.25,
                linewidth=1,
            )

        for origin_id, d in uniform_df.groupby("origin_id"):
            plt.plot(
                d["radius"],
                d["U_total"],
                color="tab:orange",
                alpha=0.25,
                linewidth=1,
            )

        cf4_label = "CF4 (all origins)"
        uniform_label = "Uniform (all origins)"

        # Dummy lines for legend
        plt.plot([], [], color="tab:blue", label=cf4_label)
        plt.plot([], [], color="tab:orange", label=uniform_label)

    # ==================================================
    # OPTION 2: mean + variance bands (default behavior)
    # ==================================================
    else:

        def aggregate_stats(d):
            return (
                d.groupby("radius", as_index=False)
                 .agg(
                     U_mean=("U_total", "mean"),
                     U_std=("U_total", "std"),
                     N=("U_total", "count"),
                 )
                 .sort_values("radius")
            )

        cf4_stats = aggregate_stats(cf4_df)
        uniform_stats = aggregate_stats(uniform_df)

        plt.plot(
            cf4_stats["radius"],
            cf4_stats["U_mean"],
            marker=marker_cf4,
            label="CF4 (mean)",
        )

        plt.plot(
            uniform_stats["radius"],
            uniform_stats["U_mean"],
            marker=marker_uniform,
            label="Uniform (mean)",
        )

        if plot_variance_band:
            plt.fill_between(
                cf4_stats["radius"],
                cf4_stats["U_mean"] - cf4_stats["U_std"],
                cf4_stats["U_mean"] + cf4_stats["U_std"],
                alpha=variance_alpha,
                label="CF4 ±1σ",
            )

            plt.fill_between(
                uniform_stats["radius"],
                uniform_stats["U_mean"] - uniform_stats["U_std"],
                uniform_stats["U_mean"] + uniform_stats["U_std"],
                alpha=variance_alpha,
                label="Uniform ±1σ",
            )

    # --------------------------------------------------
    # Theory
    # --------------------------------------------------
    if plot_theory:
        radii = np.sort(df["radius"].unique())
        sigma_v = theoretical_bulkflow_colossus(radii=radii)

        if use_mean_amplitude:
            U_theory = np.sqrt(8 / (3 * np.pi)) * sigma_v
            theory_label = r"$\Lambda$CDM $\langle |U| \rangle$"
        else:
            U_theory = sigma_v
            theory_label = r"$\Lambda$CDM $\sigma_v$"

        plt.plot(
            radii,
            U_theory,
            "--",
            linewidth=2,
            label=theory_label,
        )

    # --------------------------------------------------
    # Final styling
    # --------------------------------------------------
    plt.xlabel(r"Radius [$h^{-1}$ Mpc]")
    plt.ylabel(r"$|U|$ [km/s]")
    plt.title("Bulk Flow vs Radius")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, output_file), dpi=150)
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



# ------------------------------------------------------
# Entry point
# ------------------------------------------------------
if __name__ == "__visualise__":
    plot_bulkflow = True
    visualize(plot_bulkflow= plot_bulkflow)