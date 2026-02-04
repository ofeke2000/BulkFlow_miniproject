import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import logging

import yaml

from .data_loader import load_cf4_catalogue, load_rockstar_catalog
from .specific_utils import add_periodic_distance
from .theoretical_bulkflow import theoretical_bulkflow_colossus


logging.basicConfig(level=logging.INFO, format="%(message)s")

# ------------------------------------------------------
# Load YAML configuration
# ------------------------------------------------------
def load_config(path: str = "config.yaml") -> dict:
    logging.info(f"Loading config from {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
def visualize ():
    
    plot_bulkflow=False

    plot_histogram_bool=True
    import_MDPL2=True
    import_CF4=False

    # ===========================================
    # 1. Load configuration
    # ===========================================
    cfg = load_config("config.yaml")

    output_folder = cfg["paths"]["output_folder"]
    output_file = cfg["paths"]["output_file"]
    
    n_origins = cfg["origin_configs"]["number_of_origins"]
    radius_overdensity = int(cfg["origin_configs"]["local_overdensity_radius"])
    radius_bulkflow = int(cfg["origin_configs"]["local_bulkflow_radius"])

    # ===========================================
    # 2. Plot whatever is requested
    # ===========================================

    if plot_bulkflow:
        plot_bulkflow_from_hdf5(
            hdf_file=output_file,
            output_folder=output_folder,
            key="bulkflow",
            output_file=f"bulkflow_vs_radius_{n_origins}_points_With_Var.png",
            plot_theory=True,
            use_mean_amplitude=True,
            plot_variance_band=True,
            variance_alpha=0.25,
            plot_all_curves=False,
            show_markers=False,
        )

    if plot_histogram_bool:

        if import_MDPL2:

            path=cfg["paths"]["rockstar_catalog"]
            rockstar_df = load_rockstar_catalog(
                path=path
            )

            key = "mvir"

            delta_column = f"delta_{int(radius_overdensity)}"
            virgo_column = "near_virgo"
            bulkflow_column = f"bulkflow_{int(radius_bulkflow)}"

            # mask = (
            #     (rockstar_df[delta_column].between(- 1e-4, 1e-4)) #&
            #     #(rockstar_df[bulkflow_column].between(400.0, 600.0)) &
            #     #(rockstar_df[virgo_column] > 0)
            # )

            # candidates = rockstar_df.loc[mask]
            
            rockstar_df['mvir'] = np.log10(rockstar_df['mvir'])
            data_df = rockstar_df 

        logging.info(f"Plotting histogram for column: {key}")
        logging.info(f"Number of entries: {len(data_df)} out of {len(rockstar_df)}")
        

        plot_histogram(
        data_df=data_df, 
        output_folder=output_folder, 
        output_file="Mass Histogram.png", 
        key=key,
        origin=(0,0,0),
        bins=20,
        log_axis="y"
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
    full_df = df[df["mask"] == "full"]

    marker_cf4 = "o" if show_markers else None
    marker_uniform = "s" if show_markers else None
    marker_full = "D" if show_markers else None

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

        for origin_id, d in full_df.groupby("origin_id"):
            plt.plot(
                d["radius"],
                d["U_total"],
                color="tab:green",
                alpha=0.25,
                linewidth=1,
            )

        cf4_label = "CF4 (all origins)"
        uniform_label = "Uniform (all origins)"
        full_label = "Full (all origins)"

        # Dummy lines for legend
        plt.plot([], [], color="tab:blue", label=cf4_label)
        plt.plot([], [], color="tab:orange", label=uniform_label)
        plt.plot([], [], color="tab:green", label=full_label)

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
        full_stats = aggregate_stats(full_df)

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

        plt.plot(
            full_stats["radius"],
            full_stats["U_mean"],
            marker=marker_full,
            label="Full (mean)",
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

            plt.fill_between(
                full_stats["radius"],
                full_stats["U_mean"] - full_stats["U_std"],
                full_stats["U_mean"] + full_stats["U_std"],
                alpha=variance_alpha,
                label="Full ±1σ",
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



def plot_histogram(
        data_df, 
        output_folder="plots", 
        output_file="cf4_histogram_lin.png", 
        key="distance",
        origin=(0,0,0),
        bins=50,
        log_axis= False   # NEW
        ):

    # Check for 'distance' column; calculate if missing
    if key == "distance" and key not in data_df.columns:
        if all(col in data_df.columns for col in ["x", "y", "z"]):
            data_df = add_periodic_distance(
                df=data_df,
                origin=origin,
                box_size=1000.0,
                distance_col="distance"
            )
        else:
            raise KeyError(
                "The dataframe is missing 'distance' and cannot find 'x, y, z' to calculate it."
            )
    

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(data_df[key], bins=bins)

    if log_axis == "y":
        plt.yscale("log")
    elif log_axis == "x":
        plt.xscale("log")
    elif log_axis == "all":
        plt.yscale("log")
        plt.xscale("log")

    plt.xlabel(key.replace("_", " ").capitalize())
    plt.ylabel("Number of Objects")
    plt.title(output_file.replace("_", " ").replace(".png", ""))
    plt.grid(True, linestyle='--', alpha=0.7)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_file)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_simulation_slice_heatmap(
    df: pd.DataFrame,
    slice_axis: str = "z",
    slice_min: float = 400.0,
    slice_max: float = 500.0,
    proj_axes: tuple = ("x", "y"),
    gridsize: int = 500,
    cmap: str = "magma",
    output_folder: str = "heatmap_slices",
    output_file: str | None = None,
    dpi: int = 300,
) -> None:
    """
    Plot a hexbin heatmap of a thin slice of the simulation box.

    Parameters
    ----------
    df : pandas.DataFrame
        Halo catalogue with columns x, y, z.
    slice_axis : str
        Axis to slice on ("x", "y", or "z").
    slice_min, slice_max : float
        Slice boundaries (e.g. 400 < z < 500).
    proj_axes : tuple
        Two axes to project onto, e.g. ("x", "y"), ("x", "z").
    gridsize : int
        Hexbin resolution.
    cmap : str
        Matplotlib colormap.
    output_folder : str
        Directory to save the plot.
    output_file : str or None
        Output filename. If None, auto-generated.
    dpi : int
        Output image DPI.
    """

    # ---------------------------------------
    # Safety checks
    # ---------------------------------------
    for col in (*proj_axes, slice_axis):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

    # ---------------------------------------
    # Slice the data
    # ---------------------------------------
    slice_df = df[
        (df[slice_axis] >= slice_min) &
        (df[slice_axis] < slice_max)
    ]

    n_halos = len(slice_df)

    # ---------------------------------------
    # Plot
    # ---------------------------------------
    fig, ax = plt.subplots(figsize=(8, 8))

    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    hb = ax.hexbin(
        slice_df[proj_axes[0]],
        slice_df[proj_axes[1]],
        gridsize=gridsize,
        cmap=cmap,
    )

    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("Counts", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel(proj_axes[0], color="white")
    ax.set_ylabel(proj_axes[1], color="white")
    ax.set_title(
        f"{slice_axis} ∈ [{slice_min}, {slice_max}) | Halos: {n_halos}",
        color="white"
    )

    ax.tick_params(colors="white")

    plt.tight_layout()

    # ---------------------------------------
    # Save
    # ---------------------------------------
    os.makedirs(output_folder, exist_ok=True)

    if output_file is None:
        output_file = (
            f"heatmap_{proj_axes[0]}_{proj_axes[1]}_"
            f"{slice_axis}_{slice_min}_{slice_max}.png"
        )

    output_path = os.path.join(output_folder, output_file)
    plt.savefig(output_path, dpi=dpi)
    plt.close()


#=======================================================
# plot 
#======================================================
def plot_bulkflow_from_csv(
    csv_file: str,
    output_folder: str,
    output_file: str = "bulkflow_vs_radius_overdensity.png",
    plot_theory: bool = True,
    plot_errors: bool = False,
    error_alpha: float = 0.25,
    show_markers: bool = False,
):
    """
    Plot averaged bulk flow vs radius for multiple origin overdensity bands
    from a unified CSV file.

    Parameters
    ----------
    plot_errors : bool
        If True, plot ±1σ shaded bands using the *_std columns.
    show_markers : bool
        If True, show markers on the mean curves.
    """

    # --------------------------------------------------
    # Load CSV
    # --------------------------------------------------
    df = pd.read_csv(csv_file)

    radii = df["radius"].values

    # --------------------------------------------------
    # Identify overdensity bands from column names
    # --------------------------------------------------
    band_pattern = re.compile(r"V_band_(-?\d+\.\d)_to_(-?\d+\.\d)_mean")

    bands = []
    for col in df.columns:
        m = band_pattern.match(col)
        if m:
            low = float(m.group(1))
            high = float(m.group(2))
            bands.append((low, high, col))

    if not bands:
        raise ValueError("No overdensity band columns found in CSV.")

    # Sort bands by overdensity
    bands.sort(key=lambda x: x[0])

    # --------------------------------------------------
    # Color mapping: overdensity → color
    # --------------------------------------------------
    norm = mpl.colors.TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.5)
    cmap = plt.cm.coolwarm

    marker = "o" if show_markers else None

    plt.figure(figsize=(8, 5))

    # --------------------------------------------------
    # Plot each overdensity band
    # --------------------------------------------------
    for low, high, mean_col in bands:

        std_col = mean_col.replace("_mean", "_std")

        color = cmap(norm(0.5 * (low + high)))

        plt.plot(
            radii,
            df[mean_col],
            color=color,
            linewidth=2,
            marker=marker,
        )

        if plot_errors and std_col in df.columns:
            plt.fill_between(
                radii,
                df[mean_col] - df[std_col],
                df[mean_col] + df[std_col],
                color=color,
                alpha=error_alpha,
            )

    # --------------------------------------------------
    # Theory
    # --------------------------------------------------
    if plot_theory and "U_mean_theory" in df.columns:
        plt.plot(
            radii,
            df["U_mean_theory"],
            "k--",
            linewidth=2.5,
            label=r"$\Lambda$CDM $\langle |U| \rangle$",
        )

    # --------------------------------------------------
    # Colorbar (instead of legend spam)
    # --------------------------------------------------
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax = plt.gca()
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(r"Origin overdensity $\delta$")

    # --------------------------------------------------
    # Final styling
    # --------------------------------------------------
    plt.xlabel(r"Radius [$h^{-1}$ Mpc]")
    plt.ylabel(r"$\langle |U| \rangle$ [km/s]")
    plt.title("Bulk Flow vs Radius\nConditioned on Origin Overdensity (Full Mask)")
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, output_file), dpi=150)
    plt.close()


# ------------------------------------------------------
# Entry point
# ------------------------------------------------------
if __name__ == "__main__":
    visualize()