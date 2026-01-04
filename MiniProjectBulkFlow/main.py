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
from src.bulkflow import calculate_bulk_flow_series
from src.specific_utils import append_bulkflow_results
from src.visualize import plot_bulkflow_from_hdf5, plot_distance_histogram


# ------------------------------------------------------
# Setup logging
# ------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)


# ------------------------------------------------------
# Load YAML configuration
# ------------------------------------------------------
def load_config(path: str = "config.yaml") -> dict:
    logging.info(f"Loading config from {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------
# Main workflow
# ------------------------------------------------------
def main():

    # ----------------------------------------------------------
    # TIMING DICT
    # ----------------------------------------------------------
    timings = {}

    t0_total = time.time()


    # ===========================================
    # 1. Load configuration
    # ===========================================
    t0 = time.time()
    cfg = load_config("config.yaml")

    rockstar_path = cfg["paths"]["rockstar_catalog"]
    cf4_path = cfg["paths"]["cf4_catalog"]
    output_folder = cfg["paths"]["output_folder"]
    output_file = cfg["paths"]["output_file"]

    box_size = cfg["MDPL2"]["box_size"]
    Hubble_Parameter = cfg["MDPL2"]["HubbleParameter"]
    radius_overdensity = int(cfg["overdensity"]["radius"])
    n_lowest = cfg["overdensity"]["n_lowest_delta"]

    # bulkflow config
    r_min = int(cfg["bulkflow"]["min_radius"])
    r_max = int(cfg["bulkflow"]["max_radius"])
    r_jump = int(cfg["bulkflow"]["radii_step"])
    error_frac = cfg["bulkflow"]["error_fraction"]
    sigma_star = cfg["bulkflow"]["sigma_star"]
    sigma_min = cfg["bulkflow"]["sigma_min"]

    logging.info("Configuration loaded successfully.")

    # ===========================================
    # 2. Load catalogs
    # ===========================================
    t0 = time.time()
    logging.info("Loading Rockstar catalog...")
    halos_df = load_rockstar_catalog(rockstar_path)

    # Fit Halos into box [0, box_size)
    halos_df[['x','y','z']] %= box_size
    logging.info("Rockstar catalog loaded and prepared.")

    logging.info("Loading CF4 catalog...")
    cf4_df = load_cf4_catalogue(cf4_path, h=Hubble_Parameter)

    # -------------------------------------------
    # Filter CF4 galaxies inside r_max
    # -------------------------------------------
    n_before = len(cf4_df)

    cf4_df = cf4_df[cf4_df["distance"] <= r_max].copy()

    n_after = len(cf4_df)

    logging.info(
        f"Filtered CF4 catalogue to r <= {r_max:.1f} : "
        f"{n_before} → {n_after} galaxies"
    )

    # ===========================================
    # 3. Build cKDTree
    # ===========================================
    t0 = time.time()
    logging.info("Building cKDTree...")
    tree = cKDTree(halos_df[["x", "y", "z"]].values, boxsize=box_size)
    logging.info("cKDTree built successfully.")
    timings["load_and_prepare_data"] = time.time() - t0

    # ===========================================
    # 4. Compute overdensity
    # ===========================================

    delta_column = f"delta_{radius_overdensity}"

    t0 = time.time()
    logging.info(f"Computing overdensity ({delta_column})...")

    if delta_column not in halos_df.columns:

        halos_df = compute_overdensity(
            df=halos_df,
            radius=radius_overdensity,
            tree=tree,
            box_size=box_size,
            mass_column="mvir"
        )

        halos_df.to_csv(rockstar_path, index=False)

        logging.info(
            f"Overdensity computed and saved "
            f"(Δt = {time.time() - t0:.2f} s)"
        )
    else:
        logging.info(
            f"Column '{delta_column}' already exists — skipping overdensity computation."
        )

    # ===========================================
    # 5. Choose n_lowest halos closest to zero overdensity
    # ===========================================
    t0 = time.time()
    logging.info(f"Selecting {n_lowest} lowest-|delta| halos...")
    halos_df[f"delta_abs_{int(radius_overdensity)}"] = halos_df[delta_column].abs()
    selected_points = halos_df.nsmallest(n_lowest, f"delta_abs_{int(radius_overdensity)}")

    logging.info(f"Selected {len(selected_points)} origin points.")

    # ===========================================
    # 6. Loop over selected points
    # ===========================================
    per_origin_times = []
    t0 = time.time()

    for idx, row in selected_points.iterrows():
        origin = (row["x"], row["y"], row["z"])
        origin_id = int(row["rockstarid"])

        logging.info(f"Processing origin ID {origin_id} at {origin}")

        # ---------------------------------------
        # 6.1 Make masks
        # ---------------------------------------
        cf4_mask_df = make_cf4_mask(
            position=np.array(origin),
            halos_df=halos_df,
            cf4_df=cf4_df,
            tree=tree,
            box_size=box_size,
            radius=5.0,
            max_doublings=5
        )

        plot_distance_histogram(
            data_df=cf4_mask_df,
            output_folder=output_folder,
            output_file="cf4_mask_histogram_lin.png",
            bins=50
        )

        uniform_mask_df = make_uniform_mask(
            position=np.array(origin),
            radius=r_max,
            df_halos=halos_df,
            CF4_catalogue=cf4_df,
            tree=tree
        )

        plot_distance_histogram(
            data_df=uniform_mask_df,
            output_folder=output_folder,
            output_file="uniform_mask_histogram_lin.png",
            bins=50
        )

        t_origin = time.time()
        logging.info(f" Masks created. CF4 mask size: {len(cf4_mask_df)}, Uniform mask size: {len(uniform_mask_df)}")

        # ---------------------------------------
        # 6.2 Compute bulk flow for each mask
        # ---------------------------------------
        bf_cf4 = calculate_bulk_flow_series(
            halos_df=cf4_mask_df,
            origin=origin,
            r_max=r_max,
            r_min=r_min,
            r_jumps=r_jump,
            error_frac=error_frac,
            sigma_star=sigma_star,
            sigma_min=sigma_min
        )

        bf_uniform = calculate_bulk_flow_series(
            halos_df=uniform_mask_df,
            origin=origin,
            r_max=r_max,
            r_min=r_min,
            r_jumps=r_jump,
            error_frac=error_frac,
            sigma_star=sigma_star,
            sigma_min=sigma_min
        )

        logging.info(f" Bulk flows computed. CF4 bulk flow size: {len(bf_cf4)}, Uniform bulk flow size: {len(bf_uniform)}")
        logging.info(f" Bulk flow is {bf_cf4['U_total'].iloc[-1]}")

        # ---------------------------------------
        # 6.3 Save results
        # ---------------------------------------
        append_bulkflow_results(
            bf_cf4,
            origin_id=origin_id,
            mask_name="cf4",
            filename=output_file
        )

        append_bulkflow_results(
            bf_uniform,
            origin_id=origin_id,
            mask_name="uniform",
            filename=output_file
        )

        logging.info(f" Results appended to {output_file}.")

        per_origin_times.append(time.time() - t_origin)

    # ===========================================
    # 7. Visualize results
    # ===========================================

    plot_bulkflow_from_hdf5(
        hdf_file=output_file,
        output_folder=output_folder,
        key="bulkflow",
        output_file="bulkflow_vs_radius.png"
    )

    timings["process_all_origins"] = time.time() - t0
    timings["mean_origin_time"] = np.mean(per_origin_times)
    timings["min_origin_time"] = np.min(per_origin_times)
    timings["max_origin_time"] = np.max(per_origin_times)

    # ==========================================================
    # END — FINAL TIMING SUMMARY
    # ==========================================================
    timings["total_runtime"] = time.time() - t0_total

    logging.info("\n" + "=" * 60)
    logging.info("TIMING SUMMARY")
    logging.info("=" * 60)

    for key, value in timings.items():
        logging.info(f"{key:25s} : {value:8.3f} sec")

    logging.info("=" * 60)
    logging.info(f"TOTAL RUNTIME : {timings['total_runtime']:.3f} sec")
    logging.info("=" * 60)

    logging.info("All origins processed successfully!")
    logging.info(f"Results saved to {output_file}")


# ------------------------------------------------------
# Entry point
# ------------------------------------------------------
if __name__ == "__main__":
    main()
