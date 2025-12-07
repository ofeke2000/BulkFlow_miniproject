import yaml
import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path

from MDAnalysis.lib.pkdtree import PeriodicKDTree

# --- Local modules ---
from src.data_loader import load_rockstar_catalog, load_cf4_catalogue
from src.overdensity import compute_overdensity
from src.masks import make_cf4_mask, make_uniform_mask
from src.bulkflow import calculate_bulk_flow_series
from src.specific_utils import append_bulkflow_results


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
    output_file = cfg["paths"]["output_file"]

    box_size = cfg["MDPL2"]["box_size"]
    radius_overdensity = cfg["overdensity"]["radius"]
    n_lowest = cfg["overdensity"]["n_lowest_delta"]

    # bulkflow config
    r_min = cfg["bulkflow"]["r_min"]
    r_max = cfg["bulkflow"]["r_max"]
    r_jump = cfg["bulkflow"]["radii_step"]
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

    logging.info("Loading CF4 catalog...")
    cf4_df = load_cf4_catalogue(cf4_path)

    # ===========================================
    # 3. Build PeriodicKDTree
    # ===========================================
    t0 = time.time()
    logging.info("Building PeriodicKDTree...")
    tree = PeriodicKDTree(halos_df[["x", "y", "z"]].values, boxsize=box_size)

    # ===========================================
    # 4. Compute overdensity
    # ===========================================
    t0 = time.time()
    logging.info("Computing overdensity...")
    halos_df = compute_overdensity(
        df=halos_df,
        radius=radius_overdensity,
        tree=tree,
        box_size=box_size,
        mass_column="mvir"
    )
    
    # Save updated CSV, overwriting the original
    halos_df.to_csv(os.path.join(rockstar_path), index=False)

    delta_column = f"delta_{int(radius_overdensity)}"

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
            radius=1.0,
            max_doublings=4
        )

        uniform_mask_df = make_uniform_mask(
            position=np.array(origin),
            radius=1.0,
            df_halos=halos_df,
            CF4_catalogue=cf4_df,
            tree=tree
        )

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

        per_origin_times.append(time.time() - t_origin)

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
