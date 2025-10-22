# main.py
import os
import pandas as pd
from utils import setup_logger, ensure_dir, timing
from data_loader import load_data
from overdensity import compute_delta5
from masks import create_cf4_mask, create_uniform_mask
from experiment import run_bulkflow_experiment
from visualize import (
    scatter_overdensity,
    projection_overdensity,
    histogram_delta5,
)
from utils import save_dataframe


# ================================================================
# CONFIGURATION
# ================================================================

INPUT_CSV = os.path.expanduser(
    "~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid-1_mvir12/mdpl2_rockstar_125_pid-1_mvir12.csv"
)
OUTPUT_DIR = os.path.expanduser(
    "~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid-1_mvir12/bulk_flow_radius_series"
)

LOG_PATH = os.path.join(OUTPUT_DIR, "run.log")
ensure_dir(OUTPUT_DIR)
logger = setup_logger(LOG_PATH)


# ================================================================
# MAIN PIPELINE
# ================================================================

@timing
def main():
    logger.info("=== Bulk Flow Analysis Pipeline Started ===")

    # ------------------------------------------------------------
    # 1. Load simulation halo data
    # ------------------------------------------------------------
    usecols = ["mvir", "rvir", "rs", "rockstarid", "pid", "x", "y", "z", "vx", "vy", "vz"]
    df = load_data(INPUT_CSV, usecols=usecols)
    logger.info(f"Loaded {len(df)} halos from Rockstar catalogue.")

    # ------------------------------------------------------------
    # 2. Compute overdensity δ₅ and find the lowest δ₅ halos
    # ------------------------------------------------------------
    logger.info("Computing local overdensity δ₅ ...")
    df = compute_delta5(df, radius=5.0)
    save_dataframe(df[["rockstarid", "delta_5"]], os.path.join(OUTPUT_DIR, "delta5_table.csv"))

    histogram_delta5(df, OUTPUT_DIR)
    scatter_overdensity(df, OUTPUT_DIR)
    projection_overdensity(df, OUTPUT_DIR, plane='xy')

    df_sorted = df.sort_values(by="delta_5", key=abs).head(50)
    start_points = df_sorted[["x", "y", "z"]].values
    logger.info(f"Selected 50 lowest-|δ₅| halos as starting points.")

    # ------------------------------------------------------------
    # 3. Create masks
    # ------------------------------------------------------------
    logger.info("Creating CF4 and uniform masks ...")
    cf4_mask = create_cf4_mask(df)
    uniform_mask = create_uniform_mask(df, size=len(cf4_mask))

    # Save mask info (optional, for reproducibility)
    save_dataframe(cf4_mask, os.path.join(OUTPUT_DIR, "cf4_mask.csv"))
    save_dataframe(uniform_mask, os.path.join(OUTPUT_DIR, "uniform_mask.csv"))

    # ------------------------------------------------------------
    # 4. Run bulk flow experiment (ML-based) for both masks
    # ------------------------------------------------------------
    logger.info("Running bulk flow experiment with both masks ...")
    radii = list(range(5, 255, 5))
    error_fraction = 0.2

    # CF4 mask
    logger.info("Running experiment with CF4 mask...")
    run_bulkflow_experiment(
        df=df,
        start_points=start_points,
        mask=cf4_mask,
        radii=radii,
        error_fraction=error_fraction,
        output_path=os.path.join(OUTPUT_DIR, "bulkflow_CF4.csv"),
        parallel=True
    )

    # Uniform mask
    logger.info("Running experiment with uniform mask...")
    run_bulkflow_experiment(
        df=df,
        start_points=start_points,
        mask=uniform_mask,
        radii=radii,
        error_fraction=error_fraction,
        output_path=os.path.join(OUTPUT_DIR, "bulkflow_uniform.csv"),
        parallel=True
    )

    # ------------------------------------------------------------
    # 5. Done
    # ------------------------------------------------------------
    logger.info("All experiments completed successfully.")
    logger.info(f"Results saved in: {OUTPUT_DIR}")
    logger.info("=== Bulk Flow Analysis Finished ===")


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    main()
