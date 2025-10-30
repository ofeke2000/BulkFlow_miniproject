# main.py
import os
import pandas as pd
import yaml
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

CONFIG_PATH = "config.yaml"

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


config = load_config(CONFIG_PATH)

INPUT_CSV = os.path.expanduser(config["input_csv"])
OUTPUT_DIR = os.path.expanduser(config["output_dir"])
ensure_dir(OUTPUT_DIR)
LOG_PATH = os.path.join(OUTPUT_DIR, "run.log")
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
    df = load_data(INPUT_CSV, usecols=config["columns"])
    logger.info(f"Loaded {len(df)} halos from Rockstar catalogue.")

    # ------------------------------------------------------------
    # 2. Compute overdensity δ₅ and find the lowest δ₅ halos
    # ------------------------------------------------------------
    logger.info("Computing local overdensity δ₅ ...")
    radius = config["overdensity"]["radius"]
    df = compute_delta5(df, radius=radius)
    save_dataframe(df[["rockstarid", "delta_5"]], os.path.join(OUTPUT_DIR, "delta5_table.csv"))

    histogram_delta5(df, OUTPUT_DIR)
    scatter_overdensity(df, OUTPUT_DIR)
    projection_overdensity(df, OUTPUT_DIR, plane=config["visualization"]["projection_plane"])

    n_lowest = config["experiment"]["n_lowest_delta"]
    df_sorted = df.sort_values(by="delta_5", key=abs).head(n_lowest)
    start_points = df_sorted[["x", "y", "z"]].values
    logger.info(f"Selected {n_lowest} lowest-|δ₅| halos as starting points.")

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
    radii = list(range(
        config["experiment"]["min_radius"],
        config["experiment"]["max_radius"] + config["experiment"]["radii_step"],
        config["experiment"]["radii_step"]
    ))

    error_fraction=config["experiment"]["error_fraction"]
    parallel=config["experiment"]["parallel"]

    # CF4 mask
    logger.info("Running experiment with CF4 mask...")
    run_bulkflow_experiment(
        df=df,
        start_points=start_points,
        mask=cf4_mask,
        radii=radii,
        error_fraction=error_fraction,
        output_path=os.path.join(OUTPUT_DIR, "bulkflow_CF4.csv"),
        parallel=parallel
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
        parallel=parallel
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
