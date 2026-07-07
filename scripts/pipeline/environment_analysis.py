"""
environment_analysis.py
-----------------------
Compute environmental properties for halos: overdensity, Virgo proximity, local bulk flow.
"""

import logging
import pandas as pd
from scipy.spatial import cKDTree

from src.physics.overdensity import OverdensityCalculator
from src.physics.near_virgo import near_virgo
from src.physics.bulkflow import calculate_local_bulkflow
from src.classes import AppConfig


def compute_environmental_tests(halos_df: pd.DataFrame, tree: cKDTree, cfg: AppConfig) -> pd.DataFrame:
    """
    Compute environmental tests for halos.

    Parameters:
        halos_df: Halo catalog DataFrame
        tree: cKDTree for spatial queries
        cfg: AppConfig instance

    Returns:
        Updated halos_df with environmental columns
    """
    box_size = cfg.MDPL2.box_size
    radius_overdensity = cfg.origin_configs.local_overdensity_radius
    radius_bulkflow = cfg.origin_configs.local_bulkflow_radius
    use_virgo = cfg.origin_configs.use_virgo_criteria

    delta_column = f"delta_{int(radius_overdensity)}"
    virgo_column = "near_virgo"
    bulkflow_column = f"bulkflow_{int(radius_bulkflow)}"

    # Check which tests are missing
    missing_tests = []
    if delta_column not in halos_df.columns:
        missing_tests.append("overdensity")
    if use_virgo and virgo_column not in halos_df.columns:
        missing_tests.append("virgo")
    if bulkflow_column not in halos_df.columns:
        missing_tests.append("bulkflow")

    if not missing_tests:
        logging.info("All environmental test columns already exist — skipping.")
        return halos_df

    logging.info(f"Computing missing tests: {', '.join(missing_tests)}")

    # Compute overdensity
    if "overdensity" in missing_tests:
        logging.info(f"Computing overdensity ({delta_column})...")
        calculator = OverdensityCalculator(
            radius=radius_overdensity,
            box_size=box_size,
            tree=tree,
        )
        halos_df = calculator.compute(halos_df)
        logging.info("Overdensity computed.")

    # Compute Virgo proximity
    if "virgo" in missing_tests:
        logging.info("Running Virgo environment test...")
        halos_df = near_virgo(
            df=halos_df,
            box_size=box_size,
            virgo_cfg=cfg.virgo_test,
        )
        logging.info("Virgo test completed.")

    # Compute local bulk flow
    if "bulkflow" in missing_tests:
        logging.info("Computing local bulk flow environment...")
        halos_df = calculate_local_bulkflow(
            df=halos_df,
            tree=tree,
            radius=radius_bulkflow,
            velocity_columns=("vx", "vy", "vz"),
        )
        logging.info("Local bulk flow test completed.")

    return halos_df
