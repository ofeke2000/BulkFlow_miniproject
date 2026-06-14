"""
bulkflow_computation.py
-----------------------
Compute bulk flows for selected origin points.
"""

from typing import Optional

import numpy as np
import pandas as pd
import logging
import time
from scipy.spatial import cKDTree

from src.bulkflow import calculate_bulk_flow_series
from src.classes import AppConfig, Vector3D
from src.masks import MaskMaker
from src.specific_utils import append_bulkflow_results

LOG_PERCENT_CADENCE = 5
PERCENT_FACTOR = 100
ETA_MIN_FACTOR = 60


def compute_bulkflows_for_origins(
    halos_df: pd.DataFrame,
    selected_points: pd.DataFrame,
    tree: cKDTree,
    cfg: AppConfig,
    cf4_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute bulk flows for each selected origin point.

    Parameters:
        halos_df: Full halo catalog
        selected_points: DataFrame of selected origins
        tree: cKDTree for spatial queries
        cfg: Configuration dictionary
        cf4_df: CF4 catalog DataFrame when CF4/uniform masks are requested

    Returns:
        timings: Dictionary with timing information
    """
    output_file = cfg.paths.output_file
    r_min = cfg.bulkflow.min_radius
    r_max = cfg.bulkflow.max_radius
    r_jump = cfg.bulkflow.radii_step
    error_frac = cfg.bulkflow.error_fraction
    sigma_star = cfg.bulkflow.sigma_star
    sigma_min = cfg.bulkflow.sigma_min
    calculation_method = cfg.bulkflow.calculation_method
    mask_mode = cfg.bulkflow.masks or "full"
    cf4_match_radius = cfg.bulkflow.cf4_match_radius
    cf4_match_max_doublings = cfg.bulkflow.cf4_match_max_doublings
    uniform_radius = cfg.bulkflow.uniform_radius
    box_size = cfg.MDPL2.box_size

    if mask_mode in ("cf4", "uniform", "all") and cf4_df is None:
        raise ValueError("CF4 DataFrame is required when bulkflow.masks is 'cf4', 'uniform', or 'all'.")

    use_cf4 = mask_mode in ("cf4", "all")
    use_uniform = mask_mode in ("uniform", "all")
    use_full = mask_mode in ("full", "all")

    mask_maker = MaskMaker(
        halos_df=halos_df,
        tree=tree,
        box_size=box_size,
        cf4_df=cf4_df,
    )

    per_origin_times = []
    n_origins = len(selected_points)

    logging.info("Starting bulk flow computation for selected origins...")

    t0_bulk = time.time()
    for i, (_, row) in enumerate(selected_points.iterrows()):
        origin = Vector3D.from_sequence((row["x"], row["y"], row["z"]))
        origin_id = int(row["rockstarid"])

        if n_origins > 0 and i > 0 and (PERCENT_FACTOR * i / n_origins) % LOG_PERCENT_CADENCE == 0:
            logging.info(f"Processing origin ID {origin_id} at {origin}")
            avg_time = np.mean(per_origin_times)
            eta = avg_time * (n_origins - i)
            logging.info(
                f"=== Processing origin {i}/{n_origins} "
                f"({PERCENT_FACTOR*i/n_origins:.1f}%), "
                f"ETA ~ {eta/ETA_MIN_FACTOR:.1f} min ==="
            )

        t_origin = time.time()

        if use_cf4:
            cf4_mask_df = mask_maker.make_cf4_mask(
                position=origin.to_array(),
                radius=cf4_match_radius,
                max_doublings=cf4_match_max_doublings,
            )
            bf_cf4 = calculate_bulk_flow_series(
                halos_df=cf4_mask_df,
                origin=origin,
                r_max=r_max,
                r_min=r_min,
                r_jumps=r_jump,
                box_size=box_size,
                calculation_method=calculation_method,
                error_frac=error_frac,
                sigma_star=sigma_star,
                sigma_min=sigma_min,
            )
            append_bulkflow_results(
                bf_cf4,
                origin_id=origin_id,
                mask_name="cf4",
                filename=output_file,
            )

        if use_uniform:
            uniform_mask_df = mask_maker.make_uniform_mask(
                position=origin.to_array(),
                radius=uniform_radius,
            )
            bf_uniform = calculate_bulk_flow_series(
                halos_df=uniform_mask_df,
                origin=origin,
                r_max=r_max,
                r_min=r_min,
                r_jumps=r_jump,
                box_size=box_size,
                calculation_method=calculation_method,
                error_frac=error_frac,
                sigma_star=sigma_star,
                sigma_min=sigma_min,
            )
            append_bulkflow_results(
                bf_uniform,
                origin_id=origin_id,
                mask_name="uniform",
                filename=output_file,
            )

        if use_full:
            idx_full = tree.query_ball_point(origin.to_array(), r=r_max)
            full_mask_df = halos_df.iloc[idx_full].copy()
            bf_full = calculate_bulk_flow_series(
                halos_df=full_mask_df,
                origin=origin,
                r_max=r_max,
                r_min=r_min,
                r_jumps=r_jump,
                box_size=box_size,
                calculation_method=calculation_method,
                error_frac=error_frac,
                sigma_star=sigma_star,
                sigma_min=sigma_min,
            )
            append_bulkflow_results(
                bf_full,
                origin_id=origin_id,
                mask_name="full",
                filename=output_file,
            )

        per_origin_times.append(time.time() - t_origin)

    timings = {
        "process_all_origins": time.time() - t0_bulk,
        "mean_origin_time": np.mean(per_origin_times) if per_origin_times else 0.0,
        "min_origin_time": np.min(per_origin_times) if per_origin_times else 0.0,
        "max_origin_time": np.max(per_origin_times) if per_origin_times else 0.0,
    }

    logging.info("All origins processed successfully!")
    logging.info(f"Results saved to {output_file}")

    return timings
