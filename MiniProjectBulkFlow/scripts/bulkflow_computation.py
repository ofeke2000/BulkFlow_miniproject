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
from src.masks import make_cf4_mask, make_uniform_mask
from src.specific_utils import append_bulkflow_results


def compute_bulkflows_for_origins(
    halos_df: pd.DataFrame,
    selected_points: pd.DataFrame,
    tree: cKDTree,
    cfg: dict,
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
    output_file = cfg["paths"]["output_file"]
    r_min = cfg["bulkflow"]["min_radius"]
    r_max = cfg["bulkflow"]["max_radius"]
    r_jump = cfg["bulkflow"]["radii_step"]
    error_frac = cfg["bulkflow"]["error_fraction"]
    sigma_star = cfg["bulkflow"]["sigma_star"]
    sigma_min = cfg["bulkflow"]["sigma_min"]
    calculation_method = cfg["bulkflow"]["calculation_method"]
    mask_mode = cfg["bulkflow"].get("masks", cfg.get("mask_type", "full"))
    cf4_match_radius = cfg["bulkflow"].get("cf4_match_radius", 5.0)
    cf4_match_max_doublings = cfg["bulkflow"].get("cf4_match_max_doublings", 5)
    uniform_radius = cfg["bulkflow"].get("uniform_radius", r_max)
    box_size = cfg["MDPL2"]["box_size"]

    if mask_mode in ("cf4", "uniform", "all") and cf4_df is None:
        raise ValueError("CF4 DataFrame is required when bulkflow.masks is 'cf4', 'uniform', or 'all'.")

    use_cf4 = mask_mode in ("cf4", "all")
    use_uniform = mask_mode in ("uniform", "all")
    use_full = mask_mode in ("full", "all")

    per_origin_times = []
    n_origins = len(selected_points)

    logging.info("Starting bulk flow computation for selected origins...")

    t0_bulk = time.time()
    for i, (_, row) in enumerate(selected_points.iterrows()):
        origin = (row["x"], row["y"], row["z"])
        origin_id = int(row["rockstarid"])

        if n_origins > 0 and i > 0 and (100 * i / n_origins) % 5 == 0:
            logging.info(f"Processing origin ID {origin_id} at {origin}")
            avg_time = np.mean(per_origin_times)
            eta = avg_time * (n_origins - i)
            logging.info(f"=== Processing origin {i}/{n_origins} ({100*i/n_origins:.1f}%), ETA ~ {eta/60:.1f} min ===")

        t_origin = time.time()

        if use_cf4:
            cf4_mask_df = make_cf4_mask(
                position=np.array(origin),
                halos_df=halos_df,
                cf4_df=cf4_df,
                tree=tree,
                box_size=box_size,
                radius=cf4_match_radius,
                max_doublings=cf4_match_max_doublings,
            )
            bf_cf4 = calculate_bulk_flow_series(
                halos_df=cf4_mask_df,
                origin=origin,
                r_max=r_max,
                r_min=r_min,
                r_jumps=r_jump,
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
            uniform_mask_df = make_uniform_mask(
                position=np.array(origin),
                radius=uniform_radius,
                df_halos=halos_df,
                CF4_catalogue=cf4_df,
                tree=tree,
            )
            bf_uniform = calculate_bulk_flow_series(
                halos_df=uniform_mask_df,
                origin=origin,
                r_max=r_max,
                r_min=r_min,
                r_jumps=r_jump,
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
            idx_full = tree.query_ball_point(np.array(origin), r=r_max)
            full_mask_df = halos_df.iloc[idx_full].copy()
            bf_full = calculate_bulk_flow_series(
                halos_df=full_mask_df,
                origin=origin,
                r_max=r_max,
                r_min=r_min,
                r_jumps=r_jump,
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