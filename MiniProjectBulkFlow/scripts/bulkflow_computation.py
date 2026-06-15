"""
bulkflow_computation.py
-----------------------
Compute bulk flows for selected origin points and write results to a single
self-describing netCDF file via BulkFlowDataset.
"""

import subprocess
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import logging
from scipy.spatial import cKDTree

from src.bulkflow import calculate_bulk_flow_series
from src.classes import AppConfig, Vector3D
from src.masks import MaskMaker
from src.data.bulkflow_dataset import BulkFlowDataset, BulkFlowResult

LOG_PERCENT_CADENCE = 5
PERCENT_FACTOR = 100
ETA_MIN_FACTOR = 60


def _build_dataset_attrs(cfg: AppConfig) -> dict:
    """Assemble provenance + configuration attributes for the Dataset."""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        git_commit = "unknown"

    oc = cfg.origin_configs
    bf = cfg.bulkflow

    return {
        "selection_variable": cfg.postprocessing.selection_variable,
        "sigma_star": bf.sigma_star,
        "sigma_min": bf.sigma_min,
        "error_fraction": bf.error_fraction,
        "calculation_method": bf.calculation_method,
        "box_size": cfg.MDPL2.box_size,
        "min_radius": bf.min_radius,
        "max_radius": bf.max_radius,
        "radii_step": bf.radii_step,
        "cf4_match_radius": bf.cf4_match_radius,
        "cf4_match_max_doublings": bf.cf4_match_max_doublings,
        "uniform_radius": bf.uniform_radius,
        "selection_mass_min": float(oc.selection_mass_min or 0),
        "selection_mass_max": float(oc.selection_mass_max or 0),
        "number_of_origins": oc.number_of_origins,
        "local_overdensity_radius": oc.local_overdensity_radius,
        "local_bulkflow_radius": oc.local_bulkflow_radius,
        "git_commit": git_commit,
        "timestamp": datetime.now().isoformat(),
    }


def _build_result(
    bf_df: pd.DataFrame,
    origin: Vector3D,
    origin_id: int,
    mask_name: str,
    method: str,
    row: pd.Series,
    delta_col: str,
    bulkflow_col: str,
) -> BulkFlowResult:
    """Construct a BulkFlowResult from a calculate_bulk_flow_series output."""
    return BulkFlowResult(
        origin_id=origin_id,
        origin=origin,
        mask=mask_name,
        method=method,
        overdensity=float(row.get(delta_col, np.nan)),
        local_bulkflow=float(row.get(bulkflow_col, np.nan)),
        mvir=float(row["mvir"]),
        near_virgo=bool(row.get("near_virgo", False)),
        radii=bf_df["radius"].values,
        u_x=bf_df["u_x"].values,
        u_y=bf_df["u_y"].values,
        u_z=bf_df["u_z"].values,
        U_tot=bf_df["U_total"].values,
        U_deb=bf_df["U_debiased"].values,
        sigma_U=bf_df["sigma_U"].values,
        n_used=bf_df["n_used"].values,
    )


def compute_bulkflows_for_origins(
    halos_df: pd.DataFrame,
    selected_points: pd.DataFrame,
    tree: cKDTree,
    cfg: AppConfig,
    cf4_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Compute bulk flows for each selected origin point and write a single netCDF file.

    Parameters
    ----------
    halos_df : DataFrame
        Full halo catalog.
    selected_points : DataFrame
        Selected origin points (must include 'x', 'y', 'z', 'rockstarid', 'mvir').
    tree : cKDTree
        Spatial index for the full halo catalog.
    cfg : AppConfig
        Pipeline configuration.
    cf4_df : DataFrame, optional
        CF4 catalog (required when masks include 'cf4' or 'uniform').

    Returns
    -------
    timings : dict
        Wall-clock times for monitoring.
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

    # Column names for per-origin environment (mirroring environment_analysis.py:34-36)
    delta_col = f"delta_{int(cfg.origin_configs.local_overdensity_radius)}"
    bulkflow_col = f"bulkflow_{int(cfg.origin_configs.local_bulkflow_radius)}"

    if mask_mode in ("cf4", "uniform", "all") and cf4_df is None:
        raise ValueError(
            "CF4 DataFrame is required when bulkflow.masks is 'cf4', 'uniform', or 'all'."
        )

    use_cf4 = mask_mode in ("cf4", "all")
    use_uniform = mask_mode in ("uniform", "all")
    use_full = mask_mode in ("full", "all")

    mask_maker = MaskMaker(
        halos_df=halos_df,
        tree=tree,
        box_size=box_size,
        cf4_df=cf4_df,
    )

    dataset = BulkFlowDataset()
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
            dataset.add(_build_result(
                bf_cf4, origin, origin_id, "cf4",
                calculation_method, row, delta_col, bulkflow_col,
            ))

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
            dataset.add(_build_result(
                bf_uniform, origin, origin_id, "uniform",
                calculation_method, row, delta_col, bulkflow_col,
            ))

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
            dataset.add(_build_result(
                bf_full, origin, origin_id, "full",
                calculation_method, row, delta_col, bulkflow_col,
            ))

        per_origin_times.append(time.time() - t_origin)

    dataset.set_attrs(_build_dataset_attrs(cfg))
    dataset.write(output_file)
    logging.info(f"Results written to {output_file}")

    timings = {
        "process_all_origins": time.time() - t0_bulk,
        "mean_origin_time": np.mean(per_origin_times) if per_origin_times else 0.0,
        "min_origin_time": np.min(per_origin_times) if per_origin_times else 0.0,
        "max_origin_time": np.max(per_origin_times) if per_origin_times else 0.0,
    }

    logging.info("All origins processed successfully!")

    return timings
