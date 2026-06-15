"""
origin_selection.py
-------------------
Select origin points based on environmental criteria.
"""

import numpy as np
import pandas as pd
import logging

from src.visualize import plot_histogram
from src.classes import AppConfig


def select_origin_points(halos_df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
    """
    Select origin points based on configuration criteria.

    Parameters:
        halos_df: Halo catalog with environmental columns
        cfg: Configuration dictionary

    Returns:
        selected_points: DataFrame of selected origin points
    """
    radius_overdensity = cfg.origin_configs.local_overdensity_radius
    radius_bulkflow = cfg.origin_configs.local_bulkflow_radius
    overdensity_upper_cut = cfg.origin_configs.local_overdensity_upper_cut
    overdensity_lower_cut = cfg.origin_configs.local_overdensity_lower_cut
    apply_overdensity_cut = cfg.origin_configs.apply_overdensity_cut
    apply_mass_selection = cfg.origin_configs.apply_mass_selection
    selection_mass_min = cfg.origin_configs.selection_mass_min
    selection_mass_max = cfg.origin_configs.selection_mass_max
    bulkflow_upper_cut = cfg.origin_configs.local_bulkflow_upper_cut
    bulkflow_lower_cut = cfg.origin_configs.local_bulkflow_lower_cut
    use_virgo_criteria = cfg.origin_configs.use_virgo_criteria
    n_origins = cfg.origin_configs.number_of_origins
    lowest_delta = cfg.origin_configs.select_lowest_delta
    select_random = cfg.origin_configs.select_random
    output_folder = cfg.paths.output_folder

    delta_column = f"delta_{int(radius_overdensity)}"
    virgo_column = "near_virgo"
    bulkflow_column = f"bulkflow_{int(radius_bulkflow)}"

    # Create absolute overdensity column for ranking
    delta_abs_col = f"delta_abs_{int(radius_overdensity)}"
    halos_df[delta_abs_col] = halos_df[delta_column].abs()

    # Apply selection criteria
    mask = halos_df[bulkflow_column].between(bulkflow_lower_cut, bulkflow_upper_cut)

    if use_virgo_criteria:
        mask &= halos_df[virgo_column] > 0

    if apply_overdensity_cut:
        mask &= halos_df[delta_column].between(overdensity_lower_cut, overdensity_upper_cut)

    if apply_mass_selection:
        if selection_mass_min is not None:
            mask &= halos_df["mvir"] >= selection_mass_min
        if selection_mass_max is not None:
            mask &= halos_df["mvir"] <= selection_mass_max

    candidates = halos_df.loc[mask]
    logging.info(f"Candidates after cuts: {len(candidates)}")

    # Plot histogram of candidates
    plot_histogram(
        data_df=candidates,
        output_folder=output_folder,
        output_file=f"overdensity_histogram_[{bulkflow_lower_cut:.0f}<local_V<{bulkflow_upper_cut:.0f}]_for_candidates.png",
        key=delta_column,
        origin=cfg.visualization.plot_histogram_origin,
        bins=cfg.visualization.style.mass_histogram_bins,
        box_size=cfg.MDPL2.box_size,
        log_axis="none",
        style=cfg.visualization.style,
    )

    # Select origins
    if lowest_delta:
        selected_points = candidates.sort_values(delta_abs_col).head(n_origins)
    elif select_random:
        selected_points = candidates.sample(n=min(n_origins, len(candidates)), replace=False)
    else:
        raise ValueError("Must specify either select_lowest_delta or select_random as True")

    selected_df = halos_df.loc[selected_points.index].copy()

    # Plot histogram of selected points
    plot_histogram(
        data_df=selected_df,
        output_folder=output_folder,
        output_file=f"overdensity_histogram_[{bulkflow_lower_cut:.0f}<local_V<{bulkflow_upper_cut:.0f}]_for_selected_points.png",
        key=delta_column,
        origin=cfg.visualization.plot_histogram_origin,
        bins=cfg.visualization.style.mass_histogram_bins,
        box_size=cfg.MDPL2.box_size,
        log_axis="none",
        style=cfg.visualization.style,
    )

    logging.info(f"Selected {len(selected_points)} origin points")
    if lowest_delta:
        logging.info(f"(min |δ| = {selected_points[delta_abs_col].min():.3e})")

    return selected_points