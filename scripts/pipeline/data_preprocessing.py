"""
data_preprocessing.py
---------------------
Handles data loading and initial preprocessing steps.
"""

from typing import Optional

import yaml
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from scipy.spatial import cKDTree

from src.io.data_loader import load_rockstar_catalog, load_cf4_catalogue
from src.viz.visualize import plot_simulation_slice_heatmap
from src.classes import AppConfig


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load YAML configuration and return a typed config object."""
    logging.info(f"Loading config from {path}")
    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    return AppConfig.from_dict(raw_cfg or {})


def preprocess_data(cfg: AppConfig) -> tuple[pd.DataFrame, cKDTree, Optional[pd.DataFrame]]:
    """
    Load and preprocess the halo catalog.

    Returns:
        halos_df: Preprocessed DataFrame
        tree: cKDTree for spatial queries
        cf4_df: CF4 catalog DataFrame if requested, otherwise None
    """
    rockstar_path = cfg.paths.rockstar_catalog
    box_size = cfg.MDPL2.box_size
    mass_cut_bool = cfg.origin_configs.mass_cut_bool
    mass_cut = cfg.origin_configs.mass_cut
    mask_mode = cfg.bulkflow.masks or "full"

    # Load catalog
    logging.info("Loading Rockstar catalog...")
    halos_df = load_rockstar_catalog(rockstar_path)

    # Fit halos into box [0, box_size)
    halos_df[['x','y','z']] %= box_size
    logging.info("Rockstar catalog loaded and prepared.")

    # Apply mass cut if requested
    if mass_cut_bool:
        logging.info(f"Applying mass cut: mvir >= {mass_cut:.2e}")
        n_before = len(halos_df)
        halos_df = halos_df[halos_df["mvir"] >= mass_cut].copy()
        n_after = len(halos_df)
        logging.info(f"Mass cut applied: {n_before} → {n_after} halos")

    # Build cKDTree
    logging.info("Building cKDTree...")
    tree = cKDTree(halos_df[["x", "y", "z"]].values, boxsize=box_size)
    logging.info("cKDTree built successfully.")

    cf4_df = None
    if cfg.bulkflow.masks in ("cf4", "uniform", "all"):
        cf4_path = cfg.paths.cf4_catalog
        logging.info("Loading CF4 catalog for optional masks...")
        cf4_df = load_cf4_catalogue(cf4_path, h=cfg.MDPL2.HubbleParameter)
        logging.info("CF4 catalog loaded.")

    # Optional slice heatmap plot for diagnostics
    heatmap_cfg = cfg.visualization.simulation_slice_heatmap
    if heatmap_cfg.enabled:
        plot_simulation_slice_heatmap(
            df=halos_df,
            slice_axis=heatmap_cfg.slice_axis,
            slice_min=heatmap_cfg.slice_min,
            slice_max=heatmap_cfg.slice_max,
            proj_axes=tuple(heatmap_cfg.proj_axes),
            gridsize=heatmap_cfg.gridsize,
            cmap=heatmap_cfg.cmap,
            output_folder=cfg.paths.output_folder,
            output_file=heatmap_cfg.output_file,
            dpi=heatmap_cfg.dpi,
        )

    return halos_df, tree, cf4_df


def save_catalog_checkpoint(halos_df: pd.DataFrame, rockstar_path: str):
    """Save the catalog with computed environmental columns."""
    halos_df.to_csv(rockstar_path, index=False)
    logging.info("Catalog saved with environmental test results.")