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

from src.data_loader import load_rockstar_catalog, load_cf4_catalogue
from src.visualize import plot_simulation_slice_heatmap


def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    logging.info(f"Loading config from {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def preprocess_data(cfg: dict) -> tuple[pd.DataFrame, cKDTree, Optional[pd.DataFrame]]:
    """
    Load and preprocess the halo catalog.

    Returns:
        halos_df: Preprocessed DataFrame
        tree: cKDTree for spatial queries
        cf4_df: CF4 catalog DataFrame if requested, otherwise None
    """
    rockstar_path = cfg["paths"]["rockstar_catalog"]
    box_size = cfg["MDPL2"]["box_size"]
    mass_cut_bool = cfg["origin_configs"]["mass_cut_bool"]
    mass_cut = cfg["origin_configs"]["mass_cut"]
    mask_mode = cfg["bulkflow"].get("masks", cfg.get("mask_type", "full"))

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
    if mask_mode in ("cf4", "uniform", "all"):
        cf4_path = cfg["paths"]["cf4_catalog"]
        logging.info("Loading CF4 catalog for optional masks...")
        cf4_df = load_cf4_catalogue(cf4_path, h=cfg["MDPL2"]["HubbleParameter"])
        logging.info("CF4 catalog loaded.")

    # Optional slice heatmap plot for diagnostics
    heatmap_cfg = cfg.get("visualization", {}).get("simulation_slice_heatmap", {})
    if heatmap_cfg.get("enabled", False):
        plot_simulation_slice_heatmap(
            df=halos_df,
            slice_axis=heatmap_cfg.get("slice_axis", "z"),
            slice_min=heatmap_cfg.get("slice_min", 400.0),
            slice_max=heatmap_cfg.get("slice_max", 500.0),
            proj_axes=tuple(heatmap_cfg.get("proj_axes", ("x", "y"))),
            gridsize=heatmap_cfg.get("gridsize", 500),
            cmap=heatmap_cfg.get("cmap", "magma"),
            output_folder=cfg["paths"]["output_folder"],
            output_file=heatmap_cfg.get("output_file", "simulation_slice_heatmap.png"),
            dpi=heatmap_cfg.get("dpi", 300),
        )

    return halos_df, tree, cf4_df


def save_catalog_checkpoint(halos_df: pd.DataFrame, rockstar_path: str):
    """Save the catalog with computed environmental columns."""
    halos_df.to_csv(rockstar_path, index=False)
    logging.info("Catalog saved with environmental test results.")