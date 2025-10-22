"""
experiment.py
--------------
Orchestrate the full experiment:
 - find candidate origins (halos with |delta_5| closest to 0)
 - for each origin:
     * place CF4 groups around that origin (shift CF4 positions into sim frame)
     * match CF4 groups -> halos (one-to-one) to form CF4-like mask
     * build uniform mask with same N
     * compute ML bulk flow series for both masks
 - aggregate and save results (one CSV per mask)

Notes
-----
- This script expects the following modules to be importable:
    data_loader.load_rockstar_catalog, data_loader.load_cf4_catalog, data_loader.cf4_to_cartesian
    analysis.overdensity (if you want to re-compute, but here we expect delta_5 already present)
    masking.make_cf4_like_mask, masking.make_uniform_mask
    analysis.bulkflow.compute_bulkflow_series
- Edit PATHS and CONFIG below to match your environment.
"""

import os
import sys
import math
import logging
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

# ----- Project modules (adjust imports if you put files into different package layout) -----
# Example paths assume `data_loader`, `masking`, `analysis` packages as discussed
from data_loader import load_rockstar_catalog, load_cf4_catalog, cf4_to_cartesian
from masking import make_cf4_like_mask, make_uniform_mask
from analysis.bulkflow import compute_bulkflow_series

# -------------------------
# CONFIG / PATHS (edit these)
# -------------------------
# Paths: change these to your actual files if needed
PATHS = {
    "rockstar_csv": os.path.expanduser(
        "~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid-1_mvir12/mdpl2_rockstar_125_pid-1_mvir12.csv"
    ),
    # This should be the merged rockstar file that already includes 'delta_5' column.
    "rockstar_with_delta5_csv": os.path.expanduser(
        "~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid-1_mvir12/bulk_flow_radius_series/mdpl2_rockstar_with_delta5_full.csv"
    ),
    "cf4_groups_csv": os.path.expanduser(
        "~/bulk-flow-Rockstar/Data/cf4/CF4_groups_example.csv"   # <- replace with your CF4 groups file
    ),
    "output_dir": os.path.expanduser(
        "~/bulk-flow-Rockstar/Data/mdpl2_rockstar_125_pid-1_mvir12/bulk_flow_radius_series/experiments"
    )
}

# Numeric config
CONFIG = {
    "n_origins": 50,               # number of origins to select (closest to delta_5 == 0)
    "match_radius_start": 1.0,     # initial r_match for CF4->halo matching (h^-1 Mpc)
    "max_doublings": 3,            # doubling steps: 1 -> 2 -> 4 -> 8 (max)
    "radii": np.arange(5, 251, 5), # 5, 10, ..., 250
    "error_frac": 0.20,            # 20% fractional error on radial velocities
    "min_sigma": 50.0,             # min velocity sigma (km/s)
    "min_count": 10,               # minimal objects per radius to compute BF
    "box_size": 1000.0,            # simulation box size (h^-1 Mpc) - adjust if needed
    "n_workers": max(1, cpu_count() - 1)  # parallel workers (set to 1 to run single-threaded)
}

# Ensure output dir exists
os.makedirs(PATHS["output_dir"], exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("experiment")


# -------------------------
# Helper utilities
# -------------------------
def wrap_positions_to_box(df: pd.DataFrame, box_size: float, pos_cols=("x", "y", "z")) -> pd.DataFrame:
    """Wrap x,y,z coordinates into [0, box_size) using modulo (assumes non-negative positions)."""
    df = df.copy()
    for c in pos_cols:
        df[c] = np.mod(df[c], box_size)
    return df


def shift_cf4_positions(cf4_cart: pd.DataFrame, origin: tuple, box_size: float) -> pd.DataFrame:
    """
    Shift CF4 cartesian positions (which are observer-centered at (0,0,0)) to simulation frame
    where the CF4 observer sits at `origin`. Then wrap positions into the simulation box.
    """
    shifted = cf4_cart.copy()
    shifted[['x', 'y', 'z']] = shifted[['x', 'y', 'z']].values + np.array(origin).reshape((1, 3))
    shifted = wrap_positions_to_box(shifted, box_size)
    return shifted


def select_candidate_origins(halo_df: pd.DataFrame, n_origins: int = 50, delta_col: str = "delta_5") -> pd.DataFrame:
    """
    Select n_origins halos with smallest |delta_col| (closest to zero).
    Returns DataFrame with the selected origin halo rows.
    """
    if delta_col not in halo_df.columns:
        raise KeyError(f"{delta_col} not present in halo dataframe. Run overdensity module first.")

    df = halo_df.copy()
    df['abs_delta'] = np.abs(df[delta_col])
    df_sorted = df.sort_values(by='abs_delta', ascending=True)
    selected = df_sorted.head(n_origins).reset_index(drop=True)
    logger.info(f"Selected {len(selected)} origin halos (smallest |{delta_col}|).")
    return selected


# -------------------------
# Worker function for parallel processing
# -------------------------
def process_origin(origin_row: pd.Series,
                   halos_df: pd.DataFrame,
                   cf4_cart: pd.DataFrame,
                   cfg: dict,
                   paths: dict) -> dict:
    """
    Process a single origin:
      - Shift CF4 groups to the origin position
      - Match CF4 -> halos to make cf4-like mask
      - Make uniform mask with same count
      - Compute bulkflow series for both masks
    Returns a dict with two DataFrames: results_cf4 and results_uniform (each with R values)
    """
    origin_id = origin_row['rockstarid']
    origin_coords = (float(origin_row['x']), float(origin_row['y']), float(origin_row['z']))
    logger.info(f"Origin {origin_id}: coords = {origin_coords}")

    # shift CF4 to simulation frame
    cf4_shifted = shift_cf4_positions(cf4_cart, origin_coords, cfg['box_size'])

    # Build CF4-like mask (one-to-one)
    cf4_mask = make_cf4_like_mask(
        halos_df,
        cf4_shifted,
        radius=cfg['match_radius_start'],
        max_doublings=cfg['max_doublings']
    )

    # In case CF4 matching returned 0 (unlikely), handle gracefully
    n_cf4 = len(cf4_mask)
    if n_cf4 == 0:
        logger.warning(f"Origin {origin_id}: CF4 matching returned 0 halos. Returning empty results.")
        return {
            'origin_id': origin_id,
            'origin_coords': origin_coords,
            'cf4_mask_count': 0,
            'results_cf4': pd.DataFrame(),
            'results_uniform': pd.DataFrame()
        }

    # Build uniform mask with same N
    uniform_mask = make_uniform_mask(halos_df, cf4_mask, seed=int(origin_id % (2**31 - 1)))

    # For bulkflow, we need halos with velocity columns; both mask DFs currently have rockstarid + x,y,z.
    # We join to the halos_df to get vx,vy,vz and any other required columns.
    cf4_halos_full = cf4_mask.merge(halos_df, on='rockstarid', how='left')
    uniform_halos_full = uniform_mask.merge(halos_df, on='rockstarid', how='left')

    # compute bulk flow series for both masks
    radii = cfg['radii']
    try:
        results_cf4 = compute_bulkflow_series(
            cf4_halos_full,
            origin=origin_coords,
            radii=radii,
            error_frac=cfg['error_frac'],
            min_sigma=cfg['min_sigma'],
            min_count=cfg['min_count'],
            cumulative=True
        )
    except Exception as e:
        logger.exception(f"Error computing bulkflow for CF4 mask for origin {origin_id}: {e}")
        results_cf4 = pd.DataFrame()

    try:
        results_uniform = compute_bulkflow_series(
            uniform_halos_full,
            origin=origin_coords,
            radii=radii,
            error_frac=cfg['error_frac'],
            min_sigma=cfg['min_sigma'],
            min_count=cfg['min_count'],
            cumulative=True
        )
    except Exception as e:
        logger.exception(f"Error computing bulkflow for uniform mask for origin {origin_id}: {e}")
        results_uniform = pd.DataFrame()

    # add metadata to results (origin id and coords)
    if not results_cf4.empty:
        results_cf4.insert(0, 'origin_rockstarid', origin_id)
        results_cf4.insert(1, 'origin_x', origin_coords[0])
        results_cf4.insert(2, 'origin_y', origin_coords[1])
        results_cf4.insert(3, 'origin_z', origin_coords[2])
        results_cf4.insert(4, 'mask_count', len(cf4_halos_full))

    if not results_uniform.empty:
        results_uniform.insert(0, 'origin_rockstarid', origin_id)
        results_uniform.insert(1, 'origin_x', origin_coords[0])
        results_uniform.insert(2, 'origin_y', origin_coords[1])
        results_uniform.insert(3, 'origin_z', origin_coords[2])
        results_uniform.insert(4, 'mask_count', len(uniform_halos_full))

    logger.info(f"Origin {origin_id}: CF4_mask N={len(cf4_halos_full)}, uniform_mask N={len(uniform_halos_full)}")

    return {
        'origin_id': origin_id,
        'origin_coords': origin_coords,
        'cf4_mask_count': len(cf4_halos_full),
        'results_cf4': results_cf4,
        'results_uniform': results_uniform
    }


# -------------------------
# Main orchestration
# -------------------------
def run_experiments(paths: dict = PATHS, cfg: dict = CONFIG):
    # Load data
    logger.info("Loading halo catalog (with delta_5)...")
    halos = pd.read_csv(paths['rockstar_with_delta5_csv'])
    required_cols = {'rockstarid', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'delta_5'}
    if not required_cols.issubset(set(halos.columns)):
        missing = required_cols - set(halos.columns)
        raise RuntimeError(f"Missing required columns in halo file: {missing}")

    logger.info("Loading CF4 groups...")
    cf4 = pd.read_csv(paths['cf4_groups_csv'])
    # Ensure CF4 has RA, Dec, distance (in h^-1 Mpc); convert to cartesian
    if not set(['ra', 'dec', 'distance']).issubset(set(cf4.columns)):
        raise RuntimeError("CF4 groups file must contain 'ra','dec','distance' columns (in degrees and h^-1 Mpc).")
    cf4_cart = cf4.copy()
    # convert RA/Dec/distance to x,y,z in observer-centered frame
    cf4_cart = cf4_to_cartesian(cf4_cart)   # adds x,y,z in same distance units as 'distance'

    # find candidate origins: halos with smallest |delta_5|
    origins = select_candidate_origins(halos, n_origins=cfg['n_origins'], delta_col='delta_5')

    # Prepare parallel worker
    worker = partial(process_origin, halos_df=halos, cf4_cart=cf4_cart, cfg=cfg, paths=paths)

    n_workers = cfg.get('n_workers', 1)
    logger.info(f"Running experiments on {len(origins)} origins with {n_workers} workers...")

    results_list = []
    if n_workers == 1:
        # serial loop
        for _, origin_row in origins.iterrows():
            results_list.append(worker(origin_row))
    else:
        # parallel
        origin_rows = [row for _, row in origins.iterrows()]
        with Pool(processes=n_workers) as pool:
            for r in pool.imap_unordered(worker, origin_rows):
                results_list.append(r)

    # Aggregate results per mask
    cf4_results_frames = []
    uniform_results_frames = []
    for res in results_list:
        if isinstance(res.get('results_cf4'), pd.DataFrame) and not res['results_cf4'].empty:
            cf4_results_frames.append(res['results_cf4'])
        if isinstance(res.get('results_uniform'), pd.DataFrame) and not res['results_uniform'].empty:
            uniform_results_frames.append(res['results_uniform'])

    if cf4_results_frames:
        df_cf4_all = pd.concat(cf4_results_frames, ignore_index=True)
    else:
        df_cf4_all = pd.DataFrame()

    if uniform_results_frames:
        df_uniform_all = pd.concat(uniform_results_frames, ignore_index=True)
    else:
        df_uniform_all = pd.DataFrame()

    # Save to CSV files (one file per mask)
    out_cf4 = os.path.join(paths['output_dir'], "bulkflow_results_cf4_mask.csv")
    out_uniform = os.path.join(paths['output_dir'], "bulkflow_results_uniform_mask.csv")

    if not df_cf4_all.empty:
        df_cf4_all.to_csv(out_cf4, index=False)
        logger.info(f"Saved CF4-mask results to {out_cf4}")
    else:
        logger.warning("No CF4 results to save.")

    if not df_uniform_all.empty:
        df_uniform_all.to_csv(out_uniform, index=False)
        logger.info(f"Saved uniform-mask results to {out_uniform}")
    else:
        logger.warning("No uniform results to save.")

    return {
        'cf4_results_path': out_cf4 if not df_cf4_all.empty else None,
        'uniform_results_path': out_uniform if not df_uniform_all.empty else None,
        'n_origins_processed': len(results_list)
    }


# -------------------------
# CLI entrypoint
# -------------------------
if __name__ == "__main__":
    summary = run_experiments(PATHS, CONFIG)
    logger.info(f"Experiment finished. Summary: {summary}")
