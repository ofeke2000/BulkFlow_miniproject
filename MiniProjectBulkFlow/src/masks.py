"""
masks.py
--------
Functions to construct spatial masks for bulk flow analysis.

Includes:
1. CF4-like mask: select halos near CF4 group positions
2. Uniform mask: select random halos matching the CF4-like count

The goal is to reproduce the observational selection function in the simulation.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from specific_utils import periodic_distance

##################################################################
# CF4-like mask (halo properties at CF4 positions)
##################################################################

def make_cf4_mask(
    position: np.ndarray,
    halos_df: pd.DataFrame,
    cf4_df: pd.DataFrame,
    tree: cKDTree,
    box_size: float = 1000.0,
    radius: float = 1.0,
    max_doublings: int = 4
) -> pd.DataFrame:
    """
    Create a CF4-like matched sample:
    - match each CF4 galaxy to a Rockstar halo
    - keep halo properties
    - overwrite halo position with CF4 position

    Returns
    -------
    matched_halos : pandas.DataFrame
        Columns include halo properties + CF4 position + match_distance
    """

    # Shift CF4 positions to the observer location
    cf4_shifted_xyz = (cf4_df[['x', 'y', 'z']].values + position) % box_size
    cf4_ids = cf4_df['id'].values

    matched_rows = []
    used_indices = set()
    match_distances = []

    print(f"Starting CF4-like matching (R0 = {radius} h⁻¹ Mpc)...")

    for i, (cf4_id, pos_cf4) in enumerate(
        zip(cf4_ids, cf4_shifted_xyz)
    ):
        search_radius = radius
        idx = []

        # Try R → 2R → 4R → ...
        for _ in range(max_doublings + 1):
            idx = tree.query_ball_point(pos_cf4, search_radius)
            idx = [j for j in idx if j not in used_indices]

            if idx:
                break

            search_radius *= 2.0

        if not idx:
            continue

        # --- compute periodic distances ---
        halo_pos = halos_df.iloc[idx][['x', 'y', 'z']].values
        delta = halo_pos - pos_cf4
        delta -= box_size * np.round(delta / box_size)
        distances = np.linalg.norm(delta, axis=1)

        j_closest = idx[np.argmin(distances)]
        d_min = distances.min()

        used_indices.add(j_closest)
        match_distances.append(d_min)

        halo = halos_df.iloc[j_closest]

        # --- IMPORTANT PART ---
        # Keep halo properties, but place at CF4 position
        matched_rows.append({
            'rockstarid': halo['rockstarid'],
            'x': pos_cf4[0],
            'y': pos_cf4[1],
            'z': pos_cf4[2],
            'vx': halo['vx'],
            'vy': halo['vy'],
            'vz': halo['vz'],
            'mvir': halo.get('mvir', np.nan),
            'cf4_id': cf4_id,
            'match_distance': d_min
        })

        if (i + 1) % 10_000 == 0:
            print(f"  Matched {i+1:,}/{len(cf4_df):,}")

    matched_halos = pd.DataFrame(matched_rows)

    # ---------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------
    if match_distances:
        mean_d = np.mean(match_distances)
        median_d = np.median(match_distances)

        print("\nCF4–Rockstar matching statistics:")
        print(f"  Mean distance   = {mean_d:.3f} h⁻¹ Mpc")
        print(f"  Median distance = {median_d:.3f} h⁻¹ Mpc")
        print(f"  Max distance    = {np.max(match_distances):.3f} h⁻¹ Mpc")

    print(f"\nMatched {len(matched_halos):,} CF4 galaxies to halos.")

    return matched_halos


##################################################################################
# Uniform mask
###################################################################################

def make_uniform_mask(
    position: np.ndarray,
    radius: float,
    df_halos: pd.DataFrame,
    CF4_catalogue: pd.DataFrame,
    tree: cKDTree
) -> pd.DataFrame:
    """
    Select a uniform random set of halos within a sphere of given radius
    around a center point. The number of halos returned matches len(df_reference).

    Uses MDAnalysis.lib.pkdtree.PeriodicKDTree for fast PBC search.

    Parameters
    ----------
    position : array-like, shape (3,)
        (x,y,z) center in h^-1 Mpc.
    radius : float
        Radius of sphere selection.
    df_halos : DataFrame
        Simulation halo catalog (with x,y,z and other columns).
    CF4_catalogue : DataFrame
        Reference catalog used only to set the number of halos.
    tree : cKDTree
        Pre-built periodic KDTree for df_halos.

    Returns
    -------
    DataFrame
        A subset of df_halos with full halo properties.
    """

    # Query using periodic boundary conditions
    neighbor_indices = tree.query_ball_point(position, radius)

    # candidate halos: FULL rows, not just coordinates
    df_candidates = df_halos.iloc[neighbor_indices]

    # Ensure 'distance' column exists in CF4_catalogue
    if 'distance' not in CF4_catalogue.columns:
        raise ValueError("CF4_catalogue must contain column 'distance'.")

    # Number of CF4 galaxies inside the radius
    n_pick = np.sum(CF4_catalogue['distance'] <= radius)

    if n_pick == 0:
        # No galaxies inside this radius -> return empty frame
        return df_halos.iloc[0:0].copy()

    # Sample with replacement if not enough halos
    replace = len(df_candidates) < n_pick

    selected = df_candidates.sample(n=n_pick, replace=replace)

    print(f"Randomized {len(selected):,} Uniform halos.")

    return selected.reset_index(drop=True)