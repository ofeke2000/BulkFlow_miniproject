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

##################################################################
# CF4-like mask
##################################################################

def make_cf4_mask(position: np.ndarray,
                  halos_df: pd.DataFrame,
                  cf4_df: pd.DataFrame,
                  tree: cKDTree,
                  box_size: float = 1000.0,
                  radius: float = 1.0,
                  max_doublings: int = 4) -> pd.DataFrame:
    """
    Create a CF4-like matched sample using a pre-built PeriodicKDTree.

    Parameters
    ----------
    tree : MDAnalysis.lib.pkdtree.PeriodicKDTree
        Pre-built periodic KDTree built from halos_df[['x', 'y', 'z']].
    halos_df : pandas.DataFrame
        Simulation halos (must contain ['rockstarid','x','y','z','vx','vy','vz']).
    cf4_df : pandas.DataFrame
        CF4 catalog (must contain ['id','x','y','z']).
    radius : float, optional
        Initial search radius (h⁻¹ Mpc). Default = 1.0
    max_doublings : int, optional
        Number of times to double search radius upon failure.

    Returns
    -------
    matched_halos : pandas.DataFrame
        One matched halo per CF4 object, columns:
        ['rockstarid','x','y','z','vx','vy','vz','cf4_id','match_distance']
    """

    cf4_shifted_xyz = cf4_df[['x', 'y', 'z']].values + position # Shift CF4 coordinates
    cf4_shifted_xyz = np.mod(cf4_shifted_xyz, box_size) # Apply periodic boundaries

    matched_rows = []
    used_indices = set()

    print(f"Starting CF4-like matching with radius = {radius} h^-1 Mpc...")

    for i, (cf4_id, pos_cf4) in enumerate(
            zip(cf4_df['id'], cf4_df[['x', 'y', 'z']].values)):
        
        search_radius = radius
        idx = []

        # Try radius → 2R → 4R → ...
        for attempt in range(max_doublings + 1):
            idx = tree.query_ball_point(pos_cf4, search_radius)

            # Remove halos already matched to earlier CF4 entries
            idx = [j for j in idx if j not in used_indices]

            if idx:
                break  # success

            search_radius *= 2.0

        if not idx:
            continue  # no match after all doublings

        # Compute nearest halo
        halo_positions = halos_df.iloc[idx][['x', 'y', 'z']].values
        distances = np.linalg.norm(halo_positions - pos_cf4, axis=1)

        j_closest = idx[np.argmin(distances)]
        used_indices.add(j_closest)

        # Store result
        row = halos_df.iloc[j_closest].copy()
        matched_rows.append({
            'rockstarid': row['rockstarid'],
            'x': row['x'],
            'y': row['y'],
            'z': row['z'],
            'vx': row['vx'],
            'vy': row['vy'],
            'vz': row['vz'],
            'cf4_id': cf4_id,
            'match_distance': distances.min()
        })

        if (i + 1) % 10_000 == 0:
            print(f"  Matched {i+1:,}/{len(cf4_df):,}")

    matched_halos = pd.DataFrame(matched_rows)
    print(f"Matched {len(matched_halos):,} CF4 groups to halos.")
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