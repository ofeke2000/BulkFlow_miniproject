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
from MDAnalysis.lib.pkdtree import PeriodicKDTree

##################################################################
# Broken functions
##################################################################


def make_cf4_mask(halos_df: pd.DataFrame,
                       cf4_df: pd.DataFrame,
                       radius: float = 1.0,
                       max_doublings: int = 4,
                       tree: PeriodicKDTree, 
                       box_size: float = 1000.0) -> pd.DataFrame:
    """
    Create a one-to-one CF4-like halo sample from the simulation.

    For each CF4 group, find the nearest halo within `radius` (h^-1 Mpc).
    If none are found, double the radius (up to `max_doublings` times)
    until a match is made.

    Parameters
    ----------
    halos_df : pandas.DataFrame
        Simulation halos (with columns ['rockstarid', 'x', 'y', 'z']).
    cf4_df : pandas.DataFrame
        CF4 group catalog (with columns ['id', 'x', 'y', 'z']).
    radius : float, optional
        Initial search radius in h^-1 Mpc. Default = 1.0.
    max_doublings : int, optional
        Number of times to double radius if no match is found.

    Returns
    -------
    matched_halos : pandas.DataFrame
        Subset of halos matched one-to-one to CF4 groups.
        Columns: ['rockstarid', 'x', 'y', 'z', 'cf4_id', 'match_distance']
    """

    matched_rows = []
    used_indices = set()

    print(f"Starting CF4-like matching with radius = {radius} h^-1 Mpc...")

    for i, (cf4_id, pos_cf4) in enumerate(zip(cf4_df['id'], cf4_df[['x', 'y', 'z']].values)):
        search_radius = radius
        idx = []

        for attempt in range(max_doublings + 1):
            idx = tree.query_ball_point(pos_cf4, r=search_radius)
            idx = [j for j in idx if j not in used_indices]  # avoid duplicates
            if len(idx) > 0:
                break
            search_radius *= 2.0

        if len(idx) == 0:
            # no match found even after doublings
            continue

        # choose the closest halo
        halo_positions = halos_df.iloc[idx][['x', 'y', 'z']].values
        distances = np.linalg.norm(halo_positions - pos_cf4, axis=1)
        j_closest = idx[np.argmin(distances)]
        used_indices.add(j_closest)

        matched_rows.append({
            'rockstarid': halos_df.iloc[j_closest]['rockstarid'],
            'x': halos_df.iloc[j_closest]['x'],
            'y': halos_df.iloc[j_closest]['y'],
            'z': halos_df.iloc[j_closest]['z'],
            'vx': halos_df.iloc[j_closest]['vx'],
            'vy': halos_df.iloc[j_closest]['vy'],
            'vz': halos_df.iloc[j_closest]['vz'],
            'cf4_id': cf4_id,
            'match_distance': np.min(distances)
        })

        if (i + 1) % 10000 == 0:
            print(f"  Matched {i+1:,}/{len(cf4_df):,} CF4 groups")

    matched_halos = pd.DataFrame(matched_rows)
    print(f"Matched {len(matched_halos):,} CF4 groups to halos.")
    return matched_halos

def make_cf4_mask_pbc(halos_df: pd.DataFrame,
                      cf4_df: pd.DataFrame,
                      box_size: float,
                      tree: PeriodicKDTree,
                      radius: float = 1.0,
                      max_doublings: int = 4) -> pd.DataFrame:
    """
    Create a one-to-one CF4-like halo sample from the simulation using a PeriodicKDTree.

    For each CF4 group, find the nearest halo within `radius` (h^-1 Mpc) using periodic boundaries.
    If none are found, double the radius (up to `max_doublings` times) until a match is made.

    Parameters
    ----------
    halos_df : pandas.DataFrame
        Simulation halos (columns: 'rockstarid', 'x', 'y', 'z', 'vx', 'vy', 'vz', ...).
    cf4_df : pandas.DataFrame
        CF4 group catalog (columns: 'id', 'x', 'y', 'z').
    box_size : float
        Simulation box size (assumed cubic, h^-1 Mpc).
    radius : float
        Initial search radius (h^-1 Mpc).
    max_doublings : int
        Number of times to double the search radius if no match is found.

    Returns
    -------
    matched_halos : pd.DataFrame
        Subset of halos matched one-to-one to CF4 groups.
    """

    matched_rows = []
    used_indices = set()

    print(f"Starting CF4-like matching with initial radius = {radius} h^-1 Mpc...")

    for i, (cf4_id, pos_cf4) in enumerate(zip(cf4_df['id'], cf4_df[['x', 'y', 'z']].values)):
        search_radius = radius
        neighbor_indices = []

        for attempt in range(max_doublings + 1):
            neighbor_indices = tree.search(pos_cf4, search_radius)
            neighbor_indices = [j for j in neighbor_indices if j not in used_indices]  # avoid duplicates
            if neighbor_indices:
                break
            search_radius *= 2.0

        if not neighbor_indices:
            # no match found even after doublings
            continue

        # choose the closest halo
        halo_positions = halos_df.iloc[neighbor_indices][['x', 'y', 'z']].values
        distances = np.linalg.norm(halo_positions - pos_cf4, axis=1)
        j_closest = neighbor_indices[np.argmin(distances)]
        used_indices.add(j_closest)

        halo = halos_df.iloc[j_closest]
        matched_rows.append({
            'rockstarid': halo['rockstarid'],
            'x': halo['x'],
            'y': halo['y'],
            'z': halo['z'],
            'vx': halo['vx'],
            'vy': halo['vy'],
            'vz': halo['vz'],
            'cf4_id': cf4_id,
            'match_distance': np.min(distances)
        })

        if (i + 1) % 1000 == 0:
            print(f"  Matched {i+1:,}/{len(cf4_df):,} CF4 groups")

    matched_halos = pd.DataFrame(matched_rows)
    print(f"Matched {len(matched_halos):,} CF4 groups to halos.")
    return matched_halos

def make_uniform_mask(
    position: np.ndarray,
    radius: float,
    df_halos: pd.DataFrame,
    CF4_catalogue: pd.DataFrame,
    tree: PeriodicKDTree
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
    tree : PeriodicKDTree
        Pre-built periodic KDTree for df_halos.

    Returns
    -------
    DataFrame
        A subset of df_halos with full halo properties.
    """

    # Query using periodic boundary conditions
    neighbor_indices = tree.search(position, radius)

    # candidate halos: FULL rows, not just coordinates
    df_candidates = df_halos.iloc[neighbor_indices]

    n_pick = len(CF4_catalogue)

    # Sample with replacement if not enough halos
    replace = len(df_candidates) < n_pick

    selected = df_candidates.sample(n=n_pick, replace=replace)

    return selected.reset_index(drop=True)