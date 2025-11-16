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


def make_cf4_mask(halos_df: pd.DataFrame,
                       cf4_df: pd.DataFrame,
                       radius: float = 1.0,
                       max_doublings: int = 4) -> pd.DataFrame:
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
    print(f"Building KDTree for {len(halos_df):,} halos...")
    tree = cKDTree(halos_df[['x', 'y', 'z']].values)

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


def make_uniform_mask(halos_df: pd.DataFrame,
                      cf4_like_mask: pd.DataFrame,
                      seed: int = 42) -> pd.DataFrame:
    """
    Create a uniform random halo sample with the same count as the CF4-like mask.

    Parameters
    ----------
    halos_df : pandas.DataFrame
        Full halo catalog.
    cf4_like_mask : pandas.DataFrame
        The CF4-like matched sample.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    uniform_mask : pandas.DataFrame
        Randomly chosen halos (same length as cf4_like_mask).
    """
    np.random.seed(seed)
    n = len(cf4_like_mask)
    print(f"Selecting {n:,} random halos for uniform mask...")

    sample = halos_df.sample(n=n, random_state=seed)
    sample = sample[['rockstarid', 'x', 'y', 'z']].copy()
    sample['mask_type'] = 'uniform'

    print(f"Created uniform mask with {len(sample):,} halos.")
    return sample

def pick_uniform_halos(position, radius, df_source, df_target, tree=None):
    """
    Efficiently pick halos uniformly within radius around a position.

    Parameters
    ----------
    position : array-like, shape (3,)
        The (x, y, z) coordinates of the center.
    radius : float
        Radius of the selection region.
    df_source : pd.DataFrame
        Source halos with columns ['x', 'y', 'z'].
    df_target : pd.DataFrame
        Target DataFrame whose length determines number of halos to pick.
    tree : scipy.spatial.cKDTree, optional
        Precomputed KDTree of df_source[['x', 'y', 'z']]. If not provided, one will be built (slower).

    Returns
    -------
    pd.DataFrame
        Subset of df_source containing the sampled halos.
    """
    # Build tree if not provided
    if tree is None:
        coords = df_source[['x', 'y', 'z']].to_numpy()
        tree = cKDTree(coords)

    # Query indices of points within radius
    idx = tree.query_ball_point(position, radius)
    if not idx:
        raise ValueError(f"No halos found within radius {radius:.2f}")

    candidates = df_source.iloc[idx]

    n_target = len(df_target)
    replace = len(candidates) < n_target

    sampled = candidates.sample(n=n_target, replace=replace, random_state=np.random.default_rng())

    return sampled.reset_index(drop=True)