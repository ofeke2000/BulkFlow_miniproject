"""
masks.py
--------
Construct spatial masks for bulk flow analysis.

Includes:
1. CF4-like mask: select halos near CF4 group positions
2. Uniform mask: select random halos matching the CF4-like count

The goal is to reproduce the observational selection function in the simulation.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from .config.bulkflow_config import BulkFlowConfig


class MaskMaker:
    """
    Build CF4-like and uniform halo masks for a fixed catalog and spatial index.

    Instantiate once per bulk-flow run (outside the origin loop), then call
    make_cf4_mask / make_uniform_mask for each origin.
    """

    CF4_PROGRESS_INTERVAL = 10_000
    CF4_SEARCH_RADIUS_FACTOR = 2.0

    def __init__(
        self,
        halos_df: pd.DataFrame,
        tree: cKDTree,
        box_size: float,
        cf4_df: pd.DataFrame | None = None,
    ):
        self.halos_df = halos_df
        self.tree = tree
        self.box_size = box_size
        self.cf4_df = cf4_df

    def make_cf4_mask(
        self,
        position: np.ndarray,
        radius: float | None = None,
        max_doublings: int = 4,
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
        if self.cf4_df is None:
            raise ValueError("cf4_df must be provided at construction for CF4 masking.")

        if radius is None:
            radius = BulkFlowConfig().cf4_match_radius

        cf4_shifted_xyz = (self.cf4_df[['x', 'y', 'z']].values + position) % self.box_size
        cf4_ids = self.cf4_df['id'].values

        matched_rows = []
        used_indices = set()
        match_distances = []

        print(f"Starting CF4-like matching (R0 = {radius} h⁻¹ Mpc)...")

        for i, (cf4_id, pos_cf4) in enumerate(zip(cf4_ids, cf4_shifted_xyz)):
            search_radius = radius
            idx = []

            for _ in range(max_doublings + 1):
                idx = self.tree.query_ball_point(pos_cf4, search_radius)
                idx = [j for j in idx if j not in used_indices]

                if idx:
                    break

                search_radius *= self.CF4_SEARCH_RADIUS_FACTOR

            if not idx:
                continue

            halo_pos = self.halos_df.iloc[idx][['x', 'y', 'z']].values
            delta = halo_pos - pos_cf4
            delta -= self.box_size * np.round(delta / self.box_size)
            distances = np.linalg.norm(delta, axis=1)

            j_closest = idx[np.argmin(distances)]
            d_min = distances.min()

            used_indices.add(j_closest)
            match_distances.append(d_min)

            halo = self.halos_df.iloc[j_closest]

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
                'match_distance': d_min,
            })

            if (i + 1) % self.CF4_PROGRESS_INTERVAL == 0:
                print(f"  Matched {i+1:,}/{len(self.cf4_df):,}")

        matched_halos = pd.DataFrame(matched_rows)

        if match_distances:
            print("\nCF4–Rockstar matching statistics:")
            print(f"  Mean distance   = {np.mean(match_distances):.3f} h⁻¹ Mpc")
            print(f"  Median distance = {np.median(match_distances):.3f} h⁻¹ Mpc")
            print(f"  Max distance    = {np.max(match_distances):.3f} h⁻¹ Mpc")

        print(f"\nMatched {len(matched_halos):,} CF4 galaxies to halos.")
        return matched_halos

    def make_uniform_mask(
        self,
        position: np.ndarray,
        radius: float,
    ) -> pd.DataFrame:
        """
        Select a uniform random set of halos within a sphere of given radius
        around a center point. The number of halos returned matches the CF4
        galaxy count inside that radius.

        Parameters
        ----------
        position : array-like, shape (3,)
            (x,y,z) center in h^-1 Mpc.
        radius : float
            Radius of sphere selection.

        Returns
        -------
        DataFrame
            A subset of halos_df with full halo properties.
        """
        if self.cf4_df is None:
            raise ValueError("cf4_df must be provided at construction for uniform masking.")
        if 'distance' not in self.cf4_df.columns:
            raise ValueError("cf4_df must contain column 'distance'.")

        neighbor_indices = self.tree.query_ball_point(position, radius)
        df_candidates = self.halos_df.iloc[neighbor_indices]

        n_pick = int(np.sum(self.cf4_df['distance'] <= radius))

        if n_pick == 0:
            return self.halos_df.iloc[0:0].copy()

        replace = len(df_candidates) < n_pick
        selected = df_candidates.sample(n=n_pick, replace=replace)

        print(f"Randomized {len(selected):,} Uniform halos.")
        return selected.reset_index(drop=True)
