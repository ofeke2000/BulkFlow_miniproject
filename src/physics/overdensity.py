"""
overdensity.py
---------------
Compute the local overdensity around each halo within a given radius (e.g., 5 h^-1 Mpc).

Definitions:
    delta_R = (rho_local - rho_mean) / rho_mean

where:
    rho_local = mass density within radius R around each halo
    rho_mean  = global mean mass density of all halos in the catalog

The result can be used to identify halos in 'average' environments (|delta_R| ≈ 0).
"""

import numpy as np
import pandas as pd
import sys
from scipy.spatial import cKDTree


class OverdensityCalculator:
    """Compute local overdensity delta_R for each halo using a periodic KDTree."""

    PROGRESS_INTERVAL = 500000

    def __init__(
        self,
        radius: float,
        box_size: float,
        tree: cKDTree,
        mass_column: str = "mvir",
    ):
        if tree is None:
            sys.exit("No tree provided. Please provide a pre-built cKDTree.")
        self.radius = radius
        self.box_size = box_size
        self.tree = tree
        self.mass_column = mass_column

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add column ``delta_{radius}`` to df, or return df unchanged if it exists.

        Parameters
        ----------
        df : DataFrame
            Halo catalog containing 'x', 'y', 'z', and mass column.

        Returns
        -------
        df : DataFrame
            Same DataFrame with an added column delta_{radius}.
        """
        delta_col = f"delta_{int(self.radius)}"

        if delta_col in df.columns:
            print(f"[INFO] Column '{delta_col}' already exists. Skipping computation.")
            return df

        box_volume = self.box_size ** 3
        total_mass = df[self.mass_column].sum()
        rho_mean = total_mass / box_volume
        print(f"Mean mass density: {rho_mean:.3e}")

        overdensity = np.zeros(len(df), dtype=np.float64)
        V_R = (4.0 / 3.0) * np.pi * self.radius ** 3

        print(f"Computing overdensity for R = {self.radius} h^-1 Mpc...")
        positions = df[['x', 'y', 'z']].values

        for i in range(len(df)):
            indices = self.tree.query_ball_point(positions[i], self.radius)
            local_mass = df.iloc[indices][self.mass_column].sum()
            overdensity[i] = (local_mass / V_R - rho_mean) / rho_mean

            if i % self.PROGRESS_INTERVAL == 0 and i > 0:
                print(f"  Processed {i:,} halos...")

        print("Done computing overdensity.")

        df[delta_col] = overdensity
        return df
