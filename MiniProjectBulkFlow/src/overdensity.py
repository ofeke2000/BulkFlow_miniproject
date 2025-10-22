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
from scipy.spatial import cKDTree


def compute_overdensity(df: pd.DataFrame, radius: float = 5.0, mass_column: str = "mvir") -> pd.DataFrame:
    """
    Compute overdensity delta_R for each halo within a sphere of given radius.

    Parameters
    ----------
    df : pandas.DataFrame
        Halo catalog with 'x', 'y', 'z', and 'mvir' columns.
    radius : float, optional
        Radius of sphere in h^-1 Mpc for local density computation. Default is 5.
    mass_column : str, optional
        Column to use as mass indicator (e.g. 'mvir').

    Returns
    -------
    delta_df : pandas.DataFrame
        DataFrame with ['rockstarid', 'delta_R'].
    """
    print(f"Building KDTree for {len(df):,} halos...")
    tree = cKDTree(df[['x', 'y', 'z']].values)

    # Compute mean mass density (in simulation box units)
    print("Computing mean density...")
    box_volume = (df[['x', 'y', 'z']].max() - df[['x', 'y', 'z']].min()).prod()
    total_mass = df[mass_column].sum()
    rho_mean = total_mass / box_volume

    print(f"Mean mass density = {rho_mean:.3e}")

    # Prepare results
    delta_values = np.zeros(len(df), dtype=np.float64)

    # Precompute volume of sphere
    V_R = (4.0 / 3.0) * np.pi * radius**3

    # Query neighbors
    print(f"Computing local overdensity for R = {radius} h^-1 Mpc...")
    for i, (pos, mvir) in enumerate(zip(df[['x', 'y', 'z']].values, df[mass_column].values)):
        idx = tree.query_ball_point(pos, r=radius)
        local_mass = df.iloc[idx][mass_column].sum()
        rho_local = local_mass / V_R
        delta_values[i] = (rho_local - rho_mean) / rho_mean

        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i:,} halos...")

    print("Done computing overdensity values.")
    return pd.DataFrame({
        'rockstarid': df['rockstarid'],
        f'delta_{int(radius)}': delta_values
    })


def merge_overdensity(df_original: pd.DataFrame, df_delta: pd.DataFrame, radius: float = 5.0) -> pd.DataFrame:
    """
    Merge overdensity results into the original halo catalog.

    Parameters
    ----------
    df_original : pandas.DataFrame
        Original halo catalog.
    df_delta : pandas.DataFrame
        DataFrame from compute_overdensity().
    radius : float
        Radius of overdensity computation (used for column name).

    Returns
    -------
    merged : pandas.DataFrame
        Catalog with new column delta_<radius>.
    """
    delta_col = f'delta_{int(radius)}'
    merged = df_original.merge(df_delta, on='rockstarid', how='left')
    print(f"Added column '{delta_col}' to catalog.")
    return merged
