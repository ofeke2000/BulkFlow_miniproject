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
from MDAnalysis.lib.pkdtree import PeriodicKDTree

#################################################################
# compute_overdensity
################################################################


def compute_overdensity(
    df: pd.DataFrame,
    radius: float = 5.0,
    tree: PeriodicKDTree | None = None,
    box_size: float = 1000.0,
    mass_column: str = "mvir"
) -> pd.DataFrame:
    """
    Compute the local overdensity delta_R for each halo using a periodic KDTree.

    Parameters
    ----------
    df : DataFrame
        Halo catalog containing 'x', 'y', 'z', and mass column.
    radius : float
        Sphere radius in h^-1 Mpc.
    tree : PeriodicKDTree
        Pre-built tree (optional). 
    box_size : float
        Size of the simulation box in h^-1 Mpc.
    mass_column : str
        Column representing halo mass (default 'mvir').

    Returns
    -------
    df : DataFrame
        Same DataFrame with an added column delta_{radius}.
    """

    delta_col = f"delta_{int(radius)}"

    # --- Early skip if overdensity was already computed ---
    if delta_col in df.columns:
        print(f"[INFO] Column '{delta_col}' already exists. Skipping computation.")
        return df

    # --- Build tree if not provided ---
    if tree is None:
        sys.exit("No tree proveded. Please provide a pre-built PeriodicKDTree.")

    # --- Mean mass density ---
    box_volume = box_size**3
    total_mass = df[mass_column].sum()
    rho_mean = total_mass / box_volume
    print(f"Mean mass density: {rho_mean:.3e}")

    # --- Prepare output ---
    overdensity = np.zeros(len(df), dtype=np.float64)
    V_R = (4.0 / 3.0) * np.pi * radius**3

    print(f"Computing overdensity for R = {radius} h^-1 Mpc...")
    positions = df[['x', 'y', 'z']].values

    # --- For-loop over halos ---
    for i in range(len(df)):
        pos = positions[i]

        # periodic tree search
        indices = tree.search(pos, radius)

        local_mass = df.iloc[indices][mass_column].sum()
        rho_local = local_mass / V_R

        overdensity[i] = (rho_local - rho_mean) / rho_mean

        if i  == len(df)/2 and i > 0:
            print(f"  Processed {i:,} halos...")

    print("Done computing overdensity.")

    # --- Store result ---
    df[delta_col] = overdensity
    return df


################################################################
# NOT IN USE
################################################################

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
