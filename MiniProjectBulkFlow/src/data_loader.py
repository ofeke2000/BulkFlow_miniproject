"""
data_loader.py
----------------
Handles loading of simulation (Rockstar) and observational (CosmicFlows-4) catalogs.
Performs basic cleaning and coordinate conversions.

The goal: provide ready-to-use pandas DataFrames for analysis and masking.
"""

import pandas as pd
import numpy as np


def load_rockstar_catalog(path: str, columns=None) -> pd.DataFrame:
    """
    Load the Rockstar halo catalog.

    Parameters
    ----------
    path : str
        Path to the CSV file containing the halo catalog.
    columns : list[str], optional
        Columns to load (for memory efficiency). If None, load all.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns: mvir, rvir, rs, rockstarid, pid, x, y, z, vx, vy, vz
    """
    print(f"Loading Rockstar catalog from {path}...")
    df = pd.read_csv(path, usecols=columns)
    print(f"Loaded {len(df):,} halos.")
    return df


def load_cf4_catalog(path: str) -> pd.DataFrame:
    """
    Load the CosmicFlows-4 (CF4) groups catalog.

    Parameters
    ----------
    path : str
        Path to the CF4 groups CSV file.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with relevant columns such as:
        'GroupID', 'RA', 'Dec', 'Dist_Mpc', 'Vpec', 'sigma_D', etc.
    """
    print(f"Loading CF4 catalog from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} CF4 entries.")

    # Rename for consistency
    rename_map = {
        'GroupID': 'id',
        'RA': 'ra',
        'Dec': 'dec',
        'Dist_Mpc': 'distance',
        'Vpec': 'vpec'
    }
    df = df.rename(columns=rename_map)

    # Remove entries with missing distances
    df = df.dropna(subset=['distance'])
    print(f"After cleaning: {len(df):,} groups remain.")
    return df


def cf4_to_cartesian(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert RA, Dec, distance [Mpc/h] to Cartesian coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        CF4 catalog with 'ra', 'dec', 'distance' columns.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with added 'x', 'y', 'z' in h^-1 Mpc.
    """
    ra_rad = np.deg2rad(df['ra'])
    dec_rad = np.deg2rad(df['dec'])

    df['x'] = df['distance'] * np.cos(dec_rad) * np.cos(ra_rad)
    df['y'] = df['distance'] * np.cos(dec_rad) * np.sin(ra_rad)
    df['z'] = df['distance'] * np.sin(dec_rad)
    return df
