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


def load_cf4_catalogue(path: str) -> pd.DataFrame:
    """
    Load the CosmicFlows-4 (CF4) groups catalogue.

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
    print(f"Loading CF4 catalogue from {path}...")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} CF4 entries.")

    # Rename for consistency
    rename_map = {
        'PGC': 'id',
        'RA': 'ra',
        'DE': 'dec',
        'DM_av': 'distance_modulus',
        'Vcmb': 'Vcmb'
    }
    df = df.rename(columns=rename_map)

    # Remove entries with missing distances
    df = df.dropna(subset=['distance_modulus'])
    print(f"After cleaning: {len(df):,} groups remain.")

    # Convert distance modulus to distance in Mpc/h
    df['distance'] = distance_modulus_to_mpc(df['distance_modulus'].values)

    # Convert coordinates to Cartesian
    df = cf4_to_cartesian(df)
    print("Converted RA/Dec/Dist → Cartesian (x, y, z).")


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


def distance_modulus_to_mpc(mu: float | np.ndarray, h: float = 0.7) -> np.ndarray:
    """
    Convert distance modulus to comoving distance in Mpc/h.

    Parameters
    ----------
    mu : float or array-like
        Distance modulus (m - M).
    h : float, optional
        Hubble parameter normalization (H0 / 100). Default is 0.7.

    Returns
    -------
    distance_mpch : numpy.ndarray
        Comoving distance in Mpc/h.
    """
    # Convert to luminosity distance in Mpc
    d_l_mpc = (10 ** ((mu - 25) / 5))/(10**6)

    # Convert to comoving distance in Mpc/h
    distance = d_l_mpc * h
    return distance
