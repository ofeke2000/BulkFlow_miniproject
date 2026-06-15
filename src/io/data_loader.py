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


def _read_cf4_csv(path: str) -> pd.DataFrame:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
        second_line = f.readline()
        third_line = f.readline()
        fourth_line = f.readline()
        fifth_line = f.readline()

    # Many CF4 files use a 5-line format header:
    #   1) human-readable title row
    #   2) machine-readable column names row
    #   3) FORTRAN-style format row
    #   4) units row
    #   5) description row
    if second_line.lower().startswith('pgc') and third_line.strip().startswith('%'):
        df = pd.read_csv(path, header=1, skiprows=[2, 3, 4])
    else:
        df = pd.read_csv(path)

    return df


def load_cf4_catalogue(path: str, h: float = 0.7) -> pd.DataFrame:
    """
    Load the CosmicFlows-4 (CF4) groups catalogue.

    Parameters
    ----------
    path : str
        Path to the CF4 groups CSV file.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with Cartesian coordinates in columns ['x', 'y', 'z'].
    """
    print(f"Loading CF4 catalogue from {path}...")
    df = _read_cf4_csv(path)
    print(f"Loaded {len(df):,} CF4 entries.")

    # Normalize column names
    rename_map = {
        'pgc': 'id',
        'PGC': 'id',
        'RA': 'ra',
        'DE': 'dec',
        'Vcmb': 'vcmb',
        'Vcmbm': 'vcmb',
    }
    df = df.rename(columns=rename_map)

    # Normalize distance fields
    if 'D' in df.columns and 'distance' not in df.columns:
        df['distance'] = df['D'] * h
    elif 'distance_modulus' not in df.columns:
        for candidate in ('DM_av', 'DM_av', 'DM_zp', 'DMzp', 'DM_zp'):
            if candidate in df.columns:
                df['distance_modulus'] = df[candidate]
                break

    if 'distance' not in df.columns and 'distance_modulus' in df.columns:
        df['distance'] = distance_modulus_to_mpc(df['distance_modulus'].values, h=h)

    required = {'ra', 'dec', 'distance'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CF4 catalogue is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=['distance', 'ra', 'dec'])
    print(f"After cleaning: {len(df):,} groups remain.")

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
    d_l_mpc = (10 ** ((mu + 5) / 5))/(10**6)

    # Convert to comoving distance in Mpc/h
    distance = d_l_mpc * h
    return distance
