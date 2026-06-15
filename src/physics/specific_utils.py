# specific_utils.py
import numpy as np
import pandas as pd
import os

from .config.bulkflow_config import BulkFlowConfig
from .config.cosmology_config import CosmologyConfig
from .config.mdpl2_config import MDPL2Config

ORIGIN_EPS = 1e-8

# ================================================================
# Periodic distance computations
# ================================================================

def periodic_distance(
    p1: np.ndarray,
    p2: np.ndarray,
    box_size: float
) -> float:
    """
    Compute periodic (minimum-image) distance between two points.

    Parameters
    ----------
    p1, p2 : array-like, shape (3,)
        Cartesian coordinates of the two points.
    box_size : float
        Size of the periodic box.

    Returns
    -------
    float
        Periodic Euclidean distance between p1 and p2.
    """

    p1 = np.asarray(p1)
    p2 = np.asarray(p2)

    if p1.shape != (3,) or p2.shape != (3,):
        raise ValueError("p1 and p2 must be 3-element vectors")

    # Displacement
    d = p1 - p2

    # Minimum-image convention
    d -= box_size * np.round(d / box_size)

    return np.linalg.norm(d)

def add_periodic_distance(
    df: pd.DataFrame,
    origin: np.ndarray,
    box_size: float,
    distance_col: str = "r"
) -> pd.DataFrame:
    """
    Add periodic (minimum-image) distance from an origin to a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['x', 'y', 'z'].
    origin : array-like, shape (3,)
        Origin point (x0, y0, z0).
    box_size : float
        Size of the periodic box.
    distance_col : str
        Name of the distance column to add.

    Returns
    -------
    pandas.DataFrame
        Copy of df with an added column `distance_col`.
    """

    origin = np.asarray(origin)

    if origin.shape != (3,):
        raise ValueError("origin must be a 3-element vector")

    # Displacement vector
    dx = df["x"].values - origin[0]
    dy = df["y"].values - origin[1]
    dz = df["z"].values - origin[2]

    # Minimum-image convention
    dx -= box_size * np.round(dx / box_size)
    dy -= box_size * np.round(dy / box_size)
    dz -= box_size * np.round(dz / box_size)

    # Euclidean distance
    r = np.sqrt(dx**2 + dy**2 + dz**2)

    df_out = df.copy()
    df_out[distance_col] = r

    return df_out

# ================================================================
# Create Radial velocity Column
# ================================================================

def radial_velocity_and_error_pbc(
        halos_df: pd.DataFrame,
        origin: tuple[float, float, float],
        box_size: float | None = None,
        error_frac: float | None = None,
        min_sigma: float | None = None) -> pd.DataFrame:
    """
    Compute line-of-sight unit vectors r_hat, radial velocities v_rad,
    per-object errors sigma, and PBC radius.

    Includes full periodic boundary conditions.

    Error model (distance-based):
        sigma_i = max( (H0/h) * r_i * error_frac , min_sigma )
    where r_i is the PBC distance to halo i (h^-1 Mpc) and H0/h converts
    distance to a Hubble velocity (km/s per h^-1 Mpc). This models surveys
    where fractional distance errors grow with distance.

    New columns added:
        radius_from_origin  -- PBC distance from origin (h^-1 Mpc)
        r_hat_x, r_hat_y, r_hat_z  -- line-of-sight unit vector components
        v_rad               -- radial (line-of-sight) velocity (km/s)
        sigma_v_rad         -- per-object measurement uncertainty (km/s)
    """

    required = ('x', 'y', 'z', 'vx', 'vy', 'vz')
    if not all(col in halos_df.columns for col in required):
        raise ValueError(f"halos_df must contain columns: {required}")

    if box_size is None:
        box_size = MDPL2Config().box_size
    if error_frac is None:
        error_frac = BulkFlowConfig().error_fraction
    if min_sigma is None:
        min_sigma = BulkFlowConfig().sigma_min

    pos = halos_df[['x', 'y', 'z']].values.astype(float)
    vel = halos_df[['vx', 'vy', 'vz']].values.astype(float)
    origin = np.array(origin, dtype=float)

    # -----------------------------
    # Apply periodic displacement
    # -----------------------------
    disp = pos - origin   # naïve displacement

    # Minimum‐image convention (PBC)
    disp -= box_size * np.round(disp / box_size)

    # Radius (PBC distance)
    r_norm = np.linalg.norm(disp, axis=1)

    # Handle halos exactly at the origin
    zero_mask = (r_norm == 0.0)
    if np.any(zero_mask):
        r_norm[zero_mask] = ORIGIN_EPS
        disp[zero_mask] += ORIGIN_EPS

    # Unit vector r̂
    r_hat = disp / r_norm[:, None]

    # Radial velocity
    v_rad = np.sum(vel * r_hat, axis=1)

    # Distance-based error model:
    #   sigma_i = max( (H0/h) * r_i * error_frac , min_sigma )
    # H0/h converts h^-1 Mpc to km/s (≈ 100 km/s per h^-1 Mpc for MDPL2).
    hubble_velocity_per_dist = CosmologyConfig().H0 / MDPL2Config().HubbleParameter
    sigma = np.maximum(hubble_velocity_per_dist * r_norm * error_frac, min_sigma)

    # -----------------------------
    # Add columns to the DataFrame
    # -----------------------------
    halos_df['radius_from_origin'] = r_norm
    halos_df['r_hat_x'] = r_hat[:, 0]
    halos_df['r_hat_y'] = r_hat[:, 1]
    halos_df['r_hat_z'] = r_hat[:, 2]
    halos_df['v_rad'] = v_rad
    halos_df['sigma_v_rad'] = sigma

    return halos_df

