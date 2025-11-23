# specific_utils.py
import numpy as np
import pandas as pd


def weighted_average(values, weights):
    """Compute weighted average safely."""
    values = np.asarray(values)
    weights = np.asarray(weights)
    if np.sum(weights) == 0:
        return np.nan
    return np.sum(values * weights) / np.sum(weights)

# ================================================================
# Distance computations
# ================================================================

def distance(x1, y1, z1, x2, y2, z2):
    """Compute Euclidean distance between two points or arrays."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

# ================================================================
# Create Radial velocity Column
# ================================================================

def create_radial_velocities(df: pd.DataFrame, position: np.ndarray, calculate_radial_velocities_vector: int = 0) -> pd.DataFrame:
    """
    Compute the radial velocity VECTOR of halos relative to a given position
    and add it as 3 new columns: vr_x, vr_y, vr_z.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain: x, y, z, vx, vy, vz
    position : np.ndarray
        3-element array (x0, y0, z0)

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with new columns: vr_x, vr_y, vr_z
    """

    # Extract positions and velocities
    pos = df[['x', 'y', 'z']].values          # shape (N,3)
    vel = df[['vx', 'vy', 'vz']].values       # shape (N,3)

    # Vector from reference position to halo
    r_vec = pos - position                    # shape (N,3)

    # Distance
    r = np.linalg.norm(r_vec, axis=1)

    # Unit vector r_hat
    r_hat = np.zeros_like(r_vec)
    nonzero = r > 0
    r_hat[nonzero] = r_vec[nonzero] / r[nonzero, np.newaxis]

    # Radial velocity scalar: vr = v · r_hat
    vr_scalar = np.sum(vel * r_hat, axis=1)   # shape (N,)

    # Add radial velocity scalar column
    df['vr'] = vr_scalar

    if calculate_radial_velocities_vector == 1:
        # Radial velocity vector: vr_vec = vr * r_hat
        vr_vec = vr_scalar[:, np.newaxis] * r_hat  # shape (N,3)

        # Add radial velocity vector columns
        df['vr_x'] = vr_vec[:, 0]
        df['vr_y'] = vr_vec[:, 1]
        df['vr_z'] = vr_vec[:, 2]


    return df