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

def radial_velocity_and_error_pbc(
        halos: pd.DataFrame,
        origin: Tuple[float, float, float],
        box_size: float = 1000.0,
        error_frac: float = 0.20,
        min_sigma: float = 50.0) -> pd.DataFrame:
    """
    Compute line-of-sight unit vectors r_hat, radial velocities v_rad,
    per-object errors sigma, and PBC radius.
    
    Includes full periodic boundary conditions.
    
    New columns added:
        radius
        rhat_x, rhat_y, rhat_z
        v_rad
        sigma_vrad
    """

    required = ('x', 'y', 'z', 'vx', 'vy', 'vz')
    if not all(col in halos.columns for col in required):
        raise ValueError(f"halos must contain columns: {required}")

    pos = halos[['x', 'y', 'z']].values.astype(float)
    vel = halos[['vx', 'vy', 'vz']].values.astype(float)
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
        print(f"Warning: {zero_mask.sum()} halos at origin — adding tiny offset")
        r_norm[zero_mask] = 1e-8
        disp[zero_mask] += 1e-8

    # Unit vector r̂
    r_hat = disp / r_norm[:, None]

    # Radial velocity
    v_rad = np.sum(vel * r_hat, axis=1)

    # Error model
    sigma = np.maximum(np.abs(error_frac * v_rad), min_sigma)

    # -----------------------------
    # Add columns to the DataFrame
    # -----------------------------
    halos['radius_from_origin'] = r_norm
    halos['rhat_x'] = r_hat[:, 0]
    halos['rhat_y'] = r_hat[:, 1]
    halos['rhat_z'] = r_hat[:, 2]
    halos['v_rad'] = v_rad
    halos['sigma_vrad'] = sigma

    return halos
