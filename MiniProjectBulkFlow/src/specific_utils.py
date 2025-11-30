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

def radial_velocity_and_error(halos: pd.DataFrame,
                              origin: Tuple[float, float, float],
                              error_frac: float = 0.20,
                              min_sigma: float = 50.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute line-of-sight unit vectors r_hat, radial velocities v_rad, and per-object errors sigma.

    Parameters
    ----------
    halos : pd.DataFrame
        halo dataframe with columns ['x','y','z','vx','vy','vz'] (positions in h^-1 Mpc,
        velocities in km/s or same units as desired).
    origin : 3-tuple of floats
        (x0, y0, z0) observer / origin coordinates in the same units as halo positions.
    error_frac : float
        Fractional error to apply to radial velocities (default 0.20 for 20%).
        sigma_i = max(error_frac * |v_rad_i|, min_sigma).
    min_sigma : float
        Minimum per-object error to avoid zero uncertainties (in same units as velocity).

    Returns
    -------
    v_rad : (N,) ndarray
        radial velocities (v · r_hat)
    r_hat : (N,3) ndarray
        unit vectors from origin to halo
    sigma : (N,) ndarray
        per-halo uncertainties
    """
    if not all(col in halos.columns for col in ('x', 'y', 'z', 'vx', 'vy', 'vz')):
        raise ValueError("halos must contain columns: 'x','y','z','vx','vy','vz'")

    pos = halos[['x', 'y', 'z']].values.astype(float)
    vel = halos[['vx', 'vy', 'vz']].values.astype(float)

    # displacement vector from origin to object
    disp = pos - np.array(origin, dtype=float).reshape((1, 3))
    r_norm = np.linalg.norm(disp, axis=1)

    ###########################################################
    # Need to check what to do for objects exactly at origin
    ###########################################################

    # handle objects exactly at origin (avoid divide-by-zero)
    zero_mask = (r_norm == 0.0)
    if np.any(zero_mask):
        logging.warning("Found {0} halos exactly at the origin. Setting tiny offset to avoid singular rhat."
                        .format(zero_mask.sum()))
        r_norm[zero_mask] = 1e-8
        disp[zero_mask] += 1e-8

    r_hat = disp / r_norm[:, None]  # shape (N,3)

    # radial velocity: projection of velocity onto line-of-sight unit vector
    v_rad = np.sum(vel * r_hat, axis=1)

    # simple fractional error model (user-specified)
    sigma = np.maximum(np.abs(error_frac * v_rad), min_sigma)

    return v_rad, r_hat, sigma