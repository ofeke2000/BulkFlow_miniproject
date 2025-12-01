"""
bulkflow.py
-----------
Maximum-likelihood bulk flow estimator (weighted least-squares) using radial velocities.

Functions
---------
- radial_velocity_and_error(halos, origin, error_frac=0.20, min_sigma=50.0)
    compute line-of-sight unit vectors, radial velocities and per-object errors

- ml_bulk_flow(v_rad, r_hat, sigma)
    compute ML estimate of bulk flow vector U and its covariance for given inputs

- compute_bulkflow_series(halos_df, origin, radii, error_frac=0.20, min_sigma=50.0,
                          min_count=10, cumulative=True)
    compute bulk flow (and uncertainties) for a series of radii (cumulative by default)

Notes / assumptions
-------------------
* The module assumes halo velocities (vx, vy, vz) are in the same units as you want the
  bulk flow result (typically km/s). Positions (x,y,z) are in comoving h^-1 Mpc.
* "error_frac" is interpreted as a fractional uncertainty on the radial velocity:
    sigma_i = max(error_frac * |v_rad_i|, min_sigma)
  This is a conservative and simple choice. If you meant a fractional distance error
  or some other observational uncertainty model, see the notes below and I can adapt.
* The ML estimator fits v_rad_i = r_hat_i . U + noise with Gaussian noise sigma_i.
  The solution is U = A^{-1} b, where
    A = sum_i ( r_hat_i r_hat_i^T / sigma_i^2 ),   b = sum_i ( v_rad_i r_hat_i / sigma_i^2 )
  Cov(U) = A^{-1}.
"""

from typing import Iterable, Tuple
import numpy as np
import pandas as pd
import logging
import os
from specific_utils import radial_velocity_and_error

logging.getLogger(__name__).addHandler(logging.NullHandler())

###########################################################################
# Compute A^-1 for multiple radii
###########################################################################

import numpy as np

def compute_Ainv(r_hat: np.ndarray, 
                 sigma: np.ndarray, 
                 r_sorted: np.ndarray, 
                 r_jumps: Iterable[float]) -> dict:
    """
    Compute A^{-1}(R) incrementally assuming halos are sorted by radius.

    Parameters
    ----------
    r_hat      : array (N, 3)
        Unit vectors.
    sigma      : array (N,)
        Velocity uncertainties.
    r_sorted   : array (N,)
        Radii of halos, sorted in increasing order.
    r_jumps    : list/array
        Radii at which A^{-1}(R) is saved (sorted ascending).

    Returns
    -------
    dict : {R : Ainv_R}
    """

    small_scale_velocities = 250.0  # km/s
    Sigma_Star = small_scale_velocities * np.ones(len(sigma), dtype=float)
    # Weight = 1 / sigma^2
    W = 1.0 / (sigma**2 + Sigma_Star**2)

    # Running A matrix (3x3)
    A = np.zeros((3, 3), dtype=float)

    Ainv_dict = {}
    jump_index = 0
    current_jump = r_jumps[jump_index]

    for i in range(len(r_sorted)):

        # If we crossed a jump radius, compute and save A^{-1}
        while jump_index < len(r_jumps) and r_sorted[i] >= current_jump:

            # Try to invert the matrix
            try:
                Ainv_dict[current_jump] = np.linalg.inv(A.copy())
            except np.linalg.LinAlgError:
                Ainv_dict[current_jump] = np.full((3, 3), np.nan)

            jump_index += 1
            if jump_index < len(r_jumps):
                current_jump = r_jumps[jump_index]
            else:
                break

        # Add this halo to A
        A += W[i] * np.outer(r_hat[i], r_hat[i])    

        if jump_index >= len(r_jumps):
            break

    return Ainv_dict


##########################################################################
# Compute w_i(R) for multiple radii
##########################################################################

def compute_w_mle(
    r_hat: np.ndarray,
    sigma: np.ndarray,
    Ainv_dict: dict,
    sigma_star: float = 250.0
):
    """
    Compute the MLE weights w_{i,n} for all radii in Ainv_dict.

    Parameters
    ----------
    r_hat : (N, 3)
        Unit vectors for each halo.
    sigma : (N,)
        Measurement uncertainties for each halo.
    Ainv_dict : dict
        Dictionary mapping radius R -> A^{-1}(R)  (3x3 matrices).
    sigma_star : float
        Nonlinear dispersion term σ* (default 250 km/s).

    Returns
    -------
    dict : {R : w_R}
        w_R is an array of shape (N, 3), with w_R[n,i] = w_{i,n}.
    """

    N = len(sigma)
    denom = sigma**2 + sigma_star**2  # denominator for all halos
    s = 1.0 / denom                   # (N,)

    w_dict = {}

    for R, Ainv in Ainv_dict.items():

        # (N,3) * (3,3) → (N,3)
        # Each halo row gets multiplied by A^{-1}
        projected = r_hat @ Ainv.T   # shape: (N, 3)

        # Multiply each row by s[n]
        w = projected * s[:, None]   # broadcast (N,1)

        w_dict[R] = w

    return w_dict


##########################################################################
# Maximum-likelihood bulk flow estimator
##########################################################################


def MLE_bulk_flow(v_rad: np.ndarray, r_hat: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Maximum-likelihood bulk flow estimator for a single sample.

    Parameters
    ----------
    v_rad : (N,) array
        radial velocities
    r_hat : (N,3) array
        line-of-sight unit vectors
    sigma : (N,) array
        per-object uncertainties

    Returns
    -------
    U : (3,) array
        estimated bulk flow vector (Ux,Uy,Uz)
    covU : (3,3) array
        covariance matrix of U
    """
    if len(v_rad) != len(r_hat) or len(v_rad) != len(sigma):
        raise ValueError("v_rad, r_hat and sigma must have matching lengths")

    # Build A and w
    A = np.zeros((3, 3), dtype=float)
    w = np.zeros(3, dtype=float)

    inv_sigma2 = 1.0 / (sigma ** 2)

    # accumulate
    for i in range(len(v_rad)):
        ri = r_hat[i].reshape(3, 1)  # column
        A += inv_sigma2[i] * (ri @ ri.T)
        w += inv_sigma2[i] * v_rad[i] * r_hat[i]

    # solve
    # Regularize if A is nearly singular (tiny Tikhonov)
    try:
        covU = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        logging.warning("A matrix singular or ill-conditioned: adding small regularization.")
        reg = np.eye(3) * 1e-8 * np.mean(np.diag(A))
        covU = np.linalg.inv(A + reg)

    U = covU @ w
    return U, covU

##########################################################################
# Bulk flow series (Not Good)
##########################################################################

def compute_bulkflow_series(halos_df: pd.DataFrame,
                            origin: Tuple[float, float, float],
                            radii: Iterable[float],
                            error_frac: float = 0.20,
                            min_sigma: float = 50.0,
                            min_count: int = 10,
                            cumulative: bool = True) -> pd.DataFrame:
    """
    Compute ML bulk-flow estimates for a series of radii.

    Parameters
    ----------
    halos_df : pd.DataFrame
        Halo catalog with columns ['x','y','z','vx','vy','vz'].
    origin : tuple
        (x0,y0,z0) coordinates (h^-1 Mpc) of the observer / origin.
    radii : iterable of floats
        Radii at which to compute bulk flow (h^-1 Mpc). If cumulative=True, halos with
        distance <= R are used for radius R. Otherwise (cumulative=False) only halos
        within shell (R-dR/2, R+dR/2) are used; but here we implement simple shells
        by computing between consecutive radii if desired (user can pass appropriate bins).
    error_frac : float
        Fractional error for radial velocities (default 0.20).
    min_sigma : float
        Minimum sigma in velocity units.
    min_count : int
        Minimum number of objects required to compute a result; otherwise NaNs are returned.
    cumulative : bool
        If True (default), use cumulative sphere up to R. If False, use halos with distance
        strictly within (prev_R, R] for successive radii; in that case pass radii as bin-edges.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns:
        ['R', 'N', 'Ux','Uy','Uz','U_mag','sigma_U', 'sigma_Ux','sigma_Uy','sigma_Uz']
    """
    # precompute displacements and radii of all halos relative to origin
    pos = halos_df[['x', 'y', 'z']].values.astype(float)
    disp = pos - np.array(origin, dtype=float).reshape((1, 3))
    r_vals = np.linalg.norm(disp, axis=1)

    results = []
    radii_list = list(radii)

    for idx_R, R in enumerate(radii_list):
        if cumulative:
            mask = (r_vals <= R)
        else:
            # use shells defined by previous radius (if idx_R == 0, use (0,R])
            if idx_R == 0:
                mask = (r_vals <= R)
            else:
                Rprev = radii_list[idx_R - 1]
                mask = (r_vals > Rprev) & (r_vals <= R)

        sub = halos_df.loc[mask]
        N = len(sub)
        if N < min_count:
            logging.info(f"R={R}: N={N} < min_count ({min_count}) -> returning NaNs.")
            results.append({
                'R': R, 'N': N,
                'Ux': np.nan, 'Uy': np.nan, 'Uz': np.nan,
                'U_mag': np.nan, 'sigma_U': np.nan,
                'sigma_Ux': np.nan, 'sigma_Uy': np.nan, 'sigma_Uz': np.nan
            })
            continue

        # compute v_rad, r_hat, sigma for this sub-sample
        v_rad, r_hat, sigma = radial_velocity_and_error(sub, origin, error_frac=error_frac, min_sigma=min_sigma)

        # ML estimate
        U, covU = ml_bulk_flow(v_rad, r_hat, sigma)

        U_mag = np.linalg.norm(U)
        # sigma_U taken as sqrt(trace(covU))/sqrt(3) as a single-number uncertainty (rms)
        sigma_U = np.sqrt(np.trace(covU) / 3.0)
        sigma_Ux, sigma_Uy, sigma_Uz = np.sqrt(np.diag(covU))

        results.append({
            'R': R, 'N': N,
            'Ux': U[0], 'Uy': U[1], 'Uz': U[2],
            'U_mag': U_mag, 'sigma_U': sigma_U,
            'sigma_Ux': sigma_Ux, 'sigma_Uy': sigma_Uy, 'sigma_Uz': sigma_Uz
        })

    results_df = pd.DataFrame(results)
    return results_df
