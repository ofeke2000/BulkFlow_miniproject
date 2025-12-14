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

from typing import Iterable
import numpy as np
import pandas as pd
from .specific_utils import radial_velocity_and_error_pbc

########################################################################
# Compute A^-1 for multiple radii
########################################################################
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
# Compute u_i(R) for single radii
##########################################################################

def compute_bulk_flow_MLE_single_radius(
    r_hat: np.ndarray,
    r_sorted: np.ndarray,
    radius: float,
    v_rad: np.ndarray,
    sigma: np.ndarray,
    Ainv: np.ndarray,
    sigma_star: float = 250.0,
):
    """
    Compute the MLE bulk flow vector u(R) for a single radius,
    using only halos inside the radius.

    Parameters
    ----------
    r_hat : (N, 3)
        Unit vectors for each halo.
    r_sorted : (N,)
        Radii of halos (must be sorted ascending).
    radius : float
        Radius R at which to compute the bulk flow.
    v_rad : (N,)
        Radial peculiar velocities.
    sigma : (N,)
        Measurement uncertainties.
    Ainv : (3,3)
        Inverse A matrix for this radius.
    sigma_star : float
        Nonlinear dispersion parameter (default 250 km/s).

    Returns
    -------
    dict : {radius : u_R}
        u_R is a numpy array of shape (3,) with [u_x, u_y, u_z].
    """

    # ---- Filter halos within R ----
    mask = r_sorted <= radius

    r_hat_R   = r_hat[mask]
    v_rad_R   = v_rad[mask]
    sigma_R   = sigma[mask]

    # ---- MLE Weights ----
    denom = sigma_R**2 + sigma_star**2
    s = 1.0 / denom   # shape (N_R,)

    # (N_R, 3) @ (3,3) → (N_R, 3)
    projected = r_hat_R @ Ainv.T

    # Multiply rows by s[n]
    w = projected * s[:, None]

    # Multiply by radial velocities
    weighted_v = w * v_rad_R[:, None]

    # ---- Bulk flow: sum over all halos ----
    u_R = np.sum(weighted_v, axis=0)

    return {radius: u_R}

##########################################################################
# Compute u_i(R) for multiple radii
##########################################################################

def compute_bulk_flow_table(
    r_hat: np.ndarray,
    r_sorted: np.ndarray,
    r_jumps: np.ndarray,
    v_rad: np.ndarray,
    sigma: np.ndarray,
    Ainv_dict: dict,
    sigma_star: float = 250.0
):
    """
    Compute bulk flow (MLE) at multiple radii and return a DataFrame.

    Parameters
    ----------
    r_hat : (N,3)
        Unit vectors for halos.
    r_sorted : (N,)
        Radii of halos (sorted ascending).
    r_jumps : array-like
        Radii at which to compute the bulk flow.
    v_rad : (N,)
        Radial velocities.
    sigma : (N,)
        Measurement uncertainties.
    Ainv_dict : dict
        Mapping radius -> 3x3 A^{-1}(R) matrix.
    sigma_star : float
        Nonlinear dispersion term σ*.

    Returns
    -------
    DataFrame
        Columns: [radius, u_x, u_y, u_z, U_total]
    """

    results = {
        "radius": [],
        "u_x": [],
        "u_y": [],
        "u_z": [],
        "U_total": []
    }

    for R in r_jumps:

        Ainv = Ainv_dict[R]   # 3×3 inverse A at this radius

        # --- compute bulk flow for R ---
        u_R = compute_bulk_flow_MLE_single_radius(
            r_hat=r_hat,
            r_sorted=r_sorted,
            radius=R,
            v_rad=v_rad,
            sigma=sigma,
            Ainv=Ainv,
            sigma_star=sigma_star,
        )[R]

        ux, uy, uz = u_R
        Utot = np.sqrt(ux**2 + uy**2 + uz**2)

        results["radius"].append(R)
        results["u_x"].append(ux)
        results["u_y"].append(uy)
        results["u_z"].append(uz)
        results["U_total"].append(Utot)

    # convert to DataFrame
    df = pd.DataFrame(results)
    return df


##########################################################################
# Bulk flow series (only function used in main code)
##########################################################################

def calculate_bulk_flow_series(
    halos_df: pd.DataFrame,
    origin: tuple,
    r_max: float,
    r_min: float,
    r_jumps: float,
    error_frac: float = 0.20,
    sigma_star: float = 250.0,
    sigma_min: float = 50.0
):
    """
    Compute the bulk flow as a function of radius, using the series (incremental)
    calculation of A and A^{-1}.

    Parameters
    ----------
    halos_df : DataFrame
        Must contain columns ['x','y','z','vx','vy','vz'].
    origin : tuple of 3 floats
        Point around which the radial velocities and radii are computed.
    r_max, r_min : float
        Minimum and maximum radius to evaluate.
    r_jumps : float
        Radius step between evaluations.
    error_frac : float
        Fractional velocity error used by radial_velocity_and_error_pbc.
    sigma_star : float
        Small-scale velocity parameter.
    sigma_min : float
        Floor on sigma (passed into radial_velocity_and_error_pbc).

    Returns
    -------
    velocities_df : DataFrame
        Columns: [radius, u_x, u_y, u_z, U_total]
    """

    # ---------------------------------------------------------
    # (1) Compute radial velocity, error, radii, unit vector
    # ---------------------------------------------------------
    halos_df = radial_velocity_and_error_pbc(
            halos_df,
            origin=origin,
            box_size=1000.0,
            error_frac=error_frac,
            min_sigma=sigma_min
        )

    # ---------------------------------------------------------
    # (2) radius list: r_min → r_max in steps of r_jumps
    # ---------------------------------------------------------
    r_list = np.arange(r_min, r_max + r_jumps, r_jumps)

    # ---------------------------------------------------------
    # (3) extract arrays
    # ---------------------------------------------------------
    r_hat = halos_df[["r_hat_x","r_hat_y","r_hat_z"]].values
    v_rad = halos_df["v_rad"].values
    sigma = halos_df["sigma"].values
    r_sorted = halos_df["r"].values

    # Ensure halos are sorted by radius (required for series Ainv construction)
    sort_idx = np.argsort(r_sorted)
    r_sorted = r_sorted[sort_idx]
    r_hat    = r_hat[sort_idx]
    v_rad    = v_rad[sort_idx]
    sigma    = sigma[sort_idx]

    # ---------------------------------------------------------
    # (4) Compute A^{-1}(R) dict
    # ---------------------------------------------------------
    Ainv_dict = compute_Ainv(
        r_hat=r_hat,
        sigma=sigma,
        r_sorted=r_sorted,
        r_jumps=r_list
    )

    # ---------------------------------------------------------
    # (5) Compute bulk flow table
    # ---------------------------------------------------------
    velocities_df = compute_bulk_flow_table(
        r_hat=r_hat,
        r_sorted=r_sorted,
        r_list=r_list,
        v_rad=v_rad,
        sigma=sigma,
        Ainv_dict=Ainv_dict,
        sigma_star=sigma_star
    )

    return velocities_df