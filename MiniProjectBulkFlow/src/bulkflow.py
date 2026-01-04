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
from scipy.linalg import lu_factor, lu_solve
from .specific_utils import radial_velocity_and_error_pbc

def bulk_flow_chi2_cumulative(
    r_hat: np.ndarray,
    r_sorted: np.ndarray,
    r_list: list,
    v_rad: np.ndarray,
    sigma: np.ndarray,
    sigma_star: float = 250.0
) -> pd.DataFrame:
    """
    Cumulative chi^2 bulk-flow estimator.

    Parameters
    ----------
    r_hat : (N,3) array
        Unit vectors to halos.
    r_sorted : (N,) array
        Radii of halos (sorted ascending).
    r_list : array-like
        Radii at which to compute the bulk flow.
    v_rad : (N,) array
        Radial velocities.
    sigma : (N,) array
        Measurement uncertainties.
    sigma_star : float
        Nonlinear dispersion term.

    Returns
    -------
    pandas.DataFrame
        Columns: [radius, u_x, u_y, u_z, U_total]
    """

    r_list = np.asarray(r_list)

    # Total variance
    sigma2 = sigma**2 + sigma_star**2
    w = 1.0 / sigma2

    # Initialize cumulative A and b
    A = np.zeros((3, 3))
    b = np.zeros(3)

    results = []
    idx = 0
    N = len(r_sorted)

    for R in r_list:
        # Accumulate until we reach this radius
        while idx < N and r_sorted[idx] <= R:
            rh = r_hat[idx]
            weight = w[idx]

            A += weight * np.outer(rh, rh)
            b += weight * v_rad[idx] * rh

            idx += 1

        # Solve A u = b using LU decomposition
        try:

            lu, piv = lu_factor(A)
            u = lu_solve((lu, piv), b)
            
        except Exception:
            print("\n=== BULK FLOW SOLVER FAILURE ===")
            print(f"Radius R = {R}")
            print(f"Number of objects used = {idx}")

            print("\nMatrix A:")
            print(f"idx = {idx}")
            print(f"Radius R = {R}")
            print(A)

            detA = np.linalg.det(A)
            print(f"\ndet(A) = {detA:.3e}")

            eigvals = np.linalg.eigvalsh(A)  # symmetric → more stable
            print("\nEigenvalues of A:")
            print(eigvals)

            cond = np.inf
            if np.min(np.abs(eigvals)) > 0:
                cond = np.max(np.abs(eigvals)) / np.min(np.abs(eigvals))
            print(f"\nCondition number estimate = {cond:.3e}")

            print("================================\n")

            u = np.array([np.nan, np.nan, np.nan])


        U = np.linalg.norm(u)

        results.append([R, u[0], u[1], u[2], U])

    return pd.DataFrame(
        results,
        columns=["radius", "u_x", "u_y", "u_z", "U_total"]
    )



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
    
    print("[INFO] Radial velocities and errors computed.")

    # ---------------------------------------------------------
    # (2) radius list: r_min → r_max in steps of r_jumps
    # ---------------------------------------------------------
    r_list = np.arange(r_min, r_max + r_jumps, r_jumps)
    r_list = np.round(r_list).astype(int)

    # ---------------------------------------------------------
    # (3) extract arrays
    # ---------------------------------------------------------
    r_hat = halos_df[["r_hat_x","r_hat_y","r_hat_z"]].values
    v_rad = halos_df["v_rad"].values
    sigma = halos_df["sigma_v_rad"].values
    r_sorted = halos_df["radius_from_origin"].values

    # Ensure halos are sorted by radius (required for series Ainv construction)
    sort_idx = np.argsort(r_sorted)
    r_sorted = r_sorted[sort_idx]
    r_hat    = r_hat[sort_idx]
    v_rad    = v_rad[sort_idx]
    sigma    = sigma[sort_idx]

    # ---------------------------------------------------------
    # (4) Compute bulk flow table
    # ---------------------------------------------------------
    velocities_df = bulk_flow_chi2_cumulative(
        r_hat=r_hat,
        r_sorted=r_sorted,
        r_list=r_list,
        v_rad=v_rad,
        sigma=sigma,
        sigma_star=sigma_star
    )

    return velocities_df