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
* "error_frac" is interpreted as a fractional distance error used to compute the
  measurement uncertainty on the radial velocity:
    sigma_i = max( (H0/h) * r_i * error_frac , min_sigma )
  where r_i is the PBC distance to halo i (h^-1 Mpc) and H0/h ≈ 100 km/s per h^-1 Mpc
  converts to a Hubble velocity. This models peculiar-velocity surveys where distance
  errors grow with distance.
* The ML estimator fits v_rad_i = r_hat_i . U + noise with Gaussian noise sigma_i.
  The solution is U = A^{-1} b, where
    A = sum_i ( r_hat_i r_hat_i^T / sigma_i^2 ),   b = sum_i ( v_rad_i r_hat_i / sigma_i^2 )
  Cov(U) = A^{-1}.
"""

from typing import Iterable
import numpy as np
import pandas as pd
import logging
from scipy.spatial import cKDTree
from scipy.linalg import lu_factor, lu_solve
from ..config.bulkflow_config import BulkFlowConfig
from .specific_utils import radial_velocity_and_error_pbc

_EYE3 = np.eye(3)  # pre-allocated identity for covariance computation

###########################################################################
# Bulk flow series (cumulative chi^2 ML estimator)
###########################################################################

def bulk_flow_chi2_cumulative(
    r_hat: np.ndarray,
    r_sorted: np.ndarray,
    r_list: list,
    v_rad: np.ndarray,
    sigma: np.ndarray,
    sigma_star: float | None = None
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
        Columns: [radius, u_x, u_y, u_z, U_total, U_debiased, sigma_U, n_used]
        sigma_U is the propagated uncertainty on the bulk-flow magnitude:
            sigma_U = sqrt(u^T · A^{-1} · u) / |u|
        U_debiased removes the noise bias from the magnitude:
            U_debiased = sqrt( max(U_total^2 - Tr(A^{-1}), 0) )
        n_used is the cumulative count of halos used at each radius.
    """

    r_list = np.asarray(r_list)

    if sigma_star is None:
        sigma_star = BulkFlowConfig().sigma_star

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
            U = np.linalg.norm(u)
            if U > 0:
                # Covariance of u is A^{-1}; propagate to scalar sigma_U
                cov = lu_solve((lu, piv), _EYE3)
                sigma_U = float(np.sqrt(u @ cov @ u)) / U
                # Noise-bias-debiased magnitude: E[|U|^2] = |B_true|^2 + Tr(A^{-1})
                # A valid covariance matrix must have Tr(A^{-1}) > 0; a negative
                # trace signals a near-singular A (too few halos), so we return NaN.
                trace_cov = float(np.trace(cov))
                if trace_cov > 0:
                    U_debiased = float(np.sqrt(max(U**2 - trace_cov, 0.0)))
                else:
                    U_debiased = np.nan
            else:
                sigma_U = np.nan
                U_debiased = np.nan
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
            U = np.nan
            sigma_U = np.nan
            U_debiased = np.nan

        results.append([R, u[0], u[1], u[2], U, U_debiased, sigma_U, idx])

    return pd.DataFrame(
        results,
        columns=["radius", "u_x", "u_y", "u_z", "U_total", "U_debiased", "sigma_U", "n_used"]
    )

##########################################################################
# Bulk flow series (cumulative mean estimator)
##########################################################################

def bulk_flow_mean_cumulative(
    r_hat: np.ndarray,
    r_sorted: np.ndarray,
    r_list: list,
    v_rad: np.ndarray,
    sigma: np.ndarray,
    vel: np.ndarray,
    sigma_star: float | None = None
) -> pd.DataFrame:
    """
    Cumulative mean bulk-flow estimator.

    Computes the unweighted average of the true 3D velocities inside each
    radius R (cumulative):
        u = mean( [vx, vy, vz] )  for halos with r <= R

    Parameters
    ----------
    r_hat : (N,3) array
        Unit vectors to halos (sorted ascending by radius, unused here but
        kept for a consistent call signature).
    r_sorted : (N,) array
        Radii of halos (sorted ascending).
    r_list : array-like
        Radii at which to compute the bulk flow.
    v_rad : (N,) array
        Radial velocities (unused in mean estimator, kept for call-signature
        consistency).
    sigma : (N,) array
        Measurement uncertainties (unused in mean estimator).
    vel : (N,3) array
        True 3D velocities [vx, vy, vz] of halos, sorted by radius in the
        same order as r_sorted / r_hat / v_rad.
    sigma_star : float, optional
        Nonlinear dispersion term (unused in mean estimator).

    Returns
    -------
    pandas.DataFrame
        Columns: [radius, u_x, u_y, u_z, U_total, U_debiased, sigma_U, n_used]
        sigma_U and U_debiased are NaN for the mean estimator (no analytic
        covariance available).
        n_used is the cumulative count of halos at each radius.
    """

    r_list = np.asarray(r_list)

    if sigma_star is None:
        sigma_star = BulkFlowConfig().sigma_star

    results = []
    idx = 0
    N = len(r_sorted)

    # Cumulative sum of true 3D velocities
    sum_v = np.zeros(3)
    count = 0

    for R in r_list:

        # Accumulate until radius threshold
        while idx < N and r_sorted[idx] <= R:
            sum_v += vel[idx]
            count += 1
            idx += 1

        if count > 0:
            u = sum_v / count
        else:
            u = np.array([np.nan, np.nan, np.nan])

        U = np.linalg.norm(u)

        results.append([R, u[0], u[1], u[2], U, np.nan, np.nan, count])

    return pd.DataFrame(
        results,
        columns=["radius", "u_x", "u_y", "u_z", "U_total", "U_debiased", "sigma_U", "n_used"]
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
    box_size: float,
    calculation_method: str = "chi2",
    error_frac: float | None = None,
    sigma_star: float | None = None,
    sigma_min: float | None = None
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
        Columns: [radius, u_x, u_y, u_z, U_total, U_debiased, sigma_U, n_used]
    """

    if error_frac is None:
        error_frac = BulkFlowConfig().error_fraction
    if sigma_star is None:
        sigma_star = BulkFlowConfig().sigma_star
    if sigma_min is None:
        sigma_min = BulkFlowConfig().sigma_min

    # ---------------------------------------------------------
    # (1) Compute radial velocity, error, radii, unit vector
    # ---------------------------------------------------------
    halos_df = radial_velocity_and_error_pbc(
            halos_df,
            origin=origin,
            box_size=box_size,
            error_frac=error_frac,
            min_sigma=sigma_min
        )
    
    # print("[INFO] Radial velocities and errors computed.")

    # ---------------------------------------------------------
    # (2) radius list: r_min → r_max in steps of r_jumps
    # ---------------------------------------------------------
    r_list = np.arange(r_min, r_max + r_jumps, r_jumps)

    # ---------------------------------------------------------
    # (3) extract arrays
    # ---------------------------------------------------------
    r_hat = halos_df[["r_hat_x","r_hat_y","r_hat_z"]].values
    v_rad = halos_df["v_rad"].values
    sigma = halos_df["sigma_v_rad"].values
    r_sorted = halos_df["radius_from_origin"].values
    vel = halos_df[["vx", "vy", "vz"]].values

    # Ensure halos are sorted by radius (required for series Ainv construction)
    sort_idx = np.argsort(r_sorted)
    r_sorted = r_sorted[sort_idx]
    r_hat    = r_hat[sort_idx]
    v_rad    = v_rad[sort_idx]
    sigma    = sigma[sort_idx]
    vel      = vel[sort_idx]

    # ---------------------------------------------------------
    # (4) Compute bulk flow table
    # ---------------------------------------------------------
    if calculation_method == "chi2":
        velocities_df = bulk_flow_chi2_cumulative(
            r_hat=r_hat,
            r_sorted=r_sorted,
            r_list=r_list,
            v_rad=v_rad,
            sigma=sigma,
            sigma_star=sigma_star
        )
    elif calculation_method == "mean":
        velocities_df = bulk_flow_mean_cumulative(
            r_hat=r_hat,
            r_sorted=r_sorted,
            r_list=r_list,
            v_rad=v_rad,
            sigma=sigma,
            vel=vel,
            sigma_star=sigma_star
        )

    return velocities_df


###########################################################################
# Local bulk flow (neighbor averaging)
###########################################################################

def calculate_local_bulkflow(
    df,
    tree: cKDTree,
    radius: float,
    velocity_columns=("vx", "vy", "vz"),
    output_prefix="bulkflow"
):
    """
    Calculate local bulk flow for each halo by averaging velocities of neighbors
    within a given radius.

    Parameters
    ----------
    df : pandas.DataFrame
        Halo catalog containing positions and velocities.
    tree : scipy.spatial.cKDTree
        KDTree built from halo positions (x, y, z).
    radius : float
        Radius within which to compute the local bulk flow.
    velocity_columns : tuple of str, optional
        Names of velocity columns (vx, vy, vz).
    output_prefix : str, optional
        Prefix for output column name.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with an added column:
        f"{output_prefix}_{int(radius)}"
    """

    colname = f"{output_prefix}_{int(radius)}"

    # ------------------------------------------
    # Skip if already exists
    # ------------------------------------------
    if colname in df.columns:
        logging.info(
            f"Column '{colname}' already exists — skipping local bulk flow calculation."
        )
        return df

    logging.info(f"Computing local bulk flow within R = {radius}...")

    vx, vy, vz = velocity_columns
    N = len(df)

    bulkflow = np.zeros(N, dtype=np.float64)

    # ------------------------------------------
    # Loop over halos
    # ------------------------------------------
    for i in range(N):
        # Find neighbors within radius
        idx = tree.query_ball_point(tree.data[i], r=radius)

        if len(idx) == 0:
            bulkflow[i] = np.nan
            continue

        # Average velocity components
        v_mean = df.iloc[idx][[vx, vy, vz]].mean().values

        # Total bulk flow magnitude
        bulkflow[i] = np.linalg.norm(v_mean)

    # ------------------------------------------
    # Save to DataFrame
    # ------------------------------------------
    df[colname] = bulkflow

    logging.info(
        f"Local bulk flow '{colname}' computed for {N} halos."
    )

    return df