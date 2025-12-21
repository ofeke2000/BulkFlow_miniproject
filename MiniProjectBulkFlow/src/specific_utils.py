# specific_utils.py
import numpy as np
import pandas as pd
import os


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
        halos_df: pd.DataFrame,
        origin: tuple[float, float, float],
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
    if not all(col in halos_df.columns for col in required):
        raise ValueError(f"halos_df must contain columns: {required}")

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
    halos_df['radius_from_origin'] = r_norm
    halos_df['r_hat_x'] = r_hat[:, 0]
    halos_df['r_hat_y'] = r_hat[:, 1]
    halos_df['r_hat_z'] = r_hat[:, 2]
    halos_df['v_rad'] = v_rad
    halos_df['sigma_v_rad'] = sigma

    return halos_df

########################################################
# Save bulk flow results to HDF5
########################################################

def append_bulkflow_results(
    results_df: pd.DataFrame,
    origin_id: int,
    mask_name: str,
    filename: str = "bulkflow_results.h5"
):
    """
    Append bulk flow results for a single origin & mask to a tidy HDF5 database.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with columns:
            ["radius", "u_x", "u_y", "u_z", "U_total"]
        All rows correspond to the same origin and mask.

    origin_id : int
        Identifier of the origin point.

    mask_name : str
        Name of the mask, e.g. "uniform" or "cf4".

    filename : str, optional
        Path to the HDF5 database.
        Default: "bulkflow_results.h5"
    """

    # Validate input columns
    expected_cols = {"radius", "u_x", "u_y", "u_z", "U_total"}
    if not expected_cols.issubset(results_df.columns):
        raise ValueError(f"Results df must contain columns: {expected_cols}")

    # Build a tidy dataframe to append
    df_to_store = results_df.copy()
    df_to_store["origin_id"] = origin_id
    df_to_store["mask"] = mask_name

    # Ensure correct dtypes
    df_to_store["origin_id"] = df_to_store["origin_id"].astype(int)
    df_to_store["mask"] = df_to_store["mask"].astype(str)  # will set min_itemsize later
    df_to_store[["radius", "u_x", "u_y", "u_z", "U_total"]] = df_to_store[
        ["radius", "u_x", "u_y", "u_z", "U_total"]
    ].astype(float)

    # Reorder columns for cleanliness (HDF5 is strict about column order)
    df_to_store = df_to_store[
        ["origin_id", "mask", "radius", "u_x", "u_y", "u_z", "U_total"]
    ]

    # Append to HDF5 database
    df_to_store.to_hdf(
        filename,
        key="bulkflow",
        format="table",     # allow append and filtering
        mode="a",           # append mode
        append=True,
        min_itemsize={"mask": 8}  # set large enough to hold all strings
    )