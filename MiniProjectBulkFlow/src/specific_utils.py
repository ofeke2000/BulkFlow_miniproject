# specific_utils.py
import numpy as np
import pandas as pd
import os

from .theoretical_bulkflow import theoretical_bulkflow_colossus
from .config.bulkflow_config import BulkFlowConfig
from .config.cosmology_config import CosmologyConfig
from .config.mdpl2_config import MDPL2Config
from .config.theory_config import TheoryConfig

ORIGIN_EPS = 1e-8
HDF5_MASK_MIN_ITEMSIZE = {"mask": 8}

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

    New columns added:
        radius
        rhat_x, rhat_y, rhat_z
        v_rad
        sigma_vrad
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
    df_to_store["mask"] = df_to_store["mask"].astype(str)
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
        format="table",
        mode="a",
        append=True,
        min_itemsize=HDF5_MASK_MIN_ITEMSIZE,
    )


# ================================================================
# Save bulk flow averages results to csv
# ================================================================

def save_average_bulkflow_to_csv(
    hdf_file: str,
    csv_file: str,
    column_name: str,
    mask_type: str = "full",
    key: str = "bulkflow",
    cosmology_cfg: CosmologyConfig = None,
    theory_cfg: TheoryConfig = None,
) -> None:
    """
    Calculate mean and std bulk flow per radius (averaged over origins)
    and store them as band-specific columns in a CSV file.
    """
    if cosmology_cfg is None:
        cosmology_cfg = CosmologyConfig()
    if theory_cfg is None:
        theory_cfg = TheoryConfig()

    if not os.path.exists(hdf_file):
        print(f"Error: HDF5 file {hdf_file} not found.")
        return

    df_hdf = pd.read_hdf(hdf_file, key=key)

    mask_df = df_hdf[df_hdf["mask"] == mask_type]

    if mask_df.empty:
        print(f"Warning: No data found for mask '{mask_type}' in {hdf_file}.")
        return

    stats = (
        mask_df.groupby("radius", as_index=False)
               .agg(
                   U_mean=("U_total", "mean"),
                   U_std=("U_total", "std"),
               )
               .sort_values("radius")
    )

    # Load or create CSV
    if os.path.exists(csv_file):
        target_df = pd.read_csv(csv_file)
    else:
        print(f"Creating new CSV file: {csv_file}")

        radii = stats["radius"].values
        target_df = pd.DataFrame({"radius": stats["radius"].values})
        sigma_v = theoretical_bulkflow_colossus(
            radii=radii,
            cosmology_cfg=cosmology_cfg,
            theory_cfg=theory_cfg,
        )
        target_df["U_mean_theory"] = cosmology_cfg.bulk_flow_amplitude_factor * sigma_v

    # Rename columns to encode overdensity band
    stats = stats.rename(
        columns={
            "U_mean": f"{column_name}_mean",
            "U_std":  f"{column_name}_std",
        }
    )

    # Drop existing columns if overwriting
    for col in stats.columns:
        if col != "radius" and col in target_df.columns:
            print(f"Warning: Column '{col}' already exists and will be overwritten.")
            target_df = target_df.drop(columns=[col])

    updated_df = pd.merge(target_df, stats, on="radius", how="left")

    updated_df.to_csv(csv_file, index=False)
    print(f"Saved band '{column_name}' to {csv_file} (mask={mask_type})")
