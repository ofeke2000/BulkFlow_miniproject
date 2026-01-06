import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import logging


def near_virgo(
    df: pd.DataFrame,
    box_size: float = 1000.0,
    mass_threshold: float = 1e14,
    r_min: float = 10.0,
    r_max: float = 20.0,
    column_name: str = "near_virgo"
) -> pd.DataFrame:
    """
    Add a Virgo-like environment flag to halos.

    A halo is flagged if there exists a halo with
    mvir > mass_threshold within [r_min, r_max] Mpc.

    Parameters
    ----------
    df : pandas.DataFrame
        Halo catalog with columns ['x','y','z','mvir']
    box_size : float
        Periodic box size
    mass_threshold : float
        Mass threshold for Virgo-like halos
    r_min, r_max : float
        Distance shell for Virgo test
    column_name : str
        Name of output column

    Returns
    -------
    pandas.DataFrame
        Updated dataframe with column `column_name`
    """

    # --------------------------------------------------
    # 0. Skip if column already exists
    # --------------------------------------------------
    if column_name in df.columns:
        logging.info(f"Column '{column_name}' already exists — skipping Virgo test.")
        return df

    logging.info("Running Virgo environment test...")

    # --------------------------------------------------
    # 1. Select massive halos
    # --------------------------------------------------
    massive_df = df[df["mvir"] > mass_threshold]

    if len(massive_df) == 0:
        logging.warning("No massive halos found — setting all flags to 0.")
        df[column_name] = 0
        return df

    massive_positions = massive_df[["x", "y", "z"]].values

    logging.info(f"Found {len(massive_positions)} massive halos (m > {mass_threshold:.1e})")

    # --------------------------------------------------
    # 2. Build periodic KDTree of massive halos
    # --------------------------------------------------
    massive_tree = cKDTree(
        massive_positions,
        boxsize=box_size
    )

    # --------------------------------------------------
    # 3. Query for each halo
    # --------------------------------------------------
    positions = df[["x", "y", "z"]].values
    near_virgo = np.zeros(len(df), dtype=np.int8)

    for i, pos in enumerate(positions):
        # Find massive halos within r_max
        idxs = massive_tree.query_ball_point(pos, r_max)

        if not idxs:
            continue

        # Compute periodic distances to candidates
        deltas = massive_positions[idxs] - pos
        deltas -= box_size * np.round(deltas / box_size)
        distances = np.linalg.norm(deltas, axis=1)

        # Check shell condition
        if np.any((distances >= r_min) & (distances <= r_max)):
            near_virgo[i] = 1

    # --------------------------------------------------
    # 4. Attach column
    # --------------------------------------------------
    df[column_name] = near_virgo

    logging.info(
        f"Virgo test complete: {near_virgo.sum()} / {len(df)} halos flagged."
    )

    return df
