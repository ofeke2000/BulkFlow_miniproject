import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import logging

from ..config.virgo_config import VirgoTestConfig


def near_virgo(
    df: pd.DataFrame,
    box_size: float,
    virgo_cfg: VirgoTestConfig = None,
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
    virgo_cfg : VirgoTestConfig, optional
        Test parameters. Defaults to VirgoTestConfig() values.
    column_name : str
        Name of output column

    Returns
    -------
    pandas.DataFrame
        Updated dataframe with column `column_name`
    """
    if virgo_cfg is None:
        virgo_cfg = VirgoTestConfig()

    mass_threshold = virgo_cfg.mass_threshold
    r_min = virgo_cfg.r_min
    r_max = virgo_cfg.r_max

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
    near_virgo_flags = np.zeros(len(df), dtype=np.int8)

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
            near_virgo_flags[i] = 1

    # --------------------------------------------------
    # 4. Attach column
    # --------------------------------------------------
    df[column_name] = near_virgo_flags

    logging.info(
        f"Virgo test complete: {near_virgo_flags.sum()} / {len(df)} halos flagged."
    )

    return df
