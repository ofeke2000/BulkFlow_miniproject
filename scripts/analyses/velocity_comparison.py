"""
velocity_comparison.py
----------------------
Standalone analysis comparing the CF4 bulk flow inferred from two different
observed line-of-sight peculiar-velocity estimators in the CF4 "All Group
Velocities" catalogue: ``Vpds`` and ``Vpwf``.

Unlike the simulation pipeline, this measures the bulk flow *directly* on the
real CF4 groups, with no MDPL2 halos involved:

  - The observer sits at the supergalactic origin (us). Positions come from the
    catalogue's RA/Dec/D, converted to h^-1 Mpc by ``load_cf4_catalogue``.
  - The velocities fed to the chi^2 estimator are the catalogue's *own* radial
    peculiar velocities (Vpds or Vpwf) -- they are already line-of-sight, so
    there is no projection of a 3-D velocity.
  - No periodic boundary conditions: this is the real sky, not the box.

Per-object uncertainties are propagated from the distance-modulus error eDM:

    sigma_v = max( (ln10/5) * (H0/h) * r * eDM , sigma_min )

where r is the distance in h^-1 Mpc and (H0/h) = 100 km/s per h^-1 Mpc, so
(H0/h)*r = H0*D is the Hubble velocity. The (ln10/5) factor converts a
distance-modulus error into a fractional distance error (d proportional to
10^(mu/5), hence d ln d / d mu = ln(10)/5).

The two estimators are stored as the ``method`` dimension of a single netCDF
(via ``BulkFlowDataset``), so one self-describing file backs the comparison
plot. The chi^2 weights keep the configured small-scale dispersion
``sigma_star`` for consistency with the rest of the pipeline.

Caveat: ``x, y, z`` are built from equatorial RA/Dec, so the bulk-flow vector
*components* (u_x, u_y, u_z) are in the equatorial frame. The plotted magnitude
|U| is rotation-invariant, so the Vpds-vs-Vpwf comparison is unaffected.
"""

import logging
import math
import os

import numpy as np
import pandas as pd

from src.classes import AppConfig, Vector3D
from src.config import BulkFlowPlotConfig
from src.config.paths_config import (
    DEFAULT_CF4_VELOCITIES_CATALOG,
    DEFAULT_OUTPUT_FOLDER,
)
from src.data.bulkflow_dataset import BulkFlowDataset, BulkFlowResult
from src.io.data_loader import load_cf4_catalogue
from src.physics.bulkflow import bulk_flow_chi2_cumulative
from src.viz.visualize import BulkFlowPlotter, plot_overlaid_histogram


class VelocityComparison:
    """
    Compare the CF4 chi^2 bulk flow from two radial-peculiar-velocity columns.

    Run-specific parameters live here as class attributes (per the project's
    no-bare-numbers rule); shared physics, cosmology, theory, and style settings
    come from the AppConfig.
    """

    # --- Radial grid for this comparison (h^-1 Mpc) ---
    R_MIN: float = 5.0
    R_MAX: float = 250.0
    R_STEP: float = 5.0

    # --- CF4 radial-peculiar-velocity estimators compared (become `method`) ---
    VELOCITY_COLUMNS: tuple[str, ...] = ("Vpds", "Vpwf")

    # --- eDM (distance-modulus error) -> fractional distance error ---
    #   d proportional to 10^(mu/5)  =>  d ln(d) / d(mu) = ln(10) / 5
    LN10_OVER_5: float = math.log(10.0) / 5.0

    # --- Single CF4 observer (us) at the supergalactic origin ---
    OBSERVER_ID: int = 0
    MASK_NAME: str = "cf4"

    # --- Output ---
    OUTPUT_SUBFOLDER: str = "velocity comparison"
    DATASET_FILE: str = "velocity_comparison.nc"
    PLOT_FILE: str = "velocity_comparison_vpds_vpwf.png"
    HISTOGRAM_FILE: str = "vpds_vpwf_histogram.png"
    HISTOGRAM_XLABEL: str = "Radial peculiar velocity [km/s]"

    def __init__(self, cfg: AppConfig, cf4_path: str | None = None) -> None:
        self._cfg = cfg
        self._cf4_path = cf4_path or DEFAULT_CF4_VELOCITIES_CATALOG
        self._h = cfg.MDPL2.HubbleParameter

        bf = cfg.bulkflow
        self._sigma_star = bf.sigma_star
        self._sigma_min = bf.sigma_min
        self._hubble_velocity = cfg.cosmology.hubble_velocity_per_hinv_mpc

        self._output_folder = os.path.join(
            DEFAULT_OUTPUT_FOLDER, self.OUTPUT_SUBFOLDER, ""
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_groups(self) -> pd.DataFrame:
        """Load the CF4 'All Group Velocities' catalogue and check columns."""
        df = load_cf4_catalogue(self._cf4_path, h=self._h)
        for col in (*self.VELOCITY_COLUMNS, "eDM"):
            if col not in df.columns:
                raise KeyError(
                    f"CF4 velocities catalogue is missing required column '{col}'."
                )
        return df

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame) -> BulkFlowDataset:
        """Run the chi^2 estimator once per velocity column (observer at origin)."""
        r_list = np.arange(self.R_MIN, self.R_MAX + self.R_STEP, self.R_STEP)

        pos = df[["x", "y", "z"]].values.astype(float)
        radius = np.linalg.norm(pos, axis=1)

        # Observer at the origin; guard against a group exactly at r = 0.
        safe_radius = np.where(radius == 0.0, np.nan, radius)
        r_hat = pos / safe_radius[:, None]

        eDM = df["eDM"].values.astype(float)
        sigma_v = np.maximum(
            self.LN10_OVER_5 * self._hubble_velocity * radius * eDM,
            self._sigma_min,
        )

        dataset = BulkFlowDataset()
        for col in self.VELOCITY_COLUMNS:
            v_rad = df[col].values.astype(float)

            good = (
                np.isfinite(v_rad)
                & np.isfinite(radius)
                & np.isfinite(sigma_v)
                & (radius > 0.0)
            )
            order = np.argsort(radius[good])

            bf_df = bulk_flow_chi2_cumulative(
                r_hat=r_hat[good][order],
                r_sorted=radius[good][order],
                r_list=r_list,
                v_rad=v_rad[good][order],
                sigma=sigma_v[good][order],
                sigma_star=self._sigma_star,
            )
            dataset.add(self._build_result(bf_df, col))
            logging.info(
                f"Computed CF4 chi^2 bulk flow for '{col}' "
                f"({int(good.sum()):,} groups)."
            )

        dataset.set_attrs(self._build_attrs())
        return dataset

    def _build_result(self, bf_df: pd.DataFrame, method: str) -> BulkFlowResult:
        """Wrap a chi^2 series as a single-observer BulkFlowResult."""
        return BulkFlowResult(
            origin_id=self.OBSERVER_ID,
            origin=Vector3D.from_sequence((0.0, 0.0, 0.0)),
            mask=self.MASK_NAME,
            method=method,
            overdensity=np.nan,
            local_bulkflow=np.nan,
            mvir=np.nan,
            near_virgo=False,
            radii=bf_df["radius"].values,
            u_x=bf_df["u_x"].values,
            u_y=bf_df["u_y"].values,
            u_z=bf_df["u_z"].values,
            U_tot=bf_df["U_total"].values,
            U_deb=bf_df["U_debiased"].values,
            sigma_U=bf_df["sigma_U"].values,
            n_used=bf_df["n_used"].values,
        )

    def _build_attrs(self) -> dict:
        """Provenance / configuration attributes for the dataset."""
        return {
            # Single observer -> no banding; kept for plotter compatibility.
            "selection_variable": "overdensity",
            "sigma_star": self._sigma_star,
            "sigma_min": self._sigma_min,
            "min_radius": self.R_MIN,
            "max_radius": self.R_MAX,
            "radii_step": self.R_STEP,
            "number_of_origins": 1,
            "calculation_method": "chi2",
            "velocity_estimators": ",".join(self.VELOCITY_COLUMNS),
            "hubble_parameter": self._h,
            "analysis": "velocity_comparison",
            "catalog": os.path.basename(self._cf4_path),
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_and_plot(self, dataset: BulkFlowDataset) -> None:
        """Persist the dataset and render the Vpds-vs-Vpwf comparison plot."""
        os.makedirs(self._output_folder, exist_ok=True)
        out_file = os.path.join(self._output_folder, self.DATASET_FILE)

        dataset.write(out_file)
        logging.info(f"Wrote dataset: {out_file}")

        BulkFlowPlotter(
            nc_file=out_file,
            output_folder=self._output_folder,
            output_file=self.PLOT_FILE,
            methods=list(self.VELOCITY_COLUMNS),
            plot_cfg=BulkFlowPlotConfig(plot_debiased=False, show_markers=False),
            cosmology_cfg=self._cfg.cosmology,
            theory_cfg=self._cfg.theory,
            style=self._cfg.visualization.style,
        ).plot()

    def plot_velocity_histograms(self, df: pd.DataFrame) -> None:
        """Overlay the CF4 velocity columns (count vs velocity) on one figure."""
        plot_overlaid_histogram(
            data_df=df,
            keys=list(self.VELOCITY_COLUMNS),
            output_folder=self._output_folder,
            output_file=self.HISTOGRAM_FILE,
            xlabel=self.HISTOGRAM_XLABEL,
            style=self._cfg.visualization.style,
        )
        logging.info(
            "Wrote overlaid "
            f"{' vs '.join(self.VELOCITY_COLUMNS)} histogram."
        )

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        df = self.load_groups()
        dataset = self.compute(df)
        self.write_and_plot(dataset)
        self.plot_velocity_histograms(df)
