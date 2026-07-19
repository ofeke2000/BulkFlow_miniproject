"""
velocity_comparison.py
----------------------
Standalone analysis comparing the CF4 bulk flow inferred from two different
observed line-of-sight peculiar-velocity estimators in the CF4 "All Group
Velocities" catalogue: ``Vpds`` and ``Vpwf``, each run through *both* bulk
flow estimators (``chi2`` and ``mean``).

Unlike the simulation pipeline, this measures the bulk flow *directly* on the
real CF4 groups, with no MDPL2 halos involved:

  - The observer sits at the supergalactic origin (us). Positions come from the
    catalogue's RA/Dec/D, converted to h^-1 Mpc by ``load_cf4_catalogue``.
  - The velocities fed to the estimators are the catalogue's *own* radial
    peculiar velocities (Vpds or Vpwf) -- they are already line-of-sight, so
    there is no projection of a 3-D velocity onto them.
  - No periodic boundary conditions: this is the real sky, not the box.

Per-object uncertainties are propagated from the distance-modulus error eDM,
combined with a distance-proportional floor:

    sigma_edm   = (ln10/5) * (H0_CF4/h) * r * eDM
    sigma_v     = max( sigma_edm , sigma_min_fraction * (H0_CF4/h) * r_scheme )

where r is the *measured* distance in h^-1 Mpc, r_scheme is the radial
coordinate of whichever placement scheme (see "Radial placement schemes"
below) sigma_v is being used for, and (H0_CF4/h) is CF4's own calibrated
Hubble constant (Tully et al. 2023, H0_CF4 = 74.6 km/s/Mpc) divided by the
MDPL2 h, i.e. ``cfg.cosmology.cf4_hubble_velocity_per_hinv_mpc`` (~110 km/s
per h^-1 Mpc, NOT the MDPL2 H0/h = 100), so (H0_CF4/h)*r = H0_CF4*D is the
CF4-calibrated Hubble velocity. The (ln10/5) factor converts a
distance-modulus error into a fractional distance error (d proportional to
10^(mu/5), hence d ln d / d mu = ln(10)/5). ``sigma_min_fraction`` (default
0.10, from ``cfg.bulkflow.sigma_min_fraction``) replaces the previous flat
``sigma_min`` floor: unlike a fixed km/s floor, this floor scales with how
far out a group is placed, so it does not artificially dominate sigma_v for
distant groups the way a flat floor would.

For each velocity column (Vpds, Vpwf) we compute both estimators:

  - ``chi2``: the ML / chi^2 estimator (``bulk_flow_chi2_cumulative``), fed
    the radial velocity and its per-object sigma directly -- no 3-D velocity
    is needed.
  - ``mean``: the unweighted cumulative mean estimator
    (``bulk_flow_mean_cumulative``). That estimator's signature was written
    for the simulation pipeline and requires a *true* 3-D velocity ``vel``
    (it averages ``[vx, vy, vz]``). CF4 groups only have a single radial
    velocity, so there is no true 3-D velocity to give it. We reuse the
    estimator as-is by synthesizing a **projection proxy**:

        vel_proxy = v_rad[:, None] * r_hat        # shape (N, 3)

    i.e. the radial velocity re-projected back along the line-of-sight unit
    vector. This is NOT a true 3-D velocity -- it has no transverse
    component -- so ``bulk_flow_mean_cumulative`` is, in effect, averaging
    line-of-sight vectors rather than actual 3-D peculiar velocities. Treat
    the resulting ``*_mean`` curves as an approximation for comparing against
    the ``*_chi2`` curves, not as an independent measurement of the true bulk
    flow direction/magnitude.

Each (velocity column, estimator) pair is stored as its own entry on the
``method`` dimension of a single netCDF (via ``BulkFlowDataset``), labeled
``f"{column}_{estimator}"`` (e.g. ``Vpds_chi2``, ``Vpds_mean``), so one
self-describing file backs the comparison plots. The chi^2 weights keep the
configured small-scale dispersion ``sigma_star`` for consistency with the
rest of the pipeline.

Radial placement schemes
-------------------------
Every group also has to be placed at *some* radial coordinate for the
cumulative-in-radius sums, and there are two natural choices, both computed
here (``DISTANCE_SCHEMES``):

  - ``"d"`` -- the **measured distance** ``r = |x, y, z|``, i.e. the
    catalogue's own ``D`` / distance-modulus distance (as before -- this is
    the original, sole scheme prior to this feature);
  - ``"cz"`` -- the **redshift distance** ``r_z = vcmb / (H0_CF4/h)``, where
    ``vcmb`` is the catalogue's CMB-frame velocity column and ``H0_CF4/h``
    (~110 km/s per h^-1 Mpc, CF4's own calibrated Hubble constant divided by
    the MDPL2 h) is ``cfg.cosmology.cf4_hubble_velocity_per_hinv_mpc``. This
    sidesteps the (possibly noisy) distance-modulus distance entirely and
    instead places each group along its own line of sight using its
    redshift.

The line-of-sight unit vectors ``r_hat`` (built from x, y, z / measured
radius) are the **same for both schemes** -- only the radial coordinate used
for the validity cut, sorting, and cumulative binning changes. The "d" scheme
requires ``r > 0``; the "cz" scheme requires ``vcmb > 0``.

``sigma_v`` has two ingredients that are treated differently across schemes.
The ``sigma_edm`` term (from ``eDM`` and the *measured* distance -- see the
``sigma_v`` formula above) is **shared unchanged across both schemes**: it is
a property of the measurement itself (how well the catalogue pins down the
group's peculiar velocity), not of which radial coordinate that group is
subsequently placed at for the bulk-flow sum, so it is computed once from the
measured distance and reused as-is. The floor, however, is a property of
*placement*: it is ``sigma_min_fraction`` of the Hubble velocity at each
group's own **scheme radius**, so it is recomputed per scheme (for ``"cz"``
this reduces to ``sigma_min_fraction * vcmb``, since
``(H0_CF4/h) * r_cz = vcmb`` by construction). ``sigma_v`` itself -- the
``maximum`` of the two -- is therefore per-scheme even though its ``eDM``
ingredient is shared.

For each (velocity column, estimator, scheme) combination the result is
stored on the ``method`` dimension: the "d" scheme keeps the original,
unsuffixed label shape ``f"{column}_{estimator}"`` for backward
compatibility (e.g. ``Vpds_chi2``); the "cz" scheme appends a ``"_cz"``
suffix, e.g. ``Vpds_chi2_cz``. That gives 2 columns x 2 estimators x 2
schemes = 8 series in the one netCDF.

From that one dataset we render **three** bulk-flow comparison plots plus a
fourth sigma diagnostic. In the three comparison plots, **color
encodes the velocity column** (``Vpds`` blue, ``Vpwf`` orange):

  - a measured-distance chi2-only plot with just ``Vpds_chi2`` and
    ``Vpwf_chi2`` (two solid curves, blue vs orange) -- linestyle encodes the
    estimator (chi2 solid);
  - a both-methods plot (measured distance) with all four curves (per
    column: solid chi2 + dashed mean, same color) -- linestyle encodes the
    estimator;
  - a redshift-distance chi2-only plot, the "cz" counterpart of the first:
    ``Vpds_chi2_cz`` and ``Vpwf_chi2_cz`` (two dashed curves, blue vs
    orange) -- here linestyle encodes the **distance scheme** rather than
    the estimator (redshift "cz" dashed), which visually distinguishes it
    from the solid measured-distance plot even though both only contain
    chi2-estimator curves.

Each style scheme is supplied to ``BulkFlowPlotter`` via its optional
``method_colors`` / ``method_linestyles`` override maps.

The fourth plot, ``plot_sigma_terms``, is a diagnostic rather than a
bulk-flow comparison: it shows the two ``sigma_v`` ingredients side by side
(one panel per distance scheme) -- the shared ``sigma_edm`` term scattered
against that scheme's radius, the new per-scheme ``sigma_min_fraction``
floor as a straight line through the origin, and the old flat ``sigma_min``
as a dotted reference -- so it is easy to see, per scheme, how often the
distance-proportional floor now dominates over the eDM term.

Caveat: ``x, y, z`` are built from equatorial RA/Dec, so the bulk-flow vector
*components* (u_x, u_y, u_z) are in the equatorial frame. The plotted magnitude
|U| is rotation-invariant, so the Vpds-vs-Vpwf comparison is unaffected.
"""

import logging
import math
import os

import matplotlib.pyplot as plt
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
from src.physics.bulkflow import bulk_flow_chi2_cumulative, bulk_flow_mean_cumulative
from src.viz.visualize import BulkFlowPlotter, plot_overlaid_histogram


class VelocityComparison:
    """
    Compare the CF4 bulk flow from two radial-peculiar-velocity columns
    (Vpds, Vpwf), each run through both the chi^2 and mean estimators.

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

    # --- Bulk-flow calculation methods run for each velocity column ---
    #   Combined with VELOCITY_COLUMNS (and DISTANCE_SCHEMES, see below) via
    #   `_make_label` to label results (e.g. "Vpds_chi2", "Vpds_mean"). See
    #   module docstring for the "mean" estimator's vel_proxy caveat.
    CALCULATION_METHODS: tuple[str, ...] = ("chi2", "mean")

    # --- Radial placement schemes: measured distance (from D / distance
    #     modulus) vs redshift distance (vcmb / (H0/h)) -- see module
    #     docstring "Radial placement schemes" section. r_hat and the eDM
    #     sigma term are shared across both; the radial coordinate used for
    #     the validity cut / sorting / cumulative binning and the
    #     sigma_min_fraction floor (see `_sigma_model`) are per-scheme. ---
    #   "d"  -> measured distance r = |x, y, z| (original scheme; r > 0 cut)
    #   "cz" -> redshift distance r_z = vcmb / (H0/h) (vcmb > 0 cut)
    DISTANCE_SCHEMES: tuple[str, ...] = ("d", "cz")

    # --- Label suffix per distance scheme (see `_make_label`/`_parse_label`) ---
    #   "d" keeps the original, unsuffixed f"{col}_{estimator}" label shape
    #   for backward compatibility; "cz" appends "_cz".
    SCHEME_LABEL_SUFFIX: dict[str, str] = {"d": "", "cz": "_cz"}

    # --- Plot style scheme (color = velocity column, linestyle = estimator
    #     or distance scheme, depending on the plot -- see `_style_maps`) ---
    #   Color encodes the velocity column (Vpds vs Vpwf), consistent across
    #   all plots; linestyle encodes either the estimator (chi2 solid, mean
    #   dashed) -- used by CHI2_PLOT_FILE / BOTH_METHODS_PLOT_FILE -- or the
    #   distance scheme (measured "d" solid, redshift "cz" dashed) -- used by
    #   CHI2_CZ_PLOT_FILE to visually distinguish it from CHI2_PLOT_FILE even
    #   though both only contain chi2-estimator curves. Keys mirror
    #   VELOCITY_COLUMNS / CALCULATION_METHODS / DISTANCE_SCHEMES.
    COLUMN_COLORS: dict[str, str] = {"Vpds": "tab:blue", "Vpwf": "tab:orange"}
    METHOD_LINESTYLES: dict[str, str] = {"chi2": "-", "mean": "--"}
    SCHEME_LINESTYLES: dict[str, str] = {"d": "-", "cz": "--"}

    # --- `_style_maps` dispatch keys: which facet drives linestyle ---
    _LINESTYLE_BY_ESTIMATOR: str = "estimator"
    _LINESTYLE_BY_SCHEME: str = "scheme"

    # --- eDM (distance-modulus error) -> fractional distance error ---
    #   d proportional to 10^(mu/5)  =>  d ln(d) / d(mu) = ln(10) / 5
    LN10_OVER_5: float = math.log(10.0) / 5.0

    # --- Single CF4 observer (us) at the supergalactic origin ---
    OBSERVER_ID: int = 0
    MASK_NAME: str = "cf4"

    # --- Output ---
    OUTPUT_SUBFOLDER: str = "velocity comparison"
    DATASET_FILE: str = "velocity_comparison.nc"
    #   Comparison plots: chi2-only at measured distance (Vpds/Vpwf), both
    #   estimators at measured distance, and chi2-only at redshift distance
    #   (Vpds/Vpwf) -- the "cz" counterpart of the first plot, dashed to
    #   distinguish it from the solid measured-distance plot.
    CHI2_PLOT_FILE: str = "velocity_comparison_vpds_vpwf_chi2.png"
    BOTH_METHODS_PLOT_FILE: str = "velocity_comparison_vpds_vpwf_chi2_mean.png"
    CHI2_CZ_PLOT_FILE: str = "velocity_comparison_vpds_vpwf_chi2_cz.png"
    HISTOGRAM_FILE: str = "vpds_vpwf_histogram.png"
    HISTOGRAM_XLABEL: str = "Radial peculiar velocity [km/s]"
    # --- Comparison-plot titles / x-axis labels announcing the distance
    #     space. The "d" scheme places each group at its measured
    #     (real-space) distance; the "cz" scheme uses its redshift-inferred
    #     distance. Passed through to BulkFlowPlotter's title/xlabel. ---
    REAL_SPACE_TITLE: str = "CF4 bulk flow vs radius — real-space (measured) distance"
    REDSHIFT_SPACE_TITLE: str = "CF4 bulk flow vs radius — redshift-space distance"
    REAL_SPACE_XLABEL: str = r"Real-space radius $r$ [$h^{-1}$ Mpc]"
    REDSHIFT_SPACE_XLABEL: str = r"Redshift-space radius $r_{cz}$ [$h^{-1}$ Mpc]"
    #   Sigma diagnostic plot (see `plot_sigma_terms`): one panel per
    #   distance scheme -- the shared eDM sigma term scattered against that
    #   scheme's radius, the per-scheme distance-proportional floor as a
    #   solid line through the origin, and the old flat sigma_min as a
    #   dotted gray reference.
    SIGMA_PLOT_FILE: str = "sigma_edm_vs_distance_floor.png"
    SIGMA_PLOT_FIGSIZE: tuple[float, float] = (12.0, 5.0)
    SIGMA_SCATTER_SIZE: float = 4.0
    SIGMA_SCATTER_ALPHA: float = 0.25
    SIGMA_SCATTER_COLOR: str = "tab:blue"
    SIGMA_SCATTER_LABEL: str = "eDM term"
    SIGMA_FLOOR_COLOR: str = "black"
    SIGMA_FLOOR_LINESTYLE: str = "-"
    #   The floor-legend / panel-title percentages are rendered from
    #   sigma_min_fraction at plot time (e.g. 0.10 -> "10%"), so the labels
    #   track the configured fraction instead of hardcoding "10".
    SIGMA_FLOOR_LABEL_TEMPLATE: str = "{pct:.0f}% distance floor"
    SIGMA_FLAT_REF_COLOR: str = "gray"
    SIGMA_FLAT_REF_LINESTYLE: str = ":"
    SIGMA_FLAT_REF_LABEL_TEMPLATE: str = "old flat sigma_min ({value:.0f} km/s)"
    SIGMA_TITLE_TEMPLATE: str = "{space}: floor active for {pct:.0f}% of groups"
    #   Human-readable distance-space name per scheme, for the panel titles.
    SIGMA_SPACE_NAMES: dict[str, str] = {"d": "Real space", "cz": "Redshift space"}
    SIGMA_YLABEL: str = "sigma_v contribution [km/s]"
    SIGMA_XLABELS: dict[str, str] = {
        "d": "Real-space (measured) distance $r$ [$h^{-1}$ Mpc]",
        "cz": "Redshift-space distance $r_{cz}$ [$h^{-1}$ Mpc]",
    }
    #   Fraction -> percent conversion for the labels/titles above.
    PERCENT_SCALE: float = 100.0

    def __init__(self, cfg: AppConfig, cf4_path: str | None = None) -> None:
        self._cfg = cfg
        self._cf4_path = cf4_path or DEFAULT_CF4_VELOCITIES_CATALOG
        self._h = cfg.MDPL2.HubbleParameter

        bf = cfg.bulkflow
        self._sigma_star = bf.sigma_star
        self._sigma_min_fraction = bf.sigma_min_fraction
        self._hubble_velocity = cfg.cosmology.cf4_hubble_velocity_per_hinv_mpc

        self._output_folder = os.path.join(
            DEFAULT_OUTPUT_FOLDER, self.OUTPUT_SUBFOLDER, ""
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_groups(self) -> pd.DataFrame:
        """Load the CF4 'All Group Velocities' catalogue and check columns."""
        df = load_cf4_catalogue(self._cf4_path, h=self._h)
        for col in (*self.VELOCITY_COLUMNS, "eDM", "vcmb"):
            if col not in df.columns:
                raise KeyError(
                    f"CF4 velocities catalogue is missing required column '{col}'."
                )
        return df

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def _scheme_radii(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]:
        """
        Build the measured radius and the per-scheme ``(radius, validity)``
        pairs shared by `compute` and `plot_sigma_terms` -- see the module
        docstring's "Radial placement schemes" section.
        """
        pos = df[["x", "y", "z"]].values.astype(float)
        radius = np.linalg.norm(pos, axis=1)  # measured distance ("d" scheme)

        # Redshift distance r_z = vcmb / (H0_CF4/h): same units (h^-1 Mpc) as
        # `radius` since self._hubble_velocity = H0_CF4/h (CF4's own
        # calibrated Hubble constant, not the MDPL2 H0). See module docstring.
        vcmb = df["vcmb"].values.astype(float)
        radius_cz = vcmb / self._hubble_velocity

        # Per-scheme radial coordinate + validity cut. "d" excludes r <= 0
        # (as before); "cz" excludes vcmb <= 0 (its natural extension).
        d_scheme, cz_scheme = self.DISTANCE_SCHEMES
        scheme_radii: dict[str, tuple[np.ndarray, np.ndarray]] = {
            d_scheme: (radius, radius > 0.0),
            cz_scheme: (radius_cz, vcmb > 0.0),
        }
        return radius, scheme_radii

    def _sigma_model(
        self,
        eDM: np.ndarray,
        radius: np.ndarray,
        scheme_radii: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Build the two ``sigma_v`` ingredients shared by `compute` and
        `plot_sigma_terms`: the eDM measurement-error term (always from the
        *measured* distance, shared across schemes) and the per-scheme
        distance-proportional floor (``sigma_min_fraction`` of the Hubble
        velocity at each scheme's own radius). Returns
        ``(sigma_edm, scheme_sigma)``, where ``scheme_sigma[scheme]`` is
        ``maximum(sigma_edm, floor_scheme)`` -- the actual per-scheme
        ``sigma_v`` fed to the estimators in `compute`.
        """
        # eDM-based measurement error term: always from the MEASURED distance.
        sigma_edm = self.LN10_OVER_5 * self._hubble_velocity * radius * eDM

        # Distance-proportional floor, per scheme: sigma_min_fraction of the
        # Hubble velocity at each group's own *scheme* radius. For "cz" this
        # reduces to sigma_min_fraction * vcmb (since
        # (H0_CF4/h) * radius_cz = vcmb).
        scheme_sigma: dict[str, np.ndarray] = {
            scheme: np.maximum(
                sigma_edm,
                self._sigma_min_fraction
                * self._hubble_velocity
                * scheme_radii[scheme][0],
            )
            for scheme in self.DISTANCE_SCHEMES
        }
        return sigma_edm, scheme_sigma

    def compute(self, df: pd.DataFrame) -> BulkFlowDataset:
        """
        Run both bulk-flow estimators for each velocity column, at each
        radial placement scheme (observer at origin).

        r_hat (line-of-sight unit vectors, from measured x/y/z) and the eDM
        ingredient of sigma_v (the measurement-error term, from eDM and the
        *measured* distance) are shared across both distance-placement
        schemes -- see the module docstring's "Radial placement schemes"
        section. The floor ingredient, however, is placement-based:
        sigma_min_fraction of the Hubble velocity at each group's own
        *scheme* radius (see `_sigma_model`), so sigma_v itself -- the
        maximum of the two -- is per-scheme. The radial coordinate used for
        the validity cut / sorting / cumulative binning also differs between
        schemes, as before.
        """
        r_list = np.arange(self.R_MIN, self.R_MAX + self.R_STEP, self.R_STEP)

        radius, scheme_radii = self._scheme_radii(df)

        pos = df[["x", "y", "z"]].values.astype(float)
        # Observer at the origin; guard against a group exactly at r = 0.
        safe_radius = np.where(radius == 0.0, np.nan, radius)
        r_hat = pos / safe_radius[:, None]
        # A group at measured r = 0 has a NaN r_hat row. The "d" scheme's
        # radius > 0 cut excludes it implicitly, but the "cz" scheme's cut is
        # on vcmb only -- so require a finite r_hat explicitly in both
        # schemes' validity masks to keep NaNs out of the estimators.
        r_hat_finite = np.all(np.isfinite(r_hat), axis=1)

        eDM = df["eDM"].values.astype(float)
        # sigma_edm (the eDM measurement-error term, from the measured
        # distance) is shared across schemes; scheme_sigma[scheme] is the
        # per-scheme sigma_v = max(sigma_edm, distance-proportional floor).
        # See `_sigma_model` and the module docstring.
        sigma_edm, scheme_sigma = self._sigma_model(eDM, radius, scheme_radii)

        # Unpack rather than compare against inline "chi2"/"mean" literals.
        chi2_method, mean_method = self.CALCULATION_METHODS

        dataset = BulkFlowDataset()
        for col in self.VELOCITY_COLUMNS:
            v_rad = df[col].values.astype(float)

            for scheme in self.DISTANCE_SCHEMES:
                scheme_radius, scheme_valid = scheme_radii[scheme]
                sigma_v = scheme_sigma[scheme]

                good = (
                    np.isfinite(v_rad)
                    & np.isfinite(scheme_radius)
                    & np.isfinite(sigma_v)
                    & r_hat_finite
                    & scheme_valid
                )
                order = np.argsort(scheme_radius[good])

                r_hat_sorted = r_hat[good][order]
                r_sorted = scheme_radius[good][order]
                v_rad_sorted = v_rad[good][order]
                sigma_sorted = sigma_v[good][order]

                for method in self.CALCULATION_METHODS:
                    if method == chi2_method:
                        bf_df = bulk_flow_chi2_cumulative(
                            r_hat=r_hat_sorted,
                            r_sorted=r_sorted,
                            r_list=r_list,
                            v_rad=v_rad_sorted,
                            sigma=sigma_sorted,
                            sigma_star=self._sigma_star,
                        )
                    elif method == mean_method:
                        # No true 3-D velocity exists for CF4 groups: synthesize a
                        # line-of-sight projection proxy so the "mean" estimator
                        # (written for the simulation's true [vx, vy, vz]) can be
                        # reused as-is. See module docstring for the caveat.
                        vel_proxy = v_rad_sorted[:, None] * r_hat_sorted
                        bf_df = bulk_flow_mean_cumulative(
                            r_hat=r_hat_sorted,
                            r_sorted=r_sorted,
                            r_list=r_list,
                            v_rad=v_rad_sorted,
                            sigma=sigma_sorted,
                            vel=vel_proxy,
                            sigma_star=self._sigma_star,
                        )

                    label = self._make_label(col, method, scheme)
                    dataset.add(self._build_result(bf_df, label))
                    logging.info(
                        f"Computed CF4 '{method}' bulk flow for '{col}' "
                        f"(scheme='{scheme}', {int(good.sum()):,} groups)."
                    )

        dataset.set_attrs(self._build_attrs())
        return dataset

    def _build_result(self, bf_df: pd.DataFrame, method: str) -> BulkFlowResult:
        """Wrap a bulk-flow series as a single-observer BulkFlowResult."""
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
            "sigma_min_fraction": self._sigma_min_fraction,
            "min_radius": self.R_MIN,
            "max_radius": self.R_MAX,
            "radii_step": self.R_STEP,
            "number_of_origins": 1,
            "calculation_method": ",".join(self.CALCULATION_METHODS),
            "velocity_estimators": ",".join(self.VELOCITY_COLUMNS),
            "distance_schemes": ",".join(self.DISTANCE_SCHEMES),
            "hubble_parameter": self._h,
            "cf4_hubble_constant": self._cfg.cosmology.H0_CF4,
            "analysis": "velocity_comparison",
            "catalog": os.path.basename(self._cf4_path),
        }

    # ------------------------------------------------------------------
    # Method-label helpers
    # ------------------------------------------------------------------

    def _make_label(self, col: str, estimator: str, scheme: str) -> str:
        """
        Build the combined ``method`` label for a (velocity column,
        estimator, distance scheme) triple. The "d" (measured-distance)
        scheme keeps the original, unsuffixed ``f"{col}_{estimator}"`` shape
        for backward compatibility (e.g. ``Vpds_chi2``); other schemes append
        their ``SCHEME_LABEL_SUFFIX`` (e.g. ``Vpds_chi2_cz``). Inverse of
        `_parse_label`.
        """
        return f"{col}_{estimator}{self.SCHEME_LABEL_SUFFIX[scheme]}"

    def _parse_label(self, label: str) -> tuple[str, str, str]:
        """
        Recover ``(col, estimator, scheme)`` from a combined method label
        built by `_make_label`. Labels are generated from a small fixed set
        of components (VELOCITY_COLUMNS, CALCULATION_METHODS,
        DISTANCE_SCHEMES), none of which contain underscores, so checking
        each scheme's known (non-empty) suffix first -- rather than blindly
        splitting on the last underscore -- stays robust as soon as a label
        can have a 3rd, scheme-suffix segment.
        """
        for scheme, suffix in self.SCHEME_LABEL_SUFFIX.items():
            if suffix and label.endswith(suffix):
                col, estimator = label[: -len(suffix)].rsplit("_", 1)
                return col, estimator, scheme
        # No known non-empty suffix matched -> the unsuffixed ("d") scheme.
        col, estimator = label.rsplit("_", 1)
        return col, estimator, self.DISTANCE_SCHEMES[0]

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _style_maps(
        self, method_labels: list[str], linestyle_by: str = _LINESTYLE_BY_ESTIMATOR
    ) -> tuple[dict[str, str], dict[str, str]]:
        """
        Build the per-method color/linestyle override maps for a set of
        combined method labels (see `_make_label`/`_parse_label`). Color
        always encodes the velocity column. ``linestyle_by`` selects whether
        linestyle encodes the bulk-flow estimator (default -- used by the
        two original chi2/mean plots) or the distance-placement scheme (used
        by the per-column measured-vs-redshift-distance comparison plots).
        """
        method_colors: dict[str, str] = {}
        method_linestyles: dict[str, str] = {}
        for label in method_labels:
            col, estimator, scheme = self._parse_label(label)
            method_colors[label] = self.COLUMN_COLORS[col]
            if linestyle_by == self._LINESTYLE_BY_ESTIMATOR:
                method_linestyles[label] = self.METHOD_LINESTYLES[estimator]
            elif linestyle_by == self._LINESTYLE_BY_SCHEME:
                method_linestyles[label] = self.SCHEME_LINESTYLES[scheme]
            else:
                raise ValueError(f"Unknown linestyle_by='{linestyle_by}'.")
        return method_colors, method_linestyles

    def _plot_comparison(
        self,
        out_file: str,
        method_labels: list[str],
        plot_file: str,
        linestyle_by: str = _LINESTYLE_BY_ESTIMATOR,
        title: str | None = None,
        xlabel: str | None = None,
    ) -> None:
        """Render one comparison plot for the given combined method labels."""
        method_colors, method_linestyles = self._style_maps(method_labels, linestyle_by=linestyle_by)
        BulkFlowPlotter(
            nc_file=out_file,
            output_folder=self._output_folder,
            output_file=plot_file,
            title=title,
            xlabel=xlabel,
            methods=method_labels,
            plot_cfg=BulkFlowPlotConfig(plot_debiased=False, show_markers=False),
            cosmology_cfg=self._cfg.cosmology,
            theory_cfg=self._cfg.theory,
            style=self._cfg.visualization.style,
            method_colors=method_colors,
            method_linestyles=method_linestyles,
        ).plot()

    def write_and_plot(self, dataset: BulkFlowDataset) -> None:
        """
        Persist the dataset and render the comparison plots.

        A single netCDF holds all eight (column, estimator, distance scheme)
        methods. Three fixed plots are drawn from it:

          - chi2-only (measured distance): just ``Vpds_chi2`` and
            ``Vpwf_chi2`` (two solid curves) -- color = column, linestyle =
            estimator;
          - both-methods (measured distance): all four curves (solid chi2 +
            dashed mean per column) -- color = column, linestyle = estimator;
          - chi2-only (redshift distance): just ``Vpds_chi2_cz`` and
            ``Vpwf_chi2_cz`` (two dashed curves) -- color = column,
            linestyle = distance scheme (redshift dashed), so this plot is
            visually distinct from the solid measured-distance chi2-only
            plot even though both only contain chi2-estimator curves.
        """
        os.makedirs(self._output_folder, exist_ok=True)
        out_file = os.path.join(self._output_folder, self.DATASET_FILE)

        dataset.write(out_file)
        logging.info(f"Wrote dataset: {out_file}")

        # CALCULATION_METHODS[0] is the chi2 estimator (avoid an inline "chi2").
        chi2_method = self.CALCULATION_METHODS[0]
        # DISTANCE_SCHEMES[0] is the measured-distance ("d") scheme;
        # DISTANCE_SCHEMES[1] is the redshift-distance ("cz") scheme.
        d_scheme = self.DISTANCE_SCHEMES[0]
        cz_scheme = self.DISTANCE_SCHEMES[1]

        chi2_labels = [
            self._make_label(col, chi2_method, d_scheme) for col in self.VELOCITY_COLUMNS
        ]
        both_labels = [
            self._make_label(col, method, d_scheme)
            for col in self.VELOCITY_COLUMNS
            for method in self.CALCULATION_METHODS
        ]
        chi2_cz_labels = [
            self._make_label(col, chi2_method, cz_scheme) for col in self.VELOCITY_COLUMNS
        ]

        self._plot_comparison(
            out_file,
            chi2_labels,
            self.CHI2_PLOT_FILE,
            title=self.REAL_SPACE_TITLE,
            xlabel=self.REAL_SPACE_XLABEL,
        )
        self._plot_comparison(
            out_file,
            both_labels,
            self.BOTH_METHODS_PLOT_FILE,
            title=self.REAL_SPACE_TITLE,
            xlabel=self.REAL_SPACE_XLABEL,
        )
        self._plot_comparison(
            out_file,
            chi2_cz_labels,
            self.CHI2_CZ_PLOT_FILE,
            linestyle_by=self._LINESTYLE_BY_SCHEME,
            title=self.REDSHIFT_SPACE_TITLE,
            xlabel=self.REDSHIFT_SPACE_XLABEL,
        )

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

    def plot_sigma_terms(self, df: pd.DataFrame) -> None:
        """
        Diagnostic figure for the two ``sigma_v`` ingredients (see the module
        docstring's sigma formula): one panel per distance scheme, sharing
        the y-axis. Each panel scatters the shared eDM measurement-error
        term against that scheme's radius (valid groups only), overlays the
        per-scheme ``sigma_min_fraction`` floor as a straight line through
        the origin, and marks the old flat ``sigma_min`` as a dotted gray
        reference. The panel title reports the floor-domination percentage:
        the fraction of that scheme's valid groups whose ``sigma_edm`` falls
        below the floor. Radii and sigmas are rebuilt via the same
        `_scheme_radii` / `_sigma_model` helpers `compute` uses, so the
        plotted values match the estimators' inputs; the validity mask here
        is geometry-only, though (it skips `compute`'s additional
        `r_hat_finite` and per-velocity-column finiteness cuts), so the
        plotted population can differ marginally from any one estimator run.
        """
        radius, scheme_radii = self._scheme_radii(df)
        eDM = df["eDM"].values.astype(float)
        sigma_edm, _ = self._sigma_model(eDM, radius, scheme_radii)

        style = self._cfg.visualization.style
        floor_pct = self.PERCENT_SCALE * self._sigma_min_fraction
        flat_sigma_min = self._cfg.bulkflow.sigma_min

        fig, axes = plt.subplots(
            ncols=len(self.DISTANCE_SCHEMES),
            figsize=self.SIGMA_PLOT_FIGSIZE,
            sharey=True,
        )
        for ax, scheme in zip(axes, self.DISTANCE_SCHEMES):
            scheme_radius, scheme_valid = scheme_radii[scheme]
            valid = (
                scheme_valid
                & np.isfinite(scheme_radius)
                & np.isfinite(sigma_edm)
            )
            r_valid = scheme_radius[valid]
            edm_valid = sigma_edm[valid]

            # Floor at each valid group's own scheme radius; a group is
            # "floor-dominated" when its eDM term falls below it.
            floor_valid = (
                self._sigma_min_fraction * self._hubble_velocity * r_valid
            )
            dominated_pct = self.PERCENT_SCALE * np.mean(edm_valid < floor_valid)

            ax.scatter(
                r_valid,
                edm_valid,
                s=self.SIGMA_SCATTER_SIZE,
                alpha=self.SIGMA_SCATTER_ALPHA,
                color=self.SIGMA_SCATTER_COLOR,
                label=self.SIGMA_SCATTER_LABEL,
            )
            # The floor is linear in r, so its two endpoints (origin -> the
            # panel's largest valid radius) define the whole line.
            r_line = np.array([0.0, r_valid.max()])
            ax.plot(
                r_line,
                self._sigma_min_fraction * self._hubble_velocity * r_line,
                color=self.SIGMA_FLOOR_COLOR,
                linestyle=self.SIGMA_FLOOR_LINESTYLE,
                label=self.SIGMA_FLOOR_LABEL_TEMPLATE.format(pct=floor_pct),
            )
            ax.axhline(
                flat_sigma_min,
                color=self.SIGMA_FLAT_REF_COLOR,
                linestyle=self.SIGMA_FLAT_REF_LINESTYLE,
                label=self.SIGMA_FLAT_REF_LABEL_TEMPLATE.format(value=flat_sigma_min),
            )
            ax.set_xlabel(self.SIGMA_XLABELS[scheme])
            ax.set_title(
                self.SIGMA_TITLE_TEMPLATE.format(
                    space=self.SIGMA_SPACE_NAMES[scheme], pct=dominated_pct
                )
            )
            ax.grid(True, alpha=style.grid_alpha)
            ax.legend()
        axes[0].set_ylabel(self.SIGMA_YLABEL)
        fig.tight_layout()

        os.makedirs(self._output_folder, exist_ok=True)
        out_path = os.path.join(self._output_folder, self.SIGMA_PLOT_FILE)
        fig.savefig(out_path, dpi=style.dpi_normal)
        plt.close(fig)
        logging.info(f"Wrote sigma diagnostic plot: {out_path}")

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        df = self.load_groups()
        dataset = self.compute(df)
        self.write_and_plot(dataset)
        self.plot_velocity_histograms(df)
        self.plot_sigma_terms(df)
