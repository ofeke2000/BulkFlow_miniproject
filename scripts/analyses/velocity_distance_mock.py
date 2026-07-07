"""
velocity_distance_mock.py
--------------------------
Mock CF4 "peculiar velocity vs measured distance" analysis built from MDPL2
Rockstar halos, to test whether the trends seen in the real CF4 plots
(``velocity_redshift_binning.py``) are consistent with LambdaCDM once
measurement errors and the CF4 selection function are included.

Per observer (a Local-Universe-like halo, or a seeded-random halo as a
fallback):

  1. A CF4-like mock catalogue is built around the observer with
     ``MaskMaker.make_cf4_mask`` -- CF4 sky geometry (positions shifted to
     the observer, mod box) + matched halo velocities. The CF4-mask
     positions are treated as the *true* positions.
  2. The true PBC distance ``r_true`` and true line-of-sight velocity
     ``v_r_true`` are computed with ``radial_velocity_and_error_pbc``.
  3. The observed (CMB-frame analog) redshift is forged from the *exact*
     FLRW truth at the MDPL2 cosmology: ``z_cos_true(r_true)`` inverts the
     exact comoving-distance relation (colossus, interpolated on a fine z
     grid), and the peculiar velocity composes multiplicatively,
     ``(1 + z_obs) = (1 + z_cos_true)(1 + v_r_true/c)``, so
     ``cz_obs = c z_obs``. The box frame stands in for the CMB frame, so the
     observer's own halo velocity is *not* subtracted.
  4. The measured distance is forged in distance-modulus space:
     ``D_meas = r_true * 10**(delta_mu / 5)``, ``delta_mu ~ N(0, eDM_av)``,
     where ``eDM_av`` is the matched CF4 group's own distance-modulus error
     (carried through the mask via ``MaskMaker.make_cf4_mask``'s
     ``cf4_carry_columns``, not resampled from the eDM distribution). The RNG
     is seeded per observer (``seed + observer_id``) for reproducibility.
  5. ``Vpds``/``Vpwf`` analogs are re-derived from ``(cz_obs, D_meas)`` using
     the same second-order cosmographic correction (Davis & Scrimgeour 2014,
     their Eq. 14; Watkins & Feldman 2015), parameterized by the deceleration
     parameter ``q0 = Om0/2 - Ode0`` and jerk ``j0 = 1`` (flat LambdaCDM).
     The true ``v_r_true`` is kept as a zero-error reference series.

Analysis/plotting reuses ``VelocityRedshiftBinning``'s binning, table, and
figure machinery by subclassing it (``VELOCITY_COLUMNS``, ``X_AXES``,
``LABELS``/``MARKERS``, and the file templates are all class attributes
there already); only the loading/generation step is CF4-mock-specific. Three
views are produced:

  - per-observer binned means + scatter, one subfolder per observer, exactly
    mirroring the real CF4 script's figures;
  - pooled binned means over all observers (shared bin edges), with each
    observer's own binned curve overlaid as a thin line, to show
    observer-to-observer scatter;
  - a CF4-overlay figure: the mock binned means +/- SEM *across observers*
    (one sample per observer, not per galaxy) plotted against the real CF4
    binned means from ``VelocityRedshiftBinning`` on the same axes -- the key
    deliverable of this analysis.

No pipeline checkpoint is touched; this is read-only on the Rockstar catalog.

Note on conventions: the truth redshift is *exact* FLRW (multiplicative
peculiar-velocity composition, step 3) -- deliberately NOT generated with the
estimators' own second-order cosmographic bracket, so each estimator exhibits
its honest intrinsic residual bias rather than being zeroed by construction.
``Vpds_mock``'s cosmographic inversion tracks exact FLRW very closely at
these redshifts (residual ~0 km/s), while ``Vpwf_mock`` carries a genuine
positive residual growing with distance (~ +20 km/s at r = 50, ~ +190 km/s at
r = 150 h^-1 Mpc, at zero true velocity and zero eDM) -- a real feature of
the estimator that this mock is designed to expose, on top of the
distance-error-driven trends.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from scripts.pipeline.data_preprocessing import load_config
from scripts.pipeline.origin_selection import select_origin_points
from scripts.analyses.velocity_redshift_binning import VelocityRedshiftBinning
from src.classes import AppConfig, ROCKSTAR_COLUMNS
from src.io.data_loader import load_cf4_catalogue, load_rockstar_catalog
from src.physics.masks import MaskMaker
from src.physics.specific_utils import radial_velocity_and_error_pbc


class VelocityDistanceMock(VelocityRedshiftBinning):
    """
    Mock CF4 "velocity vs measured distance" analysis from MDPL2 halos.

    Subclasses ``VelocityRedshiftBinning`` to reuse its binning, table, and
    figure machinery (``bin_and_aggregate``, ``save_table``, ``plot``,
    ``plot_abs``, ``plot_scatter``, ``_draw_series``, ``_plot_binned``,
    ``_save_figure``); only the data-generation step (this class's whole
    reason to exist) and the extra CF4-overlay/per-observer-overlay figures
    are new. Run-specific numbers live here as class attributes, per the
    project's no-bare-numbers rule; simulation/cosmology parameters come from
    ``config.yaml`` via ``AppConfig``.
    """

    # --- Mock generation defaults (overridable via constructor / CLI) ---
    DEFAULT_N_OBSERVERS: int = 5
    DEFAULT_SEED: int = 1234

    # --- Cosmographic (Visser 2004; Davis & Scrimgeour 2014 Eq. 14; Watkins
    # & Feldman 2015) low-z expansion of H0/h * D_comoving(z) in {q0, j0}:
    #   B(z) = 1 + (1/2)(1-q0) z - (1/6)(1-q0-3 q0^2+j0) z^2
    #   (H0/h) D_C(z) = c z / (1+z) * B(z)
    # The SAME bracket B(z) is (a) evaluated at z_obs to build z_mod for the
    # Vpwf estimator, and (b) inverted (fixed-point solve) at z_cos, given
    # D_meas, for the Vpds estimator. ---
    JERK_PARAMETER: float = 1.0             # j0; exactly 1 for flat LambdaCDM (w=-1)
    COSMOGRAPHIC_HALF: float = 0.5          # 1/2 coefficient of the linear term
    COSMOGRAPHIC_SIXTH: float = 1.0 / 6.0   # 1/6 coefficient of the quadratic term
    COSMOGRAPHIC_Q0_SQUARED_COEFF: float = 3.0  # the "3 q0^2" term
    Z_COS_SOLVER_ITERATIONS: int = 8        # fixed-point iterations inverting D_meas -> z_cos

    # --- Exact-FLRW truth redshift (colossus): interpolation grid inverting the
    # exact comoving-distance relation D_C(z) -> z_cos_true(r_true). The grid
    # tops out well beyond the largest possible PBC distance in the box
    # (sqrt(3)/2 * 1000 ~ 866 h^-1 Mpc ~ z ~ 0.31 for MDPL2). ---
    TRUE_Z_GRID_MAX: float = 0.5
    TRUE_Z_GRID_POINTS: int = 4096
    COLOSSUS_COSMOLOGY_NAME: str = "mdpl2MockCosmo"

    # --- Distance-modulus definition: d propto 10^(mu/5) ---
    DISTANCE_MODULUS_BASE: float = 10.0
    DISTANCE_MODULUS_SLOPE: float = 5.0

    # --- CF4 columns carried through MaskMaker.make_cf4_mask for the error model ---
    CF4_ERROR_COLUMN: str = "eDM_av"
    CF4_CATALOG_DISTANCE_COLUMN: str = "distance"
    CF4_CARRY_COLUMNS: tuple[str, ...] = (CF4_ERROR_COLUMN, CF4_CATALOG_DISTANCE_COLUMN)

    # --- Cached environmental-column names checked for the observer-selection path
    # (mirrors environment_analysis.py / origin_selection.py) ---
    VIRGO_COLUMN: str = "near_virgo"

    # --- Mock per-object dataframe column names ---
    TRUE_RADIUS_COLUMN: str = "r_true"
    TRUE_VELOCITY_COLUMN: str = "v_r_true"
    EDM_COLUMN: str = "eDM"
    CZ_OBS_COLUMN: str = "cz_obs"
    Z_OBS_COLUMN: str = "z_obs"
    D_MEAS_COLUMN: str = "D_meas"
    VPDS_MOCK_COLUMN: str = "Vpds_mock"
    VPWF_MOCK_COLUMN: str = "Vpwf_mock"
    CF4_ID_COLUMN: str = "cf4_id"
    OBSERVER_ID_COLUMN: str = "observer_id"
    HUBBLE_DISTANCE_KEY: str = "d_cz_obs"   # cz_obs / (H0/h) [h^-1 Mpc]

    # --- Estimator columns rendered on every figure. Overrides the parent's
    # (Vpds, Vpwf); adds the zero-error true v_r as a reference series. ---
    VELOCITY_COLUMNS: tuple[str, ...] = (VPDS_MOCK_COLUMN, VPWF_MOCK_COLUMN, TRUE_VELOCITY_COLUMN)
    LABEL_VPDS_MOCK: str = "Vpds mock (Davis & Scrimgeour 2014)"
    LABEL_VPWF_MOCK: str = "Vpwf mock (Watkins & Feldman 2015)"
    LABEL_TRUE: str = "True $v_r$ (zero error)"
    MARKER_TRUE: str = "^"
    LABELS: dict[str, str] = {
        VPDS_MOCK_COLUMN: LABEL_VPDS_MOCK,
        VPWF_MOCK_COLUMN: LABEL_VPWF_MOCK,
        TRUE_VELOCITY_COLUMN: LABEL_TRUE,
    }
    MARKERS: dict[str, str] = {
        VPDS_MOCK_COLUMN: VelocityRedshiftBinning.MARKER_VPDS,
        VPWF_MOCK_COLUMN: VelocityRedshiftBinning.MARKER_VPWF,
        TRUE_VELOCITY_COLUMN: MARKER_TRUE,
    }

    # --- X-axes (mock analogs of the parent's z / D / d_cz axes; same tags,
    # so the CF4-overlay figure can pair each mock axis with its real-CF4
    # counterpart). All three are h^-1 Mpc / dimensionless within the mock
    # itself; the CF4-overlay applies the MDPL2 h to the real catalogue's
    # Mpc-based D and d_cz axes. ---
    X_AXES: tuple[dict, ...] = (
        {
            "column": Z_OBS_COLUMN,
            "name": "observed redshift",
            "xlabel": "Observed redshift $z_{obs}$",
            "tag": "redshift",
            "scatter_fractions": (1.0, 0.5, 0.1),
        },
        {
            "column": D_MEAS_COLUMN,
            "name": "measured distance",
            "xlabel": "Measured distance $D_{meas}$ [$h^{-1}$ Mpc]",
            "tag": "distance",
            "scatter_fractions": (0.1,),
        },
        {
            "column": HUBBLE_DISTANCE_KEY,
            "name": "Hubble distance",
            "xlabel": "Hubble distance $cz_{obs}/(H_0/h)$ [$h^{-1}$ Mpc]",
            "tag": "hubble_distance",
            "scatter_fractions": (0.1,),
        },
    )

    # --- CF4-overlay: which mock axis tags need the Mpc -> h^-1 Mpc conversion
    # when placed alongside the real CF4 D / d_cz axes (redshift needs none),
    # and the mock -> real velocity-column name mapping. ---
    AXES_REQUIRING_H_CONVERSION: tuple[str, ...] = ("distance", "hubble_distance")
    MOCK_TO_REAL_VELOCITY_COLUMN: dict[str, str] = {
        VPDS_MOCK_COLUMN: "Vpds",
        VPWF_MOCK_COLUMN: "Vpwf",
    }

    # --- Per-observer overlay styling (pooled-with-observers figure) ---
    OBSERVER_CURVE_ALPHA: float = 0.35
    OBSERVER_CURVE_LINEWIDTH: float = 1.0
    OBSERVER_LINE_COLORS: dict[str, str] = {
        VPDS_MOCK_COLUMN: "tab:blue",
        VPWF_MOCK_COLUMN: "tab:orange",
        TRUE_VELOCITY_COLUMN: "tab:green",
    }

    # --- Output ---
    OUTPUT_SUBFOLDER: str = "velocity comparison mock"
    OBSERVER_FOLDER_TEMPLATE: str = "observer_{observer_id:03d}"
    POOLED_FOLDER: str = "pooled"
    OVERLAY_FOLDER: str = "cf4_overlay"
    POOLED_PLOT_TITLE_TEMPLATE: str = "Pooled mock peculiar velocity vs {name} ({n} observers)"
    POOLED_PLOT_FILE_TEMPLATE: str = "velocity_vs_{tag}_pooled_binned.png"
    OVERLAY_PLOT_TITLE_TEMPLATE: str = "Mock vs real CF4 peculiar velocity vs {name}"
    OVERLAY_FILE_TEMPLATE: str = "velocity_vs_{tag}_cf4_overlay.png"
    OVERLAY_TABLE_TEMPLATE: str = "velocity_vs_{tag}_cf4_overlay_table.csv"

    def __init__(
        self,
        config_path: str = "config.yaml",
        n_observers: int | None = None,
        seed: int | None = None,
        n_bins: int | None = None,
        binning_mode: str | None = None,
        min_n_per_bin: int | None = None,
        force_random_observers: bool = False,
        output_folder: str | None = None,
    ) -> None:
        super().__init__(
            n_bins=n_bins,
            binning_mode=binning_mode,
            min_n_per_bin=min_n_per_bin,
            output_folder=output_folder,
        )
        self._config_path = config_path
        self._n_observers = n_observers or self.DEFAULT_N_OBSERVERS
        self._seed = seed if seed is not None else self.DEFAULT_SEED
        self._force_random_observers = force_random_observers
        self._have_cached_env_columns = False
        # Lazy (z_grid, D_C_grid) tables for the exact-FLRW truth redshift.
        self._true_z_grid: np.ndarray | None = None
        self._true_dc_grid: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_halos(self, cfg: AppConfig) -> tuple[pd.DataFrame, float]:
        """Load the Rockstar catalog, wrapped into the box; detect cached env columns."""
        box_size = cfg.MDPL2.box_size
        rockstar_path = cfg.paths.rockstar_catalog

        header_cols = pd.read_csv(rockstar_path, nrows=0).columns.tolist()
        delta_col = f"delta_{int(cfg.origin_configs.local_overdensity_radius)}"
        bulkflow_col = f"bulkflow_{int(cfg.origin_configs.local_bulkflow_radius)}"
        env_cols = [c for c in (delta_col, bulkflow_col, self.VIRGO_COLUMN) if c in header_cols]

        self._have_cached_env_columns = (
            not self._force_random_observers
            and delta_col in header_cols
            and bulkflow_col in header_cols
            and (not cfg.origin_configs.use_virgo_criteria or self.VIRGO_COLUMN in header_cols)
        )

        load_cols = list(ROCKSTAR_COLUMNS) + (env_cols if self._have_cached_env_columns else [])
        halos_df = load_rockstar_catalog(rockstar_path, columns=load_cols)
        halos_df[["x", "y", "z"]] %= box_size
        return halos_df, box_size

    def select_observers(self, halos_df: pd.DataFrame, cfg: AppConfig) -> pd.DataFrame:
        """
        Local-Universe-like observers via ``select_origin_points`` if the
        catalog's cached environmental columns are present, else seeded
        random halos.
        """
        if self._have_cached_env_columns:
            logging.info(
                "Cached environmental columns (delta_*/bulkflow_*/near_virgo) found on "
                "the catalog -> selecting Local-Universe-like observers via select_origin_points."
            )
            origin_cfg = dataclasses.replace(cfg.origin_configs, number_of_origins=self._n_observers)
            paths_cfg = dataclasses.replace(cfg.paths, output_folder=self._output_folder)
            run_cfg = dataclasses.replace(cfg, origin_configs=origin_cfg, paths=paths_cfg)
            observers = select_origin_points(halos_df, run_cfg)
        else:
            logging.info(
                "Cached environmental columns not found on the catalog (or "
                "--random-observers was set) -> falling back to seeded random halos as observers."
            )
            observers = halos_df.sample(
                n=min(self._n_observers, len(halos_df)), replace=False, random_state=self._seed
            )
        return observers.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Mock physics -- exact-FLRW truth
    # ------------------------------------------------------------------

    def _z_cos_true(self, r_true: np.ndarray, cosmology_cfg) -> np.ndarray:
        """
        Exact FLRW cosmological redshift of the true comoving distance
        ``r_true`` (h^-1 Mpc) at the MDPL2 cosmology, by inverting colossus's
        exact comoving-distance relation on a fine z grid (colossus's native
        comoving-distance unit is h^-1 Mpc, matching ``r_true``).

        Deliberately independent of the estimators' second-order
        ``_cosmographic_bracket``: the truth must be exact so each estimator
        exhibits its honest intrinsic residual, not zero by construction.
        """
        if self._true_z_grid is None:
            from colossus.cosmology import cosmology as colossus_cosmology

            colossus_cosmology.addCosmology(
                self.COLOSSUS_COSMOLOGY_NAME, cosmology_cfg.to_colossus_dict()
            )
            cosmo = colossus_cosmology.setCosmology(self.COLOSSUS_COSMOLOGY_NAME)
            self._true_z_grid = np.linspace(0.0, self.TRUE_Z_GRID_MAX, self.TRUE_Z_GRID_POINTS)
            self._true_dc_grid = cosmo.comovingDistance(z_min=0.0, z_max=self._true_z_grid)

        if np.any(r_true > self._true_dc_grid[-1]):
            raise ValueError(
                f"r_true exceeds the z-grid range (max D_C = {self._true_dc_grid[-1]:.1f} "
                f"h^-1 Mpc at z = {self.TRUE_Z_GRID_MAX}); increase TRUE_Z_GRID_MAX."
            )
        return np.interp(r_true, self._true_dc_grid, self._true_z_grid)

    # ------------------------------------------------------------------
    # Mock physics -- estimators (mimicking what CF4 does to data)
    # ------------------------------------------------------------------

    def _cosmographic_bracket(self, z: np.ndarray, q0: float, j0: float) -> np.ndarray:
        """
        Second-order cosmographic bracket shared by the Vpwf z_mod correction
        and the Vpds z_cos inversion (Davis & Scrimgeour 2014, Eq. 14;
        Watkins & Feldman 2015):
            B(z) = 1 + (1/2)(1-q0) z - (1/6)(1-q0-3 q0^2+j0) z^2
        """
        return (
            1.0
            + self.COSMOGRAPHIC_HALF * (1.0 - q0) * z
            - self.COSMOGRAPHIC_SIXTH
            * (1.0 - q0 - self.COSMOGRAPHIC_Q0_SQUARED_COEFF * q0 ** 2 + j0)
            * z ** 2
        )

    def _z_mod(self, z_obs: np.ndarray, q0: float, j0: float) -> np.ndarray:
        """Watkins & Feldman (2015) z_mod = z_obs * B(z_obs)."""
        return z_obs * self._cosmographic_bracket(z_obs, q0, j0)

    def _solve_z_cos(self, hubble_distance_ratio: np.ndarray, q0: float, j0: float) -> np.ndarray:
        """
        Invert ``(H0/h) D_meas = c * z_cos/(1+z_cos) * B(z_cos)`` for
        ``z_cos``, via fixed-point iteration ``z <- x (1+z) / B(z)`` seeded at
        the zeroth-order guess ``z = x`` (Davis & Scrimgeour 2014, Eq. 14,
        with ``(H0/h) D_comoving(z)`` on the left-hand side).

        Parameters
        ----------
        hubble_distance_ratio : (H0/h) * D_meas / c -- dimensionless ("x").
        """
        x = hubble_distance_ratio
        z = x.copy()
        for _ in range(self.Z_COS_SOLVER_ITERATIONS):
            z = x * (1.0 + z) / self._cosmographic_bracket(z, q0, j0)
        return z

    def build_observer_catalog(
        self,
        mask_maker: MaskMaker,
        position: np.ndarray,
        observer_id: int,
        box_size: float,
        cfg: AppConfig,
    ) -> pd.DataFrame:
        """Forge one observer's mock CF4 'velocity vs measured distance' catalogue."""
        matched = mask_maker.make_cf4_mask(
            position=position,
            radius=cfg.bulkflow.cf4_match_radius,
            max_doublings=cfg.bulkflow.cf4_match_max_doublings,
            cf4_carry_columns=self.CF4_CARRY_COLUMNS,
        )
        if matched.empty:
            logging.warning(f"Observer {observer_id}: no CF4-matched halos found; skipping.")
            return pd.DataFrame()

        before = len(matched)
        matched = matched.dropna(subset=[self.CF4_ERROR_COLUMN]).copy()
        matched = matched[matched[self.CF4_ERROR_COLUMN] > 0.0]
        dropped = before - len(matched)
        if dropped:
            logging.info(
                f"Observer {observer_id}: dropped {dropped:,} matched galaxies with "
                f"missing/non-positive {self.CF4_ERROR_COLUMN}."
            )
        if matched.empty:
            logging.warning(f"Observer {observer_id}: no usable eDM_av after cleaning; skipping.")
            return pd.DataFrame()

        matched = radial_velocity_and_error_pbc(matched, origin=tuple(position), box_size=box_size)

        r_true = matched["radius_from_origin"].values
        v_r_true = matched["v_rad"].values
        eDM = matched[self.CF4_ERROR_COLUMN].values

        match_quality = np.median(np.abs(matched[self.CF4_CATALOG_DISTANCE_COLUMN].values - r_true))
        logging.info(
            f"Observer {observer_id}: {len(matched):,} CF4-matched galaxies "
            f"(median |CF4 catalog distance - r_true| = {match_quality:.3f} h^-1 Mpc)."
        )

        hubble_v = cfg.cosmology.hubble_velocity_per_hinv_mpc

        # Exact-FLRW truth: z_cos_true(r_true), then multiplicative
        # peculiar-velocity composition (1+z_obs) = (1+z_cos_true)(1+v_r/c).
        z_cos_true = self._z_cos_true(r_true, cfg.cosmology)
        z_obs = (1.0 + z_cos_true) * (1.0 + v_r_true / self.SPEED_OF_LIGHT_KM_S) - 1.0
        cz_obs = self.SPEED_OF_LIGHT_KM_S * z_obs

        # Vpwf's log(cz_obs / ...) is undefined for blueshifted (cz_obs <= 0)
        # nearby infalling objects; a real survey would not treat them as
        # distance-ladder velocity tracers either. Drop them (and log).
        positive_cz = cz_obs > 0.0
        n_blueshifted = int((~positive_cz).sum())
        if n_blueshifted:
            logging.info(
                f"Observer {observer_id}: dropped {n_blueshifted:,} blueshifted "
                f"(cz_obs <= 0) mock galaxies before estimating velocities."
            )
            r_true = r_true[positive_cz]
            v_r_true = v_r_true[positive_cz]
            eDM = eDM[positive_cz]
            z_cos_true = z_cos_true[positive_cz]
            z_obs = z_obs[positive_cz]
            cz_obs = cz_obs[positive_cz]
            matched = matched[positive_cz]
        if len(matched) == 0:
            logging.warning(f"Observer {observer_id}: no mock galaxies left after cz_obs > 0 cut.")
            return pd.DataFrame()

        rng = np.random.default_rng(self._seed + observer_id)
        delta_mu = rng.normal(loc=0.0, scale=eDM)
        D_meas = r_true * self.DISTANCE_MODULUS_BASE ** (delta_mu / self.DISTANCE_MODULUS_SLOPE)

        q0 = cfg.cosmology.Om0 * self.COSMOGRAPHIC_HALF - cfg.cosmology.Ode0
        j0 = self.JERK_PARAMETER

        z_mod = self._z_mod(z_obs, q0, j0)
        cz_mod = self.SPEED_OF_LIGHT_KM_S * z_mod
        Vpwf_mock = cz_mod * np.log(cz_obs / (hubble_v * D_meas))

        z_cos = self._solve_z_cos(hubble_v * D_meas / self.SPEED_OF_LIGHT_KM_S, q0, j0)
        Vpds_mock = self.SPEED_OF_LIGHT_KM_S * (z_obs - z_cos) / (1.0 + z_cos)

        out = pd.DataFrame({
            self.TRUE_RADIUS_COLUMN: r_true,
            self.TRUE_VELOCITY_COLUMN: v_r_true,
            self.EDM_COLUMN: eDM,
            self.CZ_OBS_COLUMN: cz_obs,
            self.Z_OBS_COLUMN: z_obs,
            self.D_MEAS_COLUMN: D_meas,
            self.VPDS_MOCK_COLUMN: Vpds_mock,
            self.VPWF_MOCK_COLUMN: Vpwf_mock,
            self.CF4_ID_COLUMN: matched["cf4_id"].values,
            self.OBSERVER_ID_COLUMN: observer_id,
        })
        out[self.HUBBLE_DISTANCE_KEY] = out[self.CZ_OBS_COLUMN] / hubble_v
        return out

    # ------------------------------------------------------------------
    # Cross-observer aggregation
    # ------------------------------------------------------------------

    def _observer_sem_table(
        self, per_observer_binned: dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Mock binned means +/- SEM *across observers* (one sample per
        observer, not per galaxy) -- the quantity compared to the real CF4
        binned means in the CF4-overlay figure.
        """
        tagged = [df for df in per_observer_binned.values() if not df.empty]
        if not tagged:
            return pd.DataFrame()
        stacked = pd.concat(tagged, ignore_index=True)

        rows = []
        for bin_center, group in stacked.groupby("bin_center"):
            row = {"bin_center": bin_center, "n_observers": len(group)}
            for col in self.MOCK_TO_REAL_VELOCITY_COLUMN:
                values = group[f"mean_{col}"].values
                row[f"mean_{col}"] = np.mean(values)
                row[f"sem_{col}"] = stats.sem(values) if len(values) > 1 else np.nan
            rows.append(row)
        return pd.DataFrame(rows).sort_values("bin_center").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Extra plots (per-observer overlay + CF4 overlay)
    # ------------------------------------------------------------------

    def plot_pooled_with_observers(
        self,
        pooled_df: pd.DataFrame,
        per_observer_binned: dict[int, pd.DataFrame],
        axis: dict,
    ) -> str:
        """Bold pooled binned-mean curve (reusing `_draw_series`) with each observer's own curve overlaid."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=self.FIGSIZE)
        for obs_df in per_observer_binned.values():
            if obs_df.empty:
                continue
            for col in self.VELOCITY_COLUMNS:
                ax.plot(
                    obs_df["x_mean"],
                    obs_df[f"mean_{col}"],
                    color=self.OBSERVER_LINE_COLORS[col],
                    alpha=self.OBSERVER_CURVE_ALPHA,
                    linewidth=self.OBSERVER_CURVE_LINEWIDTH,
                )
        self._draw_series(ax, pooled_df, stat_infix="")

        observer_proxy = Line2D(
            [], [], color="gray", alpha=self.OBSERVER_CURVE_ALPHA,
            linewidth=self.OBSERVER_CURVE_LINEWIDTH,
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [observer_proxy], labels=labels + ["individual observers"])

        ax.set_xlabel(axis["xlabel"])
        ax.set_ylabel(self.YLABEL)
        ax.set_title(
            self.POOLED_PLOT_TITLE_TEMPLATE.format(name=axis["name"], n=len(per_observer_binned))
        )
        ax.grid(True, alpha=self.GRID_ALPHA)
        fig.tight_layout()

        out_path = self._save_figure(fig, self.POOLED_PLOT_FILE_TEMPLATE.format(tag=axis["tag"]))
        plt.close(fig)
        return out_path

    def plot_cf4_overlay(
        self,
        mock_sem_df: pd.DataFrame,
        real_binned_df: pd.DataFrame,
        axis: dict,
        h: float,
    ) -> str:
        """Overlay the mock binned means (+/- SEM across observers) on the real CF4 binned means."""
        import matplotlib.pyplot as plt

        scale = h if axis["tag"] in self.AXES_REQUIRING_H_CONVERSION else 1.0

        fig, ax = plt.subplots(figsize=self.FIGSIZE)
        for mock_col, real_col in self.MOCK_TO_REAL_VELOCITY_COLUMN.items():
            if not mock_sem_df.empty:
                ax.errorbar(
                    mock_sem_df["bin_center"],
                    mock_sem_df[f"mean_{mock_col}"],
                    yerr=mock_sem_df[f"sem_{mock_col}"],
                    marker=self.MARKERS[mock_col],
                    linestyle="-",
                    capsize=self.ERRORBAR_CAPSIZE,
                    label=f"{self.LABELS[mock_col]} [mock]",
                )
            ax.errorbar(
                real_binned_df["x_mean"] * scale,
                real_binned_df[f"mean_{real_col}"],
                yerr=real_binned_df[f"sem_{real_col}"],
                marker=VelocityRedshiftBinning.MARKERS[real_col],
                linestyle="--",
                capsize=self.ERRORBAR_CAPSIZE,
                label=f"{VelocityRedshiftBinning.LABELS[real_col]} [CF4 real]",
            )
        ax.axhline(0.0, color="k", linestyle="--", linewidth=self.REFERENCE_LINE_WIDTH)
        ax.set_xlabel(axis["xlabel"])
        ax.set_ylabel(self.YLABEL)
        ax.set_title(self.OVERLAY_PLOT_TITLE_TEMPLATE.format(name=axis["name"]))
        ax.legend()
        ax.grid(True, alpha=self.GRID_ALPHA)
        fig.tight_layout()

        out_path = self._save_figure(fig, self.OVERLAY_FILE_TEMPLATE.format(tag=axis["tag"]))
        plt.close(fig)
        return out_path

    def save_overlay_table(
        self,
        mock_sem_df: pd.DataFrame,
        real_binned_df: pd.DataFrame,
        axis: dict,
        h: float,
    ) -> str:
        os.makedirs(self._output_folder, exist_ok=True)
        out_path = os.path.join(
            self._output_folder, self.OVERLAY_TABLE_TEMPLATE.format(tag=axis["tag"])
        )
        scale = h if axis["tag"] in self.AXES_REQUIRING_H_CONVERSION else 1.0

        mock_display_cols = ["bin_center", "n_observers"] + [
            f"{stat}_{col}" for col in self.MOCK_TO_REAL_VELOCITY_COLUMN for stat in ("mean", "sem")
        ]
        if mock_sem_df.empty:
            mock_table = pd.DataFrame(columns=[f"mock_{c}" for c in mock_display_cols])
        else:
            mock_table = mock_sem_df[mock_display_cols].add_prefix("mock_")

        real_display_cols = ["x_mean", "N"] + [
            f"{stat}_{col}"
            for col in self.MOCK_TO_REAL_VELOCITY_COLUMN.values()
            for stat in ("mean", "sem")
        ]
        real_table = real_binned_df[real_display_cols].copy()
        real_table["x_mean"] = real_table["x_mean"] * scale
        real_table = real_table.add_prefix("cf4_real_")

        combined = pd.concat(
            [mock_table.reset_index(drop=True), real_table.reset_index(drop=True)], axis=1
        )
        combined.to_csv(out_path, index=False)
        logging.info(f"Wrote CF4-overlay table ({axis['name']}): {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Output-folder helper
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _output_subfolder(self, name: str):
        """Temporarily redirect `_save_figure`/`save_table` into an output subfolder."""
        previous = self._output_folder
        self._output_folder = os.path.join(previous, name, "")
        os.makedirs(self._output_folder, exist_ok=True)
        try:
            yield self._output_folder
        finally:
            self._output_folder = previous

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> dict:
        cfg = load_config(self._config_path)
        logging.info("Configuration loaded.")

        halos_df, box_size = self._load_halos(cfg)
        cf4_df = load_cf4_catalogue(cfg.paths.cf4_catalog, h=cfg.MDPL2.HubbleParameter)
        tree = cKDTree(halos_df[["x", "y", "z"]].values, boxsize=box_size)
        mask_maker = MaskMaker(halos_df=halos_df, tree=tree, box_size=box_size, cf4_df=cf4_df)

        observers = self.select_observers(halos_df, cfg)
        logging.info(f"Selected {len(observers)} observer(s) for the mock.")

        per_observer_catalogs: dict[int, pd.DataFrame] = {}
        for obs_idx, (_, row) in enumerate(observers.iterrows()):
            position = row[["x", "y", "z"]].values.astype(float)
            obs_df = self.build_observer_catalog(mask_maker, position, obs_idx, box_size, cfg)
            if not obs_df.empty:
                per_observer_catalogs[obs_idx] = obs_df

        if not per_observer_catalogs:
            raise RuntimeError("No observer produced a usable mock catalogue; aborting.")

        all_df = pd.concat(per_observer_catalogs.values(), ignore_index=True)
        logging.info(
            f"Pooled mock catalogue: {len(all_df):,} mock measurements across "
            f"{len(per_observer_catalogs)} observer(s)."
        )

        # --- View 1: per-observer binned means + scatter (mirrors the CF4 script) ---
        for obs_id, obs_df in per_observer_catalogs.items():
            with self._output_subfolder(self.OBSERVER_FOLDER_TEMPLATE.format(observer_id=obs_id)):
                for axis in self.X_AXES:
                    binned = self.bin_and_aggregate(obs_df, axis)
                    self.save_table(binned, axis)
                    self.plot(binned, axis)
                    self.plot_abs(binned, axis)
                    self.plot_scatter(obs_df, axis)

        # --- View 2: pooled binned means with per-observer curves overlaid ---
        pooled_by_axis: dict[str, tuple[pd.DataFrame, dict[int, pd.DataFrame]]] = {}
        with self._output_subfolder(self.POOLED_FOLDER):
            for axis in self.X_AXES:
                edges = self._bin_edges(all_df[axis["column"]].values)
                pooled = self.bin_and_aggregate(all_df, axis, edges=edges)
                per_observer_binned = {
                    obs_id: self.bin_and_aggregate(obs_df, axis, edges=edges)
                    for obs_id, obs_df in per_observer_catalogs.items()
                }
                self.save_table(pooled, axis)
                self.plot_pooled_with_observers(pooled, per_observer_binned, axis)
                pooled_by_axis[axis["tag"]] = (pooled, per_observer_binned)

        # --- View 3: CF4-overlay (the key deliverable) ---
        real_binning = VelocityRedshiftBinning(
            n_bins=self._n_bins,
            binning_mode=self._binning_mode,
            min_n_per_bin=self._min_n_per_bin,
        )
        real_df = real_binning.load_groups()
        h = cfg.MDPL2.HubbleParameter
        with self._output_subfolder(self.OVERLAY_FOLDER):
            for axis in self.X_AXES:
                _, per_observer_binned = pooled_by_axis[axis["tag"]]
                mock_sem_df = self._observer_sem_table(per_observer_binned)
                real_axis = next(a for a in VelocityRedshiftBinning.X_AXES if a["tag"] == axis["tag"])
                real_binned = real_binning.bin_and_aggregate(real_df, real_axis)
                self.save_overlay_table(mock_sem_df, real_binned, axis, h)
                self.plot_cf4_overlay(mock_sem_df, real_binned, axis, h)

        return {"all_observers": all_df, "pooled": pooled_by_axis}
