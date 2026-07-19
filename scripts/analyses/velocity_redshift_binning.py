"""
velocity_redshift_binning.py
-----------------------------
Standalone analysis comparing two CF4 radial-peculiar-velocity estimators
-- ``Vpds`` (Davis & Scrimgeour 2014) and ``Vpwf`` (Watkins & Feldman 2015) --
against redshift.

Unlike ``velocity_comparison.py`` (which feeds these columns into the chi^2
bulk-flow estimator), this analysis is purely descriptive. Every view is
produced against each of the x-axes configured in ``X_AXES``:

  - ``z``           -- redshift from the CMB-frame velocity, z = Vcmb / c.
  - ``D``           -- the catalogue's measured distance (Mpc, from the
                       distance modulus). Binning by D is *not* equivalent to
                       binning by z: D carries the large distance errors that
                       also drive the Vpds/Vpwf differences, so points are
                       genuinely reordered between the two axes.
  - ``d_cz``        -- Hubble distance cz/H0 (Mpc, H0 from ``H0_KM_S_MPC``).
                       A pure rescaling of the z axis, included so the two
                       distance axes can be compared in the same units.

Two views are produced per axis:

  - Binned means: groups are binned in the x variable, and the mean +/-
    standard error of the mean of each velocity column is computed per bin,
    both the signed mean and the mean of the absolute value.
  - Raw scatter: velocity vs x for every group, at the per-axis point-
    retention fractions configured in ``X_AXES`` (100%/50%/10% for z;
    10% only for the distance axes).

Each figure carries a ``v_pec = 0`` reference line.

``Vcmb`` is kept as the redshift frame throughout, since it is the frame in
which both ``Vpds`` and ``Vpwf`` are defined (not ``Vhel`` or ``Vls``).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
from scipy import stats

from src.config.cosmology_config import CosmologyConfig
from src.config.paths_config import DEFAULT_CF4_VELOCITIES_CATALOG, DEFAULT_OUTPUT_FOLDER
from src.config.physical_constants import PhysicalConstants
from src.io.data_loader import read_cf4_csv


class VelocityRedshiftBinning:
    """
    Bin CF4 groups in redshift and compare the binned-mean ``Vpds``/``Vpwf``.

    Run-specific parameters live here as class attributes (per the project's
    no-bare-numbers rule); this analysis has no dependency on ``AppConfig``,
    since it does not touch the simulation, cosmology, or theory machinery.
    """

    # --- Physical constants (SPEED_OF_LIGHT_KM_S) / cosmology (H0_KM_S_MPC)
    # are set in __init__ from PhysicalConstants / CosmologyConfig.H0_CF4
    # rather than hardcoded here, per the project's no-bare-numbers rule.
    # H0_KM_S_MPC is CF4's own distance-ladder calibration (Tully et al.
    # 2023), so the cz/H0 axis is in the same Mpc scale as the catalogue's D
    # column. This script works in plain Mpc (not h^-1 Mpc), so it uses the
    # plain H0_CF4, not the /h property used by the h^-1-Mpc-based analyses. ---

    # --- Columns read from the raw CF4 "All Group Velocities" catalogue ---
    ID_COLUMN: str = "pgc"
    REDSHIFT_FRAME_COLUMN: str = "Vcmb"
    DISTANCE_COLUMN: str = "D"  # measured distance from the distance modulus [Mpc]
    VELOCITY_COLUMNS: tuple[str, ...] = ("Vpds", "Vpwf")

    # --- Derived-column names ---
    REDSHIFT_KEY: str = "z"
    HUBBLE_DISTANCE_KEY: str = "d_cz"  # cz/H0 [Mpc]

    # --- X-axes every view is rendered against ---
    #   column            : dataframe column plotted on x
    #   name              : human-readable name (titles)
    #   xlabel            : axis label (with units)
    #   tag               : filename fragment
    #   scatter_fractions : point-retention levels for the raw scatter view
    X_AXES: tuple[dict, ...] = (
        {
            "column": REDSHIFT_KEY,
            "name": "redshift",
            "xlabel": "Redshift $z$",
            "tag": "redshift",
            "scatter_fractions": (1.0, 0.5, 0.1),
        },
        {
            "column": DISTANCE_COLUMN,
            "name": "measured distance",
            "xlabel": "Measured distance $D$ [Mpc]",
            "tag": "distance",
            "scatter_fractions": (0.1,),
        },
        {
            "column": HUBBLE_DISTANCE_KEY,
            "name": "Hubble distance",
            "xlabel": "Hubble distance $cz/H_0$ [Mpc]",
            "tag": "hubble_distance",
            "scatter_fractions": (0.1,),
        },
    )

    # --- Binning defaults (overridable via constructor / CLI) ---
    DEFAULT_N_BINS: int = 18
    DEFAULT_BINNING_MODE: str = "equal_width"  # "equal_width" or "quantile"
    DEFAULT_MIN_N_PER_BIN: int = 5

    # --- Known non-physical sentinel values in the relevant columns, if any.
    # Empty by default: this CF4 release uses NaN/blank for missing data, so
    # numeric coercion + dropna already catches it. Extend if a future
    # release reintroduces an explicit sentinel (e.g. -9999). ---
    SENTINEL_VALUES: tuple[float, ...] = ()

    # --- Plot / table labels ---
    LABEL_VPDS: str = "Vpds (Davis & Scrimgeour 2014)"
    LABEL_VPWF: str = "Vpwf (Watkins & Feldman 2015)"
    MARKER_VPDS: str = "o"
    MARKER_VPWF: str = "s"
    LABELS: dict[str, str] = {"Vpds": LABEL_VPDS, "Vpwf": LABEL_VPWF}
    MARKERS: dict[str, str] = {"Vpds": MARKER_VPDS, "Vpwf": MARKER_VPWF}
    PLOT_TITLE_TEMPLATE: str = "CF4 peculiar velocity vs {name} (binned means)"
    ABS_PLOT_TITLE_TEMPLATE: str = "CF4 |peculiar velocity| vs {name} (binned means)"
    YLABEL: str = "Peculiar velocity [km/s]"
    ABS_YLABEL: str = "|Peculiar velocity| [km/s]"

    # --- Scatter (raw, unbinned points) settings ---
    SCATTER_RANDOM_SEED: int = 42
    SCATTER_MARKER_SIZE: float = 6.0
    SCATTER_ALPHA: float = 0.25
    SCATTER_TITLE_TEMPLATE: str = "CF4 peculiar velocity vs {name} ({pct}% of points, N={n:,})"
    SCATTER_FILE_TEMPLATE: str = "velocity_vs_{tag}_scatter_{pct}pct.png"

    # --- Output ---
    OUTPUT_SUBFOLDER: str = "velocity comparison"
    PLOT_FILE_TEMPLATE: str = "velocity_vs_{tag}_binned.png"
    ABS_PLOT_FILE_TEMPLATE: str = "abs_velocity_vs_{tag}_binned.png"
    TABLE_FILE_TEMPLATE: str = "velocity_vs_{tag}_binned_table.csv"
    DPI: int = 150
    FIGSIZE: tuple[float, float] = (8.0, 6.0)
    ERRORBAR_CAPSIZE: float = 3.0
    GRID_ALPHA: float = 0.7
    REFERENCE_LINE_WIDTH: float = 1.0

    def __init__(
        self,
        input_path: str | None = None,
        n_bins: int | None = None,
        binning_mode: str | None = None,
        min_n_per_bin: int | None = None,
        output_folder: str | None = None,
    ) -> None:
        self.SPEED_OF_LIGHT_KM_S = PhysicalConstants().SPEED_OF_LIGHT_KM_S
        self.H0_KM_S_MPC = CosmologyConfig().H0_CF4
        self._input_path = input_path or DEFAULT_CF4_VELOCITIES_CATALOG
        self._n_bins = n_bins or self.DEFAULT_N_BINS
        self._binning_mode = binning_mode or self.DEFAULT_BINNING_MODE
        self._min_n_per_bin = (
            min_n_per_bin if min_n_per_bin is not None else self.DEFAULT_MIN_N_PER_BIN
        )
        if self._binning_mode not in ("equal_width", "quantile"):
            raise ValueError(
                f"binning_mode must be 'equal_width' or 'quantile', got '{self._binning_mode}'."
            )

        self._output_folder = os.path.join(
            output_folder or DEFAULT_OUTPUT_FOLDER, self.OUTPUT_SUBFOLDER, ""
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_groups(self) -> pd.DataFrame:
        """Load the raw CF4 catalogue and extract the clean id/Vcmb/D/velocity columns."""
        raw = read_cf4_csv(self._input_path)
        logging.info(f"Loaded {len(raw):,} CF4 entries from {self._input_path}.")

        by_lower = {c.lower(): c for c in raw.columns}
        wanted = (
            self.ID_COLUMN,
            self.REDSHIFT_FRAME_COLUMN,
            self.DISTANCE_COLUMN,
            *self.VELOCITY_COLUMNS,
        )
        missing = [c for c in wanted if c.lower() not in by_lower]
        if missing:
            raise KeyError(f"CF4 catalogue is missing required column(s): {missing}")

        df = raw[[by_lower[c.lower()] for c in wanted]].copy()
        df.columns = list(wanted)

        numeric_cols = (
            self.REDSHIFT_FRAME_COLUMN,
            self.DISTANCE_COLUMN,
            *self.VELOCITY_COLUMNS,
        )
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if self.SENTINEL_VALUES:
                df.loc[df[col].isin(self.SENTINEL_VALUES), col] = np.nan

        before = len(df)
        df = df.dropna(subset=list(numeric_cols))
        dropped = before - len(df)
        if dropped:
            logging.info(f"Dropped {dropped:,} rows with missing/sentinel values.")

        vcmb = df[self.REDSHIFT_FRAME_COLUMN]
        df[self.REDSHIFT_KEY] = vcmb / self.SPEED_OF_LIGHT_KM_S
        df[self.HUBBLE_DISTANCE_KEY] = vcmb / self.H0_KM_S_MPC
        return df

    # ------------------------------------------------------------------
    # Binning
    # ------------------------------------------------------------------

    def _bin_edges(self, z: np.ndarray) -> np.ndarray:
        if self._binning_mode == "equal_width":
            edges = np.linspace(z.min(), z.max(), self._n_bins + 1)
        else:
            edges = np.quantile(z, np.linspace(0.0, 1.0, self._n_bins + 1))
            edges = np.unique(edges)
            if len(edges) - 1 < self._n_bins:
                logging.warning(
                    f"Quantile binning collapsed to {len(edges) - 1} unique bin(s) "
                    f"(requested {self._n_bins}); the z distribution has repeated values."
                )
        return edges

    def bin_and_aggregate(
        self, df: pd.DataFrame, axis: dict, edges: np.ndarray | None = None
    ) -> pd.DataFrame:
        """
        Bin groups in the axis variable; per-bin mean +/- SEM per velocity column.

        Parameters
        ----------
        edges : np.ndarray, optional
            Explicit bin edges to reuse across multiple dataframes (e.g. one
            shared binning applied to several observers' catalogues in
            ``VelocityDistanceMock``, so their curves are directly
            comparable). Defaults to ``None``, which computes edges from
            ``df`` itself -- the original, unchanged behaviour.
        """
        x = df[axis["column"]]
        if edges is None:
            edges = self._bin_edges(x.values)
        n_bins = len(edges) - 1

        bin_index = pd.cut(x, bins=edges, labels=False, include_lowest=True)

        rows = []
        skipped = []
        for i in range(n_bins):
            sub = df[bin_index == i]
            n = len(sub)
            if n == 0:
                skipped.append((i, 0))
                continue
            if n < self._min_n_per_bin:
                skipped.append((i, n))
                continue

            row = {
                "bin_left": edges[i],
                "bin_right": edges[i + 1],
                "bin_center": 0.5 * (edges[i] + edges[i + 1]),
                "x_mean": x[bin_index == i].mean(),
                "N": n,
            }
            for col in self.VELOCITY_COLUMNS:
                row[f"mean_{col}"] = sub[col].mean()
                row[f"sem_{col}"] = stats.sem(sub[col].values)
                row[f"mean_abs_{col}"] = sub[col].abs().mean()
                row[f"sem_abs_{col}"] = stats.sem(sub[col].abs().values)
            rows.append(row)

        if skipped:
            logging.warning(
                f"Skipped {len(skipped)} {axis['name']} bin(s) with "
                f"N < {self._min_n_per_bin} (bin index, N): {skipped}"
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_table(self, binned_df: pd.DataFrame, axis: dict) -> str:
        os.makedirs(self._output_folder, exist_ok=True)
        out_path = os.path.join(
            self._output_folder, self.TABLE_FILE_TEMPLATE.format(tag=axis["tag"])
        )

        display_cols = ["x_mean", "N"] + [
            f"{stat}_{col}"
            for col in self.VELOCITY_COLUMNS
            for stat in ("mean", "sem", "mean_abs", "sem_abs")
        ]
        binned_df[display_cols].to_csv(out_path, index=False)
        logging.info(f"Wrote binned table ({axis['name']}): {out_path}")
        logging.info("\n" + binned_df[display_cols].to_string(index=False))
        return out_path

    def _draw_series(self, ax, binned_df: pd.DataFrame, stat_infix: str) -> None:
        """Draw one errorbar series per velocity column for a stat infix ('' or 'abs_')."""
        for col in self.VELOCITY_COLUMNS:
            ax.errorbar(
                binned_df["x_mean"],
                binned_df[f"mean_{stat_infix}{col}"],
                yerr=binned_df[f"sem_{stat_infix}{col}"],
                marker=self.MARKERS[col],
                capsize=self.ERRORBAR_CAPSIZE,
                label=self.LABELS[col],
            )
        ax.axhline(
            0.0, color="k", linestyle="--", linewidth=self.REFERENCE_LINE_WIDTH
        )

    def _save_figure(self, fig, output_file: str) -> str:
        os.makedirs(self._output_folder, exist_ok=True)
        out_path = os.path.join(self._output_folder, output_file)
        fig.savefig(out_path, dpi=self.DPI)
        logging.info(f"Saved plot: {out_path}")
        return out_path

    def _plot_binned(
        self,
        binned_df: pd.DataFrame,
        axis: dict,
        stat_infix: str,
        ylabel: str,
        title_template: str,
        file_template: str,
    ) -> str:
        """Render one binned-mean figure for the given axis and stat infix."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=self.FIGSIZE)
        self._draw_series(ax, binned_df, stat_infix=stat_infix)
        ax.set_xlabel(axis["xlabel"])
        ax.set_ylabel(ylabel)
        ax.set_title(title_template.format(name=axis["name"]))
        ax.legend()
        ax.grid(True, alpha=self.GRID_ALPHA)
        fig.tight_layout()

        out_path = self._save_figure(fig, file_template.format(tag=axis["tag"]))
        plt.close(fig)
        return out_path

    def plot(self, binned_df: pd.DataFrame, axis: dict) -> str:
        """Plot the signed binned mean of each velocity column vs the axis variable."""
        return self._plot_binned(
            binned_df,
            axis,
            stat_infix="",
            ylabel=self.YLABEL,
            title_template=self.PLOT_TITLE_TEMPLATE,
            file_template=self.PLOT_FILE_TEMPLATE,
        )

    def plot_abs(self, binned_df: pd.DataFrame, axis: dict) -> str:
        """Plot the binned mean of |velocity| for each column vs the axis variable."""
        return self._plot_binned(
            binned_df,
            axis,
            stat_infix="abs_",
            ylabel=self.ABS_YLABEL,
            title_template=self.ABS_PLOT_TITLE_TEMPLATE,
            file_template=self.ABS_PLOT_FILE_TEMPLATE,
        )

    def plot_scatter(self, df: pd.DataFrame, axis: dict) -> list[str]:
        """
        Scatter every group's velocity vs the axis variable at each of the
        axis's point-retention fractions, one PNG per fraction. All fractions
        are sampled from the full dataframe with the same seed, so each
        smaller fraction is a subset of the larger ones.
        """
        import matplotlib.pyplot as plt

        out_paths = []
        for frac in axis["scatter_fractions"]:
            sample = (
                df if frac >= 1.0
                else df.sample(frac=frac, random_state=self.SCATTER_RANDOM_SEED)
            )
            pct = round(frac * 100)

            fig, ax = plt.subplots(figsize=self.FIGSIZE)
            for col in self.VELOCITY_COLUMNS:
                ax.scatter(
                    sample[axis["column"]],
                    sample[col],
                    s=self.SCATTER_MARKER_SIZE,
                    alpha=self.SCATTER_ALPHA,
                    marker=self.MARKERS[col],
                    label=self.LABELS[col],
                )
            ax.axhline(
                0.0, color="k", linestyle="--", linewidth=self.REFERENCE_LINE_WIDTH
            )
            ax.set_xlabel(axis["xlabel"])
            ax.set_ylabel(self.YLABEL)
            ax.set_title(
                self.SCATTER_TITLE_TEMPLATE.format(
                    name=axis["name"], pct=pct, n=len(sample)
                )
            )
            ax.legend()
            ax.grid(True, alpha=self.GRID_ALPHA)
            fig.tight_layout()

            out_path = self._save_figure(
                fig, self.SCATTER_FILE_TEMPLATE.format(tag=axis["tag"], pct=pct)
            )
            plt.close(fig)
            out_paths.append(out_path)

        return out_paths

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> dict[str, pd.DataFrame]:
        df = self.load_groups()
        binned_by_axis: dict[str, pd.DataFrame] = {}
        for axis in self.X_AXES:
            binned_df = self.bin_and_aggregate(df, axis)
            self.save_table(binned_df, axis)
            self.plot(binned_df, axis)
            self.plot_abs(binned_df, axis)
            self.plot_scatter(df, axis)
            binned_by_axis[axis["tag"]] = binned_df
        return binned_by_axis
