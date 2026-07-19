"""
cf4_depth_analysis.py
----------------------
Standalone, one-time analysis characterizing how noisy the CF4 grouped
catalogue (``CF4_Groups.csv``) gets with redshift distance (``cz`` = ``Vcmb``),
to suggest a preliminary ``cz_max`` cut for downstream CF4-facing analyses.

This is exploratory/diagnostic code, not part of the pipeline: per the user's
explicit instruction for this script, inline named numeric constants are used
directly (as class attributes, for readability) rather than being routed
through ``config.yaml``.

Three missions, driven by ``CF4DepthAnalysis.run()``:

1. **Dominant-indicator classification** (`classify_dominant_indicator`).
   Each CF4 group is tagged with the distance indicator that dominates its
   combined distance modulus, by argmax of its per-indicator N-counts
   (``Nsn``, ``Nfp``, ``Ntf``, ``Nsbfo + Nsbfi``, ``Nc``). SN II groups have
   no N-count column, so they are recovered as a fallback: a group with a
   finite ``DM_SNII`` and every N-count at 0/NaN is tagged "SNII". Ties
   between N-counts are broken by ``CATEGORY_PRIORITY`` (most-precise
   indicator first: Calibrator > SBF > SNIa > TF > FP). Groups with every
   count at 0/NaN and no ``DM_SNII`` are "unclassified" (kept only for the
   Mission 3 full-sample histogram).

2. **sigma_v vs cz per subsample** (`compute_sigma_v` / `plot_sigma_vs_cz`).
   Each group's velocity error is derived from *its own dominant indicator's*
   distance-modulus error column (the user's choice, rather than the
   combined ``eDM_av``):

       sigma_v = MAG_TO_FRAC * eDM_<dominant> * Vcmb,   MAG_TO_FRAC = ln(10)/5

   SBF picks whichever of the optical/infrared error columns corresponds to
   the larger of ``Nsbfo``/``Nsbfi`` (falling back to the other if that pick
   is NaN). Calibrator groups have no per-indicator error column at all, so
   they always use ``eDM_av`` (noted in the plot legend). Any row still NaN
   after that falls back to ``eDM_av`` too. Plotted as a scatter (one color
   per subsample) plus a binned running median (bin width
   ``CZ_BIN_WIDTH`` = 500 km/s) so the trend is legible through the scatter.
   The x-axis display is capped at ``CZ_MAX_PLOT`` = 20,000 km/s; no data are
   dropped from the computation, and the note reports how many groups lie
   beyond the cap.

3. **dN/dcz** (`plot_dNdcz`). Step histogram of ``Vcmb`` for the full grouped
   catalogue plus one overlaid step histogram per dominant-indicator
   subsample, same bin width as Mission 2. The bin at which the full-sample
   count first drops below ``PEAK_DROPOFF_FRACTION`` of its peak bin (the
   "collapse" point) is marked with a vertical line.

The two plots feed a short, data-driven note (`write_note`) that reports
per-subsample counts, where sigma_v climbs steeply, where dN/dcz turns over,
and a candidate preliminary ``cz_max`` (the smaller of the dN/dcz collapse
point and the cz beyond which the pooled classified sample's median sigma_v
exceeds ``SIGMA_V_WARN_THRESHOLD``), written to
``output/CF4_tests/cz_max_note.md``.
"""

from __future__ import annotations

import logging
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.classes import AppConfig
from src.config.paths_config import DEFAULT_OUTPUT_FOLDER
from src.io.data_loader import read_cf4_csv


class CF4DepthAnalysis:
    """
    Characterize CF4 group-catalogue noise vs cz and suggest a preliminary
    cz_max cut. See module docstring for the three missions.
    """

    # --- Unit conversion: distance-modulus error -> fractional distance
    #     error (d propto 10^(mu/5) => d ln(d)/d(mu) = ln(10)/5). ---
    MAG_TO_FRAC: float = math.log(10.0) / 5.0

    # --- Binning / display ---
    CZ_BIN_WIDTH: float = 500.0     # km/s
    CZ_MAX_PLOT: float = 20000.0    # km/s, x-axis display cap (data not clipped)

    # --- Candidate-cz_max derivation thresholds ---
    PEAK_DROPOFF_FRACTION: float = 0.10   # dN/dcz "collapse": count < 10% of peak
    SIGMA_V_WARN_THRESHOLD: float = 3000.0  # km/s; "a few thousand"

    # --- Mission 1: dominant-indicator classification ---
    # N-count column per single-column indicator (SBF is combined below).
    N_COUNT_COLUMNS: dict[str, str] = {
        "SNIa": "Nsn",
        "FP": "Nfp",
        "TF": "Ntf",
        "Calibrator": "Nc",
    }
    SBF_OPTICAL_COUNT_COL: str = "Nsbfo"
    SBF_INFRARED_COUNT_COL: str = "Nsbfi"
    SNII_DM_COL: str = "DM_SNII"
    SNII_LABEL: str = "SNII"
    UNCLASSIFIED_LABEL: str = "unclassified"
    # Tie-break order when two indicators share the max N-count (most
    # precise/reliable indicator wins first).
    CATEGORY_PRIORITY: tuple[str, ...] = ("Calibrator", "SBF", "SNIa", "TF", "FP")
    # Classified categories plotted in Missions 2 & 3 (excludes "unclassified",
    # which only enters the Mission-3 full-sample histogram).
    CATEGORIES: tuple[str, ...] = ("SNIa", "FP", "TF", "SBF", "Calibrator", "SNII")

    # --- Mission 2: per-indicator error column + SBF/fallback handling ---
    ERROR_COLUMNS: dict[str, str] = {
        "SNIa": "eDM_SNIa",
        "FP": "eDM_FP",
        "TF": "eDM_TFR",
        "SNII": "eDM_SNII",
    }
    SBF_OPTICAL_ERR_COL: str = "eDM_SBFo"
    SBF_INFRARED_ERR_COL: str = "eDM_SBFir"
    FALLBACK_ERR_COL: str = "eDM_av"
    CZ_COL: str = "Vcmb"

    # --- Plot styling ---
    CATEGORY_COLORS: dict[str, str] = {
        "SNIa": "tab:blue",
        "FP": "tab:orange",
        "TF": "tab:green",
        "SBF": "tab:red",
        "Calibrator": "tab:purple",
        "SNII": "tab:brown",
        "All": "black",
    }
    CATEGORY_LABELS: dict[str, str] = {
        "SNIa": "SNe Ia",
        "FP": "FP",
        "TF": "TF",
        "SBF": "SBF",
        "Calibrator": "Calibrator (eDM_av)",
        "SNII": "SN II",
    }
    FIGSIZE: tuple[float, float] = (10.0, 6.0)
    DPI: int = 150
    GRID_ALPHA: float = 0.3
    SCATTER_SIZE: float = 5.0
    SCATTER_ALPHA: float = 0.2
    MEDIAN_LINEWIDTH: float = 2.2
    STEP_LINEWIDTH: float = 1.6
    FULL_SAMPLE_LINEWIDTH: float = 2.2

    # --- Output ---
    # Deliberately DEFAULT_OUTPUT_FOLDER (the repo's output/ dir), not
    # cfg.paths.output_folder: the latter can be repointed to a run-specific
    # temp subfolder via config.yaml's output_folder_type: "temp" (see
    # AppConfig.from_dict), which would break this script's fixed
    # output/CF4_tests/ deliverable path. Mirrors the convention already used
    # by velocity_comparison.py.
    OUTPUT_SUBFOLDER: str = "CF4_tests"
    SIGMA_PLOT_FILE: str = "sigma_v_vs_cz.png"
    DNDCZ_PLOT_FILE: str = "dNdcz.png"
    NOTE_FILE: str = "cz_max_note.md"

    def __init__(self, cfg: AppConfig, cf4_path: str | None = None) -> None:
        self._cfg = cfg
        self._cf4_path = cf4_path or cfg.paths.cf4_catalog
        output_root = DEFAULT_OUTPUT_FOLDER
        self._output_folder = os.path.join(output_root, self.OUTPUT_SUBFOLDER, "")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_groups(self) -> pd.DataFrame:
        """Load the raw CF4_Groups.csv (all columns, no rows dropped)."""
        df = read_cf4_csv(self._cf4_path)
        required = [
            self.CZ_COL,
            self.FALLBACK_ERR_COL,
            "Nsn", "Nfp", "Ntf", "Nsbfo", "Nsbfi", "Nc",
            self.SNII_DM_COL,
            *self.ERROR_COLUMNS.values(),
            self.SBF_OPTICAL_COUNT_COL, self.SBF_INFRARED_COUNT_COL,
            self.SBF_OPTICAL_ERR_COL, self.SBF_INFRARED_ERR_COL,
        ]
        missing = [c for c in dict.fromkeys(required) if c not in df.columns]
        if missing:
            raise KeyError(f"CF4_Groups.csv is missing required columns: {missing}")
        logging.info(f"Loaded {len(df):,} CF4 groups from {self._cf4_path}")
        return df

    # ------------------------------------------------------------------
    # Mission 1: dominant-indicator classification
    # ------------------------------------------------------------------

    def classify_dominant_indicator(self, df: pd.DataFrame) -> pd.Series:
        """
        Classify each group by argmax of its N-counts (SBF = Nsbfo + Nsbfi;
        NaN treated as 0). Ties broken by CATEGORY_PRIORITY. Groups with no
        positive count but a finite DM_SNII are "SNII"; groups with neither
        are "unclassified". Returns a pandas Series of labels aligned to
        df.index.
        """
        counts = pd.DataFrame(
            {
                "SNIa": df[self.N_COUNT_COLUMNS["SNIa"]].fillna(0.0),
                "FP": df[self.N_COUNT_COLUMNS["FP"]].fillna(0.0),
                "TF": df[self.N_COUNT_COLUMNS["TF"]].fillna(0.0),
                "SBF": (
                    df[self.SBF_OPTICAL_COUNT_COL].fillna(0.0)
                    + df[self.SBF_INFRARED_COUNT_COL].fillna(0.0)
                ),
                "Calibrator": df[self.N_COUNT_COLUMNS["Calibrator"]].fillna(0.0),
            },
            index=df.index,
        )

        max_val = counts.max(axis=1)
        dominant = pd.Series(np.nan, index=df.index, dtype=object)
        remaining = max_val > 0.0
        for category in self.CATEGORY_PRIORITY:
            match = remaining & (counts[category] == max_val)
            dominant.loc[match] = category
            remaining &= ~match

        # SN II fallback: no N-count column of its own -- only groups with a
        # finite DM_SNII AND still unassigned (i.e. every N-count was 0/NaN)
        # qualify.
        has_snii_dm = df[self.SNII_DM_COL].notna()
        snii_mask = dominant.isna() & has_snii_dm
        dominant.loc[snii_mask] = self.SNII_LABEL

        dominant.loc[dominant.isna()] = self.UNCLASSIFIED_LABEL

        summary = dominant.value_counts()
        logging.info("Dominant-indicator classification counts:\n%s", summary.to_string())
        print("Dominant-indicator classification counts:")
        print(summary.to_string())

        return dominant

    # ------------------------------------------------------------------
    # Mission 2: sigma_v vs cz
    # ------------------------------------------------------------------

    def compute_sigma_v(self, df: pd.DataFrame, dominant: pd.Series) -> pd.Series:
        """
        Per-object velocity error from the dominant indicator's own eDM
        column: sigma_v = MAG_TO_FRAC * eDM_<dominant> * Vcmb. Calibrator has
        no per-indicator error column and always uses eDM_av; any other
        category whose per-indicator eDM is NaN for a given row also falls
        back to eDM_av for that row. Unclassified rows get NaN.
        """
        eDM_row = pd.Series(np.nan, index=df.index, dtype=float)

        for category, col in self.ERROR_COLUMNS.items():
            mask = dominant == category
            eDM_row.loc[mask] = df.loc[mask, col]

        # SBF: use whichever of eDM_SBFo/eDM_SBFir corresponds to the larger
        # of Nsbfo/Nsbfi; fall back to the other if that pick is NaN.
        sbf_mask = dominant == "SBF"
        optical_ge_infrared = (
            df[self.SBF_OPTICAL_COUNT_COL].fillna(0.0)
            >= df[self.SBF_INFRARED_COUNT_COL].fillna(0.0)
        )
        chosen_sbf = pd.Series(
            np.where(optical_ge_infrared, df[self.SBF_OPTICAL_ERR_COL], df[self.SBF_INFRARED_ERR_COL]),
            index=df.index,
        )
        other_sbf = pd.Series(
            np.where(optical_ge_infrared, df[self.SBF_INFRARED_ERR_COL], df[self.SBF_OPTICAL_ERR_COL]),
            index=df.index,
        )
        chosen_sbf = chosen_sbf.where(chosen_sbf.notna(), other_sbf)
        eDM_row.loc[sbf_mask] = chosen_sbf.loc[sbf_mask]

        # Calibrator has no per-indicator error column at all.
        calibrator_mask = dominant == "Calibrator"
        eDM_row.loc[calibrator_mask] = df.loc[calibrator_mask, self.FALLBACK_ERR_COL]

        # General fallback: any classified row still NaN (per-indicator eDM
        # missing) falls back to eDM_av.
        classified_mask = dominant.isin(self.CATEGORIES)
        still_nan = classified_mask & eDM_row.isna()
        eDM_row.loc[still_nan] = df.loc[still_nan, self.FALLBACK_ERR_COL]

        sigma_v = self.MAG_TO_FRAC * eDM_row * df[self.CZ_COL]
        return sigma_v

    @staticmethod
    def _bin_edges(max_value: float, bin_width: float) -> np.ndarray:
        n_bins = int(np.ceil(max_value / bin_width)) + 1
        return np.arange(n_bins + 1) * bin_width

    def _binned_median(
        self, x: np.ndarray, y: np.ndarray, edges: np.ndarray
    ) -> np.ndarray:
        """Median of y in each bin of x defined by edges (len(edges)-1 bins)."""
        bin_idx = np.digitize(x, edges) - 1
        n_bins = len(edges) - 1
        medians = np.full(n_bins, np.nan)
        grouped = pd.Series(y).groupby(bin_idx).median()
        for i, value in grouped.items():
            if 0 <= i < n_bins:
                medians[i] = value
        return medians

    def plot_sigma_vs_cz(
        self, df: pd.DataFrame, dominant: pd.Series, sigma_v: pd.Series
    ) -> int:
        """
        Scatter sigma_v vs cz per subsample (shared axes, one color each)
        plus a binned running median per subsample. Returns the number of
        classified groups whose cz lies beyond CZ_MAX_PLOT (not dropped from
        the computation, just outside the plotted x-range).
        """
        cz = df[self.CZ_COL]
        fig, ax = plt.subplots(figsize=self.FIGSIZE)

        n_beyond = 0
        for category in self.CATEGORIES:
            mask = (dominant == category) & sigma_v.notna() & cz.notna()
            if mask.sum() == 0:
                continue
            cz_cat = cz[mask].to_numpy()
            sigma_cat = sigma_v[mask].to_numpy()
            color = self.CATEGORY_COLORS[category]
            label = self.CATEGORY_LABELS[category]

            n_beyond += int(np.sum(cz_cat > self.CZ_MAX_PLOT))

            ax.scatter(
                cz_cat, sigma_cat,
                s=self.SCATTER_SIZE, alpha=self.SCATTER_ALPHA, color=color,
                label=f"{label} (N={mask.sum():,})",
            )

            edges = self._bin_edges(cz_cat.max(), self.CZ_BIN_WIDTH)
            centers = edges[:-1] + self.CZ_BIN_WIDTH / 2.0
            medians = self._binned_median(cz_cat, sigma_cat, edges)
            valid = np.isfinite(medians)
            ax.plot(
                centers[valid], medians[valid],
                color=color, linewidth=self.MEDIAN_LINEWIDTH,
            )

        ax.set_xlim(0.0, self.CZ_MAX_PLOT)
        ax.set_xlabel("cz (Vcmb) [km/s]")
        ax.set_ylabel(r"$\sigma_v$ [km/s]")
        ax.set_title("CF4 group velocity error vs cz, by dominant distance indicator")
        ax.grid(True, alpha=self.GRID_ALPHA)
        ax.legend(fontsize=8, loc="upper left")
        fig.tight_layout()

        os.makedirs(self._output_folder, exist_ok=True)
        out_path = os.path.join(self._output_folder, self.SIGMA_PLOT_FILE)
        fig.savefig(out_path, dpi=self.DPI)
        plt.close(fig)
        logging.info(
            f"Wrote sigma_v vs cz plot: {out_path} "
            f"({n_beyond:,} classified groups beyond cz={self.CZ_MAX_PLOT:,.0f} km/s, not shown)"
        )
        return n_beyond

    # ------------------------------------------------------------------
    # Mission 3: dN/dcz
    # ------------------------------------------------------------------

    def plot_dNdcz(self, df: pd.DataFrame, dominant: pd.Series) -> dict:
        """
        Overlaid step histograms of Vcmb: full sample plus one per dominant
        subsample. Marks the cz at which the full-sample count first drops
        below PEAK_DROPOFF_FRACTION of its peak bin. Returns the derived
        stats dict used by write_note.
        """
        cz_all = df[self.CZ_COL].dropna().to_numpy()
        max_cz = float(cz_all.max())
        edges = self._bin_edges(max_cz, self.CZ_BIN_WIDTH)
        centers = edges[:-1] + self.CZ_BIN_WIDTH / 2.0

        full_counts, _ = np.histogram(cz_all, bins=edges)

        fig, ax = plt.subplots(figsize=self.FIGSIZE)
        ax.step(
            centers, full_counts, where="mid",
            color=self.CATEGORY_COLORS["All"],
            linewidth=self.FULL_SAMPLE_LINEWIDTH,
            label=f"All groups (N={len(cz_all):,})",
        )

        for category in self.CATEGORIES:
            cz_cat = df.loc[dominant == category, self.CZ_COL].dropna().to_numpy()
            if len(cz_cat) == 0:
                continue
            counts, _ = np.histogram(cz_cat, bins=edges)
            ax.step(
                centers, counts, where="mid",
                color=self.CATEGORY_COLORS[category],
                linewidth=self.STEP_LINEWIDTH, alpha=0.85,
                label=f"{self.CATEGORY_LABELS[category]} (N={len(cz_cat):,})",
            )

        peak_idx = int(np.argmax(full_counts))
        peak_count = int(full_counts[peak_idx])
        dropoff_idx = None
        for i in range(peak_idx, len(full_counts)):
            if full_counts[i] < self.PEAK_DROPOFF_FRACTION * peak_count:
                dropoff_idx = i
                break
        dropoff_cz = float(centers[dropoff_idx]) if dropoff_idx is not None else None

        if dropoff_cz is not None:
            ax.axvline(
                dropoff_cz, color="black", linestyle="--", linewidth=1.2,
                label=f"count < {self.PEAK_DROPOFF_FRACTION:.0%} of peak (cz={dropoff_cz:,.0f})",
            )

        ax.set_xlim(0.0, self.CZ_MAX_PLOT)
        ax.set_xlabel("cz (Vcmb) [km/s]")
        ax.set_ylabel(f"Groups per {self.CZ_BIN_WIDTH:.0f} km/s bin")
        ax.set_title("CF4 group counts vs cz, full sample and by dominant indicator")
        ax.grid(True, alpha=self.GRID_ALPHA)
        ax.legend(fontsize=8)
        fig.tight_layout()

        os.makedirs(self._output_folder, exist_ok=True)
        out_path = os.path.join(self._output_folder, self.DNDCZ_PLOT_FILE)
        fig.savefig(out_path, dpi=self.DPI)
        plt.close(fig)
        logging.info(f"Wrote dN/dcz plot: {out_path}")

        return {
            "max_cz": max_cz,
            "peak_cz": float(centers[peak_idx]),
            "peak_count": peak_count,
            "dropoff_cz": dropoff_cz,
        }

    # ------------------------------------------------------------------
    # Note
    # ------------------------------------------------------------------

    def _derive_sigma_threshold_cz(
        self, df: pd.DataFrame, dominant: pd.Series, sigma_v: pd.Series
    ) -> float | None:
        """
        cz at which the binned median sigma_v of the pooled classified
        sample first exceeds SIGMA_V_WARN_THRESHOLD, or None if it never
        does within the data.
        """
        classified = dominant.isin(self.CATEGORIES) & sigma_v.notna()
        cz_vals = df.loc[classified, self.CZ_COL].to_numpy()
        sig_vals = sigma_v.loc[classified].to_numpy()
        valid = np.isfinite(cz_vals) & np.isfinite(sig_vals)
        cz_vals, sig_vals = cz_vals[valid], sig_vals[valid]
        if len(cz_vals) == 0:
            return None

        edges = self._bin_edges(float(cz_vals.max()), self.CZ_BIN_WIDTH)
        centers = edges[:-1] + self.CZ_BIN_WIDTH / 2.0
        medians = self._binned_median(cz_vals, sig_vals, edges)

        for center, median in zip(centers, medians):
            if np.isfinite(median) and median > self.SIGMA_V_WARN_THRESHOLD:
                return float(center)
        return None

    def write_note(
        self,
        dominant: pd.Series,
        dNdcz_stats: dict,
        sigma_threshold_cz: float | None,
        n_beyond_plot: int,
    ) -> str:
        """Compose and write the short data-driven cz_max note."""
        counts = dominant.value_counts()
        candidates = [c for c in (dNdcz_stats["dropoff_cz"], sigma_threshold_cz) if c is not None]
        candidate_cz_max = min(candidates) if candidates else None

        lines = ["# CF4 depth diagnostic: preliminary cz_max", ""]

        lines.append("## Per-subsample counts")
        for category in (*self.CATEGORIES, self.UNCLASSIFIED_LABEL):
            n = int(counts.get(category, 0))
            lines.append(f"- {category}: {n:,}")
        lines.append(f"- **Total**: {int(counts.sum()):,}")
        lines.append("")

        lines.append("## dN/dcz")
        lines.append(
            f"- Full-sample count peaks at cz ≈ {dNdcz_stats['peak_cz']:,.0f} km/s "
            f"({dNdcz_stats['peak_count']:,} groups/bin)."
        )
        if dNdcz_stats["dropoff_cz"] is not None:
            lines.append(
                f"- Count first drops below {self.PEAK_DROPOFF_FRACTION:.0%} of the peak "
                f"at cz ≈ {dNdcz_stats['dropoff_cz']:,.0f} km/s."
            )
        else:
            lines.append(
                f"- Count never drops below {self.PEAK_DROPOFF_FRACTION:.0%} of the peak "
                f"within the data (max cz ≈ {dNdcz_stats['max_cz']:,.0f} km/s)."
            )
        lines.append("")

        lines.append("## sigma_v vs cz")
        if sigma_threshold_cz is not None:
            lines.append(
                f"- Pooled classified-sample median sigma_v first exceeds "
                f"{self.SIGMA_V_WARN_THRESHOLD:,.0f} km/s at cz ≈ {sigma_threshold_cz:,.0f} km/s."
            )
        else:
            lines.append(
                f"- Pooled classified-sample median sigma_v stays below "
                f"{self.SIGMA_V_WARN_THRESHOLD:,.0f} km/s throughout the data."
            )
        lines.append(
            f"- {n_beyond_plot:,} classified groups lie beyond the plotted cz range "
            f"({self.CZ_MAX_PLOT:,.0f} km/s); they are included in all computations above "
            "but not shown in sigma_v_vs_cz.png."
        )
        lines.append("")

        lines.append("## Candidate preliminary cz_max")
        if candidate_cz_max is not None:
            lines.append(f"**cz_max ≈ {candidate_cz_max:,.0f} km/s**")
            lines.append("")
            justification = (
                "dN/dcz collapse point" if candidate_cz_max == dNdcz_stats["dropoff_cz"]
                else "sigma_v growth threshold"
            )
            dropoff_str = (
                f"{dNdcz_stats['dropoff_cz']:,.0f} km/s"
                if dNdcz_stats["dropoff_cz"] is not None else "undefined"
            )
            sigma_str = (
                f"{sigma_threshold_cz:,.0f} km/s"
                if sigma_threshold_cz is not None else "undefined"
            )
            lines.append(
                f"Justification: this is the smaller of the dN/dcz collapse cz "
                f"({dropoff_str}) and the sigma_v threshold cz ({sigma_str}) -- driven "
                f"here by the {justification}, i.e. the catalogue becomes sparse and/or "
                "individually noisy beyond this point."
            )
        else:
            lines.append(
                "No candidate cz_max found: neither the dN/dcz collapse criterion nor the "
                "sigma_v threshold was crossed within the data range."
            )
        lines.append("")

        note = "\n".join(lines)
        os.makedirs(self._output_folder, exist_ok=True)
        out_path = os.path.join(self._output_folder, self.NOTE_FILE)
        with open(out_path, "w") as f:
            f.write(note)
        logging.info(f"Wrote note: {out_path}")
        print(note)
        return note

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        os.makedirs(self._output_folder, exist_ok=True)

        df = self.load_groups()
        dominant = self.classify_dominant_indicator(df)
        sigma_v = self.compute_sigma_v(df, dominant)

        n_beyond_plot = self.plot_sigma_vs_cz(df, dominant, sigma_v)
        dNdcz_stats = self.plot_dNdcz(df, dominant)
        sigma_threshold_cz = self._derive_sigma_threshold_cz(df, dominant, sigma_v)

        self.write_note(dominant, dNdcz_stats, sigma_threshold_cz, n_beyond_plot)
