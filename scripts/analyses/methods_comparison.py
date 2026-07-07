"""
methods_comparison.py
---------------------
Standalone analysis comparing bulk-flow estimators.

Produces two plots:
  1. chi2 vs mean estimator (biased magnitude U_tot) against the LCDM theory.
  2. chi2 biased magnitude U_tot vs noise-debiased magnitude U_deb.

Both estimators are computed for the *same* set of origins (full mask) and
stored together in a single netCDF (method is a dataset dimension), so the two
plots are drawn from one self-describing file.

This entry point is intentionally separate from main.py: it does NOT run the
environmental analysis and does NOT overwrite the Rockstar catalog checkpoint.
It relies on the environmental columns (delta_*, bulkflow_*, near_virgo)
already present in the catalog CSV.
"""

import logging
import os

import pandas as pd
from scipy.spatial import cKDTree

from src.classes import AppConfig, Vector3D
from src.data.bulkflow_dataset import BulkFlowDataset
from src.physics.bulkflow import calculate_bulk_flow_series
from src.viz.visualize import BulkFlowPlotter, FacetSet
from src.config import BulkFlowPlotConfig
from scripts.pipeline.bulkflow_computation import _build_result, _build_dataset_attrs


class MethodsComparison:
    """
    Compare bulk-flow estimators over a shared set of origins.

    Run-specific parameters live here as class attributes (per the project's
    no-bare-numbers rule); shared physics, paths, cosmology, and theory settings
    are pulled from the AppConfig.
    """

    # --- Radial grid for this comparison (h^-1 Mpc) ---
    R_MIN: float = 5.0
    R_MAX: float = 250.0
    R_STEP: float = 5.0

    # --- Origin sample and the estimators compared ---
    N_ORIGINS: int = 50
    METHODS: tuple[str, ...] = ("chi2", "mean")
    MASK_NAME: str = "full"

    # --- Output plot filenames ---
    METHODS_PLOT_FILE: str = "methods_comparison_chi2_mean_theory.png"
    DEBIAS_PLOT_FILE: str = "chi2_Utot_vs_Udeb.png"

    def __init__(self, cfg: AppConfig, halos_df: pd.DataFrame, tree: cKDTree) -> None:
        self._cfg = cfg
        self._halos_df = halos_df
        self._tree = tree
        self._box_size = cfg.MDPL2.box_size

        bf = cfg.bulkflow
        self._error_frac = bf.error_fraction
        self._sigma_star = bf.sigma_star
        self._sigma_min = bf.sigma_min

        # Per-origin environment columns (mirror bulkflow_computation.py)
        self._delta_col = f"delta_{int(cfg.origin_configs.local_overdensity_radius)}"
        self._bulkflow_col = f"bulkflow_{int(cfg.origin_configs.local_bulkflow_radius)}"

    # ------------------------------------------------------------------
    # Origin selection
    # ------------------------------------------------------------------

    def select_origins(self) -> pd.DataFrame:
        """Select the N lowest-|delta| origins, with no other cuts."""
        if self._delta_col not in self._halos_df.columns:
            raise KeyError(
                f"Catalog is missing '{self._delta_col}'. Run the environmental "
                "analysis first to populate the derived columns."
            )
        abs_delta = self._halos_df[self._delta_col].abs()
        order = abs_delta.sort_values().index
        selected = self._halos_df.loc[order].head(self.N_ORIGINS)
        logging.info(
            f"Selected {len(selected)} origins by lowest |{self._delta_col}| "
            f"(range {abs_delta.loc[selected.index].min():.3e} – "
            f"{abs_delta.loc[selected.index].max():.3e})."
        )
        return selected

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------

    def compute(self, selected: pd.DataFrame) -> BulkFlowDataset:
        """Compute every estimator in METHODS for each origin (full mask)."""
        dataset = BulkFlowDataset()
        n = len(selected)

        for i, (_, row) in enumerate(selected.iterrows()):
            origin = Vector3D.from_sequence((row["x"], row["y"], row["z"]))
            origin_id = int(row["rockstarid"])

            idx_full = self._tree.query_ball_point(origin.to_array(), r=self.R_MAX)
            full_mask_df = self._halos_df.iloc[idx_full].copy()

            for method in self.METHODS:
                bf_df = calculate_bulk_flow_series(
                    halos_df=full_mask_df,
                    origin=origin,
                    r_max=self.R_MAX,
                    r_min=self.R_MIN,
                    r_jumps=self.R_STEP,
                    box_size=self._box_size,
                    calculation_method=method,
                    error_frac=self._error_frac,
                    sigma_star=self._sigma_star,
                    sigma_min=self._sigma_min,
                )
                dataset.add(_build_result(
                    bf_df, origin, origin_id, self.MASK_NAME,
                    method, row, self._delta_col, self._bulkflow_col,
                ))

            logging.info(f"Origin {i + 1}/{n} (id={origin_id}) computed.")

        dataset.set_attrs(self._build_attrs())
        return dataset

    def _build_attrs(self) -> dict:
        """Provenance attrs, overridden to reflect this run's actual grid/sample."""
        attrs = _build_dataset_attrs(self._cfg)
        attrs.update({
            "min_radius": self.R_MIN,
            "max_radius": self.R_MAX,
            "radii_step": self.R_STEP,
            "number_of_origins": self.N_ORIGINS,
            "calculation_method": ",".join(self.METHODS),
            "analysis": "methods_comparison",
        })
        return attrs

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _radii_range_str(self) -> str:
        """Compact 'min-max' radial-extent string for filenames."""
        return f"{self.R_MIN:.0f}-{self.R_MAX:.0f}"

    def _with_radii_tag(self, filename: str) -> str:
        """
        Insert the radial extent into a filename's base. The range is the
        plot x-axis (not a corner-textbox constant), so it is not part of the
        facet suffix — but it is a defining run parameter, so tagging it keeps
        runs with different ranges from overwriting each other.
        """
        root, ext = os.path.splitext(filename)
        return f"{root}_radii-{self._radii_range_str()}{ext}"

    def _dataset_filename(self, base_path: str) -> str:
        """
        Append the run's defining constants to the netCDF filename, using the
        same suffix logic (and tokens) as the plots so the dataset sorts
        alongside its figures and never overwrites a run with different
        constants.
        """
        facets = {
            "N": self.N_ORIGINS,
            "mask": self.MASK_NAME,
            "methods": ",".join(self.METHODS),
            "sigma_star": self._sigma_star,
            "selection_variable": self._cfg.postprocessing.selection_variable,
        }
        suffix = FacetSet([facets]).filename_suffix()
        root, ext = os.path.splitext(base_path)
        ext = ext or ".nc"
        return f"{root}__{suffix}{ext}" if suffix else f"{root}{ext}"

    def write_and_plot(self, dataset: BulkFlowDataset) -> None:
        """Persist the dataset and render the two comparison plots."""
        cfg = self._cfg
        out_file = self._dataset_filename(self._with_radii_tag(cfg.paths.output_file))
        out_folder = cfg.paths.output_folder

        dataset.write(out_file)
        logging.info(f"Wrote dataset: {out_file}")

        # Plot 1: chi2 vs mean (U_tot) vs LCDM theory.
        BulkFlowPlotter(
            nc_file=out_file,
            output_folder=out_folder,
            output_file=self._with_radii_tag(self.METHODS_PLOT_FILE),
            methods=list(self.METHODS),
            plot_cfg=BulkFlowPlotConfig(plot_debiased=False, show_markers=False),
            cosmology_cfg=cfg.cosmology,
            theory_cfg=cfg.theory,
            style=cfg.visualization.style,
        ).plot()

        # Plot 2: chi2 biased U_tot vs noise-debiased U_deb (+ theory).
        BulkFlowPlotter(
            nc_file=out_file,
            output_folder=out_folder,
            output_file=self._with_radii_tag(self.DEBIAS_PLOT_FILE),
            methods=["chi2"],
            plot_cfg=BulkFlowPlotConfig(show_markers=False),
            cosmology_cfg=cfg.cosmology,
            theory_cfg=cfg.theory,
            style=cfg.visualization.style,
        ).plot()

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        selected = self.select_origins()
        dataset = self.compute(selected)
        self.write_and_plot(dataset)
