"""
manual_post_processing.py
-------------------------
Re-run theoretical calculations and regenerate plots from an existing
netCDF dataset without re-running the full pipeline.
"""

import logging
import os

from src.viz.visualize import BulkFlowPlotter
from src.config import BulkFlowPlotConfig
from src.config.cosmology_config import CosmologyConfig
from src.config.theory_config import TheoryConfig
from src.config.visualization_config import PlotStyleConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


NC_FILE = os.path.expanduser(
    "~/BulkFlow_miniproject/BulkFlow_miniproject/output/methods comparison/"
    "methods_comparison_radii-5-400__N-200__mask-full__methods-chi2_mean"
    "__selection_variable-overdensity__sigma_star-250_0.nc"
)
OUTPUT_FOLDER = os.path.dirname(NC_FILE)

METHODS_PLOT_FILE = "methods_comparison_chi2_mean_theory_radii-5-400.png"
DEBIAS_PLOT_FILE = "chi2_Utot_vs_Udeb_radii-5-400.png"


def main():
    cosmology_cfg = CosmologyConfig()
    theory_cfg = TheoryConfig()
    style = PlotStyleConfig()

    # Plot 1: chi2 vs mean (U_tot) vs LCDM theory
    BulkFlowPlotter(
        nc_file=NC_FILE,
        output_folder=OUTPUT_FOLDER,
        output_file=METHODS_PLOT_FILE,
        methods=["chi2", "mean"],
        plot_cfg=BulkFlowPlotConfig(plot_debiased=False, show_markers=False),
        cosmology_cfg=cosmology_cfg,
        theory_cfg=theory_cfg,
        style=style,
    ).plot()

    # Plot 2: chi2 biased U_tot vs noise-debiased U_deb (+ theory)
    BulkFlowPlotter(
        nc_file=NC_FILE,
        output_folder=OUTPUT_FOLDER,
        output_file=DEBIAS_PLOT_FILE,
        methods=["chi2"],
        plot_cfg=BulkFlowPlotConfig(show_markers=False),
        cosmology_cfg=cosmology_cfg,
        theory_cfg=theory_cfg,
        style=style,
    ).plot()


if __name__ == "__main__":
    main()
