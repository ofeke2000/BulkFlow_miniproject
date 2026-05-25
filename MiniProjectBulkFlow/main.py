"""
main.py
-------
Main orchestration script for the bulk flow analysis pipeline.
"""

import logging
import time
import numpy as np

from scripts.data_preprocessing import load_config, preprocess_data, save_catalog_checkpoint
from scripts.environment_analysis import compute_environmental_tests
from scripts.origin_selection import select_origin_points
from scripts.bulkflow_computation import compute_bulkflows_for_origins
from scripts.postprocessing import aggregate_results, create_final_plots
from src.visualize import plot_bulkflow_from_hdf5


def main():
    """Main pipeline execution."""

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Timing dictionary
    timings = {}
    t0_total = time.time()

    # ===========================================
    # 1. Load configuration
    # ===========================================
    cfg = load_config("config.yaml")
    logging.info("Configuration loaded successfully.")

    # ===========================================
    # 2. Data preprocessing
    # ===========================================
    t0 = time.time()
    halos_df, tree, cf4_df = preprocess_data(cfg)
    timings["data_preprocessing"] = time.time() - t0

    # ===========================================
    # 3. Environmental analysis
    # ===========================================
    t0 = time.time()
    halos_df = compute_environmental_tests(halos_df, tree, cfg)
    timings["environmental_analysis"] = time.time() - t0

    # Save catalog with environmental data
    save_catalog_checkpoint(halos_df, cfg.paths.rockstar_catalog)

    # ===========================================
    # 4. Origin selection
    # ===========================================
    t0 = time.time()
    selected_points = select_origin_points(halos_df, cfg)
    timings["origin_selection"] = time.time() - t0

    # ===========================================
    # 5. Bulk flow computation
    # ===========================================
    t0 = time.time()
    computation_timings = compute_bulkflows_for_origins(halos_df, selected_points, tree, cfg, cf4_df=cf4_df)
    timings.update(computation_timings)
    timings["bulkflow_computation"] = time.time() - t0

    # ===========================================
    # 6. Final visualization
    # ===========================================
    bulkflow_lower_cut = cfg.origin_configs.local_bulkflow_lower_cut
    bulkflow_upper_cut = cfg.origin_configs.local_bulkflow_upper_cut
    output_folder = cfg.paths.output_folder
    output_file = cfg.paths.output_file

    plot_bulkflow_from_hdf5(
        hdf_file=output_file,
        output_folder=output_folder,
        key="bulkflow",
        output_file=f"bulkflow_vs_radius_local_bulkflow_[{bulkflow_lower_cut:.0f},{bulkflow_upper_cut:.0f}].png",
        plot_theory=True,
        use_mean_amplitude=True,
        plot_variance_band=True,
        show_markers=False,
        plot_all_curves=False
    )

    # ==========================================================
    # END — FINAL TIMING SUMMARY
    # ==========================================================
    timings["total_runtime"] = time.time() - t0_total

    logging.info("\n" + "=" * 60)
    logging.info("TIMING SUMMARY")
    logging.info("=" * 60)

    for key, value in timings.items():
        logging.info(f"{key:25s} : {value:8.3f} sec")

    logging.info("=" * 60)
    logging.info(f"TOTAL RUNTIME : {timings['total_runtime']:.3f} sec")
    logging.info("=" * 60)

    logging.info("Pipeline completed successfully!")


# ------------------------------------------------------
# Entry point
# ------------------------------------------------------
if __name__ == "__main__":
    main()
