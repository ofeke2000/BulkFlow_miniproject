"""
run_cf4_depth.py
-----------------
Entry point for the standalone CF4 depth/noise diagnostic.

Characterizes how noisy the CF4 grouped catalogue (CF4_Groups.csv) gets with
cz, and suggests a preliminary cz_max cut. Renders two plots plus a short
note into output/CF4_tests/. No simulation catalog is loaded and no
checkpoint is overwritten.
"""

import logging

from scripts.pipeline.data_preprocessing import load_config
from scripts.analyses.cf4_depth_analysis import CF4DepthAnalysis


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config("config.yaml")
    logging.info("Configuration loaded.")

    CF4DepthAnalysis(cfg).run()

    logging.info("CF4 depth analysis completed.")


if __name__ == "__main__":
    main()
