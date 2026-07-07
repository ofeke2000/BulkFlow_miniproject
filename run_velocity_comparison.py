"""
run_velocity_comparison.py
--------------------------
Entry point for the standalone CF4 velocity-estimator comparison.

Computes the chi^2 bulk flow directly on the CF4 "All Group Velocities"
catalogue using two radial-peculiar-velocity columns (Vpds and Vpwf), and
renders the comparison plot into output/velocity comparison/. No simulation
catalog is loaded and no checkpoint is overwritten.
"""

import logging

from scripts.pipeline.data_preprocessing import load_config
from scripts.analyses.velocity_comparison import VelocityComparison


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config("config.yaml")
    logging.info("Configuration loaded.")

    VelocityComparison(cfg).run()

    logging.info("Velocity comparison completed.")


if __name__ == "__main__":
    main()
