"""
run_velocity_distance_mock.py
------------------------------
Entry point for the mock CF4 "peculiar velocity vs measured distance"
analysis: forges mock CF4-like observations from MDPL2 Rockstar halos around
one or more Local-Universe-like (or seeded-random) observers, re-derives the
Vpds/Vpwf estimators from the forged (cz_obs, D_meas) pair, and overlays the
result against the real CF4 binned means. Output goes to
output/velocity comparison mock/. Read-only on the Rockstar catalog -- no
checkpoint is overwritten.

The CF4-matching step is the slow part of this analysis (one full CF4 sweep
per observer); start with --n-observers 1 before scaling up.
"""

import argparse
import logging

from scripts.analyses.velocity_distance_mock import VelocityDistanceMock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        dest="config_path",
        default="config.yaml",
        help="Path to the pipeline config.yaml (default: repo root config.yaml).",
    )
    parser.add_argument(
        "--n-observers",
        dest="n_observers",
        type=int,
        default=None,
        help=f"Number of mock observers (default: {VelocityDistanceMock.DEFAULT_N_OBSERVERS}).",
    )
    parser.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=None,
        help=f"Base RNG seed; observer i uses seed + i (default: {VelocityDistanceMock.DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--bins",
        dest="n_bins",
        type=int,
        default=None,
        help=f"Number of bins per axis (default: {VelocityDistanceMock.DEFAULT_N_BINS}).",
    )
    parser.add_argument(
        "--mode",
        dest="binning_mode",
        choices=("equal_width", "quantile"),
        default=None,
        help=f"Binning mode (default: {VelocityDistanceMock.DEFAULT_BINNING_MODE}).",
    )
    parser.add_argument(
        "--min-n-per-bin",
        dest="min_n_per_bin",
        type=int,
        default=None,
        help=f"Minimum group count per bin; underpopulated bins are skipped "
        f"(default: {VelocityDistanceMock.DEFAULT_MIN_N_PER_BIN}).",
    )
    parser.add_argument(
        "--random-observers",
        dest="force_random_observers",
        action="store_true",
        help="Force seeded-random halo observers even if cached environmental columns exist.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    VelocityDistanceMock(
        config_path=args.config_path,
        n_observers=args.n_observers,
        seed=args.seed,
        n_bins=args.n_bins,
        binning_mode=args.binning_mode,
        min_n_per_bin=args.min_n_per_bin,
        force_random_observers=args.force_random_observers,
    ).run()

    logging.info("Mock CF4 velocity-vs-distance analysis completed.")


if __name__ == "__main__":
    main()
