"""
run_velocity_redshift_binning.py
---------------------------------
Entry point for the standalone CF4 Vpds-vs-Vpwf binned-mean-vs-redshift
comparison. Bins CF4 groups by redshift (z = Vcmb / c), computes the per-bin
mean +/- SEM of each velocity estimator, and renders the comparison plot and
table into output/velocity comparison/. No simulation catalog is loaded and
no checkpoint is overwritten.
"""

import argparse
import logging

from scripts.analyses.velocity_redshift_binning import VelocityRedshiftBinning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        dest="input_path",
        default=None,
        help="Path to the CF4 'All Group Velocities' CSV (default: repo's data/CF4_Groups_Velocities.csv).",
    )
    parser.add_argument(
        "--bins",
        dest="n_bins",
        type=int,
        default=None,
        help=f"Number of redshift bins (default: {VelocityRedshiftBinning.DEFAULT_N_BINS}).",
    )
    parser.add_argument(
        "--mode",
        dest="binning_mode",
        choices=("equal_width", "quantile"),
        default=None,
        help=f"Binning mode (default: {VelocityRedshiftBinning.DEFAULT_BINNING_MODE}).",
    )
    parser.add_argument(
        "--min-n-per-bin",
        dest="min_n_per_bin",
        type=int,
        default=None,
        help=f"Minimum group count per bin; underpopulated bins are skipped "
        f"(default: {VelocityRedshiftBinning.DEFAULT_MIN_N_PER_BIN}).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    VelocityRedshiftBinning(
        input_path=args.input_path,
        n_bins=args.n_bins,
        binning_mode=args.binning_mode,
        min_n_per_bin=args.min_n_per_bin,
    ).run()

    logging.info("Velocity-vs-redshift binning completed.")


if __name__ == "__main__":
    main()
