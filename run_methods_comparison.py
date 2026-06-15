"""
run_methods_comparison.py
-------------------------
Entry point for the standalone bulk-flow methods comparison.

Loads the catalog (no environmental analysis, no checkpoint overwrite), then
runs MethodsComparison to compute the chi2 and mean estimators over a shared
set of origins and render the two comparison plots into the configured output
folder.
"""

import logging

from scripts.data_preprocessing import load_config, preprocess_data
from scripts.methods_comparison import MethodsComparison


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config("config.yaml")
    logging.info("Configuration loaded.")

    halos_df, tree, _ = preprocess_data(cfg)
    MethodsComparison(cfg, halos_df, tree).run()

    logging.info("Methods comparison completed.")


if __name__ == "__main__":
    main()
