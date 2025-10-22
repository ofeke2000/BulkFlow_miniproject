# utils.py
import os
import numpy as np
import pandas as pd
import logging
import time
from multiprocessing import cpu_count
from functools import wraps


# ================================================================
# Logging utilities
# ================================================================

def setup_logger(log_file: str = None):
    """Initialize and return a logger for the project."""
    logger = logging.getLogger("bulkflow")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


# ================================================================
# Timing decorator
# ================================================================

def timing(func):
    """Decorator to measure execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"[TIMER] {func.__name__} took {elapsed:.2f} seconds.")
        return result
    return wrapper


# ================================================================
# File & directory utilities
# ================================================================

def ensure_dir(path: str):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, path: str):
    """Save DataFrame safely with directory creation."""
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
    print(f"[INFO] Saved: {path}")


# ================================================================
# Distance computations
# ================================================================

def distance(x1, y1, z1, x2, y2, z2):
    """Compute Euclidean distance between two points or arrays."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


# ================================================================
# Parallelism utilities
# ================================================================

def get_available_cores(fraction: float = 0.9):
    """
    Returns the number of CPU cores to use, leaving a safety margin.
    Default uses 90% of available cores.
    """
    total = cpu_count()
    use = max(1, int(total * fraction))
    print(f"[INFO] Using {use}/{total} CPU cores.")
    return use


# ================================================================
# Miscellaneous utilities
# ================================================================

def weighted_average(values, weights):
    """Compute weighted average safely."""
    values = np.asarray(values)
    weights = np.asarray(weights)
    if np.sum(weights) == 0:
        return np.nan
    return np.sum(values * weights) / np.sum(weights)


def load_halo_data(path: str, usecols=None):
    """Convenient wrapper for reading Rockstar CSV files."""
    print(f"[INFO] Loading halo data from: {path}")
    return pd.read_csv(path, usecols=usecols)


def progress_bar(iterable, prefix="", size=60):
    """Simple text-based progress bar."""
    count = len(iterable)
    def show(j):
        x = int(size * j / count)
        print(f"{prefix}[{'#' * x}{'.' * (size - x)}] {j}/{count}", end="\r")
    for i, item in enumerate(iterable):
        yield item
        show(i + 1)
    print()

