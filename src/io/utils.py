# utils.py
import os
import pandas as pd
import time
from functools import wraps


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




