# specific_utils.py
import numpy as np


def weighted_average(values, weights):
    """Compute weighted average safely."""
    values = np.asarray(values)
    weights = np.asarray(weights)
    if np.sum(weights) == 0:
        return np.nan
    return np.sum(values * weights) / np.sum(weights)

# ================================================================
# Distance computations
# ================================================================

def distance(x1, y1, z1, x2, y2, z2):
    """Compute Euclidean distance between two points or arrays."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)