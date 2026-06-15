"""
Shared fixtures for the bulk-flow test suite.

All catalogs here are small, synthetic, and built in-memory — no real data files
are read and nothing is written to data/ or output/. Positions are in h^-1 Mpc
(periodic box coordinates) and velocities in km/s, matching the pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.config.mdpl2_config import MDPL2Config


# Box centre — far from every wall, so basic (non-PBC) tests never wrap.
CENTER = (500.0, 500.0, 500.0)


@pytest.fixture
def box_size():
    """Periodic box size from the simulation config (1000 h^-1 Mpc)."""
    return MDPL2Config().box_size


def _halos_from_offsets(offsets, velocities, origin=CENTER):
    """
    Build a halo DataFrame from displacement vectors around an origin.

    Parameters
    ----------
    offsets : (N, 3) array
        Displacements from `origin` (h^-1 Mpc). Halo positions are origin + offset.
    velocities : (N, 3) array
        Velocity vectors [vx, vy, vz] (km/s).
    origin : tuple of 3 floats
        Observer position.
    """
    offsets = np.asarray(offsets, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    pos = np.asarray(origin, dtype=float) + offsets
    return pd.DataFrame(
        {
            "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
            "vx": velocities[:, 0], "vy": velocities[:, 1], "vz": velocities[:, 2],
        }
    )


def _spanning_directions(n, seed):
    """`n` isotropic unit directions (deterministic) that span 3D."""
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs


@pytest.fixture
def origin():
    """Observer position used by the catalog fixtures."""
    return CENTER


@pytest.fixture
def make_constant_flow_catalog():
    """
    Factory: a catalog where every halo shares the SAME true velocity `U_true`.

    Halos are placed in directions that span 3D (so the ML matrix A is invertible)
    at a spread of radii in [10, 60] h^-1 Mpc. With a constant velocity field both
    estimators must recover `U_true`, which is what the recovery / agreement tests
    rely on.

    Returns a callable: make_constant_flow_catalog(U_true, n=12) -> DataFrame.
    """
    def _factory(U_true, n=12, seed=42):
        dirs = _spanning_directions(n, seed)
        rng = np.random.default_rng(seed + 1)
        radii = rng.uniform(10.0, 60.0, size=n)
        offsets = dirs * radii[:, None]
        velocities = np.tile(np.asarray(U_true, dtype=float), (n, 1))
        return _halos_from_offsets(offsets, velocities)
    return _factory


@pytest.fixture
def isotropic_catalog():
    """
    Many halos with random isotropic positions and velocities around the origin.
    The net bulk flow should be consistent with zero within the estimated error.
    """
    rng = np.random.default_rng(7)
    n = 400
    dirs = _spanning_directions(n, seed=7)
    radii = rng.uniform(10.0, 100.0, size=n)
    offsets = dirs * radii[:, None]
    velocities = rng.normal(scale=300.0, size=(n, 3))  # isotropic, zero-mean
    return _halos_from_offsets(offsets, velocities)
