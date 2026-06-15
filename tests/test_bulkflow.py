"""
Category 2 — the ML / chi2 estimator (recovery + properties), plus the
noise-free chi2-vs-mean agreement check that de-risks the method comparison.

All tests use small synthetic catalogs and drive the estimators through the same
entry point the pipeline uses: calculate_bulk_flow_series, switching only the
`calculation_method` argument.
"""

import numpy as np
import pandas as pd
import pytest

from src.physics.bulkflow import calculate_bulk_flow_series
from src.physics.specific_utils import periodic_distance


BOX = 1000.0
ORIGIN = (500.0, 500.0, 500.0)
# Radius grid that comfortably encloses the fixture catalogs (radii <= 60).
R_MIN, R_MAX, R_JUMPS = 20.0, 120.0, 20.0


def _last_u(df):
    """Bulk-flow vector at the largest radius (all halos included)."""
    row = df.iloc[-1]
    return np.array([row["u_x"], row["u_y"], row["u_z"]])


def _run(df, method, sigma_star=0.0):
    return calculate_bulk_flow_series(
        df.copy(), origin=ORIGIN, r_max=R_MAX, r_min=R_MIN, r_jumps=R_JUMPS,
        box_size=BOX, calculation_method=method, sigma_star=sigma_star,
    )


def _rotate_catalog(df, R, origin=ORIGIN):
    """Rotate halo positions (about origin) and velocities by orthogonal R."""
    o = np.array(origin)
    offsets = df[["x", "y", "z"]].values - o
    vels = df[["vx", "vy", "vz"]].values
    new_pos = o + offsets @ R.T
    new_vel = vels @ R.T
    out = df.copy()
    out[["x", "y", "z"]] = new_pos
    out[["vx", "vy", "vz"]] = new_vel
    return out


# ---------------------------------------------------------------------------
# Recovery & properties
# ---------------------------------------------------------------------------

def test_chi2_recovers_constant_flow(make_constant_flow_catalog):
    U_true = np.array([120.0, -80.0, 200.0])
    df = make_constant_flow_catalog(U_true=U_true)
    # Constant velocity field => v_rad = r_hat . U_true exactly, so the
    # weighted least-squares solution must equal U_true to solver precision.
    u = _last_u(_run(df, "chi2"))
    assert np.allclose(u, U_true, rtol=1e-6, atol=1e-6)


def test_rotational_covariance(make_constant_flow_catalog):
    U_true = np.array([120.0, -80.0, 200.0])
    df = make_constant_flow_catalog(U_true=U_true)

    rng = np.random.default_rng(3)
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))  # orthogonal matrix

    u_base = _last_u(_run(df, "chi2"))
    u_rot = _last_u(_run(_rotate_catalog(df, R), "chi2"))

    assert np.allclose(u_rot, R @ u_base, atol=1e-6)
    # Magnitude is rotation-invariant.
    assert np.linalg.norm(u_rot) == pytest.approx(np.linalg.norm(u_base), rel=1e-9)


def test_debiased_magnitude_never_negative(make_constant_flow_catalog):
    # Collinear halos (all along +x) make A singular in 3D. The estimator must
    # not emit a negative debiased magnitude — it returns NaN or a floored >=0
    # value (guard at src/bulkflow.py:126).
    n = 6
    offsets = np.zeros((n, 3))
    offsets[:, 0] = np.linspace(10.0, 60.0, n)  # all along x
    pos = np.array(ORIGIN) + offsets
    df = pd.DataFrame({
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        "vx": np.full(n, 150.0), "vy": np.zeros(n), "vz": np.zeros(n),
    })
    out = _run(df, "chi2")
    ud = out["U_debiased"].values
    assert np.all(np.isnan(ud) | (ud >= 0.0))


def test_n_used_is_cumulative_count(make_constant_flow_catalog):
    df = make_constant_flow_catalog(U_true=[100.0, 0.0, 0.0])
    out = _run(df, "chi2")

    # Non-decreasing in radius.
    n_used = out["n_used"].values
    assert np.all(np.diff(n_used) >= 0)

    # Matches the count of halos within each radius (PBC distance).
    radii = np.array([
        periodic_distance(row[["x", "y", "z"]].values, np.array(ORIGIN), BOX)
        for _, row in df.iterrows()
    ])
    for _, row in out.iterrows():
        expected = int(np.sum(radii <= row["radius"]))
        assert int(row["n_used"]) == expected


def test_isotropic_field_gives_small_flow(isotropic_catalog):
    # Zero-mean isotropic velocities -> recovered bulk flow consistent with zero,
    # and debiasing must not inflate it above the biased magnitude.
    out = _run(isotropic_catalog, "chi2", sigma_star=250.0)
    row = out.iloc[-1]
    U_total = row["U_total"]
    U_deb = row["U_debiased"]
    assert U_total < 100.0  # << the 300 km/s per-halo velocity scale
    assert np.isnan(U_deb) or U_deb <= U_total + 1e-9


# ---------------------------------------------------------------------------
# chi2-vs-mean agreement (the comparison pre-flight)
# ---------------------------------------------------------------------------

def test_chi2_and_mean_agree_in_noise_free_limit(make_constant_flow_catalog):
    # With a constant velocity field and sigma_star=0 the two estimators must
    # return the SAME bulk-flow vector. If this fails, the comparison machinery
    # has a bug and any real chi2-vs-mean difference is untrustworthy.
    U_true = np.array([90.0, 140.0, -60.0])
    df = make_constant_flow_catalog(U_true=U_true)

    u_chi2 = _last_u(_run(df, "chi2"))
    u_mean = _last_u(_run(df, "mean"))

    assert np.allclose(u_chi2, u_mean, atol=1e-6)
    assert np.allclose(u_mean, U_true, atol=1e-6)
