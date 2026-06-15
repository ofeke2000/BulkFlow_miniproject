"""
Category 1 — physics primitives with known answers.

These are the foundation: if periodic distance or radial velocity is wrong,
every downstream result is silently wrong (CLAUDE.md Hard Rule #1).
"""

import itertools

import numpy as np
import pandas as pd
import pytest

from src.physics.specific_utils import periodic_distance, radial_velocity_and_error_pbc


# ---------------------------------------------------------------------------
# periodic_distance
# ---------------------------------------------------------------------------

def test_periodic_distance_wraps_across_boundary(box_size):
    # x=1 and x=999 in a 1000 box are 2 apart through the wall, not 998.
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([999.0, 0.0, 0.0])
    assert periodic_distance(p1, p2, box_size) == pytest.approx(2.0)


def test_periodic_distance_symmetric(box_size):
    p1 = np.array([12.0, 880.0, 4.0])
    p2 = np.array([970.0, 30.0, 511.0])
    assert periodic_distance(p1, p2, box_size) == pytest.approx(
        periodic_distance(p2, p1, box_size)
    )


def test_periodic_distance_self_is_zero(box_size):
    p = np.array([123.0, 456.0, 789.0])
    assert periodic_distance(p, p, box_size) == pytest.approx(0.0)


def test_periodic_distance_matches_brute_force_min_image(box_size):
    # Brute-force minimum image: try all 27 neighbouring box translations.
    rng = np.random.default_rng(0)
    shifts = list(itertools.product((-box_size, 0.0, box_size), repeat=3))
    for _ in range(20):
        p1 = rng.uniform(0.0, box_size, size=3)
        p2 = rng.uniform(0.0, box_size, size=3)
        brute = min(np.linalg.norm(p1 - (p2 + np.array(s))) for s in shifts)
        assert periodic_distance(p1, p2, box_size) == pytest.approx(brute)


# ---------------------------------------------------------------------------
# radial_velocity_and_error_pbc
# ---------------------------------------------------------------------------

def _single_halo(pos, vel):
    return pd.DataFrame(
        {"x": [pos[0]], "y": [pos[1]], "z": [pos[2]],
         "vx": [vel[0]], "vy": [vel[1]], "vz": [vel[2]]}
    )


def test_r_hat_is_unit_length(box_size, make_constant_flow_catalog, origin):
    df = make_constant_flow_catalog(U_true=[100.0, 0.0, 0.0])
    out = radial_velocity_and_error_pbc(df, origin=origin, box_size=box_size)
    norms = np.linalg.norm(out[["r_hat_x", "r_hat_y", "r_hat_z"]].values, axis=1)
    assert np.allclose(norms, 1.0)


def test_radial_velocity_signs(box_size, origin):
    o = np.array(origin)
    # Halo offset along +x; velocity purely +x -> moving radially outward.
    out_out = radial_velocity_and_error_pbc(
        _single_halo(o + [30.0, 0.0, 0.0], [250.0, 0.0, 0.0]),
        origin=origin, box_size=box_size,
    )
    assert out_out["v_rad"].iloc[0] == pytest.approx(250.0)

    # Same position, velocity purely tangential (+y) -> ~zero radial.
    out_tan = radial_velocity_and_error_pbc(
        _single_halo(o + [30.0, 0.0, 0.0], [0.0, 250.0, 0.0]),
        origin=origin, box_size=box_size,
    )
    assert out_tan["v_rad"].iloc[0] == pytest.approx(0.0, abs=1e-9)

    # Velocity pointing back toward origin (-x) -> infall, negative radial.
    out_in = radial_velocity_and_error_pbc(
        _single_halo(o + [30.0, 0.0, 0.0], [-250.0, 0.0, 0.0]),
        origin=origin, box_size=box_size,
    )
    assert out_in["v_rad"].iloc[0] == pytest.approx(-250.0)


def test_r_hat_uses_minimum_image_across_boundary(box_size):
    # Origin near the lower wall; halo just across the upper wall.
    # Minimum-image separation is -2 in x (the halo is 2 units away in the -x
    # direction through the boundary), so r_hat must point -x, not +x, and the
    # PBC distance is 2 (not 998).
    origin = (1.0, 500.0, 500.0)
    halo = _single_halo([999.0, 500.0, 500.0], [0.0, 0.0, 0.0])
    out = radial_velocity_and_error_pbc(halo, origin=origin, box_size=box_size)
    assert out["r_hat_x"].iloc[0] == pytest.approx(-1.0)
    assert out["radius_from_origin"].iloc[0] == pytest.approx(2.0)


def test_radius_matches_periodic_distance(box_size, make_constant_flow_catalog, origin):
    df = make_constant_flow_catalog(U_true=[0.0, 0.0, 0.0])
    out = radial_velocity_and_error_pbc(df.copy(), origin=origin, box_size=box_size)
    expected = [
        periodic_distance(row[["x", "y", "z"]].values, np.array(origin), box_size)
        for _, row in df.iterrows()
    ]
    assert np.allclose(out["radius_from_origin"].values, expected)
