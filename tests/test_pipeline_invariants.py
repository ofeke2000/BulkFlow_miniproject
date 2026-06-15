"""
Category 3 — pipeline invariant tests.

These tests verify the cache-and-skip pattern (CLAUDE.md Hard Rule #2),
overdensity sanity, uniform mask sizing, and netCDF round-trip correctness.
All catalogs are synthetic and built in-memory. Nothing is read from data/ or
written to output/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from src.physics.bulkflow import calculate_local_bulkflow
from src.physics.near_virgo import near_virgo
from src.physics.overdensity import OverdensityCalculator
from src.data.bulkflow_dataset import BulkFlowDataset, BulkFlowResult
from src.data.vector3d import Vector3D

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOX = 1000.0


def _make_simple_catalog(n=50, seed=0):
    """
    Small synthetic halo catalog spread evenly inside the box.
    Has columns x, y, z, vx, vy, vz, mvir.
    """
    rng = np.random.default_rng(seed)
    pos = rng.uniform(100.0, 900.0, size=(n, 3))
    vel = rng.normal(0.0, 200.0, size=(n, 3))
    return pd.DataFrame(
        {
            "x": pos[:, 0],
            "y": pos[:, 1],
            "z": pos[:, 2],
            "vx": vel[:, 0],
            "vy": vel[:, 1],
            "vz": vel[:, 2],
            "mvir": rng.uniform(1e12, 1e14, size=n),
        }
    )


def _tree_from_df(df):
    return cKDTree(df[["x", "y", "z"]].values, boxsize=BOX)


# ===========================================================================
# Test 1a: cache idempotency — calculate_local_bulkflow
# ===========================================================================

class TestLocalBulkflowCacheIdempotency:
    """Verify the skip-if-column-exists guard in calculate_local_bulkflow."""

    RADIUS = 30.0

    def test_column_added_after_first_call(self):
        df = _make_simple_catalog(n=50, seed=1)
        tree = _tree_from_df(df)
        colname = f"bulkflow_{int(self.RADIUS)}"

        assert colname not in df.columns, "column must be absent before first call"
        out = calculate_local_bulkflow(df, tree, radius=self.RADIUS)
        assert colname in out.columns, "column must exist after first call"

    def test_second_call_returns_identical_column(self):
        df = _make_simple_catalog(n=50, seed=2)
        tree = _tree_from_df(df)
        radius = self.RADIUS
        colname = f"bulkflow_{int(radius)}"

        out1 = calculate_local_bulkflow(df, tree, radius=radius)
        values_after_first = out1[colname].copy()

        # Mutate one value to confirm that the second call does NOT recompute.
        out1.at[0, colname] = 9999.0

        out2 = calculate_local_bulkflow(out1, tree, radius=radius)
        values_after_second = out2[colname]

        # The sentinel must survive — the skip path was taken.
        assert values_after_second.iloc[0] == 9999.0, (
            "Second call overwrote the column instead of skipping"
        )
        # All other rows must still match the first result.
        np.testing.assert_array_equal(
            values_after_second.iloc[1:].values,
            values_after_first.iloc[1:].values,
        )


# ===========================================================================
# Test 1b: cache idempotency — near_virgo
# ===========================================================================

class TestNearVirgoCacheIdempotency:
    """Verify the skip-if-column-exists guard in near_virgo."""

    def _catalog_with_massive_halo(self):
        """50 halos; one is massive (> 1e14) so the flag is non-trivially set."""
        rng = np.random.default_rng(10)
        df = _make_simple_catalog(n=49, seed=10)
        # Add a well-separated massive halo at a predictable location.
        massive = pd.DataFrame(
            {
                "x": [500.0],
                "y": [500.0],
                "z": [500.0],
                "vx": [0.0],
                "vy": [0.0],
                "vz": [0.0],
                "mvir": [2e14],  # above VirgoTestConfig.mass_threshold (1e14)
            }
        )
        return pd.concat([df, massive], ignore_index=True)

    def test_column_added_after_first_call(self):
        df = self._catalog_with_massive_halo()
        assert "near_virgo" not in df.columns
        out = near_virgo(df, box_size=BOX)
        assert "near_virgo" in out.columns

    def test_second_call_returns_identical_column(self):
        df = self._catalog_with_massive_halo()
        out1 = near_virgo(df, box_size=BOX)
        values_after_first = out1["near_virgo"].copy()

        # Corrupt one entry before the second call.
        out1.at[0, "near_virgo"] = 77

        out2 = near_virgo(out1, box_size=BOX)
        values_after_second = out2["near_virgo"]

        # The sentinel must survive (skip path was taken).
        assert values_after_second.iloc[0] == 77, (
            "Second call overwrote the column instead of skipping"
        )
        # The remaining values must be unchanged from the first call.
        np.testing.assert_array_equal(
            values_after_second.iloc[1:].values,
            values_after_first.iloc[1:].values,
        )


# ===========================================================================
# Test 1c: cache idempotency — OverdensityCalculator
# ===========================================================================

class TestOverdensityCacheIdempotency:
    """Verify the skip-if-column-exists guard in OverdensityCalculator.compute."""

    RADIUS = 5.0

    def _calculator_and_df(self):
        df = _make_simple_catalog(n=40, seed=20)
        tree = cKDTree(df[["x", "y", "z"]].values, boxsize=BOX)
        calc = OverdensityCalculator(
            radius=self.RADIUS,
            box_size=BOX,
            tree=tree,
            mass_column="mvir",
        )
        return calc, df

    def test_column_added_after_first_call(self):
        calc, df = self._calculator_and_df()
        colname = f"delta_{int(self.RADIUS)}"
        assert colname not in df.columns
        out = calc.compute(df)
        assert colname in out.columns

    def test_second_call_returns_identical_column(self):
        calc, df = self._calculator_and_df()
        colname = f"delta_{int(self.RADIUS)}"

        out1 = calc.compute(df)
        values_after_first = out1[colname].copy()

        # Corrupt entry 0.
        out1.at[0, colname] = 12345.0

        out2 = calc.compute(out1)
        values_after_second = out2[colname]

        # Sentinel must survive.
        assert values_after_second.iloc[0] == 12345.0, (
            "Second call overwrote the column instead of skipping"
        )
        np.testing.assert_array_equal(
            values_after_second.iloc[1:].values,
            values_after_first.iloc[1:].values,
        )


# ===========================================================================
# Test 2: overdensity sanity
# ===========================================================================

class TestOverdensitySanity:
    """
    On a uniformly random box the mean delta ≈ 0 (loose tolerance).
    A halo inside a deliberately dense clump has higher delta than an
    isolated halo in an otherwise sparse region.
    """

    RADIUS = 5.0

    def _run_overdensity(self, df):
        tree = cKDTree(df[["x", "y", "z"]].values, boxsize=BOX)
        calc = OverdensityCalculator(
            radius=self.RADIUS,
            box_size=BOX,
            tree=tree,
            mass_column="mvir",
        )
        return calc.compute(df.copy())

    def test_uniform_grid_delta_is_constant(self):
        """
        On a perfectly regular grid of equal-mass halos every halo sees the same
        neighbourhood density, so all delta values must be identical (up to
        floating-point noise).  We verify this by checking that the range
        (max - min) of delta across all halos is negligible relative to the
        absolute value of any individual delta.

        We place 5×5×5 = 125 halos on a regular grid with spacing 30 h^-1 Mpc
        near the box centre and use a search radius of 40 h^-1 Mpc (slightly
        larger than the grid spacing) so every interior halo sees the same
        number of neighbours.  Only interior halos are tested (the 3×3×3 = 27
        interior nodes) to avoid boundary effects.
        """
        spacing = 30.0
        n_side = 5
        # Place grid starting at 350: 350, 380, 410, 440, 470 (all interior)
        coords = np.array([350.0 + i * spacing for i in range(n_side)])
        gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
        pos = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        n = len(pos)  # 125

        df = pd.DataFrame(
            {
                "x": pos[:, 0],
                "y": pos[:, 1],
                "z": pos[:, 2],
                "vx": np.zeros(n),
                "vy": np.zeros(n),
                "vz": np.zeros(n),
                "mvir": np.ones(n) * 1e12,
            }
        )

        # Use search radius = 40 h^-1 Mpc (> spacing=30, so each halo sees
        # its face-adjacent neighbours; interior halos all have the same count).
        radius = 40.0
        tree = cKDTree(df[["x", "y", "z"]].values, boxsize=BOX)
        calc = OverdensityCalculator(
            radius=radius, box_size=BOX, tree=tree, mass_column="mvir"
        )
        out = calc.compute(df.copy())
        colname = f"delta_{int(radius)}"

        # Select interior nodes only (indices where grid indices are 1..3 in each
        # dimension, i.e. the inner 3×3×3 = 27 halos).
        idx_matrix = np.arange(n).reshape(n_side, n_side, n_side)
        interior_flat = idx_matrix[1:4, 1:4, 1:4].ravel()
        delta_interior = out[colname].iloc[interior_flat].values

        delta_range = delta_interior.max() - delta_interior.min()
        delta_ref = abs(delta_interior.mean())
        # All interior halos see the same neighbourhood => delta is constant.
        # Allow a relative range of 1 % (floating-point arithmetic only).
        assert delta_range <= 0.01 * max(delta_ref, 1.0), (
            f"Interior delta values not constant: range={delta_range:.4f}, "
            f"ref={delta_ref:.4f}"
        )

    def test_dense_clump_has_higher_delta_than_isolated_halo(self):
        """
        A halo surrounded by many neighbours must have a higher delta than an
        isolated halo with no neighbours within the search radius.
        """
        # Build a base catalog of well-separated halos (grid 200 Mpc apart)
        # so that none of the base halos are within RADIUS of each other.
        base_positions = np.array([
            [200.0, 200.0, 200.0],
            [400.0, 400.0, 400.0],
            [600.0, 600.0, 600.0],
            [800.0, 800.0, 800.0],
        ])
        mass = 1e12

        # Add a tight clump of halos around a central point.
        clump_center = np.array([500.0, 200.0, 200.0])
        offsets = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        clump_positions = clump_center + offsets

        # Isolated halo far from everything.
        isolated_position = np.array([[700.0, 200.0, 200.0]])

        all_positions = np.vstack([base_positions, clump_positions, isolated_position])
        n = len(all_positions)
        df = pd.DataFrame(
            {
                "x": all_positions[:, 0],
                "y": all_positions[:, 1],
                "z": all_positions[:, 2],
                "vx": np.zeros(n),
                "vy": np.zeros(n),
                "vz": np.zeros(n),
                "mvir": np.full(n, mass),
            }
        )

        out = self._run_overdensity(df)
        colname = f"delta_{int(self.RADIUS)}"

        clump_center_idx = len(base_positions)  # first clump halo
        isolated_idx = len(all_positions) - 1

        delta_dense = out[colname].iloc[clump_center_idx]
        delta_isolated = out[colname].iloc[isolated_idx]

        assert delta_dense > delta_isolated, (
            f"Expected delta_dense ({delta_dense:.3f}) > delta_isolated "
            f"({delta_isolated:.3f})"
        )


# ===========================================================================
# Test 3: uniform mask — requires cf4_df with 'distance' column
# ===========================================================================

class TestUniformMask:
    """
    MaskMaker.make_uniform_mask:
      - returned count matches n_cf4_within_radius
      - all selected halos are within the requested radius (PBC-aware)
      - len(full catalog) >= len(uniform mask)
    """

    def _pbc_distance(self, pos, center, box_size):
        """Vectorised PBC distance from each row of pos to center."""
        delta = pos - center
        delta -= box_size * np.round(delta / box_size)
        return np.linalg.norm(delta, axis=1)

    def test_uniform_mask_size_and_containment(self):
        from src.physics.masks import MaskMaker

        rng = np.random.default_rng(77)
        center = np.array([500.0, 500.0, 500.0])
        survey_radius = 80.0

        # Build a synthetic halo catalog.
        n_halos = 300
        pos = rng.uniform(200.0, 800.0, size=(n_halos, 3))
        df_halos = pd.DataFrame(
            {
                "x": pos[:, 0],
                "y": pos[:, 1],
                "z": pos[:, 2],
                "vx": np.zeros(n_halos),
                "vy": np.zeros(n_halos),
                "vz": np.zeros(n_halos),
                "mvir": np.ones(n_halos) * 1e12,
            }
        )

        tree = cKDTree(df_halos[["x", "y", "z"]].values, boxsize=BOX)

        # Build a synthetic CF4-like catalog.
        # 'distance' column represents distance from the observer (the center).
        n_cf4 = 60
        cf4_pos = rng.uniform(-survey_radius, survey_radius, size=(n_cf4, 3))
        cf4_distances = np.linalg.norm(cf4_pos, axis=1)
        cf4_df = pd.DataFrame(
            {
                "id": np.arange(n_cf4),
                "x": cf4_pos[:, 0],
                "y": cf4_pos[:, 1],
                "z": cf4_pos[:, 2],
                "distance": cf4_distances,
            }
        )

        mask_maker = MaskMaker(
            halos_df=df_halos,
            tree=tree,
            box_size=BOX,
            cf4_df=cf4_df,
        )

        selected = mask_maker.make_uniform_mask(
            position=center,
            radius=survey_radius,
        )

        # Expected count: number of CF4 entries with distance <= survey_radius.
        n_expected = int(np.sum(cf4_df["distance"] <= survey_radius))

        # (a) size matches
        assert len(selected) == n_expected, (
            f"Expected {n_expected} halos, got {len(selected)}"
        )

        # (b) all selected halos are within survey_radius of center (PBC-aware).
        if len(selected) > 0:
            sel_pos = selected[["x", "y", "z"]].values
            dists = self._pbc_distance(sel_pos, center, BOX)
            assert np.all(dists <= survey_radius + 1e-9), (
                f"Some selected halos lie outside the survey radius. "
                f"Max dist = {dists.max():.3f}, radius = {survey_radius}"
            )

        # (c) catalog is at least as large as the mask.
        assert len(df_halos) >= len(selected), (
            f"Mask ({len(selected)}) larger than catalog ({len(df_halos)})"
        )


# ===========================================================================
# Test 4: netCDF round-trip
# ===========================================================================

class TestNetCDFRoundTrip:
    """
    Build a BulkFlowDataset, add a BulkFlowResult, write to tmp_path, reopen,
    and verify dims, variables, and attributes survive unchanged.
    """

    def _make_result(self, origin_id=0, mask="uniform", method="chi2"):
        radii = np.array([20.0, 40.0, 60.0, 80.0])
        n = len(radii)
        rng = np.random.default_rng(99)
        u = rng.normal(0, 100, size=(3, n))
        U_tot = np.linalg.norm(u, axis=0)
        return BulkFlowResult(
            origin_id=origin_id,
            origin=Vector3D(500.0, 500.0, 500.0),
            mask=mask,
            method=method,
            overdensity=0.1,
            local_bulkflow=150.0,
            mvir=1e13,
            near_virgo=False,
            radii=radii,
            u_x=u[0],
            u_y=u[1],
            u_z=u[2],
            U_tot=U_tot,
            U_deb=np.maximum(U_tot - 10.0, 0.0),
            sigma_U=rng.uniform(5, 20, size=n),
            n_used=np.array([10.0, 25.0, 45.0, 70.0]),
        )

    def test_roundtrip_dims_vars_attrs(self, tmp_path):
        bfd = BulkFlowDataset()
        result = self._make_result(origin_id=0, mask="uniform", method="chi2")
        bfd.add(result)
        bfd.set_attrs({"selection_variable": "overdensity", "run_label": "test_run"})

        nc_path = tmp_path / "roundtrip.nc"
        bfd.write(nc_path)

        # Reopen via the class API.
        bfd2 = BulkFlowDataset.open(nc_path)
        ds = bfd2.dataset

        # (a) correct dimensions
        assert set(ds.dims) == {"origin", "radius", "mask", "method"}, (
            f"Unexpected dims: {set(ds.dims)}"
        )

        # (b) required variables present
        required_vars = {"u_x", "u_y", "u_z", "U_tot", "U_deb", "sigma_U", "n_used"}
        assert required_vars.issubset(set(ds.data_vars)), (
            f"Missing variables: {required_vars - set(ds.data_vars)}"
        )

        # (c) dimension coordinates
        assert "radius" in ds.coords
        assert "mask" in ds.coords
        assert "method" in ds.coords
        assert "origin" in ds.coords

        # (d) attributes survive
        assert ds.attrs.get("selection_variable") == "overdensity"
        assert ds.attrs.get("run_label") == "test_run"

        # (e) axis values match what we wrote
        np.testing.assert_array_equal(ds.coords["radius"].values, result.radii)
        assert list(ds.coords["mask"].values) == ["uniform"]
        assert list(ds.coords["method"].values) == ["chi2"]

        # (f) data values survive the round-trip (NaN-safe).
        # Use dict-based sel() to avoid xarray misinterpreting "method" as a
        # pandas fill-method keyword.
        u_x_rt = ds["u_x"].sel({"mask": "uniform", "method": "chi2"}).values.ravel()
        np.testing.assert_allclose(u_x_rt, result.u_x, rtol=1e-6)

        U_tot_rt = ds["U_tot"].sel({"mask": "uniform", "method": "chi2"}).values.ravel()
        np.testing.assert_allclose(U_tot_rt, result.U_tot, rtol=1e-6)

    def test_roundtrip_multiple_origins_and_masks(self, tmp_path):
        """Two origins × two masks × one method — verify shape and coords."""
        bfd = BulkFlowDataset()
        radii = np.array([20.0, 40.0, 60.0])
        n_r = len(radii)
        dummy_u = np.zeros(n_r)

        for oid, mask in [(0, "uniform"), (0, "cf4"), (1, "uniform"), (1, "cf4")]:
            bfd.add(BulkFlowResult(
                origin_id=oid,
                origin=Vector3D(float(oid * 100 + 400), 500.0, 500.0),
                mask=mask,
                method="chi2",
                overdensity=float(oid) * 0.1,
                local_bulkflow=100.0,
                mvir=1e13,
                near_virgo=False,
                radii=radii,
                u_x=dummy_u, u_y=dummy_u, u_z=dummy_u,
                U_tot=dummy_u, U_deb=dummy_u,
                sigma_U=dummy_u, n_used=np.array([5.0, 10.0, 15.0]),
            ))

        bfd.set_attrs({"selection_variable": "overdensity"})
        nc_path = tmp_path / "multi.nc"
        bfd.write(nc_path)

        ds = BulkFlowDataset.open(nc_path).dataset
        assert ds.sizes["origin"] == 2
        assert ds.sizes["radius"] == 3
        assert ds.sizes["mask"] == 2
        assert ds.sizes["method"] == 1
