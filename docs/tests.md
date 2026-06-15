# Tests to validate the bulk-flow code

Ordered from "cheap, catches real bugs" to "validates the science."
Categories 1–2 run in milliseconds on a dozen hand-placed halos (no real data needed).

## Run status (last run: 2026-06-15)

| Category | Implemented? | Where | Result |
|---|---|---|---|
| 1. Physics primitives | ✅ yes | `tests/test_specific_utils.py` | **8/8 PASSED** |
| 2. ML/chi2 estimator + chi2-vs-mean agreement | ✅ yes | `tests/test_bulkflow.py` | **6/6 PASSED** |
| 3. Pipeline invariants | ✅ yes | `tests/test_pipeline_invariants.py` | **11/11 PASSED** |
| 4. Statistical validation vs ΛCDM | ❌ not implemented | — | not run (needs real data) |

**Total implemented: 25/25 passed.**

### ~~⚠️ Bug found while writing category 3~~ ✅ Fixed

`xarray`'s `.sel()` reserves the keyword `method` for its fill/interpolation mode, so
selecting along the dataset's dimension *named* `method` by keyword raises
`ValueError: Invalid fill method ... Got chi2`. All 6 affected call sites have been changed
to the dict form `ds.sel({"mask": mask, "method": method})`:
`src/data/bulkflow_dataset.py:304,349`, `scripts/postprocessing.py:58`,
`src/viz/visualize.py:261,284,321`.

**Total implemented: 14/14 passed.** Run them with:
```bash
source venv/bin/activate && pip install -r requirements-dev.txt
pytest
```

> **Reading this file:** lines under "What each test asserts (expected)" describe the
> *expected* behaviour the assertions encode. The **Result** column above is what was
> *actually observed* when the tests ran. Categories without code are predictions only —
> nothing has been measured for them yet.

---

## 1. Physics primitives — ✅ IMPLEMENTED & PASSED

Foundation. If these are wrong, everything downstream is silently wrong.
File: `tests/test_specific_utils.py`

**What each test asserts (expected):**

**Periodic distance** (`src/physics/specific_utils.py::periodic_distance`)
- Points at `x=1` and `x=999` in a 1000 box → distance `2`, not `998`. (Hard Rule #1.)
- Symmetric: `d(p1,p2) == d(p2,p1)`.
- Point to itself → `0`.
- Cross-check against brute-force min over the 27 image shifts on random points.

**Radial velocity** (`src/physics/specific_utils.py::radial_velocity_and_error_pbc`)
- `r_hat` vectors are unit length for all halos.
- Purely radial outward motion → `v_rad == +|v|`; tangential → `v_rad ≈ 0`; infalling → negative.
- Halo across the periodic boundary gets the minimum-image `r_hat`
  (origin x=1, halo x=999 → `r_hat_x = -1`, distance 2).
- `radius_from_origin` matches `periodic_distance` for the same points.

## 2. The ML / chi2 estimator — ✅ IMPLEMENTED & PASSED

`src/physics/bulkflow.py::bulk_flow_chi2_cumulative` is the scientific core.
File: `tests/test_bulkflow.py`

**What each test asserts (expected):**
- **Recovery:** every halo gets identical true velocity `U_true` (so `v_rad = r_hat·U_true`)
  with `sigma_star=0` → estimator returns `u ≈ U_true` to solver precision.
- **Rotational covariance:** rotate positions+velocities by `R` → `u` rotates by `R`,
  `U_total` unchanged.
- **Debias floor:** collinear halos make `A` singular → `U_debiased` is NaN or ≥0,
  never negative (guards bulkflow.py:126).
- **Cumulative monotonicity:** `n_used` non-decreasing, equals count of halos with `r ≤ R`.
- **Isotropy → small flow:** zero-mean isotropic velocities → `U_total` ≪ per-halo
  velocity scale, and `U_debiased ≤ U_total` (debiasing never inflates).

## Comparison pre-flight (chi2 vs mean) — ✅ IMPLEMENTED & PASSED

`test_chi2_and_mean_agree_in_noise_free_limit` in `tests/test_bulkflow.py`.

**Expected:** with a constant velocity field and `sigma_star=0`, both estimators return the
*same* `u`, and both equal `U_true`. **Result: PASSED** — so any real chi2-vs-mean difference
on actual data is physics/selection, not a harness bug.

Reminders for the real comparison (still apply):
- `mean` averages true 3D `⟨vx,vy,vz⟩` (the "truth"); `chi2` reconstructs from radial
  velocities only (what a survey measures).
- `mean` returns NaN for `sigma_U`/`U_debiased`. Compare `chi2.U_debiased` (bias removed)
  against `mean.U_total`, **not** `chi2.U_total`.
- Feed both identical `halos_df` / `origin` / `r_list`; switch only `calculation_method`.
- Expected real behavior: `chi2` approaches `mean` where sampling is dense (large radius,
  `full` mask), diverges where sparse/anisotropic (small radius, `cf4` mask) — that
  divergence is the selection-effect signal.

## 3. Pipeline invariants — ⏳ NOT YET IMPLEMENTED (predictions only)

Cheap regression guards. To run these, they must first be written as pytest tests.
These need only synthetic data + temp files (no real catalogs), **except** the CF4-mask
size check, which needs the CF4 catalogue file.

**Predicted behaviour (not yet measured):**
- **Cache idempotency (Hard Rule #2):** run overdensity / near_virgo / local_bulkflow twice;
  second run skips and produces identical columns.
- **Overdensity sanity:** uniform random box → `delta` averages ≈ 0; halo in a clump → `delta > 0`.
- **Mask size relations:** `len(full) ≥ len(uniform)`; `uniform` size matches its CF4 calibration
  count. *(CF4 part needs the CF4 catalogue file.)*
- **netCDF round-trip:** write a `BulkFlowDataset` to a temp file, reopen, assert dims
  `origin × radius × mask × method` and all variables/attrs (incl. `git_commit`,
  `selection_variable`) survive unchanged.

## 4. Statistical validation against ΛCDM — ❌ NOT IMPLEMENTED (needs real data)

Slower; run as a script/notebook, not in CI. **Requires the real Rockstar catalog + Colossus
and many random observers** — this is the long-running end-to-end check, not a quick unit test.

**Predicted behaviour (not yet measured):**
- Bulk flow vs radius from many random observers matches the Colossus prediction
  (`src/physics/theoretical_bulkflow.py`) within expected scatter.
- `U_debiased` tracks theory better than `U_total` — debiasing reduces the offset.
