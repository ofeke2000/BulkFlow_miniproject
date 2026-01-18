import yaml
import numpy as np
import pandas as pd
import logging
import time
import os
from pathlib import Path

from scipy.spatial import cKDTree

# --- Local modules ---
from src.data_loader import load_rockstar_catalog, load_cf4_catalogue
from src.overdensity import compute_overdensity
from src.masks import make_cf4_mask, make_uniform_mask
from src.bulkflow import calculate_bulk_flow_series, calculate_local_bulkflow
from src.specific_utils import append_bulkflow_results
from src.visualize import plot_bulkflow_from_hdf5, plot_histogram, plot_simulation_slice_heatmap
from src.near_virgo import near_virgo

# ------------------------------------------------------
# Setup logging
# ------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

# ------------------------------------------------------
# Load YAML configuration
# ------------------------------------------------------
def load_config(path: str = "config.yaml") -> dict:
    logging.info(f"Loading config from {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ------------------------------------------------------
# Main workflow
# ------------------------------------------------------
def main():

    # ----------------------------------------------------------
    # TIMING DICT
    # ----------------------------------------------------------
    timings = {}

    t0_total = time.time()

    # ===========================================
    # 1. Load configuration
    # ===========================================
    t0 = time.time()
    cfg = load_config("config.yaml")

    rockstar_path = cfg["paths"]["rockstar_catalog"]
    cf4_path = cfg["paths"]["cf4_catalog"]
    output_folder = cfg["paths"]["output_folder"]
    output_file = cfg["paths"]["output_file"]

    box_size = cfg["MDPL2"]["box_size"]
    Hubble_Parameter = cfg["MDPL2"]["HubbleParameter"]
    radius_overdensity = int(cfg["origin_configs"]["local_overdensity_radius"])
    overdensity_upper_cut = float(cfg["origin_configs"]["local_overdensity_upper_cut"])
    overdensity_lower_cut = float(cfg["origin_configs"]["local_overdensity_lower_cut"])
    radius_bulkflow = int(cfg["origin_configs"]["local_bulkflow_radius"])
    bulkflow_upper_cut = float(cfg["origin_configs"]["local_bulkflow_upper_cut"])
    bulkflow_lower_cut = float(cfg["origin_configs"]["local_bulkflow_lower_cut"])
    use_virgo_criteria = cfg["origin_configs"]["use_virgo_criteria"]
    mass_cut = float(cfg["origin_configs"]["mass_cut"])
    mass_cut_bool = cfg["origin_configs"]["mass_cut_bool"]
    n_origins = cfg["origin_configs"]["number_of_origins"]
    lowest_delta = cfg["origin_configs"]["select_lowest_delta"]
    select_random = cfg["origin_configs"]["select_random"]

    # bulkflow config
    r_min = int(cfg["bulkflow"]["min_radius"])
    r_max = int(cfg["bulkflow"]["max_radius"])
    r_jump = int(cfg["bulkflow"]["radii_step"])
    error_frac = cfg["bulkflow"]["error_fraction"]
    sigma_star = cfg["bulkflow"]["sigma_star"]
    sigma_min = cfg["bulkflow"]["sigma_min"]

    logging.info("Configuration loaded successfully.")

    # ===========================================
    # 2. Load catalogs
    # ===========================================
    t0 = time.time()
    logging.info("Loading Rockstar catalog...")
    halos_df = load_rockstar_catalog(rockstar_path)

    # Fit Halos into box [0, box_size)
    halos_df[['x','y','z']] %= box_size
    logging.info("Rockstar catalog loaded and prepared.")

    logging.info("Loading CF4 catalog...")
    # cf4_df = load_cf4_catalogue(cf4_path, h=Hubble_Parameter)

    # -------------------------------------------
    # Filter CF4 galaxies inside r_max
    # -------------------------------------------
    # n_before = len(cf4_df)

    # cf4_df = cf4_df[cf4_df["distance"] <= r_max].copy()

    # n_after = len(cf4_df)

    # logging.info(
    #     f"Filtered CF4 catalogue to r <= {r_max:.1f} : "
    #     f"{n_before} → {n_after} galaxies"
    # )

    
    # -------------------------------------------
    # Mass cut
    # -------------------------------------------

    if mass_cut_bool:

        mass_order = np.log10(mass_cut)

        logging.info(f"Applying mass cut: mvir >= {mass_order:.2e}")

        n_before = len(halos_df)

        halos_df = halos_df[halos_df["mvir"] >= mass_cut].copy()

        n_after = len(halos_df)

        logging.info(
            f"Mass cut applied: {n_before} → {n_after} halos "
        )

        plot_simulation_slice_heatmap(
            df=halos_df,
            slice_axis="z",
            slice_min = 400.0,
            slice_max = 500.0,
            proj_axes = ("x", "y"),
            gridsize = 500,
            cmap = "magma",
            output_folder = output_folder,
            output_file = f"simulation_slice_heatmap_m={mass_order}.png",
            dpi= 300,
        )

    # ===========================================
    # 3. Build cKDTree
    # ===========================================
    t0 = time.time()
    logging.info("Building cKDTree...")
    tree = cKDTree(halos_df[["x", "y", "z"]].values, boxsize=box_size)
    logging.info("cKDTree built successfully.")
    timings["load_and_prepare_data"] = time.time() - t0

    # ===========================================
    # 4. Environmental tests at origin positions
    # ===========================================

    delta_column = f"delta_{int(radius_overdensity)}"
    virgo_column = "near_virgo"
    bulkflow_column = f"bulkflow_{int(radius_bulkflow)}"

    t0 = time.time()
    logging.info("Checking environmental test columns...")

    # --------------------------------------------------
    # 4.0 Check which tests already exist
    # --------------------------------------------------
    missing_tests = []

    if delta_column not in halos_df.columns:
        missing_tests.append("overdensity")

    if virgo_column not in halos_df.columns:
        missing_tests.append("virgo")

    if bulkflow_column not in halos_df.columns:
        missing_tests.append("bulkflow")

    if not missing_tests:
        logging.info(
            "All environmental test columns already exist "
            "— skipping section 4 entirely."
        )
    else:
        logging.info(
            f"Missing tests detected: {', '.join(missing_tests)}"
        )

        # --------------------------------------------------
        # 4.1 Local overdensity
        # --------------------------------------------------
        if "overdensity" in missing_tests:
            logging.info(f"Computing overdensity ({delta_column})...")

            halos_df = compute_overdensity(
                df=halos_df,
                radius=radius_overdensity,
                tree=tree,
                box_size=box_size,
                mass_column="mvir"
            )

            logging.info(
                f"Overdensity computed "
                f"(Δt = {time.time() - t0:.2f} s)"
            )
        else:
            logging.info(
                f"Column '{delta_column}' exists — skipping overdensity."
            )

        # --------------------------------------------------
        # 4.2 Near-Virgo test
        # --------------------------------------------------
        if "virgo" in missing_tests:
            logging.info("Running Virgo environment test...")

            halos_df = near_virgo(
                df=halos_df,
                box_size=box_size,
                mass_threshold=1e14,  # h^-1 Msun
                r_min=7.0,  # h^-1 Mpc, 10 in Mpc
                r_max=14.0  # h^-1 Mpc, 20 in Mpc
            )

            logging.info("Virgo test completed.")
        else:
            logging.info(
                f"Column '{virgo_column}' exists — skipping Virgo test."
            )

        # --------------------------------------------------
        # 4.3 Local bulk flow test
        # --------------------------------------------------
        if "bulkflow" in missing_tests:
            logging.info("Computing local bulk flow environment...")

            halos_df = calculate_local_bulkflow(
                df=halos_df,
                tree=tree,
                radius=radius_bulkflow,
                velocity_columns=("vx", "vy", "vz"),
            )

            logging.info("Local bulk flow test completed.")
        else:
            logging.info(
                f"Column '{bulkflow_column}' exists — skipping bulk flow test."
            )

        # --------------------------------------------------
        # 4.4 Save once if anything was computed
        # --------------------------------------------------
        halos_df.to_csv(rockstar_path, index=False)

        logging.info(
            f"Environmental tests updated and saved "
            f"(total Δt = {time.time() - t0:.2f} s)"
        )
        

    # ===========================================
    # 5. Select Virgo-like, quiet overdensity, proper bulk flow
    # ===========================================
    t0 = time.time()
    logging.info("Selecting Earth-like local environments...")

    # --- absolute overdensity (for ranking only) ---
    delta_abs_col = f"delta_abs_{int(radius_overdensity)}"
    halos_df[delta_abs_col] = halos_df[delta_column].abs()

    # --- physical filters only ---
    mask = (
        # (halos_df["mvir"] >= mass_cut) &
        (halos_df[delta_column].between(overdensity_lower_cut, overdensity_upper_cut)) #&
        # (halos_df[bulkflow_column].between(bulkflow_lower_cut, bulkflow_upper_cut)) &
        # (halos_df[virgo_column] > 0)
    )

    candidates = halos_df.loc[mask]
    candidates_df = halos_df.loc[candidates.index].copy()

    plot_histogram(
        data_df=candidates_df, 
        output_folder=output_folder, 
        output_file=f"overdensity_histogram_[{overdensity_lower_cut},{overdensity_upper_cut}]_for_candidates.png", 
        key=delta_column,
        origin=(0,0,0),
        bins=20,
        log_axis="none"
        )

    logging.info(f"Candidates after cuts: {len(candidates)}")

    if lowest_delta:
        # --- rank by |delta| and select ---
        selected_points = (
            candidates
            .sort_values(delta_abs_col)
            .head(n_origins)
        )
    elif select_random:
        selected_points = candidates.sample(
            n=min(n_origins, len(candidates)),
            replace=False
        )


    selected_df = halos_df.loc[selected_points.index].copy()
    logging.info(f"Selected halo sample size: {len(selected_df)}")


    plot_histogram(
        data_df=selected_df, 
        output_folder=output_folder, 
        output_file=f"overdensity_histogram_[{overdensity_lower_cut},{overdensity_upper_cut}]_for_selected_points.png", 
        key=delta_column,
        origin=(0,0,0),
        bins=20,
        log_axis= "none"
        )

    logging.info(
        f"Selected {len(selected_points)} origin points "
        f"(min |δ| = {selected_points[delta_abs_col].min():.3e})."
    )

    logging.info(f"Selection completed in {time.time() - t0:.2f} s.")


    # ===========================================
    # 6. Loop over selected points
    # ===========================================
    per_origin_times = []
    t0 = time.time()
    i=0

    for _, row in selected_points.iterrows():

        origin = (row["x"], row["y"], row["z"])
        origin_id = int(row["rockstarid"])

        i += 1

        if 100*(i/n_origins) % 5 == 0 and i > 1:
            logging.info(f"Processing origin ID {origin_id} at {origin}")
            avg_time = np.mean(per_origin_times)
            eta = avg_time * (n_origins - i)
            logging.info(
                f"=== Processing origin {i}/{n_origins} "
                f"({100*i/n_origins:.1f}%), ETA ~ {eta/60:.1f} min ==="
            )

        # ---------------------------------------
        # 6.1 Make masks
        # ---------------------------------------
        # cf4_mask_df = make_cf4_mask(
        #     position=np.array(origin),
        #     halos_df=halos_df,
        #     cf4_df=cf4_df,
        #     tree=tree,
        #     box_size=box_size,
        #     radius=5.0,
        #     max_doublings=5
        # )

        # uniform_mask_df = make_uniform_mask(
        #     position=np.array(origin),
        #     radius=r_max,
        #     df_halos=halos_df,
        #     CF4_catalogue=cf4_df,
        #     tree=tree
        # )

        idx_full = tree.query_ball_point(
            np.array(origin),
            r=r_max
        )
        full_mask_df = halos_df.iloc[idx_full].copy()

        t_origin = time.time()
        # logging.info(f" Masks created. CF4 mask size: {len(cf4_mask_df)}, Uniform mask size: {len(uniform_mask_df)}")

        # ---------------------------------------
        # 6.2 Compute bulk flow for each mask
        # ---------------------------------------
        # bf_cf4 = calculate_bulk_flow_series(
        #     halos_df=cf4_mask_df,
        #     origin=origin,
        #     r_max=r_max,
        #     r_min=r_min,
        #     r_jumps=r_jump,
        #     error_frac=error_frac,
        #     sigma_star=sigma_star,
        #     sigma_min=sigma_min
        # )

        # bf_uniform = calculate_bulk_flow_series(
        #     halos_df=uniform_mask_df,
        #     origin=origin,
        #     r_max=r_max,
        #     r_min=r_min,
        #     r_jumps=r_jump,
        #     error_frac=error_frac,
        #     sigma_star=sigma_star,
        #     sigma_min=sigma_min
        # )

        bf_full = calculate_bulk_flow_series(
            halos_df=full_mask_df,
            origin=origin,
            r_max=r_max,
            r_min=r_min,
            r_jumps=r_jump,
            error_frac=error_frac,
            sigma_star=sigma_star,
            sigma_min=sigma_min
        )


        # logging.info(f" Bulk flows computed. CF4 bulk flow size: {len(bf_cf4)}, Uniform bulk flow size: {len(bf_uniform)}")
        # logging.info(f" Bulk flow is {bf_cf4['U_total'].iloc[-1]}")

        # ---------------------------------------
        # 6.3 Save results
        # ---------------------------------------
        # append_bulkflow_results(
        #     bf_cf4,
        #     origin_id=origin_id,
        #     mask_name="cf4",
        #     filename=output_file
        # )

        # append_bulkflow_results(
        #     bf_uniform,
        #     origin_id=origin_id,
        #     mask_name="uniform",
        #     filename=output_file
        # )

        append_bulkflow_results(
            bf_full,
            origin_id=origin_id,
            mask_name="full",
            filename=output_file
        )

        per_origin_times.append(time.time() - t_origin)

    # ===========================================
    # 7. Visualize results
    # ===========================================

    plot_bulkflow_from_hdf5(
        hdf_file=output_file,
        output_folder=output_folder,
        key="bulkflow",
        output_file=f"bulkflow_vs_radius_overdensity_[{overdensity_lower_cut},{overdensity_upper_cut}].png",
        plot_theory=True,
        use_mean_amplitude=True,
        plot_variance_band=True,
        show_markers=False,
        plot_all_curves=False
    )

    timings["process_all_origins"] = time.time() - t0
    timings["mean_origin_time"] = np.mean(per_origin_times)
    timings["min_origin_time"] = np.min(per_origin_times)
    timings["max_origin_time"] = np.max(per_origin_times)

    # ==========================================================
    # END — FINAL TIMING SUMMARY
    # ==========================================================
    timings["total_runtime"] = time.time() - t0_total

    logging.info("\n" + "=" * 60)
    logging.info("TIMING SUMMARY")
    logging.info("=" * 60)

    for key, value in timings.items():
        logging.info(f"{key:25s} : {value:8.3f} sec")

    logging.info("=" * 60)
    logging.info(f"TOTAL RUNTIME : {timings['total_runtime']:.3f} sec")
    logging.info("=" * 60)

    logging.info("All origins processed successfully!")
    logging.info(f"Results saved to {output_file}")


# ------------------------------------------------------
# Entry point
# ------------------------------------------------------
if __name__ == "__main__":
    main()
