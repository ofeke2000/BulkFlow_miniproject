"""
visualize.py
------------
Visualization utilities for bulk flow analysis.

Bulk-flow plotting is consolidated in BulkFlowPlotter (class-based, netCDF-backed).
FacetSet encapsulates the constant-vs-varying routing logic.
Histogram and simulation-slice helpers remain as module-level functions.
"""

from __future__ import annotations

import os
import re
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from ..io.data_loader import load_cf4_catalogue, load_rockstar_catalog
from ..physics.specific_utils import add_periodic_distance
from ..physics.theoretical_bulkflow import theoretical_bulkflow_colossus
from ..config.cosmology_config import CosmologyConfig
from ..config.theory_config import TheoryConfig
from ..config.mdpl2_config import MDPL2Config
from ..config.visualization_config import (
    BulkFlowPlotConfig,
    PlotStyleConfig,
    SimulationSliceHeatmapConfig,
    VisualizationConfig,
)


# ===========================================================================
# FacetSet — constant-vs-varying routing helper
# ===========================================================================

class FacetSet:
    """
    Analyses the facet dict of every curve in a figure and classifies each
    facet as CONSTANT (identical across all curves) or VARYING.

    Routing rules
    -------------
    CONSTANT facets              → output filename + corner text box.
    VARYING categorical facets   → legend labels (mask / method / estimator).
    VARYING ``sel_band`` facet   → colorbar; excluded from legend.

    ``sel_band_mid`` is an internal numeric helper for colormap normalisation;
    it is excluded from filename and text box output.
    """

    _SANITIZE: re.Pattern = re.compile(r"[^\w\-]")
    _DISPLAY_EXCLUDE: frozenset[str] = frozenset({"sel_band_mid"})
    _CATEGORICAL: frozenset[str] = frozenset({"mask", "method", "estimator"})

    def __init__(self, curve_facets: list[dict]) -> None:
        if not curve_facets:
            raise ValueError("FacetSet requires at least one curve.")
        all_keys: set[str] = set().union(*curve_facets)
        self._constant: dict[str, object] = {}
        self._varying: set[str] = set()
        for key in all_keys:
            values = {f.get(key) for f in curve_facets}
            if len(values) == 1:
                self._constant[key] = next(iter(values))
            else:
                self._varying.add(key)

    @property
    def constant_facets(self) -> dict:
        return dict(self._constant)

    @property
    def varying_facets(self) -> set[str]:
        return set(self._varying)

    def legend_label(self, curve_facet: dict) -> str:
        """Legend label built from VARYING categorical facets only."""
        varying_cat = self._varying & self._CATEGORICAL
        parts = [f"{k}={curve_facet.get(k, '?')}" for k in sorted(varying_cat)]
        return ", ".join(parts)

    def filename_suffix(self) -> str:
        """Sanitised constant-facet key-value pairs for embedding in a filename."""
        parts = []
        for key, val in sorted(self._constant.items()):
            if key in self._DISPLAY_EXCLUDE:
                continue
            safe = self._SANITIZE.sub("_", str(val))
            parts.append(f"{key}-{safe}")
        return "__".join(parts)

    def textbox_text(self) -> str:
        """Multi-line summary of all constant facets (for the corner annotation)."""
        return "\n".join(
            f"{k}: {v}"
            for k, v in sorted(self._constant.items())
            if k not in self._DISPLAY_EXCLUDE
        )


# ===========================================================================
# BulkFlowPlotter
# ===========================================================================

class BulkFlowPlotter:
    """
    Facet-aware, class-based bulk-flow plotter that reads directly from a
    BulkFlowDataset netCDF file.

    Constant facets (same value across every plotted curve) appear in the
    output filename and a corner annotation box so the saved PNG is
    self-describing.  Varying categorical facets (mask / method / estimator)
    appear in the legend.  When ``band_bins`` is supplied, origins are grouped
    by the dataset's ``selection_variable`` and the band coordinate is rendered
    as a colorbar instead of a legend entry.
    """

    # Layout/style constants — not in config.yaml (non-user-facing defaults)
    _TEXTBOX_X: float = 0.02
    _TEXTBOX_Y: float = 0.98
    _TEXTBOX_FONTSIZE: int = 8
    _TEXTBOX_ALPHA: float = 0.8
    _COLORBAR_PAD: float = 0.02

    def __init__(
        self,
        nc_file: str,
        output_folder: str,
        output_file: str | None = None,
        methods: list[str] | None = None,
        band_bins: np.ndarray | None = None,
        plot_cfg: BulkFlowPlotConfig | None = None,
        cosmology_cfg: CosmologyConfig | None = None,
        theory_cfg: TheoryConfig | None = None,
        style: PlotStyleConfig | None = None,
    ) -> None:
        self._nc_file = nc_file
        self._output_folder = output_folder
        self._output_file = output_file
        self._methods = methods
        self._band_bins = band_bins
        cfg = plot_cfg or BulkFlowPlotConfig()
        self._plot_theory = cfg.plot_theory
        self._use_mean_amplitude = cfg.use_mean_amplitude
        self._plot_variance_band = cfg.plot_variance_band
        self._plot_all_curves = cfg.plot_all_curves
        self._plot_debiased = cfg.plot_debiased
        self._show_markers = cfg.show_markers
        self._append_facets = cfg.append_facets_to_filename
        self._cosmology_cfg = cosmology_cfg or CosmologyConfig()
        self._theory_cfg = theory_cfg or TheoryConfig()
        self._style = style or PlotStyleConfig()
        self._variance_alpha = (
            cfg.variance_alpha if cfg.variance_alpha is not None else self._style.curve_alpha
        )

    def plot(self) -> str:
        """Build and save the bulk-flow plot. Returns the saved file path."""
        from ..data.bulkflow_dataset import BulkFlowDataset

        if not os.path.exists(self._nc_file):
            logging.error(f"netCDF file not found: {self._nc_file}")
            return ""

        bfd = BulkFlowDataset.open(self._nc_file)
        ds = bfd.dataset

        radii = ds.coords["radius"].values
        masks = list(ds.coords["mask"].values)
        all_methods = list(ds.coords["method"].values)
        methods = [m for m in (self._methods or all_methods) if m in all_methods]
        sel_var = ds.attrs.get("selection_variable")

        global_facets = self._extract_global_facets(ds)
        curve_specs = self._build_curve_specs(ds, radii, masks, methods, sel_var, global_facets)
        facet_set = FacetSet([cs["facets"] for cs in curve_specs])

        fig, ax = plt.subplots(figsize=self._style.bulkflow_figsize)

        colorbar_sm = self._draw_curves(ax, curve_specs, facet_set)

        if colorbar_sm is not None:
            fig.colorbar(
                colorbar_sm,
                ax=ax,
                pad=self._COLORBAR_PAD,
                label=sel_var or "selection variable",
            )

        if self._plot_theory:
            self._draw_theory(ax, radii)

        if any(k in facet_set.varying_facets for k in FacetSet._CATEGORICAL):
            ax.legend()

        self._draw_textbox(ax, facet_set)

        ax.set_xlabel(r"Radius [$h^{-1}$ Mpc]")
        ax.set_ylabel(r"$|U|$ [km/s]")
        ax.set_title("Bulk Flow vs Radius")
        ax.grid(True)
        fig.tight_layout()

        out_file = self._resolve_filename(facet_set)
        os.makedirs(self._output_folder, exist_ok=True)
        out_path = os.path.join(self._output_folder, out_file)
        fig.savefig(out_path, dpi=self._style.dpi_normal)
        plt.close(fig)

        logging.info(f"Saved bulk-flow plot: {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Dataset-level metadata extraction
    # ------------------------------------------------------------------

    def _extract_global_facets(self, ds) -> dict:
        """
        Extract dataset-wide constants from attrs and dims to embed in every
        curve's facet dict, so they always appear in the textbox.
        """
        facets: dict = {}
        sel_var = ds.attrs.get("selection_variable")
        if sel_var:
            facets["selection_variable"] = sel_var
        sigma_star = ds.attrs.get("sigma_star")
        if sigma_star is not None:
            facets["sigma_star"] = sigma_star
        n_origins = ds.attrs.get("number_of_origins", ds.dims.get("origin"))
        if n_origins is not None:
            facets["N"] = n_origins
        return facets

    # ------------------------------------------------------------------
    # Curve-spec builders
    # ------------------------------------------------------------------

    def _build_curve_specs(
        self,
        ds,
        radii: np.ndarray,
        masks: list[str],
        methods: list[str],
        sel_var: str | None,
        global_facets: dict,
    ) -> list[dict]:
        if self._band_bins is not None and sel_var is not None:
            return self._build_banded_specs(ds, radii, masks, methods, sel_var, global_facets)
        if self._plot_all_curves:
            return self._build_individual_specs(ds, radii, masks, methods, global_facets)
        return self._build_mean_specs(ds, radii, masks, methods, global_facets)

    def _build_mean_specs(self, ds, radii, masks, methods, global_facets) -> list[dict]:
        specs: list[dict] = []
        for method in methods:
            for mask in masks:
                sub = ds.sel({"mask": mask, "method": method})
                mean_tot = sub["U_tot"].mean("origin").values
                std_tot = sub["U_tot"].std("origin").values
                specs.append({
                    "data": (radii, mean_tot),
                    "std": std_tot,
                    "facets": {**global_facets, "mask": mask, "method": method, "estimator": "total"},
                    "is_individual": False,
                })
                deb_vals = sub["U_deb"].values
                if self._plot_debiased and np.any(np.isfinite(deb_vals)):
                    specs.append({
                        "data": (radii, sub["U_deb"].mean("origin").values),
                        "std": None,
                        "facets": {**global_facets, "mask": mask, "method": method, "estimator": "debiased"},
                        "is_individual": False,
                    })
        return specs

    def _build_individual_specs(self, ds, radii, masks, methods, global_facets) -> list[dict]:
        specs: list[dict] = []
        for method in methods:
            for mask in masks:
                sub = ds.sel({"mask": mask, "method": method})
                for oid in ds.coords["origin"].values:
                    s = sub.sel(origin=oid)
                    specs.append({
                        "data": (radii, s["U_tot"].values),
                        "std": None,
                        "facets": {**global_facets, "mask": mask, "method": method, "estimator": "total"},
                        "is_individual": True,
                    })
                    deb = s["U_deb"].values
                    if self._plot_debiased and np.any(np.isfinite(deb)):
                        specs.append({
                            "data": (radii, deb),
                            "std": None,
                            "facets": {**global_facets, "mask": mask, "method": method, "estimator": "debiased"},
                            "is_individual": True,
                        })
        return specs

    def _build_banded_specs(
        self, ds, radii, masks, methods, sel_var, global_facets
    ) -> list[dict]:
        coord = ds[sel_var]
        bin_dim = f"{sel_var}_bins"
        mean_per_band = (
            ds["U_tot"]
            .groupby_bins(coord, bins=self._band_bins)
            .mean("origin")
        )
        specs: list[dict] = []
        for i_bin, interval in enumerate(mean_per_band.coords[bin_dim].values):
            low, high = interval.left, interval.right
            mid = 0.5 * (low + high)
            sub = mean_per_band.isel(**{bin_dim: i_bin})
            for method in methods:
                for mask in masks:
                    specs.append({
                        "data": (radii, sub.sel({"mask": mask, "method": method}).values),
                        "std": None,
                        "facets": {
                            **global_facets,
                            "mask": mask,
                            "method": method,
                            "estimator": "total",
                            "sel_band": f"{low:.0f}-{high:.0f}",
                            "sel_band_mid": mid,
                        },
                        "is_individual": False,
                    })
        return specs

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_curves(
        self, ax, curve_specs: list[dict], facet_set: FacetSet
    ) -> mpl.cm.ScalarMappable | None:
        """
        Draw all curves. Returns a ScalarMappable when the sel_band facet
        varies (caller attaches a colorbar), else None.
        """
        uses_colorbar = "sel_band" in facet_set.varying_facets

        if uses_colorbar:
            midpoints = [cs["facets"]["sel_band_mid"] for cs in curve_specs]
            vmin, vmax = min(midpoints), max(midpoints)
            if vmin < 0 < vmax:
                norm = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            else:
                norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.coolwarm
        else:
            prop_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            cat_keys: list[tuple] = list(dict.fromkeys(
                (cs["facets"].get("mask"), cs["facets"].get("method"))
                for cs in curve_specs
            ))
            cat_color: dict[tuple, str] = {
                k: prop_colors[i % len(prop_colors)]
                for i, k in enumerate(cat_keys)
            }

        labels_drawn: set[str] = set()
        marker = "o" if self._show_markers else None

        for cs in curve_specs:
            radii, values = cs["data"]
            is_debiased = cs["facets"].get("estimator") == "debiased"
            linestyle = "--" if is_debiased else "-"
            alpha = self._style.curve_alpha if cs.get("is_individual") else 1.0

            color = (
                cmap(norm(cs["facets"]["sel_band_mid"]))
                if uses_colorbar
                else cat_color[(cs["facets"].get("mask"), cs["facets"].get("method"))]
            )

            label_str = facet_set.legend_label(cs["facets"])
            label = label_str if (label_str and label_str not in labels_drawn) else None
            if label:
                labels_drawn.add(label_str)

            ax.plot(
                radii,
                values,
                color=color,
                linestyle=linestyle,
                linewidth=self._style.curve_linewidth,
                alpha=alpha,
                marker=marker,
                label=label,
            )

            if cs.get("std") is not None and self._plot_variance_band:
                ax.fill_between(
                    radii,
                    values - cs["std"],
                    values + cs["std"],
                    color=color,
                    alpha=self._variance_alpha,
                )

        if not uses_colorbar:
            return None

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        return sm

    def _draw_theory(self, ax, radii: np.ndarray) -> None:
        sigma_v = theoretical_bulkflow_colossus(
            radii=radii,
            cosmology_cfg=self._cosmology_cfg,
            theory_cfg=self._theory_cfg,
        )
        if self._use_mean_amplitude:
            U_theory = self._cosmology_cfg.bulk_flow_amplitude_factor * sigma_v
            label = r"$\Lambda$CDM $\langle |U| \rangle$"
        else:
            U_theory = sigma_v
            label = r"$\Lambda$CDM $\sigma_v$"
        ax.plot(
            radii,
            U_theory,
            "k--",
            linewidth=self._style.theory_linewidth,
            label=label,
        )

    def _draw_textbox(self, ax, facet_set: FacetSet) -> None:
        text = facet_set.textbox_text()
        if not text:
            return
        ax.text(
            self._TEXTBOX_X,
            self._TEXTBOX_Y,
            text,
            transform=ax.transAxes,
            fontsize=self._TEXTBOX_FONTSIZE,
            verticalalignment="top",
            bbox=dict(boxstyle="round", alpha=self._TEXTBOX_ALPHA, facecolor="white"),
        )

    def _resolve_filename(self, facet_set: FacetSet) -> str:
        """
        Resolve the output filename.

        With no explicit ``output_file`` an auto name is generated. When
        ``append_facets_to_filename`` is set, the constant-facet summary (the
        same key/value pairs shown in the corner textbox) is appended to the
        base name so runs with different constants never overwrite each other.
        """
        if self._output_file is None:
            return self._auto_filename(facet_set)
        if not self._append_facets:
            return self._output_file
        base, ext = os.path.splitext(self._output_file)
        ext = ext or ".png"
        suffix = facet_set.filename_suffix()
        return f"{base}__{suffix}{ext}" if suffix else f"{base}{ext}"

    def _auto_filename(self, facet_set: FacetSet) -> str:
        suffix = facet_set.filename_suffix()
        base = "bulkflow_vs_radius"
        return f"{base}__{suffix}.png" if suffix else f"{base}.png"


# ===========================================================================
# Histogram
# ===========================================================================

def plot_histogram(
    data_df: pd.DataFrame,
    output_folder: str = "plots",
    output_file: str = "cf4_histogram_lin.png",
    key: str = "distance",
    origin: tuple[float, float, float] | None = None,
    box_size: float | None = None,
    bins=None,
    log_axis: str | bool = False,
    style: PlotStyleConfig | None = None,
) -> None:

    if style is None:
        style = PlotStyleConfig()
    if origin is None:
        origin = VisualizationConfig().plot_histogram_origin
    if box_size is None:
        box_size = MDPL2Config().box_size
    if bins is None:
        bins = style.histogram_bins

    if key == "distance" and key not in data_df.columns:
        if all(col in data_df.columns for col in ["x", "y", "z"]):
            data_df = add_periodic_distance(
                df=data_df,
                origin=origin,
                box_size=box_size,
                distance_col="distance",
            )
        else:
            raise KeyError(
                "The dataframe is missing 'distance' and cannot find 'x, y, z' to calculate it."
            )

    plt.figure(figsize=style.histogram_figsize)
    plt.hist(data_df[key], bins=bins)

    if log_axis == "y":
        plt.yscale("log")
    elif log_axis == "x":
        plt.xscale("log")
    elif log_axis == "all":
        plt.yscale("log")
        plt.xscale("log")

    plt.xlabel(key.replace("_", " ").capitalize())
    plt.ylabel("Number of Objects")
    plt.title(output_file.replace("_", " ").replace(".png", ""))
    plt.grid(True, linestyle="--", alpha=style.grid_alpha)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_file)
    plt.savefig(output_path, dpi=style.dpi_normal)
    plt.close()


def plot_overlaid_histogram(
    data_df: pd.DataFrame,
    keys: list[str],
    output_folder: str = "plots",
    output_file: str = "overlaid_histogram.png",
    xlabel: str = "Value",
    title: str | None = None,
    bins=None,
    log_axis: str | bool = False,
    style: PlotStyleConfig | None = None,
) -> None:
    """Overlay several columns as step histograms on shared bins for comparison."""
    if style is None:
        style = PlotStyleConfig()
    if bins is None:
        bins = style.histogram_bins

    # Shared bin edges across all keys so the curves are directly comparable.
    finite_values = np.concatenate(
        [data_df[key].values[np.isfinite(data_df[key].values)] for key in keys]
    )
    bin_edges = np.histogram_bin_edges(finite_values, bins=bins)

    plt.figure(figsize=style.histogram_figsize)
    for key in keys:
        values = data_df[key].values
        plt.hist(
            values[np.isfinite(values)],
            bins=bin_edges,
            histtype="step",
            label=key,
        )

    if log_axis in ("y", "all"):
        plt.yscale("log")
    if log_axis in ("x", "all"):
        plt.xscale("log")

    plt.xlabel(xlabel)
    plt.ylabel("Number of Objects")
    plt.title(title or output_file.replace("_", " ").replace(".png", ""))
    plt.legend()
    plt.grid(True, linestyle="--", alpha=style.grid_alpha)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_file)
    plt.savefig(output_path, dpi=style.dpi_normal)
    plt.close()


# ===========================================================================
# Simulation slice heatmap
# ===========================================================================

def plot_simulation_slice_heatmap(
    df: pd.DataFrame,
    slice_axis: str = "z",
    slice_min: float = 400.0,
    slice_max: float = 500.0,
    proj_axes: tuple = ("x", "y"),
    gridsize: int = 500,
    cmap: str = "magma",
    output_folder: str = "heatmap_slices",
    output_file: str | None = None,
    dpi: int = 300,
    heatmap_cfg: SimulationSliceHeatmapConfig | None = None,
) -> None:
    """Plot a hexbin heatmap of a thin slice of the simulation box."""

    for col in (*proj_axes, slice_axis):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

    slice_df = df[
        (df[slice_axis] >= slice_min) & (df[slice_axis] < slice_max)
    ]
    n_halos = len(slice_df)

    if heatmap_cfg is None:
        heatmap_cfg = SimulationSliceHeatmapConfig()

    fig, ax = plt.subplots(figsize=heatmap_cfg.heatmap_figsize)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    hb = ax.hexbin(
        slice_df[proj_axes[0]],
        slice_df[proj_axes[1]],
        gridsize=gridsize,
        cmap=cmap,
    )

    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("Counts", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel(proj_axes[0], color="white")
    ax.set_ylabel(proj_axes[1], color="white")
    ax.set_title(
        f"{slice_axis} ∈ [{slice_min}, {slice_max}) | Halos: {n_halos}",
        color="white",
    )
    ax.tick_params(colors="white")

    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)

    if output_file is None:
        output_file = (
            f"heatmap_{proj_axes[0]}_{proj_axes[1]}_"
            f"{slice_axis}_{slice_min}_{slice_max}.png"
        )

    plt.savefig(os.path.join(output_folder, output_file), dpi=dpi)
    plt.close()
