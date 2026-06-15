from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimulationSliceHeatmapConfig:
    """Fixed visualization parameters - not user-editable."""
    enabled: bool = False
    slice_axis: str = "z"
    slice_min: float = 400.0
    slice_max: float = 500.0
    proj_axes: tuple[str, str] = ("x", "y")
    gridsize: int = 500
    cmap: str = "magma"
    output_file: str = "simulation_slice_heatmap.png"
    dpi: int = 300
    heatmap_figsize: tuple[int, int] = (8, 8)


@dataclass
class PlotStyleConfig:
    """Default styling constants for all plots."""
    dpi_normal: int = 150
    curve_alpha: float = 0.25
    grid_alpha: float = 0.7
    curve_linewidth: float = 1.0
    theory_linewidth: float = 2.0
    csv_theory_linewidth: float = 2.5
    bulkflow_figsize: tuple[int, int] = (8, 5)
    histogram_figsize: tuple[int, int] = (10, 6)
    histogram_bins: int = 50
    mass_histogram_bins: int = 20


@dataclass
class VisualizationConfig:
    """Fixed visualization parameters - not user-editable."""
    projection_plane: str = "xy"
    simulation_slice_heatmap: SimulationSliceHeatmapConfig = field(default_factory=SimulationSliceHeatmapConfig)
    style: PlotStyleConfig = field(default_factory=PlotStyleConfig)
    plot_histogram_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
