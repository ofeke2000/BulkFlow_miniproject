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


@dataclass
class VisualizationConfig:
    """Fixed visualization parameters - not user-editable."""
    projection_plane: str = "xy"
    simulation_slice_heatmap: SimulationSliceHeatmapConfig = field(default_factory=SimulationSliceHeatmapConfig)
