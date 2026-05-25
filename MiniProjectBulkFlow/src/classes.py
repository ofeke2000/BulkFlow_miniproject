from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class ConfigObject:
    """Base helper for typed config objects."""

    def __getitem__(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        raise KeyError(name)

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Vector3D:
    x: float
    y: float
    z: float

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, index: int) -> float:
        return (self.x, self.y, self.z)[index]

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray((self.x, self.y, self.z), dtype=dtype)

    def to_array(self) -> np.ndarray:
        return np.array((self.x, self.y, self.z), dtype=float)

    def norm(self) -> float:
        return float(np.linalg.norm(self.to_array()))

    def __add__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vector3D) -> Vector3D:
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __neg__(self) -> Vector3D:
        return Vector3D(-self.x, -self.y, -self.z)

    @classmethod
    def from_sequence(cls, values) -> Vector3D:
        values = tuple(float(v) for v in values)
        if len(values) != 3:
            raise ValueError("Vector3D requires three components")
        return cls(*values)

    def periodic_delta(self, other: Vector3D, box_size: float) -> Vector3D:
        delta = np.array(self.to_array()) - np.array(other.to_array())
        delta -= box_size * np.round(delta / box_size)
        return Vector3D(*delta)

    def periodic_distance_to(self, other: Vector3D, box_size: float) -> float:
        return float(np.linalg.norm(self.periodic_delta(other, box_size).to_array()))


@dataclass
class MDPL2Config(ConfigObject):
    HubbleParameter: float = 0.6777
    box_size: float = 1000.0


@dataclass
class PathsConfig(ConfigObject):
    rockstar_catalog: str = ""
    cf4_catalog: str = ""
    output_folder: str = ""
    output_file: str = ""


@dataclass
class BulkFlowConfig(ConfigObject):
    masks: str = "full"
    max_radius: float = 250.0
    min_radius: float = 50.0
    radii_step: float = 5.0
    error_fraction: float = 0.2
    sigma_star: float = 250.0
    sigma_min: float = 50.0
    calculation_method: str = "chi2"
    cf4_match_radius: float = 5.0
    cf4_match_max_doublings: int = 5
    uniform_radius: Optional[float] = None

    def __post_init__(self):
        if self.uniform_radius is None:
            object.__setattr__(self, "uniform_radius", self.max_radius)
        if self.min_radius < 0 or self.max_radius <= self.min_radius:
            raise ValueError("BulkFlowConfig: min_radius must be positive and less than max_radius")
        if self.radii_step <= 0:
            raise ValueError("BulkFlowConfig: radii_step must be positive")

    @property
    def radii(self) -> np.ndarray:
        return np.arange(self.min_radius, self.max_radius + self.radii_step, self.radii_step)


@dataclass
class OriginConfig(ConfigObject):
    local_overdensity_radius: float = 3.0
    local_overdensity_upper_cut: float = 0.5
    local_overdensity_lower_cut: float = 0.4
    apply_overdensity_cut: bool = False
    apply_mass_selection: bool = False
    selection_mass_min: Optional[float] = None
    selection_mass_max: Optional[float] = None
    local_bulkflow_radius: float = 3.0
    local_bulkflow_upper_cut: float = 1500.0
    local_bulkflow_lower_cut: float = 1400.0
    use_virgo_criteria: bool = True
    number_of_origins: int = 1000
    mass_cut: float = 1e14
    mass_cut_bool: bool = False
    select_lowest_delta: bool = False
    select_random: bool = True


@dataclass
class SimulationSliceHeatmapConfig(ConfigObject):
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
class VisualizationConfig(ConfigObject):
    projection_plane: str = "xy"
    simulation_slice_heatmap: SimulationSliceHeatmapConfig = field(default_factory=SimulationSliceHeatmapConfig)


@dataclass
class PostprocessingConfig(ConfigObject):
    enabled: bool = False
    build_comparison_plots: bool = False


@dataclass
class AppConfig(ConfigObject):
    mask_type: str = "full"
    MDPL2: MDPL2Config = field(default_factory=MDPL2Config)
    paths: PathsConfig = field(default_factory=PathsConfig)
    bulkflow: BulkFlowConfig = field(default_factory=BulkFlowConfig)
    origin_configs: OriginConfig = field(default_factory=OriginConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "AppConfig":
        visualization = config.get("visualization", {})
        return cls(
            mask_type=config.get("mask_type", "full"),
            MDPL2=MDPL2Config(**config.get("MDPL2", {})),
            paths=PathsConfig(**config.get("paths", {})),
            bulkflow=BulkFlowConfig(**config.get("bulkflow", {})),
            origin_configs=OriginConfig(**config.get("origin_configs", {})),
            visualization=VisualizationConfig(
                projection_plane=visualization.get("projection_plane", "xy"),
                simulation_slice_heatmap=SimulationSliceHeatmapConfig(**visualization.get("simulation_slice_heatmap", {})),
            ),
            postprocessing=PostprocessingConfig(**config.get("postprocessing", {})),
        )


@dataclass
class BulkFlowResult:
    origin_id: int
    mask: str
    origin: Vector3D
    radii: np.ndarray
    u_x: np.ndarray
    u_y: np.ndarray
    u_z: np.ndarray
    U: np.ndarray

    def __post_init__(self):
        self.radii = np.asarray(self.radii, dtype=float)
        self.u_x = np.asarray(self.u_x, dtype=float)
        self.u_y = np.asarray(self.u_y, dtype=float)
        self.u_z = np.asarray(self.u_z, dtype=float)
        self.U = np.asarray(self.U, dtype=float)

        if self.origin is None:
            raise ValueError("BulkFlowResult requires a valid origin")
        if not self.radii.shape == self.u_x.shape == self.u_y.shape == self.u_z.shape == self.U.shape:
            raise ValueError("BulkFlowResult arrays must all have the same shape")
