from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from .bulkflow_config import BulkFlowConfig
from .cosmology_config import CosmologyConfig
from .mdpl2_config import MDPL2Config
from .origin_config import OriginConfig
from .paths_config import (
    DEFAULT_CF4_CATALOG,
    DEFAULT_CF4_VELOCITIES_CATALOG,
    DEFAULT_OUTPUT_FOLDER,
    DEFAULT_ROCKSTAR_CATALOG,
    PathsConfig,
)
from .postprocessing_config import PostprocessingConfig
from .theory_config import TheoryConfig
from .virgo_config import VirgoTestConfig
from .visualization_config import SimulationSliceHeatmapConfig, VisualizationConfig


@dataclass
class AppConfig:
    MDPL2: MDPL2Config = MDPL2Config()
    cosmology: CosmologyConfig = CosmologyConfig()
    theory: TheoryConfig = TheoryConfig()
    virgo_test: VirgoTestConfig = VirgoTestConfig()
    paths: PathsConfig = PathsConfig()
    bulkflow: BulkFlowConfig = BulkFlowConfig()
    origin_configs: OriginConfig = OriginConfig()
    visualization: VisualizationConfig = VisualizationConfig()
    postprocessing: PostprocessingConfig = PostprocessingConfig()

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "AppConfig":
        paths_config = config.get("paths", {}) or {}

        cf4_path = paths_config.get("cf4_catalog", DEFAULT_CF4_CATALOG)
        cf4_type = config.get("cf4_catalog_type", "cf4")
        if cf4_type == "cf4_velocities" and "cf4_catalog" not in paths_config:
            cf4_path = DEFAULT_CF4_VELOCITIES_CATALOG

        output_folder = paths_config.get("output_folder", DEFAULT_OUTPUT_FOLDER)
        folder_type = config.get("output_folder_type", "default")
        if folder_type == "temp":
            temp_name = config.get("temp_folder_name", "temp_bulkflow")
            output_folder = os.path.join(DEFAULT_OUTPUT_FOLDER, temp_name)
            output_folder = os.path.join(output_folder, "")
            os.makedirs(output_folder, exist_ok=True)

        output_file_name = config.get("output_file_name", "bulkflow_results.h5")
        output_file = os.path.join(output_folder, output_file_name)
        if paths_config.get("output_file"):
            output_file = paths_config["output_file"]

        paths = PathsConfig(
            rockstar_catalog=paths_config.get("rockstar_catalog", DEFAULT_ROCKSTAR_CATALOG),
            cf4_catalog=cf4_path,
            output_folder=output_folder,
            output_file=output_file,
        )

        visualization = config.get("visualization", {})
        return cls(
            MDPL2=MDPL2Config(**config.get("MDPL2", {})),
            cosmology=CosmologyConfig(),
            theory=TheoryConfig(**config.get("theory", {})),
            virgo_test=VirgoTestConfig(**config.get("virgo_test", {})),
            paths=paths,
            bulkflow=BulkFlowConfig(**config.get("bulkflow", {})),
            origin_configs=OriginConfig(**config.get("origin_configs", {})),
            visualization=VisualizationConfig(
                projection_plane=visualization.get("projection_plane", "xy"),
                simulation_slice_heatmap=SimulationSliceHeatmapConfig(**visualization.get("simulation_slice_heatmap", {})),
            ),
            postprocessing=PostprocessingConfig(**config.get("postprocessing", {})),
        )
