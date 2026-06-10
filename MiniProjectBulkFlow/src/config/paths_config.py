from __future__ import annotations

from dataclasses import dataclass
import os

# Default input/output paths used when the user does not override them.
DEFAULT_ROCKSTAR_CATALOG = "/home/ofeke2000/BulkFlow_miniproject/BulkFlow_miniproject/data/mdpl2_rockstar_125_pid-1_mvir12.csv"
DEFAULT_CF4_CATALOG = "/home/ofeke2000/BulkFlow_miniproject/BulkFlow_miniproject/data/CF4_Groups.csv"
DEFAULT_CF4_VELOCITIES_CATALOG = "/home/ofeke2000/BulkFlow_miniproject/BulkFlow_miniproject/data/CF4_Groups_Velocities.csv"
DEFAULT_OUTPUT_FOLDER = "/home/ofeke2000/BulkFlow_miniproject/BulkFlow_miniproject/output/"

ROCKSTAR_COLUMNS = [
    "rockstarid",
    "mvir",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
]


@dataclass
class PathsConfig:
    rockstar_catalog: str = DEFAULT_ROCKSTAR_CATALOG
    cf4_catalog: str = DEFAULT_CF4_CATALOG
    output_folder: str = DEFAULT_OUTPUT_FOLDER
    output_file: str = ""

    def ensure_output_folder(self) -> None:
        os.makedirs(self.output_folder, exist_ok=True)
