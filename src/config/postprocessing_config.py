from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PostprocessingConfig:
    enabled: bool = False
    build_comparison_plots: bool = False
    band_min: int = 0
    band_max: int = 1600
    band_step: int = 100
    var_min: int = 0
    var_max: int = 1500
    var_step: int = 100
    error_alpha: float = 0.25
    selection_variable: str = "overdensity"
