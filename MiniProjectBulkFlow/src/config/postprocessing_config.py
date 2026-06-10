from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PostprocessingConfig:
    enabled: bool = False
    build_comparison_plots: bool = False
