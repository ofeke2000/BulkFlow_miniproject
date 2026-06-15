from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class BulkFlowConfig:
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
