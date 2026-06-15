from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class OriginConfig:
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
