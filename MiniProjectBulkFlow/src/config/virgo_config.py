from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VirgoTestConfig:
    """Parameters for the Virgo-like environment proximity test."""
    mass_threshold: float = 1e14
    r_min: float = 7.0
    r_max: float = 14.0
