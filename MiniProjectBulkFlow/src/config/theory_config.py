from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TheoryConfig:
    """Integration settings for the theoretical bulk flow calculation (Colossus)."""
    z: float = 0.0
    k_min: float = 1e-4
    k_max: float = 10.0
    k_limit: int = 200000
