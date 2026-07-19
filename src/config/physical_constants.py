from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhysicalConstants:
    """Fixed physical constants — not user-editable, not part of config.yaml.

    Distinct from ``CosmologyConfig`` (which holds the MDPL2 simulation's
    cosmological parameters): this class holds universal physical constants
    that are the same regardless of cosmology or catalogue.
    """
    SPEED_OF_LIGHT_KM_S: float = 299792.458
