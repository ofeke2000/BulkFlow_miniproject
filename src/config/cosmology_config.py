from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CosmologyConfig:
    """Fixed MDPL2 simulation cosmology — not user-editable."""
    H0: float = 67.77
    Om0: float = 0.307115
    Ode0: float = 0.692885
    Ob0: float = 0.048206
    sigma8: float = 0.8228
    ns: float = 0.96
    flat: bool = True
    growth_index: float = 0.55

    @property
    def bulk_flow_amplitude_factor(self) -> float:
        """Conversion from RMS velocity σ_v to mean bulk-flow amplitude ⟨|U|⟩ = factor * σ_v."""
        return np.sqrt(8.0 / (3.0 * np.pi))

    def to_colossus_dict(self) -> dict:
        return {
            'flat': self.flat,
            'H0': self.H0,
            'Om0': self.Om0,
            'Ode0': self.Ode0,
            'Ob0': self.Ob0,
            'sigma8': self.sigma8,
            'ns': self.ns,
        }
