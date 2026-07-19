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

    # CF4's own distance-ladder Hubble constant (Tully et al. 2023), km/s/Mpc
    # — distinct from the MDPL2 H0 above; used only by the CF4-facing
    # velocity analyses (velocity_comparison.py, velocity_distance_mock.py,
    # velocity_redshift_binning.py), not by the simulation pipeline.
    H0_CF4: float = 74.6

    @property
    def hubble_velocity_per_hinv_mpc(self) -> float:
        """Hubble velocity in km/s per h⁻¹ Mpc: H0/h = 100 km/s per h⁻¹ Mpc.

        Colossus returns P(k) in (h⁻¹ Mpc)³ and k in h/Mpc, so the velocity
        prefactor in the bulk-flow integral must use H0/h rather than H0.
        Since h ≡ H0/100 by definition, H0/h = 100 exactly.
        """
        # h ≡ H0 / 100 is the *definition* of the reduced Hubble parameter;
        # the literal 100 here is that definition, not a tunable constant.
        h = self.H0 / 100.0
        return self.H0 / h

    @property
    def cf4_hubble_velocity_per_hinv_mpc(self) -> float:
        """CF4's calibrated Hubble velocity in km/s per h⁻¹ Mpc.

        The CF4-facing velocity analyses express CF4 distances in h⁻¹ Mpc
        (the catalogue's D is loaded as D·h, using the MDPL2 h), so
        recovering the Hubble velocity H0_CF4·D from a radius already in
        h⁻¹ Mpc requires H0_CF4/h, not H0_CF4 alone. Unlike
        ``hubble_velocity_per_hinv_mpc`` (= H0/h = 100 exactly, H0 taken over
        its *own* h), this property mixes CF4's H0_CF4 with the MDPL2 h, so
        it is **not** 100 — it evaluates to H0_CF4/h ≈ 74.6/0.6777 ≈ 110.08
        km/s per h⁻¹ Mpc.
        """
        h = self.H0 / 100.0
        return self.H0_CF4 / h

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
