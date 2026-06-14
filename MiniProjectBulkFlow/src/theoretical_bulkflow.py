import numpy as np
from scipy.integrate import quad
from colossus.cosmology import cosmology

from .config.cosmology_config import CosmologyConfig
from .config.theory_config import TheoryConfig

WINDOW_EPS = 1e-12


def theoretical_bulkflow_colossus(
    radii: np.ndarray,
    cosmology_cfg: CosmologyConfig = None,
    theory_cfg: TheoryConfig = None,
) -> np.ndarray:
    """
    Compute the theoretical RMS bulk flow using Colossus (ΛCDM).

    Parameters
    ----------
    radii : array-like
        Radii in h⁻¹ Mpc.
    cosmology_cfg : CosmologyConfig, optional
        Simulation cosmology parameters. Defaults to MDPL2 values.
    theory_cfg : TheoryConfig, optional
        Integration settings. Defaults to standard values.

    Returns
    -------
    sigma_v : np.ndarray
        RMS bulk flow [km/s] at each radius.
    """
    if cosmology_cfg is None:
        cosmology_cfg = CosmologyConfig()
    if theory_cfg is None:
        theory_cfg = TheoryConfig()

    cosmology.addCosmology('mySimCosmo', cosmology_cfg.to_colossus_dict())
    cosmology.setCosmology('mySimCosmo')
    cosmo = cosmology.getCurrent()

    f = cosmo.Om(theory_cfg.z) ** cosmology_cfg.growth_index
    H0 = cosmo.H0  # km/s/Mpc

    def W(k, R):
        x = k * R
        return 3.0 * (np.sin(x) - x * np.cos(x)) / (x**3 + WINDOW_EPS)

    def integrand(k, R):
        pk = cosmo.matterPowerSpectrum(k, theory_cfg.z)
        return pk * W(k, R)**2

    sigma_v = []

    for R in radii:
        integral, _ = quad(
            integrand,
            theory_cfg.k_min,
            theory_cfg.k_max,
            args=(R,),
            limit=theory_cfg.k_limit
        )

        sigma2_v = (H0**2 * f**2 / (2.0 * np.pi**2)) * integral
        sigma_v.append(np.sqrt(sigma2_v))

    return np.array(sigma_v)
