import numpy as np
from scipy.integrate import quad
from colossus.cosmology import cosmology


def theoretical_bulkflow_colossus(
    radii: np.ndarray,
    z: float = 0.0,
    k_min: float = 1e-4,
    k_max: float = 10.0,
    k_limit: int = 200000
) -> np.ndarray:
    """
    Compute the theoretical RMS bulk flow using Colossus (ΛCDM).

    Parameters
    ----------
    radii : array-like
        Radii in h⁻¹ Mpc.
    cosmo_params : dict
        Colossus cosmology parameters.
    z : float
        Redshift (default: 0).
    k_min, k_max : float
        Integration limits in h/Mpc.
    k_limit : int
        Max subintervals for quad integration.

    Returns
    -------
    sigma_v : np.ndarray
        RMS bulk flow [km/s] at each radius.
    """

    # Define your simulation's cosmology parameters
    my_cosmo_params = {
        'flat': True,
        'H0': 67.77,
        'Om0': 0.307115,
        'Ode0': 0.692885,
        'Ob0': 0.048206,
        'sigma8': 0.8228,
        'ns': 0.96
    }

    # Register the custom cosmology in Colossus
    cosmology.addCosmology('mySimCosmo', my_cosmo_params)
    cosmology.setCosmology('mySimCosmo')
    cosmo = cosmology.getCurrent()

    # Growth rate f ≈ Ω_m(z)^0.55
    f = cosmo.Om(z)**0.55
    H0 = cosmo.H0  # km/s/Mpc

    # --------------------------------------------------
    # Top-hat window function
    # --------------------------------------------------
    def W(k, R):
        x = k * R
        return 3.0 * (np.sin(x) - x * np.cos(x)) / (x**3 + 1e-12)

    # --------------------------------------------------
    # Integrand
    # --------------------------------------------------
    def integrand(k, R):
        pk = cosmo.matterPowerSpectrum(k, z)
        return pk * W(k, R)**2

    # --------------------------------------------------
    # Compute σ_v(R)
    # --------------------------------------------------
    sigma_v = []

    for R in radii:
        integral, _ = quad(
            integrand,
            k_min,
            k_max,
            args=(R,),
            limit=k_limit
        )

        sigma2_v = (H0**2 * f**2 / (2.0 * np.pi**2)) * integral
        sigma_v.append(np.sqrt(sigma2_v))  # RMS velocity

    return np.array(sigma_v)
