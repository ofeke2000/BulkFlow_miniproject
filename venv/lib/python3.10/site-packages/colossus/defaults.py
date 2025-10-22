###################################################################################################
#
# defaults.py 	        (c) Benedikt Diemer
#     				    	diemer@umd.edu
#
###################################################################################################

"""
Global defaults used across Colossus.
"""

###################################################################################################
# COSMOLOGY
###################################################################################################

COSMOLOGY_TCMB0 = 2.7255
"""The default CMB temperature in Kelvin."""

COSMOLOGY_NEFF = 3.046
"""The default number of effective neutrino species."""

###################################################################################################
# DEFAULT MODELS (HALO PROFILE, BIAS, CONCENTRATION, SPLASHBACK)
###################################################################################################

POWER_SPECTRUM_MODEL = 'eisenstein98'
"""The default power spectrum model."""

HALO_BIAS_MODEL = 'tinker10'
"""The default halo bias model."""

HALO_MASS_FUNCTION_MODEL = 'watson13'
"""The default halo mass function model."""

HALO_CONCENTRATION_MODEL = 'diemer19'
"""The default concentration model."""

HALO_CONCENTRATION_STATISTIC = 'median'
"""The default statistic used (``mean`` or ``median``). This only applies to those models that 
distinguish between mean and median statistics."""

HALO_MASS_CONVERSION_PROFILE = 'nfw'
"""The default profile used for mass conversions. Whenever spherical overdensity mass definitions
are converted into one another, we have to assume a form of the density profile. The simplicity
of the :doc:`halo_profile_nfw` makes this computation efficient."""

HALO_SPLASHBACK_MODEL = 'diemer20'
"""The default model for the splashback radius."""

HALO_SPLASHBACK_STATISTIC = 'median'
"""The default statistic used (``mean`` or ``median``). This only applies to splashback models 
that distinguish between mean and median statistics."""

###################################################################################################
# ARGUMENT LISTS
###################################################################################################

PS_ARGS = {'model': POWER_SPECTRUM_MODEL, 'path': None}
"""The default arguments to be passed to the power spectrum function. This argument list cannot be
empty by default because some functions store data tables depending on the underlying power 
spectrum."""

SIGMA_ARGS = {}
"""The default arguments to be passed to the variance function."""

DELTAC_ARGS = {}
"""The default arguments to be passed to the collapse overdensity function."""

###################################################################################################
# HALO PROFILE (BASE CLASS)
###################################################################################################

HALO_PROFILE_ENCLOSED_MASS_ACCURACY = 1E-6
"""Integration accuracy for enclosed mass."""

HALO_PROFILE_SURFACE_DENSITY_ACCURACY = 1E-4
"""Integration accuracy for surface density."""

HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTERPOLATE = 1E8
"""Radius to which the surface density is integrated when interpolating the density."""

HALO_PROFILE_SURFACE_DENSITY_MAX_R_INTEGRATE = 1E20
"""Radius to which the surface density is integrated when evaluating the density exactly."""

HALO_PROFILE_DELTA_SIGMA_MIN_R_INTERPOLATE = 1E-6
"""Radius from which the surface density is averaged to compute DeltaSigma."""

###################################################################################################
# HALO PROFILE (SPECIFIC INNER PROFILES)
###################################################################################################

HALO_PROFILE_SELECTED_BY = 'M'
"""The constructor of the :doc:`halo_profile_dk14` sets the profile parameters to either 
predict the mean profile of halos selected by their mass (``M``) or mass accretion rate and 
mass (``Gamma``)."""

HALO_PROFILE_ACC_RADIUS = 1E-4
"""The accuracy achieved when looking for a spherical overdensity radius numerically."""

HALO_PROFILE_ACC_WARN = 0.01
"""If the desired halo mass cannot be matched with a DK14 profile to better than this accuracy,
a warning is displayed."""

HALO_PROFILE_ACC_ERR = 0.05
"""If the desired halo mass cannot be matched with a DK14 profile to better than this accuracy,
an exception is raised."""

HALO_PROFILE_DK14_PL_NORM = 1.0
"""The default normalization of the power-law outer profile for the DK14 profile."""

HALO_PROFILE_DK14_PL_SLOPE = 1.5
"""The default slope of the power-law outer profile for the DK14 profile."""

HALO_PROFILE_D22_ETA = 0.1
"""The default nuissance parameter for the Diemer 22 Model B profile."""

###################################################################################################
# HALO PROFILE (SPECIFIC OUTER PROFILE TERMS)
###################################################################################################

HALO_PROFILE_OUTER_PL_MAXRHO = 1000.0
"""The default maximum density the power-law outer profile term can contribute to the total 
density, in units of the mean matter density. If this number is set too high, the power-law profile 
can lead to a spurious density contribution at very small radii, if it is set too high the 
power-law term will not contribute at all."""

HALO_PROFILE_OUTER_D22_DELTA_MAX = 100.0
"""The default asymptotic overdensity of the infalling term at the halo center in the formulation
of Diemer 2023."""

HALO_PROFILE_OUTER_D22_ZETA = 0.5
"""The default sharpness of the transition of the infalling profile in the formulation of 
Diemer 2023."""

###################################################################################################
# MCMC
###################################################################################################

MCMC_N_WALKERS = 100
"""The number of chains (called walkers) run in parallel."""

MCMC_INITIAL_STEP = 0.1
"""A guess at the initial step taken by the walkers."""

MCMC_CONVERGENCE_STEP = 100
"""Test the convergence of the MCMC chains every n steps."""

MCMC_CONVERGED_GR = 0.01
"""Take the chains to have converged when the Gelman-Rubin statistic is smaller than this number
in all parameters."""

MCMC_OUTPUT_EVERY_N = 100
"""Output the current state of the chain every n steps."""
