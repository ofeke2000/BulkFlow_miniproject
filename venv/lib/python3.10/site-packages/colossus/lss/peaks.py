###################################################################################################
#
# peaks.py                  (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This module contains functions related to peaks in Gaussian random fields, namely the overdensity
for collapse, peak height and curvature, and the non-linear mass.

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

The peak height of a halo quantifies how big a fluctuation in the linear density field this halo
corresponds to. This quantity is computed as the ratio of the critical overdensity of collapse
(1.686 according to the top-hat spherical collapse model in an Einstein-de Sitter universe) to 
the variance of the linear density field on the scale of the halo, 
:math:`\\nu \\equiv \\delta_c / \\sigma(M)`. For example, halos with peak height one correspond to 
peaks that have just reached a variance equal to the collapse overdensity at a given 
redshift, and should thus be collapsing. Halos of smaller peak height have, on average, already 
collapsed in the past, and halos of higher peak height will, on average, collapse in the future.
The peak height of a halo is easy to evaluate in Colossus using the :func:`peakHeight` 
function::

	from colossus.cosmology import cosmology
	from colossus.lss import peaks
	
	cosmology.setCosmology('planck15')
	nu = peaks.peakHeight(M, z)

The inverse function :func:`massFromPeakHeight` converts peak height to mass. Internally, the 
scale on which the variance is computed as the lagrangian radius :func:`lagrangianR` of halos, and
the :func:`collapseOverdensity` function offers additional options such as corrections due to 
cosmology. Finally, the variance is computed using the 
:func:`~cosmology.cosmology.Cosmology.sigma` function which takes another set of parameters. All
those options can be passed to any function related to peak height. The 
:func:`nonLinearMass` is defined as the mass where peak height is unity at a given redshift, i.e., 
the mass of a halo that is typically collapsing at the current time. Finally, 
:func:`peakCurvature` is a higher-order property of peaks that describes their shape. Please see 
the :doc:`tutorials` for more extensive code examples.

---------------------------------------------------------------------------------------------------
Module contents
---------------------------------------------------------------------------------------------------

.. autosummary::
	lagrangianR
	lagrangianM
	collapseOverdensity
	peakHeight
	massFromPeakHeight
	nonLinearMass
	peakCurvature
	powerSpectrumSlope

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
import scipy.integrate
import scipy.special

from colossus import defaults
from colossus.utils import constants
from colossus.cosmology import cosmology

###################################################################################################

def lagrangianR(M):
	"""
	The lagrangian radius of a halo of mass M.

	Converts the mass of a halo (in :math:`M_{\\odot} / h`) to the radius of its 
	comoving Lagrangian volume (in comoving Mpc/h), that is the volume that would enclose the 
	halo's mass at the mean density of the universe at z = 0.

	Parameters
	----------
	M: array_like
		Halo mass in :math:`M_{\\odot} / h`; can be a number or a numpy array.

	Returns
	-------
	R: array_like
		The lagrangian radius in comoving Mpc/h; has the same dimensions as ``M``.

	See also
	--------
	lagrangianM: The lagrangian mass of a halo of radius R.
	"""
	
	cosmo = cosmology.getCurrent()
	R = (3.0 * M / 4.0 / np.pi / cosmo.rho_m(0.0) / 1E9)**(1.0 / 3.0)
	
	return R

###################################################################################################

def lagrangianM(R):
	"""
	The lagrangian mass of a halo of radius R.

	Converts the radius of a halo (in comoving Mpc/h) to the mass in its comoving Lagrangian 
	volume (in :math:`M_{\\odot} / h`), that is the volume that would enclose the halo's mass 
	at the mean density of the universe at z = 0.

	Parameters
	----------
	R: array_like
		Halo radius in comoving Mpc/h; can be a number or a numpy array.

	Returns
	-------
	M: array_like
		The lagrangian mass; has the same dimensions as ``R``.

	See also
	--------
	lagrangianR: The lagrangian radius of a halo of mass M.
	"""
	
	cosmo = cosmology.getCurrent()
	M = 4.0 / 3.0 * np.pi * R**3 * cosmo.rho_m(0.0) * 1E9
	
	return M

###################################################################################################

def collapseOverdensity(corrections = False, z = None):
	"""
	The linear overdensity threshold for halo collapse.
	
	The linear overdensity threshold for halo collapse according to the spherical top-hat collapse 
	model (`Gunn & Gott 1972 <http://adsabs.harvard.edu/abs/1972ApJ...176....1G>`__). In an EdS
	universe, this number is :math:`3/5 (3\\pi/2)^{2/3}=1.686`.
	
	This value is modified very slightly in a non-EdS universe (by less than 3% for any realistic
	cosmology). Such corrections are applied if desired, by default this function returns the 
	constant value (see, e.g., 
	`Mo, van den Bosch & White <http://adsabs.harvard.edu/abs/2010gfe..book.....M>`__ 
	for a derivation of the corrections). Note that correction formulae are implemented for flat
	cosmologies and cosmologies without dark energy, but not for the general case (both 
	curvature and dark energy). The correction is essentially identical in effect to the Equation 
	A6 of `Kitayama & Suto 1996 <https://ui.adsabs.harvard.edu/abs/1996ApJ...469..480K/abstract>`__.

	Parameters
	----------
	corrections: bool
		If True, corrections to the collapse overdensity are applied in a non-EdS cosmology. In
		this case, a redshift must be passed.
	z: float
		Redshift where the collapse density is evaluated. Only necessary if 
		``corrections == True``.
	
	Returns
	-------
	delta_c: float
		The threshold overdensity for collapse.
	"""
	
	delta_c = constants.DELTA_COLLAPSE
	
	if corrections:
		
		if z is None:
			raise Exception('If corrections == True, a redshift must be passed.')
		
		cosmo = cosmology.getCurrent()
		Om = cosmo.Om(z)
		if cosmo.flat:
			delta_c *= Om**0.0055
		elif cosmo.Ode0 == 0.0:
			delta_c *= Om**0.0185
	
	return delta_c

###################################################################################################

def peakHeight(M, z, ps_args = defaults.PS_ARGS, sigma_args = defaults.SIGMA_ARGS, deltac_args = {}):
	"""
	The peak height corresponding to a given a halo mass.
	
	Peak height is defined as :math:`\\nu \\equiv \\delta_c / \\sigma(M)`. This function takes 
	optional parameter lists for the 
	:func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum`, 
	:func:`~cosmology.cosmology.Cosmology.sigma`, and
	:func:`collapseOverdensity` functions. Please see the documentation of those funtions for
	details.
	
	Note that the peak height does not explicitly depend on the mass definition in which ``M``
	is given, but that it corresponds to that mass definition. For example, if M200m is given,
	that mass and the corresponding peak height will be larger than R500c for the same halo.
	
	Parameters
	----------
	M: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift.
	ps_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` 
		function.
	sigma_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.sigma` function.
	deltac_args: dict
		Arguments passed to the :func:`collapseOverdensity` function.
	
	Returns
	-------
	nu: array_like
		Peak height; has the same dimensions as ``M``.

	See also
	--------
	massFromPeakHeight: Halo mass from peak height.
	"""
	
	cosmo = cosmology.getCurrent()
	R = lagrangianR(M)
	sigma = cosmo.sigma(R, z, ps_args = ps_args, **sigma_args)
	nu = collapseOverdensity(z = z, **deltac_args) / sigma

	return nu

###################################################################################################

def massFromPeakHeight(nu, z, ps_args = defaults.PS_ARGS, sigma_args = defaults.SIGMA_ARGS, 
					deltac_args = defaults.DELTAC_ARGS):
	"""
	Halo mass from peak height.
	
	Peak height is defined as :math:`\\nu \\equiv \\delta_c / \\sigma(M)`. This function takes 
	optional parameter lists for the 
	:func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum`, 
	:func:`~cosmology.cosmology.Cosmology.sigma`, and
	:func:`collapseOverdensity` functions. Please see the documentation of those funtions for
	details.
	
	Parameters
	----------
	nu: array_like
		Peak height; can be a number or a numpy array.
	z: float
		Redshift.
	ps_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` 
		function.
	sigma_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.sigma` function.
	deltac_args: dict
		Arguments passed to the :func:`collapseOverdensity` function.

	Returns
	-------
	M: array_like
		Mass in :math:`M_{\\odot}/h`; has the same dimensions as ``nu``.

	See also
	--------
	peakHeight: The peak height corresponding to a given a halo mass.
	"""

	cosmo = cosmology.getCurrent()
	sigma = collapseOverdensity(z = z, **deltac_args) / nu
	R = cosmo.sigma(sigma, z, inverse = True, ps_args = ps_args, **sigma_args)
	M = lagrangianM(R)
	
	return M

###################################################################################################

def nonLinearMass(z, ps_args = defaults.PS_ARGS, sigma_args = defaults.SIGMA_ARGS, 
				deltac_args = defaults.DELTAC_ARGS):
	"""
	The non-linear mass.
	
	:math:`M^*` is the mass for which the variance is equal to the collapse threshold, i.e.
	:math:`\\sigma(M^*) = \\delta_c` and thus :math:`\\nu(M^*) = 1`.

	This function takes optional parameter lists for the 
	:func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum`, 
	:func:`~cosmology.cosmology.Cosmology.sigma`, and
	:func:`collapseOverdensity` functions. Please see the documentation of those funtions for
	details.
	
	Parameters
	----------
	z: float
		Redshift.
	ps_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` 
		function.
	sigma_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.sigma` function.
	deltac_args: dict
		Arguments passed to the :func:`collapseOverdensity` function.

	Returns
	-------
	Mstar: float
		The non-linear mass in :math:`M_{\\odot}/h`.

	See also
	--------
	peakHeight: The peak height corresponding to a given a halo mass.
	massFromPeakHeight: Halo mass from peak height.
	"""

	return massFromPeakHeight(1.0, z, 
							ps_args = ps_args, sigma_args = sigma_args, deltac_args = deltac_args)

###################################################################################################
# Peak curvature routines
###################################################################################################

# Get the mean peak curvature, <x>, at fixed nu from the integral of Bardeen et al. 1986 
# (BBKS). Note that this function is approximated very well by the _peakCurvatureApprox() 
# function below.

def _peakCurvatureExact(nu, gamma):

	# Equation A15 in BBKS. 
	
	def curvature_fx(x):

		f1 = np.sqrt(5.0 / 2.0) * x
		t1 = scipy.special.erf(f1) + scipy.special.erf(f1 / 2.0)

		b0 = np.sqrt(2.0 / 5.0 / np.pi)
		b1 = 31.0 * x ** 2 / 4.0 + 8.0 / 5.0
		b2 = x ** 2 / 2.0 - 8.0 / 5.0
		t2 = b0 * (b1 * np.exp(-5.0 * x ** 2 / 8.0) + b2 * np.exp(-5.0 * x ** 2 / 2.0))

		res = (x ** 3 - 3.0 * x) * t1 / 2.0 + t2

		return res

	# Equation A14 in BBKS, minus the normalization which is irrelevant here. If we need the 
	# normalization, the Rstar parameter also needs to be passed.
	
	def curvature_Npk(x, nu, gamma):

		#norm = np.exp(-nu**2 / 2.0) / (2 * np.pi)**2 / Rstar**3
		norm = 1.0
		fx = curvature_fx(x)
		xstar = gamma * nu
		g2 = 1.0 - gamma ** 2
		exponent = -(x - xstar) ** 2 / (2.0 * g2)
		res = norm * fx * np.exp(exponent) / np.sqrt(2.0 * np.pi * g2)

		return res

	# Average over Npk
	
	def curvature_Npk_x(x, nu, gamma):
		return curvature_Npk(x, nu, gamma) * x

	args = nu, gamma
	norm, _ = scipy.integrate.quad(curvature_Npk, 0.0, np.inf, args, epsrel = 1E-10)
	integ, _ = scipy.integrate.quad(curvature_Npk_x, 0.0, np.inf, args, epsrel = 1E-10)
	xav = integ / norm

	return xav

###################################################################################################

# Wrapper for the function above which takes tables of sigmas. This form can be more convenient 
# when computing many different nu's. 

def _peakCurvatureExactFromSigma(sigma0, sigma1, sigma2, z, deltac_args = defaults.DELTAC_ARGS):

	nu = collapseOverdensity(z = z, **deltac_args) / sigma0
	gamma = sigma1**2 / sigma0 / sigma2

	x = nu * 0.0
	for i in range(len(nu)):
		x[i] = _peakCurvatureExact(nu[i], gamma[i])

	return nu, gamma, x

###################################################################################################

# Get peak curvature from the approximate formula in BBKS. This approx. is excellent over the 
# relevant range of nu.

def _peakCurvatureApprox(nu, gamma):

	# Compute theta according to Equation 6.14 in BBKS
	g = gamma
	gn = g * nu
	theta1 = 3.0 * (1.0 - g ** 2) + (1.216 - 0.9 * g ** 4) * np.exp(-g * gn * gn / 8.0)
	theta2 = np.sqrt(3.0 * (1.0 - g ** 2) + 0.45 + (gn / 2.0) ** 2) + gn / 2.0
	theta = theta1 / theta2

	# Equation 6.13 in BBKS
	x = gn + theta
	
	# Equation 6.15 in BBKS
	nu_tilde = nu - theta * g / (1.0 - g ** 2)

	return theta, x, nu_tilde

###################################################################################################

# Wrapper for the function above which takes tables of sigmas. This form can be more convenient 
# when computing many different nu's. For convenience, various intermediate numbers are 
# returned as well.

def _peakCurvatureApproxFromSigma(sigma0, sigma1, sigma2, z, deltac_args = defaults.DELTAC_ARGS):

	nu = collapseOverdensity(z = z, **deltac_args) / sigma0
	gamma = sigma1**2 / sigma0 / sigma2
	
	theta, x, nu_tilde = _peakCurvatureApprox(nu, gamma)
	
	return nu, gamma, x, theta, nu_tilde

###############################################################################################

def peakCurvature(M, z, exact = False, ps_args = defaults.PS_ARGS, 
				sigma_args = {'filt': 'gaussian'}, deltac_args = defaults.DELTAC_ARGS):
	"""
	The average curvature of peaks for a halo mass M.
	
	In a Gaussian random field, :math:`\\delta`, the peak height is defined as 
	:math:`\\delta / \\sigma` where :math:`\\sigma = \\sigma_0` is the rms variance. The 
	curvature of the field is defined as :math:`x = -\\nabla^2 \\delta / \\sigma_2` where 
	:math:`\\sigma_2` is the second moment of the variance (see the documentation of the
	:func:`~cosmology.cosmology.Cosmology.sigma` function).
	
	This function computes the average curvature of peaks in a Gaussian random field, <x>,
	according to `Bardeen et al. 1986 <http://adsabs.harvard.edu/abs/1986ApJ...304...15B>`__ 
	(BBKS), for halos of a certain mass M. This mass is converted to a lagrangian scale R 
	which serves as the scale on which the variance and its moments are evaluated. The 
	computation can be performed by integration of Equation A14 in BBKS (if ``exact == True``), 
	or using their fitting function in Equation 6.13 (if ``exact == False``). The fitting 
	function is excellent over the relevant range of peak heights. 
	
	Parameters
	----------
	M: array_like
		Mass in in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift.
	exact: bool
		If ``True``, evaluate the integral exactly; if ``False``, use the BBKS approximation.	
	ps_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` 
		function.
	sigma_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.sigma` function.
	deltac_args: dict
		Arguments passed to the :func:`collapseOverdensity` function.

	Returns
	-------
	nu: array_like
		Peak height; has the same dimensions as ``M``.
	gamma: array_like
		An intermediate parameter, :math:`\\gamma = \\sigma_1^2 / (\\sigma_0 \\sigma_2)` (see
		Equation 4.6a in BBKS); has the same dimensions as ``M``.
	x: array_like
		The mean peak curvature for halos of mass M (note the caveat discussed above); has the 
		same dimensions as ``M``.
	theta: array_like
		An intermediate parameter (see Equation 6.14 in BBKS; only returned if ``exact == False``); 
		has the same dimensions as ``M``.
	nu_tilde: array_like
		The modified peak height (see Equation 6.15 in BBKS; only returned if ``exact == False``); 
		has the same dimensions as ``M``.
	
	Warnings
	--------		
	While peak height quantifies how high a fluctuation over the background a halo corresponds
	to, peak curvature tells us about the shape of the initial peak. However, peak curvature 
	suffers from the cloud-in-cloud problem (BBKS): not all peaks end up forming halos, 
	especially small peaks will often get swallowed by larger peaks. Thus, the average peak 
	curvature is not necessarily equal to the average curvature of peaks that form halos.		
	"""

	cosmo = cosmology.getCurrent()

	R = lagrangianR(M)
	sigma0 = cosmo.sigma(R, z, j = 0, ps_args = ps_args, **sigma_args)
	sigma1 = cosmo.sigma(R, z, j = 1, ps_args = ps_args, **sigma_args)
	sigma2 = cosmo.sigma(R, z, j = 2, ps_args = ps_args, **sigma_args)

	if exact:
		return _peakCurvatureExactFromSigma(sigma0, sigma1, sigma2, z, deltac_args = deltac_args)
	else:
		return _peakCurvatureApproxFromSigma(sigma0, sigma1, sigma2, z, deltac_args = deltac_args)

###################################################################################################

def powerSpectrumSlope(nu, z, slope_type = 'P', scale = 1.0, 
					ps_args = defaults.PS_ARGS, sigma_args = defaults.SIGMA_ARGS):
	"""
	The slope of the linear matter power spectrum for halos of a given peak height.
	
	In a Gaussian random field, the abundance and shape of the peaks are determined only by the
	power spectrum. The slope of this power spectrum evolves with scale in realistic cosmologies,
	meaning that peaks of different peak height form in a field with a locally different slope
	which can affect their shape, the abundance of sub-structure and so on. 
	
	This function calculates an effective slope :math:`n_{\\rm eff}` using a number of possible
	methods, namely the slope of the power spectrum (``slope_type = 'P'``),

	.. math::
		n_{\\rm eff} = \\left. \\frac{d \\ln(P)}{d \\ln(k)} \\right \\vert_{k = f 2 \\pi / R_{\\rm L}}
	
	or the slope of the variance (``slope_type = 'sigma'``),
	
	.. math::
		n_{\\rm eff} = -2 \\left. \\frac{d \\ln \\sigma(R)}{d \\ln R} \\right \\vert_{R = f R_{\\rm L}} - 3
	
	where :math:`f` is the ``scale`` parameter given by the user and :math:`R_{\\rm L}` is the 
	Lagrangian radius corresponding to the given peak height and redshift. Interpolation must be
	activated in the cosmology as this function uses derivatives.
	
	Parameters
	----------
	nu: array_like
		Peak height; can be a number or a numpy array.
	z: float
		Redshift.
	slope_type: str
		The type of slope function, can be ``P`` or ``sigma`` (see above).
	scale: float
		Scales where the slope is evaluated, see above.
	ps_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` 
		function.
	sigma_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.sigma` function.

	Returns
	-------
	n_eff: array_like
		Effective slope of the power spectrum; has the same dimensions as ``nu``.	
	"""
	
	cosmo = cosmology.getCurrent()

	M_L = massFromPeakHeight(nu, z)
	R_L = lagrangianR(M_L)
	
	if slope_type == 'P':
		k_R = 2.0 * np.pi / R_L * scale
		n_eff = cosmo.matterPowerSpectrum(k_R, derivative = True, **ps_args)
	
	elif slope_type == 'sigma':
		n_eff = -2.0 * cosmo.sigma(scale * R_L, z, derivative = True, 
								ps_args = ps_args, **sigma_args) - 3.0

	else:
		raise Exception('Unknown power spectrum slope type, %s.' % slope_type)
	
	return n_eff

###################################################################################################
