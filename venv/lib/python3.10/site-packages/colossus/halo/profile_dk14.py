###################################################################################################
#
# profile_dk14.py           (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This module implements the Diemer & Kravtsov 2014 form of the density profile. Please see 
:doc:`halo_profile` for a general introduction to the Colossus density profile module.

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

The DK14 profile (`Diemer & Kravtsov 2014 <http://adsabs.harvard.edu/abs/2014ApJ...789....1D>`__)
is defined by the following density form:

.. math::
	\\rho(r) &= \\rho_{\\rm inner} \\times f_{\\rm trans} + \\rho_{\\rm outer}
	
	\\rho_{\\rm inner} &= \\rho_{\\rm Einasto} = \\rho_{\\rm s} \\exp \\left( -\\frac{2}{\\alpha} \\left[ \\left( \\frac{r}{r_{\\rm s}} \\right)^\\alpha -1 \\right] \\right)

	f_{\\rm trans} &= \\left[ 1 + \\left( \\frac{r}{r_{\\rm t}} \\right)^\\beta \\right]^{-\\frac{\\gamma}{\\beta}}

This profile corresponds to an Einasto profile at small radii, and steepens around the virial 
radius. The profile formula has 6 free parameters, but most of those can be fixed to particular 
values that depend on the mass and mass accretion rate of a halo. The parameters have the 
following meaning:

.. table::
	:widths: auto
	
	======= ==================== ===================================================================================
	Param.  Symbol               Explanation	
	======= ==================== ===================================================================================
	rhos	:math:`\\rho_s`       The density at the scale radius in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`
	rs      :math:`r_{\\rm s}`     The scale radius in physical kpc/h
	alpha   :math:`\\alpha`       Determines how quickly the slope of the inner Einasto profile steepens
	rt      :math:`r_{\\rm t}`     The truncation radius where the profile steepens beyond the Einasto profile, in physical kpc/h
	beta    :math:`\\beta`        Sharpness of the steepening
	gamma	:math:`\\gamma`       Asymptotic negative slope of the steepening term
	======= ==================== ===================================================================================

As with all profile models, the user can pass these fundamental parameters or mass and 
concentration. In the latter case, the user can give additional information to create a more
accurate profile model. In particular, the fitting function was calibrated for the median and 
mean profiles of halo samples selected by mass (``selected_by = 'M'``) and selected by both mass 
and mass accretion rate (``selected_by = 'Gamma'``). The latter option results in a more accurate 
representation of the density profile, but the mass accretion rate must be known. 

If the profile is chosen to model halo samples selected by mass, we set 
:math:`(\\beta, \\gamma) = (4, 8)`. If the sample is selected by both mass and mass 
accretion rate, we set :math:`(\\beta, \\gamma) = (6, 4)`. Those choices result in a different 
calibration of the truncation radius :math:`r_{\\rm t}`. In the latter case, both ``z`` and ``Gamma`` 
must not be ``None``. See the :func:`~halo.profile_dk14.DK14Profile.deriveParameters` function 
for more details.

The DK14 profile makes sense only if some description of the outer profile is added. This can
easily be done with the :func:`~halo.profile_composite.compositeProfile` function::

	from colossus.cosmology import cosmology
	from colossus.halo import profile_composite
	
	cosmology.setCosmology('planck18')
	p = profile_composite.compositeProfile('dk14', outer_names = ['mean', 'pl'],
				M = 1E12, c = 10.0, z = 0.0, mdef = 'vir', norm = 1.0, slope = 1.5)
	
This line will return a DK14 profile object with two outer terms: the mean density of the Universe
and a power law infalling profile (normalized at 5 R200m by default). When a parameterization 
relies on properties of the total profile (such as R200m), the constructor determines the 
normalization iteratively. This procedure can be slow or even inconsistent when fitting; in that
case, it is generally preferred to initialize the outer terms with fixed parameters (e.g., fixed
pivot radius or bias). Please see the :doc:`tutorials` for more code examples.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

import numpy as np
import warnings

from colossus import defaults
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.halo import mass_so
from colossus.halo import profile_base

###################################################################################################
# DIEMER & KRAVTSOV 2014 PROFILE
###################################################################################################

class DK14Profile(profile_base.HaloDensityProfile):
	"""
	The Diemer & Kravtsov 2014 density profile.
	
	The redshift must always be passed to this constructor, regardless of whether the 
	fundamental parameters or a mass and concentration are given.
	
	Parameters
	----------
	rhos: float
		The central scale density, in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`.
	rs: float
		The scale radius in physical kpc/h.
	rt: float
		The radius where the profile steepens, in physical kpc/h.
	alpha: float
		Determines how quickly the slope of the inner Einasto profile steepens.
	beta: float
		Sharpness of the steepening.
	gamma: float
		Asymptotic negative slope of the steepening term.
	M: float
		Halo mass in :math:`M_{\\odot}/h`.
	c: float
		Concentration in the same mass definition as ``M``.
	z: float
		Redshift
	mdef: str
		The mass definition to which ``M`` corresponds. See :doc:`halo_mass` for details.
	selected_by: str
		The halo sample to which this profile refers can be selected mass ``M`` or by accretion
		rate ``Gamma``. This parameter influences how some of the fixed parameters in the 
		profile are set, in particular those that describe the steepening term.
	Gamma: float
		The mass accretion rate as defined in DK14. This parameter only needs to be passed if 
		``selected_by == 'Gamma'``.
	"""
	
	###############################################################################################
	# CONSTRUCTOR
	###############################################################################################
	
	def __init__(self, selected_by = defaults.HALO_PROFILE_SELECTED_BY, Gamma = None, **kwargs):

		# Set the fundamental variables par_names and opt_names
		self.par_names = ['rhos', 'rs', 'rt', 'alpha', 'beta', 'gamma']
		self.opt_names = []
		
		# Run the constructor
		profile_base.HaloDensityProfile.__init__(self, allowed_mdefs = ['200m'], 
							selected_by = selected_by, Gamma = Gamma, **kwargs)

		# Sanity checks
		if self.par['rhos'] < 0.0 or self.par['rs'] < 0.0 or self.par['rt'] < 0.0:
			raise Exception('The DK14 radius parameters cannot be negative, something went wrong (%s).' % (str(self.par)))

		# We need to guess a radius when computing vmax
		self.r_guess = self.par['rs']

		return

	###############################################################################################
	# STATIC METHODS
	###############################################################################################

	@staticmethod
	def deriveParameters(selected_by, nu200m = None, z = None, Gamma = None):
		"""
		Calibration of the parameters :math:`\\alpha`, :math:`\\beta`, :math:`\\gamma`, and :math:`r_{\\rm t}`.

		This function determines the values of those parameters in the DK14 profile that can be 
		calibrated based on mass, and potentially mass accretion rate. If the profile is chosen to 
		model halo samples selected by mass (``selected_by = 'M'``), we set
		:math:`(\\beta, \\gamma) = (4, 8)`. If the sample is selected by both mass and mass 
		accretion rate (``selected_by = 'Gamma'``), we set :math:`(\\beta, \\gamma) = (6, 4)`.
		
		Those choices result in a different calibration of the truncation radius :math:`r_{\\rm t}`. 
		If ``selected_by = 'M'``, we use Equation 6 in DK14. Though this relation was originally 
		calibrated for :math:`\\nu = \\nu_{\\rm vir}`, but the difference is small. If 
		``selected_by = 'Gamma'``, :math:`r_{\\rm t}` is calibrated from ``Gamma`` and ``z``.

		Finally, the parameter that determines how quickly the Einasto profile steepens with
		radius, :math:`\\alpha`, is calibrated according to the 
		`Gao et al. 2008 <http://adsabs.harvard.edu/abs/2008MNRAS.387..536G>`__ relation. This 
		function was also originally calibrated for :math:`\\nu = \\nu_{\\rm vir}`, but the 
		difference is small.

		Parameters
		----------
		selected_by: str
			The halo sample to which this profile refers can be selected mass ``M`` or by accretion
			rate ``Gamma``.
		nu200m: float
			The peak height of the halo for which the parameters are to be calibrated. This 
			parameter only needs to be passed if ``selected_by == 'M'``.
		z: float
			Redshift
		Gamma: float
			The mass accretion rate as defined in DK14. This parameter only needs to be passed if 
			``selected_by == 'Gamma'``.
		"""

		if selected_by == 'M':
			beta = 4.0
			gamma = 8.0
			if (nu200m is not None):
				if (nu200m > 10.0):
					raise Exception('Found nu200m = %.2f, which is unrealistic and breaks the profile initialization.' \
								% nu200m)
				rt_R200m = 1.9 - 0.18 * nu200m
			else:
				raise Exception('Need nu200m to compute rt.')
			
		elif selected_by == 'Gamma':
			beta = 6.0
			gamma = 4.0
			if (Gamma is not None) and (z is not None):
				cosmo = cosmology.getCurrent()
				rt_R200m =  0.43 * (1.0 + 0.92 * cosmo.Om(z)) * (1.0 + 2.18 * np.exp(-Gamma / 1.91))
			else:
				raise Exception('Need Gamma and z to compute rt.')

		else:
			raise Exception('Unknown sample selection, %s.' % (selected_by))

		alpha = 0.155 + 0.0095 * nu200m**2

		return alpha, beta, gamma, rt_R200m

	###############################################################################################
	# METHODS BOUND TO THE CLASS
	###############################################################################################

	def setNativeParameters(self, M, c, z, mdef, selected_by = None, Gamma = None, **kwargs):
		"""
		Set the native DK14 parameters from mass and concentration (and optionally others).

		The DK14 profile has six free parameters, which are set by this function. The mass and 
		concentration must be given as :math:`M_{\\rm 200m}` and :math:`c_{\\rm 200m}`. Other 
		mass definitions demand iteration, which can be achieved with the initialization routine
		in the parent class. This function ignores the presence of outer profiles.
	
		Parameters
		----------
		M: float
			Spherical overdensity mass in :math:`M_{\\odot}/h`.
		c: float
			The concentration, :math:`c = R / r_{\\rm s}`, corresponding to the given halo mass and 
			mass definition.
		z: float
			Redshift
		mdef: str
			The mass definition in which ``M`` and ``c`` are given. See :doc:`halo_mass` for 
			details.
		selected_by: str
			The halo sample to which this profile refers can be selected mass ``M`` or by accretion
			rate ``Gamma``.
		Gamma: float
			The mass accretion rate as defined in DK14. This parameter only needs to be passed if 
			``selected_by == 'Gamma'``.
		"""

		if selected_by is None:
			raise Exception('The selected_by option must be set in DK14 profile, found None.')
		if mdef != '200m':
			raise Exception('The DK14 parameters can only be constructed from the M200m definition, found %s.' % (mdef))

		M200m = M
		R200m = mass_so.M_to_R(M200m, z, mdef)
		nu200m = peaks.peakHeight(M200m, z)

		self.par['rs'] = R200m / c
		self.par['alpha'], self.par['beta'], self.par['gamma'], rt_R200m = \
			self.deriveParameters(selected_by, nu200m = nu200m, z = z, Gamma = Gamma)
		self.par['rt'] = rt_R200m * R200m
		self.par['rhos'] = 1.0
		self.par['rhos'] *= M200m / self.enclosedMassInner(R200m)

		return

	###############################################################################################
	
	def densityInner(self, r):
		"""
		Density of the inner profile as a function of radius.
		
		Parameters
		----------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------
		density: array_like
			Density in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`; has the same dimensions 
			as ``r``.
		"""		
		
		inner = self.par['rhos'] * np.exp(-2.0 / self.par['alpha'] * ((r / self.par['rs'])**self.par['alpha'] - 1.0))
		fT = (1.0 + (r / self.par['rt'])**self.par['beta'])**(-self.par['gamma'] / self.par['beta'])
		rho_1h = inner * fT

		return rho_1h

	###############################################################################################
	
	def densityDerivativeLinInner(self, r):
		"""
		The linear derivative of the inner density, :math:`d \\rho_{\\rm inner} / dr`. 
		
		Parameters
		----------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------
		derivative: array_like
			The linear derivative in physical :math:`M_{\\odot} h / {\\rm kpc}^2`; has the same 
			dimensions as r.
		"""
		
		drho_dr = r * 0.0
		
		rhos = self.par['rhos']
		rs = self.par['rs']
		rt = self.par['rt']
		alpha = self.par['alpha']
		beta = self.par['beta']
		gamma = self.par['gamma']
		
		inner = rhos * np.exp(-2.0 / alpha * ((r / rs) ** alpha - 1.0))
		d_inner = inner * (-2.0 / rs) * (r / rs)**(alpha - 1.0)	
		fT = (1.0 + (r / rt) ** beta) ** (-gamma / beta)
		d_fT = (-gamma / beta) * (1.0 + (r / rt) ** beta) ** (-gamma / beta - 1.0) * \
			beta / rt * (r / rt) ** (beta - 1.0)
		drho_dr += inner * d_fT + d_inner * fT
		
		return drho_dr

	###############################################################################################

	def Rsp(self, search_range = 5.0):

		warnings.warn('The DK14Profile.Rsp() function is deprecated and will be removed. Please use Rsteepest() instead.')

		return self.Rsteepest()
	
	###############################################################################################

	def RMsp(self, search_range = 5.0):

		warnings.warn('The DK14Profile.RMsp() function is deprecated and will be removed. Please use Rsteepest() instead.')

		Rsp = self.Rsteepest()
		Msp = self.enclosedMass(Rsp)

		return Rsp, Msp
	
	###############################################################################################

	def Msp(self, search_range = 5.0):

		warnings.warn('The DK14Profile.Msp() function is deprecated and will be removed. Please use Rsteepest() instead.')

		_, Msp = self.RMsp(search_range = search_range)

		return Msp
	
	###############################################################################################
	
	def _fitParamDeriv_rho(self, r, mask, N_par_fit):

		x = self.getParameterArray()
		deriv = np.zeros((N_par_fit, len(r)), float)
		rho_inner = self.densityInner(r)

		rs = x[1]
		rt = x[2]
		alpha = x[3]
		beta = x[4]
		gamma = x[5]

		rrs = r / rs
		rrt = r / rt
		term1 = 1.0 + rrt**beta
		
		counter = 0
		# rho_s
		if mask[0]:
			deriv[counter] = rho_inner
			counter += 1
		# rs
		if mask[1]:
			deriv[counter] = rho_inner * rrs**alpha * 2.0
			counter += 1
		# rt
		if mask[2]:
			deriv[counter] = rho_inner * gamma / term1 * rrt**beta
			counter += 1
		# alpha
		if mask[3]:
			deriv[counter] = rho_inner * 2.0 / alpha * rrs**alpha * (1.0 - rrs**(-alpha) - alpha * np.log(rrs))
			counter += 1
		# beta
		if mask[4]:
			deriv[counter] = rho_inner * (gamma * np.log(term1) / beta**2 - gamma * \
										rrt**beta * np.log(rrt) / term1)
			counter += 1
		# gamma
		if mask[5]:
			deriv[counter] = -rho_inner * np.log(term1) / beta / gamma
			counter += 1

		return deriv

###################################################################################################
