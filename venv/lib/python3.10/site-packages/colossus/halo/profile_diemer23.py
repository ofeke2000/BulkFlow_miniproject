###################################################################################################
#
# profile_diemer23.py       (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This module implements the Diemer 2022b form of the density profile. Please see 
:doc:`halo_profile` for a general introduction to the Colossus density profile module.

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

The `Diemer (2022b) <http://adsabs.harvard.edu/abs/2014ApJ...789....1D>`__ profile was designed to 
fit the orbiting component of dark matter halos, even at large radii where the infalling component 
comes to dominate. The orbiting component has a sharp truncation that cannot be described by a
power-law steepening term (as in the DK14 profile), but it can be described by the following
"truncated exponential" form:

.. math::
	\\rho(r) = \\rho_{\\rm s} \\exp \\left\\{ -\\frac{2}{\\alpha} \\left[ \\left( \\frac{r}{r_{\\rm s}} \\right)^\\alpha - 1 \\right] -\\frac{1}{\\beta} \\left[ \\left( \\frac{r}{r_{\\rm t}} \\right)^\\beta - \\left( \\frac{r_{\\rm s}}{r_{\\rm t}} \\right)^\\beta \\right] \\right\\}

The meaning of this functional form is easiest to understand by considering its logarithmic slope:

.. math::
	\\gamma(r) \\equiv \\frac{{\\rm d} \\ln \\rho}{{\\rm d} \\ln r} = -2 \\left( \\frac{r}{r_{\\rm s}} \\right)^\\alpha - \\left( \\frac{r}{r_{\\rm t}} \\right)^\\beta

The first term is identical to an Einasto profile, and the second term causes a more or less sharp
truncation. The formula has 5 free parameters with well-defined physical interpretations:

.. table::
	:widths: auto
	
	======= ==================== ===================================================================================
	Param.  Symbol               Explanation	
	======= ==================== ===================================================================================
	rhos	:math:`\\rho_s`       Density at the scale radius, in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`
	rs      :math:`r_{\\rm s}`     The scale radius in physical kpc/h
	alpha   :math:`\\alpha`       Determines how quickly the slope of the inner Einasto profile steepens
	rt      :math:`r_{\\rm t}`     The radius where the profile steepens beyond the Einasto profile, in physical kpc/h
	beta    :math:`\\beta`        Sharpness of the truncation
	======= ==================== ===================================================================================

As with all profile models, the user can pass these fundamental parameters or mass and 
concentration to the constructor of the :class:`ModelAProfile` class (the reason for the name 
will become apparent later). In the latter case, the user can also give additional information to 
create a more accurate profile model. In particular, the fitting function was calibrated for the 
median and mean profiles of halo samples selected by mass (``selected_by = 'M'``) and selected by 
both mass and mass accretion rate (``selected_by = 'Gamma'``). The latter option results in a more 
accurate representation of the density profile, but the mass accretion rate must be known. See the 
:func:`~halo.profile_diemer23.ModelAProfile.deriveParameters` function for details.

---------------------------------------------------------------------------------------------------
Adding an infalling profile
---------------------------------------------------------------------------------------------------

In most real-world applications, we are interested in the total density rather than only that of 
orbiting matter. We thus want to add the overdensity of matter on a first infall (or "infalling 
profile"), as well as the mean density of the Universe. Such a composite orbiting+infalling model
can easily be created with the :func:`~halo.profile_composite.compositeProfile` function::

	from colossus.cosmology import cosmology
	from colossus.halo import profile_composite
	
	cosmology.setCosmology('planck18')
	p = profile_composite.compositeProfile('diemer23', outer_names = ['mean', 'infalling'],
				M = 1E12, c = 10.0, z = 0.0, mdef = 'vir', pl_delta_1 = 10.0, pl_s = 1.5)

With a single command, we have created a truncated exponential profile with two outer terms, 
the constant mean density and an infalling profile of the form

.. math::
	\\rho_{\\rm inf}(r) = \\delta_1 \\rho_{\\rm m}(z) \\left[ \\left( \\frac{\\delta_1}{\\delta_{\\rm max}} \\right)^{1/\\zeta} + \\left( \\frac{r}{r_{\\rm pivot}} \\right)^{s/\\zeta} \\right]^{-\\zeta} 

where :math:`\\delta_1` is the overdensity normalization and :math:`s` the slope. These parameters
depend on the mass, accretion rate, and cosmology of the halo sample in question (see
`Diemer 2022b <http://adsabs.harvard.edu/abs/2014ApJ...789....1D>`__). The maximum 
overdensity at the center can safely be left to its default value unless the infalling profile is
known in detail, as can :math:`\\zeta = 0.5`. The pivot radius is, by default, set to 
:math:`R_{\\rm 200m}`. This parameterization relies on the parameters of the inner profile, which
is correctly handled by the constructor. When fitting, however, such an interdependence can create
issues and it is recommended to set a fixed physical radius as a pivot. For more details, see the 
:class:`~halo.profile_outer.OuterTermInfalling` class, as well as the code :doc:`tutorials`.

---------------------------------------------------------------------------------------------------
Model variant with correction at scale radius
---------------------------------------------------------------------------------------------------

The orbiting profile model described above has one technically unaesthetic property: the 
logarithmic slope at the scale radius is no longer -2. Thus, 
`Diemer 2022b <http://adsabs.harvard.edu/abs/2014ApJ...789....1D>`__ also proposed a corrected
variant called Model B, which can be created using the :class:`ModelBProfile` class. Here, an extra
term has been inserted into the slope to ensure that it remains -2 at :math:`r_{\\rm s}`,

.. math::
	\\gamma(r) \\equiv \\frac{{\\rm d} \\ln \\rho}{{\\rm d} \\ln r} = -2 \\left( \\frac{r}{r_{\\rm s}} \\right)^\\alpha - \\left( \\frac{r}{r_{\\rm t}} \\right)^\\beta + \\left( \\frac{r_{\\rm s}}{r_{\\rm t}} \\right)^\\beta \\left( \\frac{r}{r_{\\rm s}} \\right)^\\eta

where :math:`\\eta = 0.1` is a nuissance parameter that determines how quickly the correction term
vanishes at small radii. The density function also becomes somewhat more complicated, but the user
can ignore the underlying equations and use Model B exactly as Model A. The differences are so 
small that the parameters have virtually the same meaning. The main advantage of Model B is that 
it can be more stable in fits to profiles with a poorly defined scale radius, that is, profiles
with a slope that is roughly -2 across a wide range of radii. Otherwise, we recommend using Model
A for most applications. Given the Model A/B split is implemented via an abstract class and two
specific derived classes:

* :class:`GenericD22Profile` (should never be instantiated by the user)
* :class:`ModelAProfile` (the default)
* :class:`ModelBProfile` (if correction at scale radius is needed)

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

import numpy as np

from colossus import defaults
from colossus.utils import utilities
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.halo import mass_so
from colossus.halo import profile_base

###################################################################################################
# GENERIC BASE CLASS
###################################################################################################

class GenericD22Profile(profile_base.HaloDensityProfile):
	"""
	Base class for truncated exponential profiles.
	
	Generic profile class for methods that are common to the Model A and B variants. This class
	should never be instantiated by the user.
	"""
	
	def __init__(self, selected_by = defaults.HALO_PROFILE_SELECTED_BY, Gamma = None, **kwargs):

		# Run the constructor
		profile_base.HaloDensityProfile.__init__(self, allowed_mdefs = ['200m'], 
							selected_by = selected_by, Gamma = Gamma, **kwargs)
	
		# Sanity checks
		if self.par['rhos'] < 0.0 or self.par['rs'] < 0.0 or self.par['rt'] < 0.0:
			raise Exception('The radius parameters cannot be negative, something went wrong (%s).' % (str(self.par)))

		# We need to guess a radius when computing vmax
		self.r_guess = self.par['rs']

		return

	###############################################################################################
	# STATIC METHODS
	###############################################################################################

	@staticmethod
	def deriveParameters(selected_by, nu200m = None, z = None, Gamma = None):
		"""
		Calibration of the parameters :math:`\\alpha`, :math:`\\beta`, and :math:`r_{\\rm t}`.

		This function determines the values of those parameters in the Diemer22 profile that can be 
		calibrated based on mass, and potentially mass accretion rate. The latter is the stronger
		determinant of the profile shape, but may not always be available (e.g., for mass-selected
		samples).
		
		We set :math:`\\alpha = 0.18` and :math:`\\beta = 3`, which are the default parameters for 
		individual halo profiles. However, they are not necessarily optimal for any type of 
		averaged sample, where the optimal values vary. We do not calibrate :math:`\\alpha` with 
		mass as suggested by
		`Gao et al. 2008 <http://adsabs.harvard.edu/abs/2008MNRAS.387..536G>`__ because we do 
		not reproduce this relation in our data (Diemer 2022c).
		
		The truncation ratius :math:`r_{\\rm t}` is calibrated as suggested by DK14 for
		Gamma-selected samples. If ``selected_by = 'M'``, we use a new parametrization because the 
		meaning of rt differs for the different slope parameters of the DK14 profile. If ``selected_by = 'Gamma'``, 
		:math:`r_{\\rm t}` is calibrated from ``Gamma`` and ``z``. The DK14 calibrations are based
		on slightly different definitions of peak height (:math:`\\nu = \\nu_{\\rm vir}`), 
		accretion rate, and for a different fitting function. However, the resulting :math:`r_{\\rm t}`
		values are very similar to the forthcoming analysis in Diemer 2022c. 

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
			The mass accretion rate over the past dynamical time, which is defined as the crossing 
			time (see the :func:`~halo.mass_so.dynamicalTime` function or 
			`Diemer 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJS..231....5D/abstract>`__ for 
			details). The definition in the DK14 profile is slightly different, but the definitions are close 
			enough that they can be used interchangeably without great loss of accuracy. The Gamma 
			parameter only needs to be passed if ``selected_by == 'Gamma'``.

		Returns
		-------
		alpha: float
			The Einasto steepening parameter.
		beta: float
			The steepening of the truncation term.
		rt_R200m: float
			The truncation radius in units of R200m.
		"""

		alpha = 0.18
		beta = 3.0

		if selected_by == 'M':
			if (nu200m is not None):
				if (nu200m > 10.0):
					raise Exception('Found nu200m = %.2f, which is unrealistic and breaks the profile initialization.' \
								% nu200m)
				rt_R200m = 1.4 - 0.21 * nu200m
			else:
				raise Exception('Need nu200m to compute rt.')				
			
		elif selected_by == 'Gamma':
			if (Gamma is not None) and (z is not None):
				cosmo = cosmology.getCurrent()
				rt_R200m =  0.43 * (1.0 + 0.92 * cosmo.Om(z)) * (1.0 + 2.18 * np.exp(-Gamma / 1.91))
			else:
				raise Exception('Need Gamma and z to compute rt.')

		else:
			raise Exception('Unknown sample selection, %s.' % (selected_by))

		return alpha, beta, rt_R200m

	###############################################################################################
	# METHODS BOUND TO THE CLASS
	###############################################################################################

	def setNativeParameters(self, M, c, z, mdef, selected_by = None, Gamma = None, **kwargs):
		"""
		Set the native parameters from mass and concentration (and optionally others).

		The truncated exponential profile has five free parameters, which are set by this function. 
		The mass and concentration must be given as :math:`M_{\\rm 200m}` and :math:`c_{\\rm 200m}`. 
		Other mass definitions demand iteration, which can be achieved with the initialization routine
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
			``selected_by == 'Gamma'``. See comments in :func:`deriveParameters` function above.
		"""

		if selected_by is None:
			raise Exception('The selected_by option must be set in Diemer22 profile, found None.')
		if mdef != '200m':
			raise Exception('The Diemer22 parameters can only be constructed from the M200m definition, found %s.' % (mdef))

		M200m = M
		R200m = mass_so.M_to_R(M200m, z, mdef)
		nu200m = peaks.peakHeight(M200m, z)

		self.par['rs'] = R200m / c
		self.par['alpha'], self.par['beta'], rt_R200m = \
			self.deriveParameters(selected_by, nu200m = nu200m, z = z, Gamma = Gamma)
		self.par['rt'] = rt_R200m * R200m
		self.par['rhos'] = 1.0
		self.par['rhos'] *= M200m / self.enclosedMassInner(R200m)

		return
	
	###############################################################################################
	
	def densityDerivativeLinInner(self, r):
		"""
		The linear derivative of the inner density, :math:`d \\rho_{\\rm inner} / dr`. 
		
		For the truncated exponential profile, the logarithmic derivative is much easier to 
		evaluate. Thus, this function converts the logarithmic to the linear derivative.
		
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
		
		d_lnrho_d_lnr = self.densityDerivativeLogInner(r)
		rho = self.densityInner(r)
		der = d_lnrho_d_lnr * rho / r
		
		return der

###################################################################################################
# MODEL A (STANDARD)
###################################################################################################

class ModelAProfile(GenericD22Profile):
	"""
	The Diemer 2023 (truncated exponential) density profile (default version).
	
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
	M: float
		Halo mass in :math:`M_{\\odot}/h`.
	c: float
		Concentration in the same mass definition as ``M``.
	mdef: str
		The mass definition to which ``M`` corresponds. See :doc:`halo_mass` for details.
	z: float
		Redshift
	selected_by: str
		The halo sample to which this profile refers can be selected mass ``M`` or by accretion
		rate ``Gamma``. This parameter influences how some of the fixed parameters in the 
		profile are set, in particular those that describe the steepening term.
	Gamma: float
		The mass accretion rate over the past dynamical time, which is defined as the crossing 
		time (see the :func:`~halo.mass_so.dynamicalTime` function or 
		`Diemer 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJS..231....5D/abstract>`__ for 
		details). The definition in the DK14 profile is slightly different, but the definitions are close 
		enough that they can be used interchangeably without great loss of accuracy. The Gamma 
		parameter only needs to be passed if ``selected_by == 'Gamma'``.
	"""
	
	###############################################################################################
	# CONSTRUCTOR
	###############################################################################################
	
	def __init__(self, selected_by = defaults.HALO_PROFILE_SELECTED_BY, Gamma = None, **kwargs):

		# Set the fundamental variables par_names and opt_names
		self.par_names = ['rhos', 'rs', 'rt', 'alpha', 'beta']
		self.opt_names = []

		# Run the generic constructor
		GenericD22Profile.__init__(self, selected_by = selected_by, Gamma = Gamma, **kwargs)

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
		
		rs = self.par['rs']
		rt = self.par['rt']
		alpha = self.par['alpha']
		beta = self.par['beta']
		
		S = -2.0 / alpha * ((r / rs)**alpha - 1.0) - 1.0 / beta * ((r / rt)**beta - (rs / rt)**beta)
		rho = self.par['rhos'] * utilities.safeExp(S)

		return rho

	###############################################################################################
	
	def densityDerivativeLogInner(self, r):
		"""
		The logarithmic derivative of the inner density, :math:`d \\log(\\rho_{\\rm inner}) / d \\log(r)`. 

		Parameters
		----------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------
		derivative: array_like
			The dimensionless logarithmic derivative; has the same dimensions as ``r``.
		"""
		
		der = -2.0 * (r / self.par['rs'])**self.par['alpha'] - (r / self.par['rt'])**self.par['beta']

		return der
		
	###############################################################################################
	
	def _fitParamDeriv_rho(self, r, mask, N_par_fit):

		x = self.getParameterArray()
		deriv = np.zeros((N_par_fit, len(r)), float)

		rhos = x[0]
		rs = x[1]
		rt = x[2]
		alpha = x[3]
		beta = x[4]
		
		rrs = r / rs
		rrt = r / rt
		rrsa = rrs**alpha
		rrtb = rrt**beta
		rsrt = rs / rt
		rsrtb = rsrt**beta

		s = -2.0 / alpha * (rrsa - 1.0) - 1.0 / beta * (rrtb - rsrtb)
		rho = rhos * utilities.safeExp(s)
		
		counter = 0
		# rhos
		if mask[0]:
			deriv[counter][:] = 1.0
			counter += 1
		# rs
		if mask[1]:
			deriv[counter] = 2.0 * rrsa + rsrtb
			counter += 1
		# rt
		if mask[2]:
			deriv[counter] = rrtb - rsrtb
			counter += 1
		# alpha
		if mask[3]:
			deriv[counter] = 2.0 / alpha * (rrsa * (1.0 - alpha * np.log(rrs)) - 1.0)
			counter += 1
		# beta
		if mask[4]:
			deriv[counter] = rrtb * (1.0 / beta - np.log(rrt)) - rsrtb * (1.0 / beta - np.log(rsrt))

		deriv[:, :] *= rho[None, :]

		return deriv

###################################################################################################
# MODEL B (CORRECTION FOR SCALE RADIUS)
###################################################################################################

class ModelBProfile(GenericD22Profile):
	"""
	The Diemer 2023 (truncated exponential) density profile (Model B).
	
	This version corrects a minor flaw in the default model: the logarithmic slope at the
	scale radius is not -2 in the default Model A. In this Model B, this condition
	is enforced at the cost of an extra term, which gradually adjusts the slope between the center
	(where it is still zero) and the scale radius, where it offsets the effect of the truncation
	term. However, this correction is usually very small (except for extreme values of beta or 
	rt). Thus, Model A and Model B profiles are virtually the same for almost all parameters.
	Model B can be a little more stable in fits to profiles without a clear scale radius. The
	nuissance parameter is set to :math:`\\eta = 0.1` by default; it is not recommended to change
	this parameter or to adjust it in fits.
	
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
	eta: float
		Nuissance parameter that determines how quickly the slope approaches zero at the halo
		center.  
	M: float
		Halo mass in :math:`M_{\\odot}/h`.
	c: float
		Concentration in the same mass definition as ``M``.
	mdef: str
		The mass definition to which ``M`` corresponds. See :doc:`halo_mass` for details.
	z: float
		Redshift
	selected_by: str
		The halo sample to which this profile refers can be selected mass ``M`` or by accretion
		rate ``Gamma``. This parameter influences how some of the fixed parameters in the 
		profile are set, in particular those that describe the steepening term.
	Gamma: float
		The mass accretion rate over the past dynamical time, which is defined as the crossing 
		time (see the :func:`~halo.mass_so.dynamicalTime` function or 
		`Diemer 2017 <https://ui.adsabs.harvard.edu/abs/2017ApJS..231....5D/abstract>`__ for 
		details). The definition in the DK14 profile is slightly different, but the definitions are close 
		enough that they can be used interchangeably without great loss of accuracy. The Gamma 
		parameter only needs to be passed if ``selected_by == 'Gamma'``.
	"""
	
	###############################################################################################
	# CONSTRUCTOR
	###############################################################################################
	
	def __init__(self, selected_by = defaults.HALO_PROFILE_SELECTED_BY, Gamma = None, 
				eta = defaults.HALO_PROFILE_D22_ETA, **kwargs):

		# Set the fundamental variables par_names and opt_names
		self.par_names = ['rhos', 'rs', 'rt', 'alpha', 'beta', 'eta']
		self.opt_names = []

		# Run the generic constructor
		GenericD22Profile.__init__(self, selected_by = selected_by, Gamma = Gamma, eta = eta, **kwargs)
	
		return
	
	###############################################################################################

	def setNativeParameters(self, M, c, z, mdef, eta = defaults.HALO_PROFILE_D22_ETA, 
						selected_by = None, Gamma = None, **kwargs):
		"""
		Set the native Diemer22 parameters from mass and concentration (and optionally others).

		The D22 profile has five free parameters, which are set by this function. The mass and 
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
		eta: float
			Nuissance parameter that determines how quickly the slope approaches zero at the halo
			center.
		selected_by: str
			The halo sample to which this profile refers can be selected mass ``M`` or by accretion
			rate ``Gamma``.
		Gamma: float
			The mass accretion rate as defined in DK14. This parameter only needs to be passed if 
			``selected_by == 'Gamma'``.
		"""
		
		self.par['eta'] = eta
		
		GenericD22Profile.setNativeParameters(self, M, c, z, mdef, eta = eta, 
											selected_by = selected_by, Gamma = Gamma, **kwargs)

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
		
		rs = self.par['rs']
		rt = self.par['rt']
		alpha = self.par['alpha']
		beta = self.par['beta']
		eta = self.par['eta']
		
		rrs = r / rs
		rsrtb = (rs / rt)**beta
		S = -2.0 / alpha * (rrs**alpha - 1.0) - 1.0 / beta * ((r / rt)**beta - rsrtb) \
			+ 1.0 / eta * rsrtb * (rrs**eta - 1.0)
		rho = self.par['rhos'] * utilities.safeExp(S)

		return rho

	###############################################################################################
	
	def densityDerivativeLogInner(self, r):
		"""
		The logarithmic derivative of the inner density, :math:`d \\log(\\rho_{\\rm inner}) / d \\log(r)`. 

		Parameters
		----------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------
		derivative: array_like
			The dimensionless logarithmic derivative; has the same dimensions as ``r``.
		"""

		rs = self.par['rs']
		rt = self.par['rt']
		alpha = self.par['alpha']
		beta = self.par['beta']
		eta = self.par['eta']

		rrs = 	r / self.par['rs']	
		der = -2.0 * rrs**alpha - (r / rt)**beta + (rs / rt)**beta * rrs**eta

		return der
		
	###############################################################################################
	
	def _fitParamDeriv_rho(self, r, mask, N_par_fit):

		x = self.getParameterArray()
		deriv = np.zeros((N_par_fit, len(r)), float)

		rhos = x[0]
		rs = x[1]
		rt = x[2]
		alpha = x[3]
		beta = x[4]
		eta = x[5]
		
		rrs = r / rs
		logrrs = np.log(rrs)
		rrt = r / rt
		rrsa = rrs**alpha
		rrtb = rrt**beta
		rsrt = rs / rt
		rsrtb = rsrt**beta
		rrse = rrs**eta

		s = -2.0 / alpha * (rrsa - 1.0) - 1.0 / beta * (rrtb - rsrtb)
		rho = rhos * utilities.safeExp(s)
		
		counter = 0
		# rhos
		if mask[0]:
			deriv[counter][:] = 1.0
			counter += 1
		# rs
		if mask[1]:
			deriv[counter] = 2.0 * rrsa + (beta / eta - 1.0) * rsrtb * (rrse - 1)
			counter += 1
		# rt
		if mask[2]:
			deriv[counter] = rrtb - rsrtb * (beta / eta * (rrse - 1.0) + 1.0)
			counter += 1
		# alpha
		if mask[3]:
			deriv[counter] = 2.0 / alpha * (rrsa * (1.0 - alpha * logrrs) - 1)
			counter += 1
		# beta
		if mask[4]:
			deriv[counter] = rrtb * (1.0 / beta - np.log(rrt)) - rsrtb * (1.0 / beta - np.log(rsrt) * (beta / eta * (rrse - 1.0) + 1.0))
			counter += 1
		# eta
		if mask[5]:
			deriv[counter] = 1.0 / eta * rsrtb * (rrse * (eta * logrrs - 1.0) + 1.0)
			
		deriv[:, :] *= rho[None, :]

		return deriv

###################################################################################################
