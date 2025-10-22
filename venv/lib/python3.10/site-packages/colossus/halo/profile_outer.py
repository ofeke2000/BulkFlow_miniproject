###################################################################################################
#
# profile_outer.py           (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This module implements terms that describe the outer halo density profile. Specific terms are 
derived from the :class:`OuterTerm` base class. The :doc:`tutorials` contain more detailed code
examples. For an introduction on how to use the outer terms, please see :doc:`halo_profile`.

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

The following outer terms are currently implemented:

* :class:`OuterTermMeanDensity` (shortcode ``mean``):
  The mean density of the universe at redshift ``z``. This term should generally be present, given
  that density profiles must eventually approach the mean density at very large radii. However, it
  can be advantageous to omit this term, for example when computing surface densities.
* :class:`OuterTermCorrelationFunction` (shortcode ``cf``): 
  The matter-matter correlation function times a halo bias. Here, the user has a choice
  regarding halo bias: it can enter the profile as a parameter (if ``derive_bias_from == 
  None`` or it can be derived according to the default model of halo bias based on 
  :math:`M_{\\rm 200m}` (in which case ``derive_bias_from = 'R200m'`` and the bias parameter 
  is ignored). The latter option can make the constructor slow because of the iterative 
  evaluation of bias and :math:`M_{\\rm 200m}`. Note that the CF becomes negative at some 
  (relatively large) radius, which leads to errors when computing the surface density or lensing
  signal. In this case, the integration radius must be limited manually (see the respective
  function documentation).
* :class:`OuterTermPowerLaw` (shortcode ``pl``): 
  A power-law profile in overdensity. This form was suggested to be added to the DK14 profile, 
  with a pivot radius of :math:`5 R_{\\rm 200m}`. Note that :math:`R_{\\rm 200m}` is set as a 
  profile option in the constructor once, but not adjusted thereafter unless the 
  :func:`~halo.profile_dk14.DK14Profile.update` function is called. Thus, in a fit, the fitted 
  norm and slope refer to a pivot of the original :math:`R_{\\rm 200m}` until update() is called 
  which adjusts these parameters. Thus, it is often better to fix the pivot radius by setting
  ``pivot = 'fixed'`` and ``pivot_factor = 100.0`` or some other chosen radius in physical units.
  The parameters for the power-law outer profile (norm and slope, called :math:`b_{\\rm e}` and 
  :math:`s_{\\rm e}` in DK14) exhibit a complicated dependence on halo mass, redshift and 
  cosmology. At low redshift, and for the cosmology considered in DK14, ``power_law_norm = 1.0`` 
  and ``power_law_slope = 1.5`` are reasonable values over a wide range of masses (see Figure 18 
  in DK14), but these values are by no means universal or accurate. 
* :class:`OuterTermInfalling` (shortcode ``infalling``): 
  Infalling term: another power-law profile in overdensity with an asymptotic maximum density at
  the center, but also with a parameter that controls the smoothness of the transition to this
  fixed overdensity. This parameter is usually fixed to 0.5. This profile was specifically 
  designed to fit the infalling term in simulations. It is parameterized somewhat differently
  than the ``pl`` profile. See Diemer 2022b for details.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
import abc
import collections
import six
import warnings

# TODO remove when scipy.differentiate is available in all reasonable scipy versions.
try:
	import scipy.differentiate
	HAS_SCIPY_DIFF = True
except:
	import scipy.misc
	HAS_SCIPY_DIFF = False

from colossus.utils import utilities
from colossus import defaults
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.lss import bias as halo_bias

###################################################################################################
# ABSTRACT BASE CLASS FOR OUTER PROFILE TERMS
###################################################################################################

@six.add_metaclass(abc.ABCMeta)
class OuterTerm():
	"""
	Base class for outer profile terms.
	
	All outer terms must be derived from this OuterTerm base class overwrite at least the
	the ``_density()`` routine. The derived outer terms must also, in their constructor, call the 
	constructor of this class with the parameters specified below. The user interface to such
	derived classes will, in general, be much simpler than the constructor of this super class.
	
	The user can then add one or multiple outer terms to a density profile by calling its 
	constructor and passing a list of OuterTerm objects in the ``outer_terms`` argument (see the 
	documentation of :class:`~halo.profile_base.HaloDensityProfile`). Once the profile has been 
	created, the outer terms themselves cannot be added or removed. Their parameters, however, can
	be modified in the same ways as the parameters of the inner profile.

	Parameters
	----------
	par_array: list
		A list of parameter values for the outer term.
	opt_array: list
		A list of option values for the outer term.
	par_names: list
		A list of parameter names, corresponding to the values passed in par_array. If these names
		overlap with already existing parameters, the parameter is NOT added to the profile. 
		Instead, the value of the existing parameter will be used. This behavior can be useful when
		outer profile terms rely on parameters or options of the inner profile.
	opt_names:
		A list of option names, corresponding to the values passed in opt_array.
	"""
	
	def __init__(self, par_array, opt_array, par_names, opt_names):
		
		if len(par_array) != len(par_names):
			raise Exception('Arrays with parameters and parameter names must have the same length (%d, %d).' % \
				(len(par_array), len(par_names)))
		
		if len(opt_array) != len(opt_names):
			raise Exception('Arrays with options and option names must have the same length (%d, %d).' % \
				(len(opt_array), len(opt_names)))

		self.term_par_names = par_names
		self.term_opt_names = opt_names

		# The parameters of the profile are stored in a dictionary
		self.term_par = collections.OrderedDict()
		self.N_par = len(self.term_par_names)
		for i in range(self.N_par):
			self.term_par[self.term_par_names[i]] = par_array[i]

		# Additionally to the numerical parameters, there can be options
		self.term_opt = collections.OrderedDict()
		self.N_opt = len(self.term_opt_names)
		for i in range(self.N_opt):
			self.term_opt[self.term_opt_names[i]] = opt_array[i]

		# Set pointers to par/opt arrays. These will be overwritten with the total arrays from the
		# parent class if this outer term is added to an inner term, but they allow us to use the
		# outer term standalone as well.
		self.par = self.term_par
		self.opt = self.term_opt

		return

	###############################################################################################

	# Return the density of at an array r

	@abc.abstractmethod
	def _density(self, r):
		"""
		The density due to the outer term as a function of radius.
		
		Abstract function which must be overwritten by child classes.
		
		Parameters
		----------
		r: array_like
			Radius in physical kpc/h; guaranteed to be an array, even if of length 1.

		Returns
		-------
		density: array_like
			Density in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`; has the same dimensions 
			as ``r``.
		"""

		return

	###############################################################################################

	def density(self, r):
		"""
		The density due to the outer term as a function of radius.
		
		This function provides a convenient wrapper around _density() by ensuring that the radius
		values passed are a numpy array. This function should generally not be overwritten by 
		child classes.
		
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

		r_array, is_array = utilities.getArray(r)
		r_array = r_array.astype(float)
		rho = self._density(r_array)
		if not is_array:
			rho = rho[0]
		
		return rho

	###############################################################################################

	def densityDerivativeLin(self, r):
		"""
		The linear derivative of the density due to the outer term, :math:`d \\rho / dr`. 

		This function should be overwritten by child classes if there is an analytic, faster 
		expression for the derivative.

		Parameters
		----------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.

		Returns
		-------
		derivative: array_like
			The linear derivative in physical :math:`M_{\\odot} h / {\\rm kpc}^2`; has the same 
			dimensions as ``r``.
		"""

		# The derivative functions have changed between scipy versions, so we need to accommodate
		# old and new versions. For the new version, we evaluate the derivative element by element 
		# because the scipy function sometimes (?) evaluates multiple test radii per input, which 
		# can lead to 2D arrays, for which the density function may not be prepared.
		r_use, is_array = utilities.getArray(r)
		r_use = r_use.astype(float)
		
		if HAS_SCIPY_DIFF:
			rho_der = np.zeros_like(r_use)
			for i in range(len(r_use)):
				res = scipy.differentiate.derivative(self.density, r_use[i], 
													initial_step = 0.001, preserve_shape = True)
				rho_der[i] = res['df']
		else:
			rho_der = scipy.misc.derivative(self.density, r_use, dx = 0.001, n = 1, order = 3)
		
		if not is_array:
			rho_der = rho_der[0]
			
		return rho_der

	###############################################################################################

	def update(self):
		"""
		Update the profile object after a change in parameters or cosmology.
		"""
		
		return

###################################################################################################
# OUTER TERM: MEAN DENSITY
###################################################################################################

class OuterTermMeanDensity(OuterTerm):
	"""
	An outer term that adds the mean matter density of the universe to a density profile.
	
	This is perhaps the simplest outer term one can imagine. The only parameter is the redshift at
	which the halo density profile is modeled. Note that this term is cosmology-dependent, meaning
	that a cosmology has to be set before the constructor is called.
	
	Furthermore, note that a constant term such as this one means that the surface density cannot
	be evaluated any more, since the integral over density will diverge. If the surface density
	is to be evaluated, one should always remove constant outer terms from the profile. This 
	class does overwrite the surface density function and issues a warning if it is called.
	
	In this implementation, the redshift is added to the profile options rather than parameters,
	meaning that it cannot be varied in a fit.

	Parameters
	----------
	z: float
		The redshift at which the profile is modeled.
	"""

	def __init__(self, z = None, **kwargs):
		
		if z is None:
			raise Exception('Redshift cannot be None.')
		
		OuterTerm.__init__(self, [], [z], [], ['z'])
		
		self.initialized = False

		return

	###############################################################################################

	def _getParameters(self):

		z = self.opt['z']
		
		return z

	###############################################################################################

	def update(self):

		z = self._getParameters()
		cosmo = cosmology.getCurrent()
		self.rho_m = cosmo.rho_m(z)
		
		return

	###############################################################################################

	def _density(self, r):

		if not self.initialized:
			self.update()
		
		rho = np.ones((len(r)), float) * self.rho_m
		
		return rho

	###############################################################################################

	def surfaceDensity(self, r):
		"""
		The projected surface density at radius r due to the outer profile.

		This function is overwritten for the mean density outer profile because it is ill-defined:
		as the mean density is constant out to infinite radii, the line-of-sight integral 
		diverges. In principle, this function could just return zero in order to ignore this 
		spurious contribution, but that causes an inconsistency between the 3D (rho) and 2D 
		(Sigma) density profiles.

		Parameters
		----------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
			
		Returns
		-------
		Sigma: array_like
			An array of zeros.
		"""
		
		warnings.warn('Ignoring surface density of mean-density outer profile. This term should be removed before evaluating the surface density.')
		
		return r * 0.0

###################################################################################################
# OUTER TERM: HALO-MATTER CORRELATION FUNCTION
###################################################################################################

class OuterTermCorrelationFunction(OuterTerm):
	"""
	An outer term that adds an estimate based on the matter-matter correlation function.

	On large scales, we can model the 2-halo term, i.e., the excess density due to neighboring
	halos, by assuming a linear bias. In that case, the halo-matter correlation function is a 
	multiple of the matter-matter correlation function, independent of radius:
	
	.. math::
		\\rho(r) = \\rho_{\\rm m} \\times b(\\nu) \\times \\xi_{\\rm mm}
		
	where :math:`b(\\nu)` is called the halo bias. Note that this implementation does not add the 
	constant due to the mean density of the universe which is sometimes included. If desired, this
	contribution can be added with the :class:`OuterTermMeanDensity` term.
	
	The bias should be initialized to a physically motivated value. This value can be calculated
	self-consistently, but this needs to be done iteratively because the bias depends on mass, 
	which circularly depends on the value of bias due to the inclusion of this outer term. Thus, 
	creating such a profile can be very slow. See the :mod:`~lss.bias` module for models of the
	bias as a function of halo mass.

	In this implementation, the redshift is added to the profile options rather than parameters,
	meaning it cannot be varied in a fit. The halo bias (i.e., the normalization of this outer 
	term) is a parameter though and can be varied in a fit. 
	
	Note that this outer term can be evaluated at radii outside the range where the correlation
	function is defined by the cosmology module without throwing an error or warning. In such 
	cases, the return value is the correlation function at the min/max radius. This behavior is
	convenient when initializing profiles etc, where the outer term may be insignificant at 
	some radii. However, when integrating this outer term (e.g., when evaluating the surface 
	density), care must be taken to set the correct integration limits. See the documentation of 
	the correlation function in the cosmology module for more information.
	
	Also note that the correlation function inevitably becomes negative at some radius! This can
	lead to a number of errors, for example, when computing surface density, where the density
	profile is integrated to a large radius. These errors can be prevented by manually limiting
	this integration depth.
	
	Parameters
	----------
	z: float
		The redshift at which the profile is modeled.
	derive_bias_from: str or None
		If ``None``, the bias is passed through the bias parameter and added to the profile 
		parameters. If ``derive_bias_from`` is a string, it must correspond to a profile parameter 
		or option. Furthermore, this parameter or option must represent a valid spherical overdensity 
		mass or radius such as ``'R200m'`` or ``'Mvir'`` from which the bias can be computed. If set
		to ``'R200m'``, the bias is automatically updated when the profile is changed.
	bias: float
		The halo bias.
	bias_name: str
		The internal name of the bias parameter. If this name is set to an already existing
		profile parameter, the bias is set to this other profile parameter, and thus not an
		independent parameter any more.
	"""

	def __init__(self, z = None, derive_bias_from = None, bias = None, bias_name = 'bias', **kwargs):
		
		if z is None:
			raise Exception('Redshift cannot be None.')
		
		par_array = []
		opt_array = [z, derive_bias_from, bias]
		par_names = []
		opt_names = ['z', 'derive_bias_from', bias_name]
		
		if derive_bias_from is None:
			if bias is None:
				raise Exception('Bias cannot be None if derive_bias_from is None.')
			par_array.append(bias)
			par_names.append(bias_name)
			self._derive_bias = False
		else:
			if bias is None:
				bias = 0.0
			_rm, self._bias_from_mdef, _, _ = mass_so.parseRadiusMassDefinition(derive_bias_from)
			self._bias_from_radius = (_rm == 'R')
			self._derive_bias = True
			self._rm_bias_name = derive_bias_from
			
			if derive_bias_from == 'R200m':
				opt_array.append(None)
				opt_names.append('R200m')
				
		OuterTerm.__init__(self, par_array, opt_array, par_names, opt_names)
		
		self.initialized = False
		
		return

	###############################################################################################

	def _getParameters(self):

		z = self.opt['z']

		if self._derive_bias:
			if self._rm_bias_name in self.par:
				rm_bias = self.par[self._rm_bias_name]
			elif self._rm_bias_name in self.opt:
				rm_bias = self.opt[self._rm_bias_name]
			else:
				raise Exception('Could not find the parameter or option "%s".' % (self._rm_bias_name))

			if self._bias_from_radius:
				rm_bias = mass_so.R_to_M(rm_bias, z, self._bias_from_mdef)
			
			bias = halo_bias.haloBias(rm_bias, z, self._bias_from_mdef)
			
		else:
			bias = self.par[self.term_par_names[0]]
		
		return z, bias

	###############################################################################################

	def update(self):

		z, _ = self._getParameters()
		cosmo = cosmology.getCurrent()
		
		self.rho_m = cosmo.rho_m(z)
		self.D2 = cosmo.growthFactor(z)**2
		
		self.xi_interp = cosmo._correlationFunctionInterpolator(defaults.PS_ARGS)
		self.R_min = cosmo.R_xi[0] * 1.00001
		self.R_max = cosmo.R_xi[-1] * 0.99999
		
		return

	###############################################################################################

	# We have to be a little careful when evaluating the matter-matter correlation function, since
	# it may not be defined at very small or large radii.

	def _density(self, r):

		if not self.initialized:
			self.update()
		
		z, bias = self._getParameters()
		r_com = r / 1000.0 * (1 + z)

		if np.any(r_com < self.R_min) or np.any(r_com > self.R_max):
			mask_not_small = (r_com > self.R_min)
			mask_not_large = (r_com < self.R_max)
			mask = (mask_not_small & mask_not_large)
			xi = np.zeros((len(r)), float)
			xi[mask] = self.xi_interp(r_com[mask])
			xi[np.logical_not(mask_not_small)] = self.xi_interp(self.R_min)
			xi[np.logical_not(mask_not_large)] = self.xi_interp(self.R_max)
		else:
			xi = self.xi_interp(r_com)
			
		xi *= self.D2
		rho = self.rho_m * bias * xi

		return rho

###################################################################################################
# OUTER TERM: POWER LAW
###################################################################################################

class OuterTermPowerLaw(OuterTerm):
	"""
	An outer term that describes density as a power-law in radius. 
	
	This class implements a power-law outer profile with a free normalization and slope, and a 
	fixed or variable pivot radius,
	
	.. math::
		\\rho(r) = \\frac{a \\times \\rho_{\\rm m}(z)}{\\frac{1}{m} + \\left(\\frac{r}{r_{\\rm pivot}}\\right)^{b}}
		
	where a is the normalization in units of the mean density of the universe, b the slope, and m 
	the maximum contribution to the density this term can make. Without such a limit, sufficiently 
	steep power-law profiles can lead to a spurious density contribution at the halo center. Note 
	that the slope is inverted, i.e. that a more positive slope means a steeper profile.

	This outer profile is kept for backward compatibility; for new code, please use 
	:class:`OuterTermInfalling`, which fulfils the same function if the smoothness parameter is 
	set to its default, :math:`\\zeta = 1` (although the recommended value is now 
	:math:`\\zeta = 0.5`).

	Parameters
	----------
	norm: float
		The density normalization of the term, in units of the mean matter density of the universe.
	slope: float
		The slope of the power-law profile.
	pivot: str
		There are fundamentally two ways to set the pivot radius. If ``pivot=='fixed'``, 
		``pivot_factor`` gives the pivot radius in physical kpc/h. Otherwise, ``pivot`` must 
		indicate the name of a profile parameter or option. In this case, the pivot radius is set to 
		``pivot_factor`` times the parameter or option in question. For example, for profiles based 
		on a scale radius, a pivot radius of :math:`2 r_s` can be set by passing ``pivot = 'rs'`` 
		and ``pivot_factor = 2.0``. However, only setting the pivot to ``R200m`` ensures that the 
		profile is kept consistent.
	pivot_factor: float
		See above.
	z: float
		Redshift.
	max_rho: float
		The maximum density in units of the normalization times the mean density of the universe.
		This limit prevents spurious density contributions at the very center of halos. If you are
		unsure what this parameter should be set to, it can be useful to plot the density 
		contribution of the outer profile term. It should flatten to max_rho times norm times the 
		mean density at a radius where the inner profile strongly dominates the density, i.e. 
		where the contribution from the outer term does not matter.
	norm_name: str
		The internal name of the normalization parameter. If this name is set to an already existing
		profile parameter, the normalization is set to this other profile parameter, and thus not an
		independent parameter any more.
	"""
	
	def __init__(self, norm = None, slope = None, pivot = 'R200m', pivot_factor = 5.0, 
				z = None, max_rho = defaults.HALO_PROFILE_OUTER_PL_MAXRHO, **kwargs):

		if norm is None:
			raise Exception('Normalization of power law cannot be None.')
		if slope is None:
			raise Exception('Slope of power law cannot be None.')
		if pivot is None:
			raise Exception('Pivot of power law cannot be None.')
		if pivot_factor is None:
			raise Exception('Pivot factor of power law cannot be None.')
		if z is None:
			raise Exception('Redshift of power law cannot be None.')
		if max_rho is None:
			raise Exception('Maximum of power law cannot be None.')
		
		par_array = [norm, slope]
		opt_array = [pivot, pivot_factor, z, max_rho]
		par_names = ['norm', 'slope']
		opt_names = ['pivot', 'pivot_factor', 'z', 'max_rho']
		
		if pivot == 'R200m':
			opt_array.append(None)
			opt_names.append('R200m')
		
		OuterTerm.__init__(self, par_array, opt_array, par_names, opt_names)

		return

	###############################################################################################

	def _getParameters(self):

		r_pivot_id = self.opt['pivot']
		if r_pivot_id == 'fixed':
			r_pivot = 1.0
		elif r_pivot_id in self.par:
			r_pivot = self.par[r_pivot_id]
		elif r_pivot_id in self.opt:
			r_pivot = self.opt[r_pivot_id]
		else:
			raise Exception('Could not find the parameter or option "%s".' % (r_pivot_id))

		if r_pivot is None:
			raise Exception('Outer profile was trying to use the internal radius %s, but found None.' \
						% (r_pivot_id))

		norm = self.par['norm']
		slope = self.par['slope']
		r_pivot *= self.opt['pivot_factor']
		z = self.opt['z']
		max_rho = self.opt['max_rho']
		rho_m = cosmology.getCurrent().rho_m(z)
		
		return norm, slope, r_pivot, max_rho, rho_m

	###############################################################################################

	def _density(self, r):
		
		norm, slope, r_pivot, max_rho, rho_m = self._getParameters()
		rho = rho_m * norm / (1.0 / max_rho + (r / r_pivot)**slope)

		return rho

	###############################################################################################

	def densityDerivativeLin(self, r):
		"""
		The linear derivative of the density due to the outer term, :math:`d \\rho / dr`. 

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

		norm, slope, r_pivot, max_rho, rho_m = self._getParameters()
		t1 = (r / r_pivot)**slope
		drho_dr = -rho_m * norm * slope * t1 / r / (1.0 / max_rho + t1)**2

		return drho_dr

	###############################################################################################

	def _fitParamDeriv_rho(self, r, mask, N_par_fit):
		
		deriv = np.zeros((N_par_fit, len(r)), float)
		norm, slope, r_pivot, max_rho, rho_m = self._getParameters()
		
		rro = r / r_pivot
		t1 = 1.0 / max_rho + rro**slope
		rho = rho_m * norm / t1
		
		counter = 0
		# norm
		if mask[0]:
			deriv[counter] = rho / norm
			counter += 1
		# slope
		if mask[1]:
			deriv[counter] = -rho * np.log(rro) / t1 * rro**slope
		
		return deriv

###################################################################################################
# OUTER TERM: POWER LAW
###################################################################################################

class OuterTermInfalling(OuterTerm):
	"""
	Infalling term according to Diemer 2023, modeled as a power law with smooth transition.
	
	This class implements a power-law outer profile with a free normalization and slope, a 
	fixed or variable pivot radius, and a smooth transition to a maximum value at small radii,
	
	.. math::
		\\rho(r) = \\delta_1 \\rho_{\\rm m}(z) \\left[ \\left( \\frac{\\delta_1}{\\delta_{\\rm max}} \\right)^{1/\\zeta} + \\left( \\frac{r}{r_{\\rm pivot}} \\right)^{s/\\zeta} \\right]^{-\\zeta} 
		
	where :math:`\\delta_1` is the normalization in units of the mean density of the universe, 
	:math:`s` is the slope, :math:`\\delta_{\\rm max}` is the maximum overdensity at the center of
	the halo, and :math:`\\zeta` determines how rapidly the profile transitions to this density.
	Note that a more positive slope means a steeper profile. By default, :math:`\\zeta = 0.5`.
	In the formulation of Diemer 2023, the pivot radius is :math:`R_{\\rm 200m}`; other radii
	can be chosen but then the profile is not automatically kept up to date if the parameters of
	the inner profile change.
	
	Parameters
	----------
	pl_delta_1: float
		The normalization of the infalling profile at the pivot radius (R200m by default) in units 
		of the mean matter density of the universe.
	pl_s: float
		The (negative) slope of the power-law profile.
	pl_zeta: float
		The smoothness of the transition to the asymptotic value at the halo center.
	pl_delta_max: float
		The asymptotic overdensity at the center of the halo, in units of the mean matter density 
		of the universe.
	pivot: str
		There are fundamentally two ways to set the pivot radius. If ``pivot=='fixed'``, 
		``pivot_factor`` gives the pivot radius in physical kpc/h. Otherwise, ``pivot`` must 
		indicate the name of a profile parameter or option. In this case, the pivot radius is set to 
		``pivot_factor`` times the parameter or option in question. For example, for profiles based 
		on a scale radius, a pivot radius of :math:`2 r_s` can be set by passing ``pivot = 'rs'`` 
		and ``pivot_factor = 2.0``. However, only setting the pivot to ``R200m`` ensures that the 
		profile is kept consistent. When fitting, a fixed pivot radius is recommended because even
		R200m is not updated with every iteration in a fit, leading to inconsistencies.
	pivot_factor: float
		See above.
	z: float
		Redshift.
	"""
	
	def __init__(self, pl_delta_1 = None, pl_s = None, pl_zeta = defaults.HALO_PROFILE_OUTER_D22_ZETA, 
				pl_delta_max = defaults.HALO_PROFILE_OUTER_D22_DELTA_MAX, 
				pivot = 'R200m', pivot_factor = 1.0, z = None, **kwargs):

		if pl_delta_1 is None:
			raise Exception('Normalization of power law (delta_1) cannot be None.')
		if pl_s is None:
			raise Exception('Slope of power law (s) cannot be None.')
		if pl_zeta is None:
			raise Exception('Sharpness of transition (zeta) cannot be None.')
		if pl_delta_max is None:
			raise Exception('Maximum overdensity cannot be None.')
		if pivot is None:
			raise Exception('Pivot of power law cannot be None.')
		if pivot_factor is None:
			raise Exception('Pivot factor of power law cannot be None.')
		if z is None:
			raise Exception('Redshift of power law cannot be None.')

		par_array = [pl_delta_1, pl_s, pl_zeta, pl_delta_max]
		opt_array = [pivot, pivot_factor, z]
		par_names = ['pl_delta_1', 'pl_s', 'pl_zeta', 'pl_delta_max']
		opt_names = ['pivot', 'pivot_factor', 'z']
		
		if pivot == 'R200m':
			opt_array.append(None)
			opt_names.append('R200m')
		
		OuterTerm.__init__(self, par_array, opt_array, par_names, opt_names)

		return

	###############################################################################################

	def _getParameters(self):
		
		r_pivot_id = self.opt['pivot']
		if r_pivot_id == 'fixed':
			r_pivot = 1.0
		elif r_pivot_id in self.par:
			r_pivot = self.par[r_pivot_id]
		elif r_pivot_id in self.opt:
			r_pivot = self.opt[r_pivot_id]
		else:
			raise Exception('Could not find the parameter or option "%s".' % (r_pivot_id))

		delta_1 = self.par['pl_delta_1']
		s = self.par['pl_s']
		zeta = self.par['pl_zeta']
		delta_max = self.par['pl_delta_max']
		
		r_pivot *= self.opt['pivot_factor']
		z = self.opt['z']
		rho_m = cosmology.getCurrent().rho_m(z)
		
		return delta_1, s, zeta, delta_max, r_pivot, rho_m

	###############################################################################################

	def _density(self, r):
		
		delta_1, s, zeta, delta_max, r_pivot, rho_m = self._getParameters()
		rho = rho_m * delta_1 * ((delta_1 / delta_max)**(1.0 / zeta) + (r / r_pivot)**(s / zeta))**(-zeta)

		return rho

	###############################################################################################

	def densityDerivativeLin(self, r):
		"""
		The linear derivative of the density due to the outer term, :math:`d \\rho / dr`. 

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

		delta_1, s, zeta, delta_max, r_pivot, rho_m = self._getParameters()
		
		t1 = (r / r_pivot)**(s / zeta)
		Q = (delta_1 / delta_max)**(1.0 / zeta) + t1
		
		rho = rho_m * delta_1 * Q**(-zeta)
		drho_dr = -(rho / r) * s / Q * t1

		return drho_dr

	###############################################################################################

	def _fitParamDeriv_rho(self, r, mask, N_par_fit):

		deriv = np.zeros((N_par_fit, len(r)), float)
		delta_1, s, zeta, delta_max, r_pivot, rho_m = self._getParameters()

		zeta_inv = 1.0 / zeta
		rr = r / r_pivot
		rrsz = rr**(s * zeta_inv)
		dd = delta_1 / delta_max
		dd1z = dd**zeta_inv
		Q = dd1z + rrsz
		rho = rho_m * delta_1 * Q**-zeta
		dd1zq = dd1z / Q
		logrr = np.log(rr)

		counter = 0
		# delta1
		if mask[0]:
			deriv[counter] = rho * (1.0 - dd1zq)
			counter += 1
		# s
		if mask[1]:
			deriv[counter] = -rho * s / Q * rrsz * logrr
			counter += 1
		# deltamax
		if mask[2]:
			deriv[counter] = rho * dd1zq
			counter += 1
		# zeta
		if mask[3]:
			deriv[counter] = rho * np.log(Q) / zeta * (dd1z * np.log(dd) + s * rrsz * logrr)
		
		return deriv

	###############################################################################################
