###################################################################################################
#
# profile_nfw.py                (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This module implements the Navarro-Frenk-White form of the density profile. Please see 
:doc:`halo_profile` for a general introduction to the Colossus density profile module.

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

The NFW profile (`Navarro et al. 1997 <http://adsabs.harvard.edu/abs/1997ApJ...490..493N>`__) is
defined by the density function
	
.. math::
	\\rho(r) = \\frac{\\rho_s}{\\left(\\frac{r}{r_{\\rm s}}\\right) \\left(1 + \\frac{r}{r_s}\\right)^{2}}

The profile class can be initialized by either passing its fundamental parameters 
:math:`\\rho_{\\rm s}` and :math:`r_{\\rm s}`, but the more convenient initialization is via mass 
and concentration::

	from colossus.cosmology import cosmology
	from colossus.halo import profile_nfw
	
	cosmology.setCosmology('planck18')
	p_nfw = profile_nfw.NFWProfile(M = 1E12, c = 10.0, z = 0.0, mdef = 'vir')

The NFW profile class is optimized by using analytical expressions instead of numerical
calculations wherever possible. The :func:`radiusFromPdf` function covers the common case of 
drawing random radial positions given an NFW profile.

Please see the :doc:`tutorials` for more code examples.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

import numpy as np
import scipy.optimize
import scipy.interpolate
import warnings

from colossus.utils import utilities
from colossus.halo import mass_so
from colossus.halo import profile_base

###################################################################################################
# Global variables for x-equation lookup table
###################################################################################################

x_interpolator = None
x_interpolator_min = None
x_interpolator_max = None

###################################################################################################
# NFW PROFILE
###################################################################################################

class NFWProfile(profile_base.HaloDensityProfile):
	"""
	The Navarro-Frenk-White profile.
	
	The constructor accepts either the free parameters in this formula, central density and scale 
	radius, or a spherical overdensity mass and concentration (in this case the mass definition 
	and redshift also need to be specified, and a cosmology needs to be set). The density and other 
	commonly used routines are implemented both as class and as static routines, meaning they can 
	be called without instantiating the class.

	Parameters
	----------
	rhos: float
		The central density in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`.
	rs: float
		The scale radius in physical kpc/h.
	M: float
		A spherical overdensity mass in :math:`M_{\\odot}/h` corresponding to the mass
		definition ``mdef`` at redshift ``z``. 
	c: float
		The concentration, :math:`c = R / r_{\\rm s}`, corresponding to the given halo mass and
		mass definition.
	z: float
		Redshift
	mdef: str
		The mass definition in which ``M`` and ``c`` are given. See :doc:`halo_mass` for details.
	"""
	
	###############################################################################################
	# CONSTRUCTOR
	###############################################################################################

	def __init__(self, **kwargs):
		
		self.par_names = ['rhos', 'rs']
		self.opt_names = []
		
		profile_base.HaloDensityProfile.__init__(self, **kwargs)
		
		# We need an initial radius to guess Rmax. Even though this quantity can be analytically
		# computed for an NFW profile, we might have outer terms.
		self.r_guess = self.par['rs']

		return

	###############################################################################################
	# STATIC METHODS
	###############################################################################################

	@classmethod
	def nativeParameters(cls, M, c, z, mdef):
		"""
		The native NFW parameters, :math:`\\rho_s` and :math:`r_{\\rm s}`, from mass and 
		concentration.
		
		This routine is called in the constructor of the NFW profile class (unless :math:`\\rho_s` 
		and :math:`r_{\\rm s}` are passed by the user), but can also be called without 
		instantiating an NFWProfile object.
	
		Parameters
		----------
		M: array_like
			Spherical overdensity mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
		c: array_like
			The concentration, :math:`c = R / r_{\\rm s}`, corresponding to the given halo mass and 
			mass definition; must have the same dimensions as ``M``.
		z: float
			Redshift
		mdef: str
			The mass definition in which ``M`` and ``c`` are given. See :doc:`halo_mass` for 
			details.
			
		Returns
		-------
		rhos: array_like
			The central density in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`; has the same 
			dimensions as ``M``.
		rs: array_like
			The scale radius in physical kpc/h; has the same dimensions as ``M``.
		"""
		
		rs = mass_so.M_to_R(M, z, mdef) / c
		rhos = M / rs**3 / 4.0 / np.pi / cls.mu(c)
		
		return rhos, rs

	###############################################################################################

	@classmethod
	def fundamentalParameters(cls, M, c, z, mdef):
		
		warnings.warn('The function NFWProfile.fundamentalParameters is deprecated and has been renamed to nativeParameters.')
		rhos, rs = cls.nativeParameters(M, c, z, mdef)
		
		return rhos, rs

	###############################################################################################

	def setNativeParameters(self, M, c, z, mdef, **kwargs):
		"""
		Set the native NFW parameters from mass and concentration.

		The NFW profile has :math:`\\rho_s` and :math:`r_{\\rm s}` as internal parameters, which 
		are computed from a mass and concentration. This function ignores the presence of outer 
		profiles.
	
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
		"""
		
		self.par['rhos'], self.par['rs'] = self.nativeParameters(M, c, z, mdef)
		
		return

	###############################################################################################

	@staticmethod
	def rho(rhos, x):
		"""
		The NFW density as a function of :math:`x = r/r_{\\rm s}`.
		
		This routine can be called without instantiating an NFWProfile object. In most cases, the 
		:func:`~halo.profile_base.HaloDensityProfile.density` function should be used instead.

		Parameters
		----------
		rhos: float
			The central density in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`.
		x: array_like
			The radius in units of the scale radius, :math:`x=r/r_{\\rm s}`; can be a number or a 
			numpy array.
		
		Returns
		-------
		rho: array_like
			Density in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`; has the same dimensions 
			as ``x``.

		See also
		--------
		halo.profile_base.HaloDensityProfile.density: Density as a function of radius.
		"""
		
		return rhos / x / (1.0 + x)**2
	
	###############################################################################################
	
	@staticmethod
	def mu(x):
		"""
		A function of :math:`x=r/r_{\\rm s}` that appears in the NFW enclosed mass.

		This routine can be called without instantiating an NFWProfile object.

		Parameters
		----------
		x: array_like
			The radius in units of the scale radius, :math:`x=r/r_{\\rm s}`; can be a number or 
			a numpy array.
		
		Returns
		-------
		mu: array_like
			Has the same dimensions as ``x``.

		See also
		--------
		M: The enclosed mass in an NFW profile as a function of :math:`x=r/r_{\\rm s}`.
		halo.profile_base.HaloDensityProfile.enclosedMass: The mass enclosed within radius r.
		"""
		
		return np.log(1.0 + x) - x / (1.0 + x)
	
	###############################################################################################

	@classmethod
	def M(cls, rhos, rs, x):
		"""
		The enclosed mass in an NFW profile as a function of :math:`x=r/r_{\\rm s}`.

		This routine can be called without instantiating an NFWProfile object. In most cases, the 
		:func:`~halo.profile_base.HaloDensityProfile.enclosedMass` function should be used instead.

		Parameters
		----------
		rhos: float
			The central density in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`.
		rs: float
			The scale radius in physical kpc/h.
		x: array_like
			The radius in units of the scale radius, :math:`x=r/r_{\\rm s}`; can be a number or a 
			numpy array.
		
		Returns
		-------
		M: array_like
			The enclosed mass in :math:`M_{\\odot}/h`; has the same dimensions as ``x``.

		See also
		--------
		mu: A function of :math:`x=r/r_{\\rm s}` that appears in the NFW enclosed mass.
		halo.profile_base.HaloDensityProfile.enclosedMass: The mass enclosed within radius r.
		"""
		
		return 4.0 * np.pi * rs**3 * rhos * cls.mu(x)

	###############################################################################################

	@classmethod
	def _thresholdEquationX(cls, x, rhos, density_threshold):
		
		return rhos * cls.mu(x) * 3.0 / x**3 - density_threshold

	###############################################################################################
	
	@classmethod
	def xDelta(cls, rhos, density_threshold):
		"""
		Find :math:`x=r/r_{\\rm s}` where the enclosed density has a particular value.
		
		This function is the basis for the RDelta routine, but can 
		be used without instantiating an NFWProfile object. This is preferable when the function 
		needs to be evaluated many times, for example when converting a large number of mass 
		definitions.
		
		The function uses an interpolation table that makes it orders of magnitude faster than 
		root finding (depending on the size of the ``rhos`` and ``density_threshold`` arrays).
		
		Parameters
		----------
		rhos: array_like
			The central density in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`; can be a number 
			or a numpy array.
		density_threshold: array_like
			The desired enclosed density threshold in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`. 
			This number can be generated from a mass definition and redshift using the 
			:func:`~halo.mass_so.densityThreshold` function. Can be a number or a numpy array,
			if both ``density_threshold`` and ``rhos`` are arrays, they must have the same 
			size.
		
		Returns
		-------
		x: array_like
			The radius in units of the scale radius, :math:`x=r/r_{\\rm s}`, where the enclosed 
			density reaches ``density_threshold``. Has the same dimensions as ``rhos`` and/or
			``density_threshold``.
		"""
	
		global x_interpolator
		global x_interpolator_min
		global x_interpolator_max
		
		# If the interpolator has not been created, create it. We could theoretically store the
		# table in persistent storage but creating the table is so fast that this makes little 
		# sense. The large number of evaluation points ensures a accuracy of better than 1E-7
		# in xDelta.
		if x_interpolator is None:
			table_x = np.logspace(4.0, -4.0, 1000)
			table_y = cls.mu(table_x) * 3.0 / table_x**3
			x_interpolator = scipy.interpolate.InterpolatedUnivariateSpline(table_y, table_x, k = 3)
			
			knots = x_interpolator.get_knots()
			x_interpolator_min = knots[0]
			x_interpolator_max = knots[-1]

		# Compute the density ratio that is used to look up x. If it is outside the interpolator's
		# range, throw an error.
		y = density_threshold / rhos
		
		if np.min(y) < x_interpolator_min:
			raise Exception('Requested overdensity %.2e cannot be evaluated for scale density %.2e, out of range.' \
						% (np.min(y), x_interpolator_min))
		if np.max(y) > x_interpolator_max:
			raise Exception('Requested overdensity %.2e cannot be evaluated for scale density %.2e, out of range.' \
						% (np.max(y), x_interpolator_max))
		
		x = x_interpolator(y)
		
		return x
		
	###############################################################################################
	# METHODS BOUND TO THE CLASS
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
	
		x = r / self.par['rs']
		density = self.rho(self.par['rhos'], x)
		
		return density

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
			dimensions as ``r``.
		"""

		x = r / self.par['rs']
		density_der = -self.par['rhos'] / self.par['rs'] * (1.0 / x**2 / (1.0 + x)**2 + 2.0 / x / (1.0 + x)**3)

		return density_der
	
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

		x = r / self.par['rs']
		density_der = -(1.0 + 2.0 * x / (1.0 + x))

		return density_der

	###############################################################################################

	def enclosedMassInner(self, r, accuracy = None):
		"""
		The mass enclosed within radius r due to the inner profile term.

		Parameters
		----------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
			
		Returns
		-------
		M: array_like
			The mass enclosed within radius r, in :math:`M_{\\odot}/h`; has the same dimensions as 
			``r``.
		"""		
		
		x = r / self.par['rs']
		mass = self.M(self.par['rhos'], self.par['rs'], x)
		
		return mass
	
	###############################################################################################
	
	# The surface density of an NFW profile can be computed analytically which is much faster than
	# integration. The formula below is taken from Bartelmann (1996). The case r = rs is solved in 
	# Lokas & Mamon (2001), but in their notation the density at this radius looks somewhat 
	# complicated. In the notation used here, Sigma(rs) = 2/3 * rhos * rs.
	
	def surfaceDensityInner(self, r, **kwargs):
		"""
		The projected surface density at radius r due to the inner profile.
		
		This function uses the analytical formula of 
		`Lokas & Mamon 2001 <http://adsabs.harvard.edu/abs/2001MNRAS.321..155L>`__ rather than 
		numerical integration.

		Parameters
		----------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
			
		Returns
		-------
		Sigma: array_like
			The surface density at radius r, in physical :math:`M_{\\odot} h/{\\rm kpc}^2`; has 
			the same dimensions as ``r``.
		"""
	
		xx = r / self.par['rs']
		x, is_array = utilities.getArray(xx)
		surfaceDensity = np.ones_like(x) * self.par['rhos'] * self.par['rs']
		
		# Solve separately for r < rs, r > rs, r = rs
		mask_rs = abs(x - 1.0) < 1E-4
		mask_lt = (x < 1.0) & (np.logical_not(mask_rs))
		mask_gt = (x > 1.0) & (np.logical_not(mask_rs))
		
		surfaceDensity[mask_rs] *= 2.0 / 3.0

		xi = x[mask_lt]		
		x2 = xi**2
		x2m1 = x2 - 1.0
		surfaceDensity[mask_lt] *= 2.0 / x2m1 \
			* (1.0 - 2.0 / np.sqrt(-x2m1) * np.arctanh(np.sqrt((1.0 - xi) / (xi + 1.0))))

		xi = x[mask_gt]		
		x2 = xi**2
		x2m1 = x2 - 1.0
		surfaceDensity[mask_gt] *= 2.0 / x2m1 \
			* (1.0 - 2.0 / np.sqrt(x2m1) * np.arctan(np.sqrt((xi - 1.0) / (xi + 1.0))))
			
		if not is_array:
			surfaceDensity = surfaceDensity[0]
	
		return surfaceDensity
	
	###############################################################################################
	
	# The differential mass surface density DeltaSigma according to the analytical expression of 
	# Wright & Brainerd 2000, Equation 13.
	
	def _deltaSigmaInner(self, r, **kwargs):
	
		xx = r / self.par['rs']
		x, is_array = utilities.getArray(xx)
		deltaSigma = np.ones_like(x) * 4.0 * self.par['rhos'] * self.par['rs']
		
		# Solve separately for r < rs, r > rs, r = rs
		mask_rs = abs(x - 1.0) < 1E-4
		mask_lt = (x < 1.0) & (np.logical_not(mask_rs))
		mask_gt = (x > 1.0) & (np.logical_not(mask_rs))
		
		deltaSigma[mask_rs] *= 1.0 + np.log(0.5)

		xi = x[mask_lt]
		x2 = xi**2
		x2m1 = x2 - 1.0
		deltaSigma[mask_lt] *= 1.0 / x2 \
			* (2.0 / np.sqrt(-x2m1) * np.arctanh(np.sqrt((1.0 - xi) / (xi + 1.0))) + np.log(0.5 * xi))

		xi = x[mask_gt]
		x2 = xi**2
		x2m1 = x2 - 1.0
		deltaSigma[mask_gt] *= 1.0 / x2 \
			* (2.0 / np.sqrt(x2m1) * np.arctan(np.sqrt((xi - 1.0) / (xi + 1.0))) + np.log(0.5 * xi))

		deltaSigma -= self.surfaceDensityInner(r)
		
		if not is_array:
			deltaSigma = deltaSigma[0]
	
		return deltaSigma

	###############################################################################################
	
	# For the NFW profile, rmax is a constant multiple of the scale radius since vc is maximized
	# where ln(1+x) = (2x**2 + x) / (1+x)**2. If there are outer terms, however, this does not
	# hold.
	
	def Vmax(self):
		"""
		The maximum circular velocity, and the radius where it occurs.
			
		Returns
		-------
		vmax: float
			The maximum circular velocity in km / s.
		rmax: float
			The radius where fmax occurs, in physical kpc/h.

		See also
		--------
		circularVelocity: The circular velocity, :math:`v_c \\equiv \\sqrt{GM(<r)/r}`.
		"""
		
		if self.N_outer == 0:
			rmax = 2.16258 * self.par['rs']
			vmax = self.circularVelocity(rmax)
		else:
			rmax, vmax = profile_base.HaloDensityProfile.__init__(self)
		
		return vmax, rmax

	###############################################################################################
	
	# This equation is 0 when the enclosed density matches the given density_threshold. This 
	# function matches the abstract interface in HaloDensityProfile, but for the NFW profile it is
	# easier to solve the equation in x (see the _thresholdEquationX() function). However, if there
	# are outer terms, we need to take those into account.
		
	def _thresholdEquation(self, r, density_threshold):

		if self.N_outer == 0:
			ret = self._thresholdEquationX(r / self.par['rs'], self.par['rhos'], density_threshold)
		else:
			ret = profile_base.HaloDensityProfile._thresholdEquation(self, r, density_threshold)
		
		return ret

	###############################################################################################

	# Return the spherical overdensity radius (in kpc / h) for a given mass definition and redshift. 
	# This function is overwritten for the NFW profile as we have a better guess at the resulting
	# radius, namely the scale radius. Thus, the user can specify a minimum and maximum concentra-
	# tion that is considered. If there are outer terms, we need to fall back to the general 
	# method.

	def RDelta(self, z, mdef):
		"""
		The spherical overdensity radius of a given mass definition.

		Parameters
		----------
		z: float
			Redshift
		mdef: str
			The mass definition for which the spherical overdensity radius is computed.
			See :doc:`halo_mass` for details.
			
		Returns
		-------
		R: float
			Spherical overdensity radius in physical kpc/h.

		See also
		--------
		MDelta: The spherical overdensity mass of a given mass definition.
		RMDelta: The spherical overdensity radius and mass of a given mass definition.
		"""		

		if self.N_outer == 0:
			density_threshold = mass_so.densityThreshold(z, mdef)
			x = self.xDelta(self.par['rhos'], density_threshold)
			R = x * self.par['rs']
		else:
			R = profile_base.HaloDensityProfile.RDelta(self, z, mdef)
	
		return R

	###############################################################################################

	def M4rs(self):
		"""
		The mass within 4 scale radii, :math:`M_{<4rs}`.
		
		This mass definition was suggested by 
		`More et al. 2015 <http://adsabs.harvard.edu/abs/2015ApJ...810...36M>`__, see the 
		:doc:`halo_mass_adv` section for details.

		Returns
		-------
		M4rs: float
			The mass within 4 scale radii, :math:`M_{<4rs}`, in :math:`M_{\\odot} / h`.
		"""
		
		M = self.enclosedMass(4.0 * self.par['rs'])
		
		return M
	
	###############################################################################################

	# Return and array of d rho / d ln(rhos) and d rho / d ln(rs), since parameters are fitted
	# in log space.
	
	def _fitParamDeriv_rho(self, r, mask, N_par_fit):

		x = self.getParameterArray()
		deriv = np.zeros((N_par_fit, len(r)), float)
		rrs = r / x[1]
		rho_r = x[0] / rrs / (1.0 + rrs) ** 2

		counter = 0
		if mask[0]:
			deriv[counter] = rho_r
			counter += 1
		if mask[1]:
			deriv[counter] = rho_r * rrs * (1.0 / rrs + 2.0 / (1.0 + rrs))
			
		return deriv

###################################################################################################
# OTHER FUNCTIONS
###################################################################################################

def radiusFromPdf(M, c, z, mdef, cumulativePdf,
					interpolate = True, min_interpolate_pdf = 0.01):
	"""
	Get the radius where the cumulative density distribution of a halo has a certain value, 
	assuming an NFW profile. 
	
	This function can be useful when assigning radii to satellite galaxies in mock halos, for 
	example. The function is optimized for speed when ``M`` is a large array. The density 
	distribution is cut off at the virial radius corresponding to the given mass 
	definition. For example, if ``mdef == 'vir'``, the NFW profile is cut off at 
	:math:`R_{\\rm vir}`. The accuracy achieved is about 0.2%, unless ``min_interpolate_pdf`` is 
	changed to a lower value; below 0.01, the accuracy of the interpolation drops.
	
	Parameters
	----------
	M: array_like
		Halo mass in units of :math:`M_{\\odot}/h`; can be a number or a numpy array.
	c: array_like
		Halo concentration, in the same definition as ``M``; must have the same dimensions as 
		``M``.
	z: float
		Redshift
	mdef: str
		The mass definition in which the halo mass ``M`` is given. See :doc:`halo_mass` for 
		details.
	cumulativePdf: array_like
		The cumulative pdf that we are seeking. If an array, this array needs to have the same 
		dimensions as the ``M`` array.
	c_model: str
		The model used to evaluate concentration if ``c == None``.
	interpolate: bool
		If ``interpolate == True``, an interpolation table is built before computing the radii. This 
		is much faster if ``M`` is a large array. 
	min_interpolate_pdf: float
		For values of the cumulativePdf that fall below this value, the radius is computed exactly,
		even if ``interpolation == True``. The reason is that the interpolation becomes unreliable
		for these very low pdfs. 
		
	Returns
	-------
	r: array_like
		The radii where the cumulative pdf(s) is/are achieved, in units of physical kpc/h; has the 
		same dimensions as ``M``.

	Warnings
	--------
		If many pdf values fall below ``min_interpolate_pdf``, this will slow the function
		down significantly.
	"""

	def equ(c, target):
		return NFWProfile.mu(c) - target
	
	def getX(c, p):
		
		target = NFWProfile.mu(c) * p
		x = scipy.optimize.brentq(equ, 0.0, c, args = target)
		
		return x
	
	M_array, is_array = utilities.getArray(M)
	M_array = M_array.astype(float)
	R = mass_so.M_to_R(M, z, mdef)
	N = len(M_array)
	x = np.zeros_like(M_array)
	c_array, _ = utilities.getArray(c)
	c_array = c_array.astype(float)
	p_array, _ = utilities.getArray(cumulativePdf)
	p_array = p_array.astype(float)
	
	if interpolate:

		# Create an interpolator on a regular grid in c-p space.
		bin_width_c = 0.1
		c_min = np.min(c_array) * 0.99
		c_max = np.max(c_array) * 1.01
		c_bins = np.arange(c_min, c_max + bin_width_c, bin_width_c)
		
		p_bins0 = np.arange(0.0, 0.01, 0.001)
		p_bins1 = np.arange(0.01, 0.1, 0.01)
		p_bins2 = np.arange(0.1, 1.1, 0.1)
		p_bins = np.concatenate((p_bins0, p_bins1, p_bins2))
		
		N_c = len(c_bins)
		N_p = len(p_bins)

		x_ = np.zeros((N_c, N_p), dtype = float)
		for i in range(N_c):			
			for j in range(N_p):
				p = p_bins[j]
				target = NFWProfile.mu(c_bins[i]) * p
				x_[i, j] = scipy.optimize.brentq(equ, 0.0, c_bins[i], args = target) / c_bins[i]
		
		spl = scipy.interpolate.RectBivariateSpline(c_bins, p_bins, x_)

		# For very small values, overwrite the interpolated values with the exact value.
		for i in range(N):
			if p_array[i] < min_interpolate_pdf:
				x[i] = getX(c_array[i], cumulativePdf[i]) / c_array[i]
			else:
				x[i] = spl(c_array[i], p_array[i])[0][0]

		r = R * x
	
	else:

		# A simple root-finding algorithm. 
		for i in range(N):
			x[i] = getX(c_array[i], cumulativePdf[i])
		r = R / c_array * x
			
	if not is_array:
		r = r[0]
	
	return r

###################################################################################################
