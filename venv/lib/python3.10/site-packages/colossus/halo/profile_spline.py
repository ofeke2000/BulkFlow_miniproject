###################################################################################################
#
# profile_spline.py         (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This module implements a general density profile using spline interpolation. Please see 
:doc:`halo_profile` for a general introduction to the Colossus density profile module.
	
---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

This general profile is initialized with an arbitrary array of radii and densities or enclosed 
masses as input, and interpolates them using a splines (in log space). Note that there are three 
different ways of specifying the density profile:

* density and mass: Both density and mass are interpolated using splines.
* density only: In order for the enclosed mass to be defined, the density must be specified 
  all the way to r = 0. In that case, the mass is computed numerically, stored, and interpolated.
* mass only: The density is computed as the derivative of the mass, stored, and interpolated.

In the following example, we create an NFW profile and use its density to initialize a spline 
profile::

	from colossus.cosmology import cosmology
	from colossus.halo import mass_so
	from colossus.halo import profile_spline
	from colossus.halo import profile_nfw

	Mvir = 1E12
	c = 10.0
	mdef = 'vir'
	z = 0.0
	cosmology.setCosmology('planck15')
	Rvir = mass_so.M_to_R(Mvir, z, mdef)
	
	p_nfw = profile_nfw.NFWProfile(M = Mvir, c = c, mdef = mdef, z = z)
	r = 10**np.arange(-2.0, 1.0, 0.02) * Rvir
	rho = p_nfw.density(r)
	
	p_spline = profile_spline.SplineProfile(r, rho = rho)

Of course, in this case it would be better to work with the NFW profile directly. Please see the 
:doc:`tutorials` for more code examples.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

import numpy as np
import scipy.integrate
import scipy.interpolate

from colossus import defaults
from colossus.halo import profile_base

###################################################################################################
# SPLINE DEFINED PROFILE
###################################################################################################

class SplineProfile(profile_base.HaloDensityProfile):
	"""
	An arbitrary density profile using spline interpolation.
	
	Parameters
	----------
	r: numpy array
		Radii in physical kpc/h.
	rho: array_like
		Density at radii r in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`. Does not have to be 
		passed as long as ``M`` is passed.
	M: array_like
		Enclosed mass within radii r in :math:`M_{\\odot} / h`. Does not have to be passed
		as long as ``rho`` is passed.
	spline_order: int
		The order of the spline used. By default, a cubic spline (order 3) is used, but such 
		splines can lead to ringing, especially if there are oscillating elements in the input
		mass or density array. In such cases, it can be helpful to set ``spline_order = 1``.

	Warnings
	--------
	If both mass and density are supplied to the constructor, the consistency between the two is 
	not checked! 
	"""
	
	###############################################################################################
	# CONSTRUCTOR
	###############################################################################################
	
	def __init__(self, r, rho = None, M = None, spline_order = 3):
		
		self.par_names = []
		self.opt_names = []
		profile_base.HaloDensityProfile.__init__(self, ignore_params = True)
		
		self.rmin = np.min(r)
		self.rmax = np.max(r)
		self.r_guess = np.sqrt(self.rmin * self.rmax)
		self.min_RDelta = self.rmin
		self.max_RDelta = self.rmax

		if rho is None and M is None:
			raise Exception('Either mass or density must be specified.')
		
		self.rho_spline = None
		self.M_spline = None
		if np.any(r <= 0.0):
			raise Exception('Radius may not contain negative or zero elements.')
		logr = np.log(r)
		
		if M is not None:
			if np.any(np.isnan(M)):
				raise Exception('Mass array may not contain nan elements.')
			if np.any(M < 0.0):
				raise Exception('Mass array may not contain negative elements.')
			if np.any(np.diff(M) <= 0.0):
				raise Exception('Mass array must be strictly increasing.')
			if np.any(M < 1E-20):
				print('Warning: mass array passed to spline profile contains very small or zero numbers.')
				M[M < 1E-20] = 1E-20
			logM = np.log(M)
			self.M_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logM, k = spline_order)

		if rho is not None:
			if np.any(np.isnan(rho)):
				raise Exception('Density array may not contain nan elements.')
			if np.any(rho < 0.0):
				raise Exception('Density may not contain negative elements.')
			if np.any(rho < 1E-20):
				print('Warning: density array passed to spline profile contains very small or zero numbers.')
				rho[rho < 1E-20] = 1E-20
			logrho = np.log(rho)
			self.rho_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logrho, k = spline_order)

		# Construct M(r) from density. For some reason, the spline integrator fails on the 
		# innermost bin, and the quad integrator fails on the outermost bin. 
		if self.M_spline is None:
			integrand = 4.0 * np.pi * r**2 * rho
			integrand_spline = scipy.interpolate.InterpolatedUnivariateSpline(r, integrand, k = spline_order)
			logM = 0.0 * r
			for i in range(len(logM) - 1):
				logM[i], _ = scipy.integrate.quad(integrand_spline, 0.0, r[i])
			logM[-1] = integrand_spline.integral(0.0, r[-1])
			logM = np.log(logM)
			self.M_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logM, k = spline_order)

		if self.rho_spline is None:
			deriv = self.M_spline(np.log(r), nu = 1) * M / r
			if np.any(deriv <= 0.0):
				raise Exception('Derivative of mass array contains zero or negative elements.')
			logrho = np.log(deriv / 4.0 / np.pi / r**2)
			if np.any(np.isnan(logrho)):
				raise Exception('Found nan in density array.')
			self.rho_spline = scipy.interpolate.InterpolatedUnivariateSpline(logr, logrho, k = spline_order)

		return

	###############################################################################################

	def setNativeParameters(self, M, c, z, mdef, **kwargs):
		return 
	
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
				
		return np.exp(self.rho_spline(np.log(r)))

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

		log_deriv = self.rho_spline(np.log(r), nu = 1)
		deriv = log_deriv * self.density(r) / r
		
		return deriv

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
	
		return self.rho_spline(np.log(r), nu = 1)
	
	###############################################################################################

	def enclosedMass(self, r, accuracy = defaults.HALO_PROFILE_ENCLOSED_MASS_ACCURACY):
		"""
		The mass enclosed within radius r.

		Parameters
		----------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
		accuracy: float
			The minimum accuracy of the integration.
			
		Returns
		-------
		M: array_like
			The mass enclosed within radius r, in :math:`M_{\\odot}/h`; has the same dimensions as 
			``r``.
		"""		

		return self.enclosedMassInner(r) + self.enclosedMassOuter(r, accuracy)

	###############################################################################################

	def enclosedMassInner(self, r):
		"""
		The mass enclosed within radius r due to the inner profile term.

		Parameters
		----------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
			
		Returns
		-------
		M: array_like
			The mass enclosed within radius r, in :math:`M_{\\odot}/h`; has the same dimensions as r.
		"""		

		return np.exp(self.M_spline(np.log(r)))

###################################################################################################
