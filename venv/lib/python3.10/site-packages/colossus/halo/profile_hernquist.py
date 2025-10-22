###################################################################################################
#
# profile_hernquist.py      (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This module implements the profile form of Hernquist (1990). Please see :doc:`halo_profile` for a 
general introduction to the Colossus density profile module.

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

The Hernquist profile (`Hernquist 1990 <http://adsabs.harvard.edu/abs/1990ApJ...356..359H>`__) is 
defined by the density function

.. math::
	\\rho(r) = \\frac{\\rho_s}{\\left(\\frac{r}{r_s}\\right) \\left(1 + \\frac{r}{r_s}\\right)^{3}}

The profile class can be initialized by either passing its fundamental parameters 
:math:`\\rho_{\\rm s}` and :math:`r_{\\rm s}`, but the more convenient initialization is via mass 
and concentration::

	from colossus.cosmology import cosmology
	from colossus.halo import profile_hernquist
	
	cosmology.setCosmology('planck18')
	p_hernquist = profile_einasto.HernquistProfile(M = 1E12, c = 10.0, z = 0.0, mdef = 'vir')

Please see the :doc:`tutorials` for more code examples.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

import numpy as np
import warnings

from colossus.halo import mass_so
from colossus.halo import profile_base

###################################################################################################
# HERNQUIST PROFILE
###################################################################################################

class HernquistProfile(profile_base.HaloDensityProfile):
	"""
	The Hernquist profile.
	
	The constructor accepts either the free parameters in this formula, central density and scale 
	radius, or a spherical overdensity mass and concentration (in this case the mass definition 
	and redshift also need to be specified).

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
		The mass definition in which ``M`` and ``c``` are given. See :doc:`halo_mass` for details.
	"""

	###############################################################################################
	# CONSTRUCTOR
	###############################################################################################

	def __init__(self, **kwargs):
	
		self.par_names = ['rhos', 'rs']
		self.opt_names = []
		
		profile_base.HaloDensityProfile.__init__(self, **kwargs)

		# We need an initial radius to guess Rmax.
		self.r_guess = self.par['rs']
	
		return

	###############################################################################################

	@classmethod
	def nativeParameters(cls, M, c, z, mdef):
		"""
		The fundamental Hernquist parameters, :math:`\\rho_{\\rm s}` and :math:`r_{\\rm s}`, from 
		mass and concentration.
		
		This routine is called in the constructor of the Hernquist profile class (unless 
		:math:`\\rho_{\\rm s}` and :math:`r_{\\rm s}` are passed by the user), but can also be 
		called without instantiating a HernquistProfile object.
	
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
		rhos = M / (2 * np.pi * rs**3) / c**2 * (1.0 + c)**2
		
		return rhos, rs

	###############################################################################################

	@classmethod
	def fundamentalParameters(cls, M, c, z, mdef):
		
		warnings.warn('The function HernquistProfile.fundamentalParameters is deprecated and has been renamed to nativeParameters.')
		rhos, rs = cls.nativeParameters(M, c, z, mdef)
		
		return rhos, rs

	###############################################################################################

	def setNativeParameters(self, M, c, z, mdef, **kwargs):
		"""
		Set the native Hernquist parameters from mass and concentration.

		The Hernquist profile has :math:`\\rho_s` and :math:`r_{\\rm s}` as internal parameters, 
		which are computed from a mass and concentration. This function ignores the presence of 
		outer profiles.
	
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
		density = self.par['rhos'] / (x * (1.0 + x)**3)
		
		return density

	###############################################################################################

	def enclosedMassInner(self, r, accuracy = None):
		"""
		The mass enclosed within radius r due to the inner profile term.

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
		
		rs = self.par['rs']
		x = r / rs
		mass = 2 * np.pi * self.par['rhos'] * rs**3 * x**2 / (1.0 + x)**2
		
		return mass

###################################################################################################
