###################################################################################################
#
# profile_einasto.py        (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This module implements the Einasto form of the density profile. Please see 
:doc:`halo_profile` for a general introduction to the Colossus density profile module.

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

The Einasto profile (`Einasto 1965 <http://adsabs.harvard.edu/abs/1965TrAlm...5...87E>`__) is 
defined by the function

.. math::
	\\rho(r) = \\rho_s \\exp \\left( -\\frac{2}{\\alpha} \\left[ \\left( \\frac{r}{r_s} \\right)^{\\alpha} - 1 \\right] \\right)

or, alternatively, by a logarithmic slope that evolves with radius as 

.. math::
	\\frac{d \\log(\\rho)}{d \\log(r)} = -2 \\left( \\frac{r}{r_s} \\right)^{\\alpha}

The third parameter, :math:`\\alpha`, determines how quickly the profile steepens with slope. 
The Einasto profile class can be initialized from its fundamental parameters :math:`\\rho_{\\rm s}`, 
:math:`r_{\\rm s}`, and :math:`\\alpha`, but the more convenient initialization is via mass and 
concentration::

	from colossus.cosmology import cosmology
	from colossus.halo import profile_einasto
	
	cosmology.setCosmology('planck18')
	p_einasto = profile_einasto.EinastoProfile(M = 1E12, c = 10.0, z = 0.0, mdef = 'vir')

In this case, :math:`\\alpha` is determined automatically using the formula of 
`Gao et al. 2008 <http://adsabs.harvard.edu/abs/2008MNRAS.387..536G>`__, 

.. math::
	\\alpha = 0.155 + 0.0095 \\nu_{\\rm vir}^2

Please see the :doc:`tutorials` for more code examples.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

import numpy as np
import scipy.special

from colossus.lss import peaks
from colossus.halo import mass_so
from colossus.halo import profile_base
from colossus.halo import mass_defs

###################################################################################################
# EINASTO PROFILE
###################################################################################################

class EinastoProfile(profile_base.HaloDensityProfile):
	"""
	The Einasto 1965 density profile.

	The constructor accepts either the free parameters (the density at the scale radius, the scale 
	radius, and alpha), or a spherical overdensity mass and concentration (in this case the mass 
	definition and redshift also need to be specified). In the latter case, the user can specify 
	alpha or let the constructor compute it from its tight correlation with peak height 
	(`Gao et al. 2008 <http://adsabs.harvard.edu/abs/2008MNRAS.387..536G>`__); a cosmology must 
	be set before instantiating the profile object this way.

	Parameters
	----------
	rhos: float
		The density at the scale radius in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`.
	rs: float
		The scale radius in physical kpc/h.
	alpha: float
		The radial dependence of the profile slope.
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
	
		self.par_names = ['rhos', 'rs', 'alpha']
		self.opt_names = []
		
		profile_base.HaloDensityProfile.__init__(self, **kwargs)

		# We need an initial radius to guess Rmax
		self.r_guess = self.par['rs']
		
		# Pre-compute the mass terms now that the parameters have been fixed
		self._computeMassTerms()

		return
	
	###############################################################################################
	# METHODS BOUND TO THE CLASS
	###############################################################################################

	# Passing alpha is optional since it can also be estimated from the
	# Gao et al. 2008 relation between alpha and peak height. This relation was calibrated for
	# nu_vir, so if the given mass definition is not 'vir' we convert the given mass to Mvir
	# assuming an NFW profile with the given mass and concentration. This leads to a negligible
	# inconsistency, but solving for the correct alpha iteratively would be much slower.

	def setNativeParameters(self, M, c, z, mdef, alpha = None, **kwargs):
		"""
		The fundamental Einasto parameters, :math:`\\rho_{\\rm s}`, :math:`r_{\\rm s}`, and 
		:math:`\\alpha` from mass and concentration.
		
		This routine is called in the constructor of the Einasto profile class (unless 
		:math:`\\rho_{\\rm s}`, :math:`\\rho_{\\rm s}` and :math:`\\alpha` are passed by the 
		user), and cannot be called without instantiating an EinastoProfile object.
	
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
		alpha: float
			The radial dependence of the profile slope; can be ``None`` in which case it is 
			approximated.
			
		Returns
		-------
		rhos: float
			The density at the scale radius in physical :math:`M_{\\odot} h^2 / {\\rm kpc}^3`.
		rs: float
			The scale radius in physical kpc/h.
		alpha: float
			The radial dependence of the profile slope.
		"""

		R = mass_so.M_to_R(M, z, mdef)
		self.par['rs'] = R / c
		
		if alpha is None:
			if mdef == 'vir':
				Mvir = M
			else:
				Mvir, _, _ = mass_defs.changeMassDefinition(M, c, z, mdef, 'vir')
			nu_vir = peaks.peakHeight(Mvir, z)
			alpha = 0.155 + 0.0095 * nu_vir**2
		
		self.par['alpha'] = alpha
		self.par['rhos'] = 1.0
		self._computeMassTerms()
		self.par['rhos'] *= M / self.enclosedMassInner(R)
		
		return
	
	###############################################################################################

	def _computeMassTerms(self):
		
		self.mass_norm = np.pi * self.par['rs']**3 * 2.0**(2.0 - 3.0 / self.par['alpha']) \
			* self.par['alpha']**(-1.0 + 3.0 / self.par['alpha']) * np.exp(2.0 / self.par['alpha']) \
			* scipy.special.gamma(3.0 / self.par['alpha'])
		
		return
	
	###############################################################################################

	# The enclosed mass for the Einasto profile is semi-analytical, in that it can be expressed
	# in terms of Gamma functions. We pre-compute some factors to speed up the computation 
	# later.
	
	def update(self):
		"""
		Update the profile object after a change in parameters.
		
		If the parameters dictionary has been changed (e.g. by the user or during fitting), this 
		function must be called to ensure consistency within the profile object. This involves
		deleting any pre-computed quantities (e.g., tabulated enclosed masses) and re-computing
		profile properties that depend on the parameters.
		"""
		
		profile_base.HaloDensityProfile.update(self)
		self._computeMassTerms()
		
		return
	
	###############################################################################################

	# We need to overwrite the setParameterArray function because the mass terms need to be 
	# updated when the user changes the parameters. Note that we do not call the update function
	# which triggers an update of the parent class.
	
	def setParameterArray(self, pars, mask = None):
		"""
		Set the profile parameters from an array.
		
		The profile parameters are internally stored in an ordered dictionary. For some 
		applications (e.g., fitting), setting them directly from an array might be necessary. If 
		the profile contains values that depend on the parameters, the profile class must overwrite
		this function and update according to the new parameters.
		
		Parameters
		----------
		pars: array_like
			The new parameter array.
		mask: array_like
			Optional; must be a numpy array (not a list) of booleans, with the same length as the
			parameter vector of the profile class (profile.N_par). If passed, only those 
			parameters that correspond to ``True`` values are set (meaning the pars parameter must
			be shorter than profile.N_par).
		"""
		
		profile_base.HaloDensityProfile.setParameterArray(self, pars, mask = mask)
		self._computeMassTerms()
		
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
				
		rho = self.par['rhos'] * np.exp(-2.0 / self.par['alpha'] * \
										((r / self.par['rs'])**self.par['alpha'] - 1.0))
		
		return rho

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

		rho = self.densityInner(r)
		drho_dr = rho * (-2.0 / self.par['rs']) * (r / self.par['rs'])**(self.par['alpha'] - 1.0)	
		
		return drho_dr

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

		der = -2.0 * (r / self.par['rs'])**self.par['alpha']
		
		return der

	###############################################################################################

	def enclosedMassInner(self, r, accuracy = None):
		"""
		The mass enclosed within radius r.

		Parameters
		----------
		r: array_like
			Radius in physical kpc/h; can be a number or a numpy array.
		accuracy: float
			The minimum accuracy of the integration. This parameter has no effect because the 
			function uses an analytical expression.
			
		Returns
		-------
		M: array_like
			The mass enclosed within radius r, in :math:`M_{\\odot}/h`; has the same dimensions as r.
		"""		
		
		mass = self.par['rhos'] * self.mass_norm * scipy.special.gammainc(3.0 / self.par['alpha'],
								2.0 / self.par['alpha'] * (r / self.par['rs'])**self.par['alpha'])
		
		return mass

	###############################################################################################

	# Return array of d rho / d ln(rhos), d rho / d ln(rs), and d rho / d ln(alpha), since the 
	# parameters are fitted in log space.
	
	def _fitParamDeriv_rho(self, r, mask, N_par_fit):

		x = self.getParameterArray()
		deriv = np.zeros((N_par_fit, len(r)), float)
		rrs = r / x[1]
		rho_r = self.density(r)
		
		counter = 0
		if mask[0]:
			deriv[counter] = rho_r
			counter += 1
		if mask[1]:
			deriv[counter] = 2.0 * rho_r * rrs**(x[2])
			counter += 1
		if mask[2]:
			deriv[counter] = rho_r * 2.0 / x[2] * rrs**x[2] * (1.0 - rrs**(-x[2]) - x[2] * np.log(rrs))

		return deriv

###################################################################################################
