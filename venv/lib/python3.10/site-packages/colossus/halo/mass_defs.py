###################################################################################################
#
# mass_defs.py              (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This module implements functions to convert between spherical overdensity mass definitions and to 
compute pseudo-evolution. For basic aspects of spherical overdensity mass definitions, see the 
:doc:`halo_mass_so` section.

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

The functions in this unit assume a static halo density profile and compute spherical overdensity
radius and mass definitions based on this profile. These functions include changing the overdensity
definition at fixed redshift, changing redshift at fixed overdensity ("pseudo-evolution"), or
changing both.

One common application is to convert one spherical overdensity mass definition to another (at 
fixed redshift)::
	
	from colossus.halo import mass_defs
	
	Mvir = 1E12
	cvir = 10.0
	z = 1.0
	M200m, R200m, c200m = mass_defs.changeMassDefinition(Mvir, cvir, z, 'vir', '200m')
	
Here we assumed a halo with :math:`M_{vir}=10^{12} M_{\\odot}/h` and :math:`c_{vir} = 10` at z=1, 
and converted it to the 200m mass definition. For convenience, the new radius and concentration 
are also returned.

Pseudo-evolution is the evolution of a spherical overdensity halo radius, mass, and concentration 
due to an evolving reference density (see 
`Diemer et al. 2013 <http://adsabs.harvard.edu/abs/2013ApJ...766...25D>`__ for more information). 
The :func:`pseudoEvolve` function is a general implementation of this effect. The function 
assumes a profile that is fixed in physical units, and computes how the radius, mass and 
concentration evolve due to changes in redshift (at fixed mass definition). In the following 
example we compute the pseudo-evolution of a halo with virial mass 
:math:`M_{vir}=10^{12} M_{\\odot}/h` from z=1 to z=0::

	M_ini = 1E12
	c_ini = 10.0
	z_ini = 1.0
	z_final = 0.0
	M, R, c = mass_defs.pseudoEvolve(M_ini, c_ini, 'vir', z_ini, z_final)
	
Here we have assumed that the halo has a concentration :math:`c_{vir} = 10` at z=1. By default, 
an NFW density profile is assumed, but the user can also pass another profile object. 

Often, we do not know the concentration of a halo and wish to estimate it using a concentration-
mass model. This can be done using the :doc:`halo_concentration` module, but there is also 
a convenient wrapper for the :func:`changeMassDefinition` function, 
:func:`~halo.mass_adv.changeMassDefinitionCModel`. This function is located in a different module 
to avoid circular imports. Please see the :doc:`tutorials` for more extensive code examples.

---------------------------------------------------------------------------------------------------
Module contents
---------------------------------------------------------------------------------------------------

.. autosummary:: 
	evolveSO
	changeMassDefinition
	pseudoEvolve

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
import inspect

from colossus.utils import utilities
from colossus import defaults
from colossus.halo import mass_so
from colossus.halo import profile_nfw

###################################################################################################
# FUNCTIONS THAT CAN REFER TO DIFFERENT FORMS OF THE DENSITY PROFILE
###################################################################################################

def evolveSO(M_i, c_i, z_i, mdef_i, z_f, mdef_f, 
				profile = defaults.HALO_MASS_CONVERSION_PROFILE, profile_args = {}):
	"""
	Evolve the spherical overdensity radius for a fixed density profile.
	
	This function computes the evolution of spherical overdensity mass and radius due to a changing 
	reference density, redshift, or both. The user passes the mass and concentration of the density 
	profile, together with a redshift and mass definition to which M and c refer.
	
	To evaluate the new mass, radius, and concentration, we need to assume a particular form of the
	density profile. This profile can be either the NFW profile (``profile = 'nfw'``), or any 
	other profile class (derived from :class:`~halo.profile_base.HaloDensityProfile`). In the latter
	case, the passed profile class must accept mass, mass definition, concentration, and redshift
	as parameters to its constructor. Additional parameters can be passed as well, for example
	outer profile terms.
	
	Parameters
	----------
	M_i: array_like
		The initial halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array. If both 
		``M_i`` and ``c_i`` are arrays, they must have the same dimensions.
	c_i: array_like
		The initial halo concentration; can be a number of a numpy array. If both ``M_i`` and 
		``c_i`` are arrays, they must have the same dimensions.
	z_i: float
		The initial redshift.
	mdef_i: str
		The initial mass definition.
	z_f: float
		The final redshift (can be smaller, equal to, or larger than ``z_i``).
	mdef_f: str
		The final mass definition (can be the same as ``mdef_i``, or different).
	profile: str or HaloDensityProfile
		The functional form of the profile assumed in the computation; can be ``nfw`` or an 
		instance of HaloDensityProfile (which satisfies particular conditions, see above).
	profile_args: dict
		Any other keyword args are passed to the constructor of the density profile class.
		
	Returns
	-------
	Mnew: array_like
		The new halo mass in :math:`M_{\\odot}/h`; has the same dimensions as ``M_i`` or ``c_i``.
	Rnew: array_like
		The new halo radius in physical kpc/h; has the same dimensions as ``M_i`` or ``c_i``.
	cnew: array_like
		The new concentration (now referring to the new mass definition); has the same dimensions 
		as ``M_i`` or ``c_i``.
		
	See also
	--------
	changeMassDefinition: Change the spherical overdensity mass definition.
	pseudoEvolve: Pseudo-evolve a static density profile.
	"""

	# Redshift must always be a number, not an array
	if utilities.isArray(z_i):
		raise Exception('Redshift z_i must be a float, not an array.')
	if utilities.isArray(z_f):
		raise Exception('Redshift z_f must be a float, not an array.')
	
	# Both M and c can be numbers or arrays
	M_i_array, M_is_array = utilities.getArray(M_i)
	c_i_array, c_is_array = utilities.getArray(c_i)
	M_i_array = M_i_array.astype(float)
	c_i_array = c_i_array.astype(float)
	if not M_is_array and not c_is_array:
		M_i = M_i_array
		c_i = c_i_array
		N = 1
	elif M_is_array and not c_is_array:
		M_i = M_i_array
		c_i = np.ones_like(M_i) * c_i
		N = len(M_i_array)
	elif c_is_array and not M_is_array:
		c_i = c_i_array
		M_i = np.ones_like(c_i) * M_i
		N = len(c_i_array)
	else:
		if len(M_i_array) != len(c_i_array):
			raise Exception('If both M_i and c_i are arrays, they must have the same dimensions.')
		M_i = M_i_array
		c_i = c_i_array
		N = len(M_i_array)
	Rnew = np.zeros_like(M_i)
	cnew = np.zeros_like(M_i)

	if profile == 'nfw':
		
		# We do not instantiate NFW profile objects, but instead use the faster static functions
		rhos, rs = profile_nfw.NFWProfile.nativeParameters(M_i, c_i, z_i, mdef_i)
		density_threshold = mass_so.densityThreshold(z_f, mdef_f)
		cnew = profile_nfw.NFWProfile.xDelta(rhos, density_threshold)
		Rnew = rs * cnew

	elif inspect.isclass(profile):
		
		for i in range(N):
			prof = profile(M = M_i[i], mdef = mdef_i, z = z_i, c = c_i[i], **profile_args)
			Rnew[i] = prof.RDelta(z_f, mdef_f)
			cnew[i] = Rnew[i] / prof.par['rs']

	else:
		raise Exception('This function is not defined for profile %s.' % (profile))

	if not M_is_array and not c_is_array:
		Rnew = Rnew[0]
		cnew = cnew[0]

	Mnew = mass_so.R_to_M(Rnew, z_f, mdef_f)
	
	return Mnew, Rnew, cnew

###################################################################################################

def changeMassDefinition(M, c, z, mdef_in, mdef_out, 
						profile = defaults.HALO_MASS_CONVERSION_PROFILE, profile_args = {}):
	"""
	Change the spherical overdensity mass definition assuming a fixed density profile.
	
	This function is a special case of the more general :func:`evolveSO` function. We assume a 
	density profile fixed in physical coordinates and at a fixed redshift, but we change the mass 
	definition. This leads to a different spherical overdensity radius, mass, and concentration.
	By default, the density profile is assumed to be an NFW profile, but other profiles can be 
	specified.
	
	Parameters
	----------
	M: array_like
		The initial halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array. If both 
		``M`` and ``c`` are arrays, they must have the same dimensions.
	c: array_like
		The initial halo concentration; can be a number of a numpy array. If both ``M`` and ``c`` 
		are arrays, they must have the same dimensions.
	z: float
		The initial redshift.
	mdef_in: str
		The input mass definition.
	mdef_out: str
		The output mass definition.
	profile: str
		The functional form of the profile assumed in the computation; can be ``nfw`` or ``dk14``.
	profile_args: dict
		Any other keyword args are passed to the constructor of the density profile class.

	Returns
	-------
	Mnew: array_like
		The new halo mass in :math:`M_{\\odot}/h`; has the same dimensions as ``M`` or ``c``.
	Rnew: array_like
		The new halo radius in physical kpc/h; has the same dimensions as ``M`` or ``c``.
	cnew: array_like
		The new concentration (now referring to the new mass definition); has the same dimensions 
		as ``M`` or ``c``.
		
	See also
	--------
	evolveSO: Evolve the spherical overdensity radius for a fixed profile.
	halo.mass_adv.changeMassDefinitionCModel: Change the spherical overdensity mass definition, using a model for the concentration.
	"""
	
	return evolveSO(M, c, z, mdef_in, z, mdef_out, profile = profile, profile_args = profile_args)

###################################################################################################

def pseudoEvolve(M, c, mdef, z_i, z_f,
						profile = defaults.HALO_MASS_CONVERSION_PROFILE, profile_args = {}):
	"""
	Pseudo-evolve a static density profile.

	This function computes the evolution of spherical overdensity mass and radius due to a changing 
	reference density, an effect called 'pseudo-evolution' (e.g.,
	`Diemer et al. 2013 <http://adsabs.harvard.edu/abs/2013ApJ...766...25D>`__). The user passes 
	the mass and concentration of the density profile, together with a redshift and mass definition 
	to which ``M`` and ``c`` refer.
	
	To evaluate the new mass, radius, and concentration, we need to assume a particular form of the
	density profile. This profile can be either the NFW profile (``profile = 'nfw'``), or any 
	other profile class (derived from :class:`~halo.profile_base.HaloDensityProfile`). In the latter
	case, the passed profile class must accept mass, mass definition, concentration, and redshift
	as parameters to its constructor. Additional parameters can be passed as well, for example
	outer profile terms.

	Parameters
	----------
	M: array_like
		The initial halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array. If both 
		``M`` and ``c`` are arrays, they must have the same dimensions.
	c: array_like
		The initial halo concentration; can be a number of a numpy array. If both ``M`` and ``c`` 
		are arrays, they must have the same dimensions.
	mdef: str
		The SO mass definition. See :doc:`halo_mass` for details.
	z_i: float
		The initial redshift.
	z_f: float
		The final redshift (can be smaller, equal to, or larger than ``z_i``).
	profile: str
		The functional form of the profile assumed in the computation; can be ``nfw`` or ``dk14``.
	profile_args: dict
		Any other keyword args are passed to the constructor of the density profile class.

	Returns
	-------
	Mnew: array_like
		The new halo mass in :math:`M_{\\odot}/h`; has the same dimensions as ``M`` or ``c``.
	Rnew: array_like
		The new halo radius in physical kpc/h; has the same dimensions as ``M`` or ``c``.
	cnew: array_like
		The new concentration (now referring to the new mass definition); has the same dimensions 
		as ``M`` or ``c``.
		
	See also
	--------
	evolveSO: Evolve the spherical overdensity radius for a fixed profile.
	"""
	
	return evolveSO(M, c, z_i, mdef, z_f, mdef, profile = profile, profile_args = profile_args)

###################################################################################################
