###################################################################################################
#
# mass_so.py                (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This module implements basic aspects of spherical overdensity mass definitions for dark matter 
halos (please see :doc:`halo_mass` for an introduction).

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

For example, we can compute the spherical overdensity radius of a halo with particular mass or
vice versa::
	
	from colossus.halo import mass_so
	
	R200m = mass_so.M_to_R(1E12, 0.0, '200m')
	Mvir = mass_so.R_to_M(400.0, 1.5, 'vir')

The other functions in this module allow us to parse the mass definition strings and compute the 
density thresholds, but typically the user will not need to evaluate those functions manually
since most SO-related functions in Colossus accept ``mdef`` as an argument. Please see the 
:doc:`tutorials` for more extensive code examples.

---------------------------------------------------------------------------------------------------
Module contents
---------------------------------------------------------------------------------------------------

.. autosummary:: 

	parseMassDefinition
	parseRadiusMassDefinition
    densityThreshold
    deltaVir
    M_to_R
    R_to_M
    dynamicalTime

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np

from colossus.cosmology import cosmology

###################################################################################################
# FUNCTIONS RELATED TO SPHERICAL OVERDENSITY MASSES
###################################################################################################

def parseMassDefinition(mdef):
	"""
	The type and overdensity of a given spherical overdensity mass definition.
	
	Parameters
	----------
	mdef: str
		The mass definition. See :doc:`halo_mass` for details.
		
	Returns
	-------
	mdef_type: str
		Can either be based on the mean density (``mdef_type='m'``), the critical density 
		(``mdef_type='c'``) or the virial overdensity (``mdef_type='vir'``).
	mdef_delta: int
		The overdensity; if ``mdef_type=='vir'``, the overdensity depends on redshift, and this
		parameter is ``None``.
	"""
	
	if mdef[-1] == 'c':
		mdef_type = 'c'
		mdef_delta = int(mdef[:-1])

	elif mdef[-1] == 'm':
		mdef_type = 'm'
		mdef_delta = int(mdef[:-1])

	elif mdef == 'vir':
		mdef_type = 'vir'
		mdef_delta = None

	else:
		raise Exception('Invalid mass definition, %s.' % mdef)
	
	return mdef_type, mdef_delta

###################################################################################################

def parseRadiusMassDefinition(rmdef):
	"""
	Parse a radius or mass identifier as well as the mass definition.
	
	Parameters
	----------
	rmdef: str
		The radius or mass identifier
		
	Returns
	-------
	radius_mass: str
		Can be ``R`` for radius or ``M`` for mass.
	mdef: str
		The mdef the mass or radius are based on. See :doc:`halo_mass` for details.
	mdef_type: str
		Can either be based on the mean density (``mdef_type='m'``), the critical density 
		(``mdef_type='c'``) or the virial overdensity (``mdef_type=='vir'``).
	mdef_delta: int
		The overdensity; if ``mdef_type=='vir'``, the overdensity depends on redshift, and this
		parameter is ``None``.
	"""
		
	if rmdef[0] in ['r', 'R']:
		radius_mass = 'R'
	elif rmdef[0] in ['m', 'M']:
		radius_mass = 'M'
	else:
		raise Exception('Invalid identifier, %s. Must be either R for radius or M for mass.' % rmdef[0])
	
	mdef = rmdef[1:]
	mdef_type, mdef_delta = parseMassDefinition(mdef)
	
	return radius_mass, mdef, mdef_type, mdef_delta

###################################################################################################

def densityThreshold(z, mdef):
	"""
	The threshold density for a given spherical overdensity mass definition.
	
	Parameters
	----------
	z: array_like
		Redshift; can be a number or a numpy array.
	mdef: str
		The mass definition. See :doc:`halo_mass` for details.
		
	Returns
	-------
	rho: array_like
		The threshold density in physical :math:`M_{\\odot}h^2/{\\rm kpc}^3`; has the same 
		dimensions as ``z``.

	See also
	--------
	deltaVir: The virial overdensity in units of the critical density.
	"""
	
	cosmo = cosmology.getCurrent()
	mdef_type, mdef_delta = parseMassDefinition(mdef)
	
	if mdef_type == 'c':
		rho_treshold = mdef_delta * cosmo.rho_c(z)
	elif mdef_type == 'm':
		rho_treshold = mdef_delta * cosmo.rho_m(z)
	elif mdef_type == 'vir':
		rho_treshold = deltaVir(z) * cosmo.rho_c(z)
	else:
		raise Exception('Invalid mass definition, %s.' % mdef)

	return rho_treshold

###################################################################################################

def deltaVir(z):
	"""
	The virial overdensity in units of the critical density.
	
	This function uses the fitting formula of 
	`Bryan & Norman 1998 <http://adsabs.harvard.edu/abs/1998ApJ...495...80B>`__ to determine the 
	virial overdensity. While the universe is dominated by matter, this overdensity is about 178. 
	Once dark energy starts to matter, it decreases. 
	
	Parameters
	----------
	z: array_like
		Redshift; can be a number or a numpy array.
		
	Returns
	-------
	Delta: array_like
		The virial overdensity; has the same dimensions as ``z``.

	See also
	--------
	densityThreshold: The threshold density for a given mass definition.
	"""
	
	cosmo = cosmology.getCurrent()
	x = cosmo.Om(z) - 1.0
	Delta = 18 * np.pi**2 + 82.0 * x - 39.0 * x**2

	return Delta

###################################################################################################

def M_to_R(M, z, mdef):
	"""
	Convert spherical overdensity mass to radius.
	
	This function returns a spherical overdensity halo radius for a halo mass ``M``. Note that 
	this function is independent of the form of the density profile.

	Parameters
	----------
	M: array_like
		Mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition. See :doc:`halo_mass` for details.
		
	Returns
	-------
	R: array_like
		Halo radius in physical kpc/h; has the same dimensions as ``M``.

	See also
	--------
	R_to_M: Convert spherical overdensity radius to mass.
	"""
	
	rho = densityThreshold(z, mdef)
	R = (M * 3.0 / 4.0 / np.pi / rho)**(1.0 / 3.0)

	return R

###################################################################################################

def R_to_M(R, z, mdef):
	"""
	Convert spherical overdensity radius to mass.
	
	This function returns a spherical overdensity halo mass for a halo radius ``R``. Note that 
	this function is independent of the form of the density profile.

	Parameters
	----------
	R: array_like
		Halo radius in physical kpc/h; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition. See :doc:`halo_mass` for details.
		
	Returns
	-------
	M: array_like
		Mass in :math:`M_{\\odot}/h`; has the same dimensions as ``R``.

	See also
	--------
	M_to_R: Convert spherical overdensity mass to radius.
	"""
	
	rho = densityThreshold(z, mdef)
	M = 4.0 / 3.0 * np.pi * rho * R**3

	return M

###################################################################################################

def dynamicalTime(z, mdef, definition = 'crossing'):
	"""
	The dynamical time of a halo.
	
	The dynamical time can be estimated in multiple ways, but is almost always based on the ratio
	of a distance to circular velocity. The relevant distance can be defined in different ways as
	indicated with the ``definition`` parameter. The dynamical time is more succinctly expressed 
	as a multiple of the Hubble time which depends on the overdensity threshold and redshift.
	
	Parameters
	----------
	z: array_like
		Redshift; can be a number or a numpy array.
	mdef: str
		The mass definition. See :doc:`halo_mass` for details.
	definition: str
		An identifier for a definition of the dynamical time. Valid definitions are ``crossing``
		(the crossing time), ``peri`` (the time to reach the halo center, half the crossing time)
		and ``orbit`` (the time to orbit around the halo, crossing time times :math:`\\pi`).
		
	Returns
	-------
	t_dyn: array_like
		Dynamical time in Gyr; has the same dimensions as ``z``.
	"""
	
	cosmo = cosmology.getCurrent()
	t_cross = 2**1.5 * cosmo.hubbleTime(z) * (densityThreshold(z, mdef) / cosmo.rho_c(z))**-0.5

	if definition == 'crossing':
		t_dyn = t_cross
	elif definition == 'peri':
		t_dyn = t_cross / 2.0
	elif definition == 'orbit':
		t_dyn = t_cross * np.pi
	else:
		raise Exception('Unknown definition of the dynamical time, %s.' % definition)
	
	return t_dyn

###################################################################################################
