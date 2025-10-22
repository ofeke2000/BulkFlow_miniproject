###################################################################################################
#
# bias.py                   (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
Halo bias quantifies the excess clustering of halos over the clustering of dark matter. Bias is,
in general, a function of halo mass and scale, but this module currently implements only 
scale-free bias models.

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

Bias can be evaluated as a function of either mass or peak height. The parameters that need to be 
passed depend on the bias model to some degree, and on whether the mass needs to be converted to 
peak height first::
	
	from colossus.lss import peaks
	from colossus.lss import bias
	
	M = 1E14
	z = 0.0
	nu = peaks.peakHeight(M, z)
	b = bias.haloBiasFromNu(nu, model = 'sheth01')
	b = bias.haloBias(M, model = 'tinker10', z = z, mdef = 'vir')

Please see the :doc:`tutorials` for more extensive code examples.

---------------------------------------------------------------------------------------------------
Bias models
---------------------------------------------------------------------------------------------------

The simplest bias model (``cole89``) is based on the peak-background split model and was derived in a 
number of different papers. The rest of the models was calibrated using numerical simulations:

.. table::
	:widths: auto

	============== =========================== =========================== =========================================
	ID             Parameters                  z-dependence                Reference
	============== =========================== =========================== =========================================
	cole89         M/nu                        No                          `Cole & Kaiser 1989 <http://adsabs.harvard.edu/abs/1989MNRAS.237.1127C>`__, `Mo & White 1996 <http://adsabs.harvard.edu/abs/1996MNRAS.282..347M>`__
	jing98         M/nu, z                     Through non-linear mass     `Jing 1998 <http://adsabs.harvard.edu/abs/1998ApJ...503L...9J>`__
	sheth01        M/nu                        No                          `Sheth et al. 2001 <http://adsabs.harvard.edu/abs/2001MNRAS.323....1S>`__
	seljak04       M/nu, z                     Through non-linear mass     `Seljak & Warren 2004 <http://adsabs.harvard.edu/abs/2004MNRAS.355..129S>`__
	pillepich10    M/nu                        No                          `Pillepich et al. 2010 <http://adsabs.harvard.edu/abs/2010MNRAS.402..191P>`__
	tinker10       M/nu, z, mdef               Through mass definition     `Tinker et al. 2010 <http://adsabs.harvard.edu/abs/2010ApJ...724..878T>`__    
	bhattacharya11 M/nu, z                     Yes                         `Bhattacharya et al. 2011 <http://adsabs.harvard.edu/abs/2011ApJ...732..122B>`__
	comparat17     M/nu                        No                          `Comparat et al. 2017 <https://ui.adsabs.harvard.edu//#abs/2017MNRAS.469.4157C/abstract>`__
	============== =========================== =========================== =========================================

The z-dependence column indicates whether a model predicts a bias that varies with redshift at 
fixed peak height. At fixed mass, all models predict a strongly varying bias.
The `Tinker et al. 2010 <http://adsabs.harvard.edu/abs/2010ApJ...724..878T>`__ model was 
calibrated for a range of overdensities with respect to the mean density of the universe. Thus, 
depending on the mass definition used, this model can predict a slight redshift evolution.

---------------------------------------------------------------------------------------------------
Module contents
---------------------------------------------------------------------------------------------------

.. autosummary::
	HaloBiasModel
	models
	haloBias
	haloBiasFromNu
	twoHaloTerm
	modelCole89
	modelJing98
	modelSheth01
	modelSeljak04
	modelPillepich10
	modelTinker10
	modelBhattacharya11
	modelComparat17
	
---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
from collections import OrderedDict

from colossus.utils import constants
from colossus import defaults
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.halo import mass_so

###################################################################################################

class HaloBiasModel():
	"""
	Characteristics of halo bias models.
	
	This object contains certain characteristics of a halo bias model. Currently, this object
	is empty. The :data:`models` dictionary contains one item of this class for each available 
	model.
	"""
		
	def __init__(self):
		return

###################################################################################################

models = OrderedDict()
"""
Dictionary containing a list of models.

An ordered dictionary containing one :class:`HaloBiasModel` entry for each model.
"""

models['cole89'] = HaloBiasModel()
models['jing98'] = HaloBiasModel()
models['sheth01'] = HaloBiasModel()
models['seljak04'] = HaloBiasModel()
models['pillepich10'] = HaloBiasModel()
models['tinker10'] = HaloBiasModel()
models['bhattacharya11'] = HaloBiasModel()
models['comparat17'] = HaloBiasModel()

###################################################################################################
# HALO BIAS
###################################################################################################

def haloBiasFromNu(nu, z = None, mdef = None, model = defaults.HALO_BIAS_MODEL, **kwargs):
	"""
	The halo bias at a given peak height. 

	Redshift and mass definition are necessary only for particular models (see table above).
	
	Parameters
	----------
	nu: array_like
		Peak height; can be a number or a numpy array.
	z: array_like
		Redshift; can be a number or a numpy array. Only necessary for certain models.
	mdef: str
		The mass definition corresponding to the mass that was used to evaluate the peak height.
		Only necessary for certain models. See :doc:`halo_mass` for details.
	model: str
		The bias model used.
	kwargs: kwargs
		Extra arguments passed to the function of the particular model. See the documentation of 
		those functions for valid arguments.
	
	Returns
	-------
	bias: array_like
		Halo bias; has the same dimensions as ``nu`` or ``z``.

	See also
	--------
	haloBias: The halo bias at a given mass. 
	"""
	
	if model == 'cole89':
		bias = modelCole89(nu, **kwargs)
	elif model == 'jing98':
		bias = modelJing98(nu, z, **kwargs)
	elif model == 'sheth01':
		bias = modelSheth01(nu, **kwargs)
	elif model == 'seljak04':
		bias = modelSeljak04(nu, z, **kwargs)
	elif model == 'pillepich10':
		bias = modelPillepich10(nu, **kwargs)
	elif model == 'tinker10':
		bias = modelTinker10(nu, z, mdef, **kwargs)
	elif model == 'bhattacharya11':
		bias = modelBhattacharya11(nu, z, **kwargs)
	elif model == 'comparat17':
		bias = modelComparat17(nu, **kwargs)
	else:
		raise Exception('Unkown model, %s.' % (model))

	return bias

###################################################################################################

def haloBias(M, z, mdef = None, model = defaults.HALO_BIAS_MODEL, **kwargs):
	"""
	The halo bias at a given mass. 

	This function is a wrapper around :func:`haloBiasFromNu`. The mass definition is necessary 
	only for certain models whereas the redshift is always necessary in order to convert mass to 
	peak height.
	
	Parameters
	----------
	M: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: array_like
		Redshift; can be a number or a numpy array.
	mdef: str
		The mass definition in which ``M`` is given. Only necessary for certain models.
		See :doc:`halo_mass` for details.
	model: str
		The bias model used.
	kwargs: kwargs
		Extra arguments passed to the function of the particular model. See the documentation of 
		those functions for valid arguments.

	Returns
	-------
	bias: array_like
		Halo bias; has the same dimensions as ``M`` or ``z``.

	See also
	--------
	haloBiasFromNu: The halo bias at a given peak height. 
	"""
		
	nu = peaks.peakHeight(M, z)
	bias = haloBiasFromNu(nu, z = z, mdef = mdef, model = model, **kwargs)
	
	return bias

###################################################################################################

def twoHaloTerm(r, M, z, mdef, model = defaults.HALO_BIAS_MODEL):
	"""
	The 2-halo term as a function of radius and halo mass. 

	The 2-halo term in the halo-matter correlation function describes the excess density around 
	halos due to the proximity of other peaks. This contribution can be approximated as the matter-
	matter correlation function times a linear bias which depends on the peak height of the halo.
	Sometimes this term includes an additional factor of the mean density which is omitted here. 
	
	Note that this 2-halo term is also implemented as an outer profile in the Colossus halo module,
	see the documentation of :mod:`halo.profile_outer`.
	
	Parameters
	----------
	r: array_like
		Halocentric radius in comoving Mpc/h; can be a number or a numpy array.
	M: float
		Halo mass in :math:`M_{\\odot}/h`
	z: float
		Redshift
	mdef: str
		The mass definition in which ``M`` is given. See :doc:`halo_mass` for details.
	model: str
		The bias model used.

	Returns
	-------
	rho_2h: array_like
		The density due to the 2-halo term in physical :math:`M_{\\odot}h^2/{\\rm kpc}^3`; has the 
		same dimensions as ``r``.
	"""	
	
	cosmo = cosmology.getCurrent()
	bias = haloBias(M, z, mdef, model = model)
	xi = cosmo.correlationFunction(r, z)
	rho_2h = cosmo.rho_m(z) * bias * xi
	
	return rho_2h

###################################################################################################
# SPECIFIC MODELS
###################################################################################################

def modelCole89(nu):
	"""
	The peak-background split prediction for halo bias.
	
	For a derivation of this model, see 
	`Cole & Kaiser 1989 <http://adsabs.harvard.edu/abs/1989MNRAS.237.1127C>`__ or 
	`Mo & White 1996 <http://adsabs.harvard.edu/abs/1996MNRAS.282..347M>`__.

	Parameters
	----------
	nu: array_like
		Peak height; can be a number or a numpy array.
		
	Returns
	-------
	bias: array_like
		Halo bias; has the same dimensions as ``nu``.
	"""
	
	delta_c = peaks.collapseOverdensity()
	bias = 1.0 + (nu**2 - 1.0) / delta_c
	
	return bias

###################################################################################################

def modelJing98(nu, z):
	"""
	A bias model calibrated on scale-free simulations.
	
	This bias model relies on the slope of the power spectrum. It was calibrated on scale-free
	simulations with a power-law power spectrum, but is also applicable to LCDM cosmologies. Note
	that in the original notation, the peak height is only valid for scale-free models, but it is 
	stated in Section 3 that one should use the properly defined peak height for CDM cosmologies.
	
	Moreover, we use the zero-baryon approximation to the power spectrum of Eisenstein & Hu to
	avoid wiggles in the slope.

	Parameters
	----------
	nu: array_like
		Peak height; can be a number or a numpy array.
	z: float
		Redshift.
		
	Returns
	-------
	bias: array_like
		Halo bias; has the same dimensions as ``nu``.
	"""
	
	n_eff = peaks.powerSpectrumSlope(nu, z, slope_type = 'P', scale = 1.0,
									ps_args = {'model': 'eisenstein98_zb'})
	delta_c = peaks.collapseOverdensity()

	bias = (0.5 / nu**4 + 1.0)**(0.06 - 0.02 * n_eff) * (1.0 + (nu**2 - 1.0) / delta_c)
	
	return bias

###################################################################################################

def modelSheth01(nu):
	"""
	The halo bias model of Sheth et al 2001. 
	
	Parameters
	----------
	nu: array_like
		Peak height; can be a number or a numpy array.
		
	Returns
	-------
	bias: array_like
		Halo bias; has the same dimensions as ``nu``.
	"""
	
	a = 0.707
	b = 0.5
	c = 0.6
	roota = np.sqrt(a)
	anu2 = a * nu**2
	anu2c = anu2**c
	t1 = b * (1.0 - c) * (1.0 - 0.5 * c)
	delta_sc = peaks.collapseOverdensity()
	bias = 1.0 +  1.0 / (roota * delta_sc) * (roota * anu2 + roota * b * anu2**(1.0 - c) - anu2c / (anu2c + t1))

	return bias

###################################################################################################

def modelSeljak04(nu, z, cosmo_term = False):
	"""
	A numerically calibrated bias model.
	
	This bias model corresponds to Equation 5 in 
	`Seljak & Warren 2004 <http://adsabs.harvard.edu/abs/2004MNRAS.355..129S>`__. If 
	``cosmo_term == True``, Equation 6 is used. Colossus currently does not implement a running of
	the spectral index, the corresponding parameter :math:`\\alpha_{\\rm s}` is thus set to zero.

	Parameters
	----------
	nu: array_like
		Peak height; can be a number or a numpy array.
	z: float
		Redshift.
	cosmo_term: bool
		Include the cosmological term of Equation 6.
		
	Returns
	-------
	bias: array_like
		Halo bias; has the same dimensions as ``nu``.
	"""
	
	M = peaks.massFromPeakHeight(nu, z)
	Mstar = peaks.nonLinearMass(z)
	x = M / Mstar

	bias = 0.53 + 0.39 * x**0.45 + 0.13 / (40.0 * x + 1.0) + 5E-4 * x**1.5
	
	if cosmo_term:
		cosmo = cosmology.getCurrent()
		Om = cosmo.Om0
		sigma8 = cosmo.sigma8
		h = cosmo.h
		ns = cosmo.ns
		alphas = 0.0
		t = 0.4 * (Om - 0.3 + ns - 1.0) + 0.3 * (sigma8 - 0.9 + h - 0.7) + 0.8 * alphas
		bias += np.log10(x) * t
	
	return bias

###################################################################################################

def modelPillepich10(nu):
	"""
	The halo bias model of Pillepich et al 2010. 
	
	Parameters
	----------
	nu: array_like
		Peak height; can be a number or a numpy array.
		
	Returns
	-------
	bias: array_like
		Halo bias; has the same dimensions as ``nu``.
	"""
	
	sigma = peaks.collapseOverdensity() / nu
	
	bias = 0.647 -0.540 / sigma + 1.614 / sigma**2

	return bias

###################################################################################################

def modelTinker10(nu, z, mdef):
	"""
	The halo bias model of Tinker et al 2010. 

	The mass definition ``mdef`` must correspond to the mass that was used to evaluate the peak 
	height. Note that the Tinker bias function is universal in redshift at fixed peak height, but 
	only for mass definitions defined wrt. the mean density of the universe. For other definitions, 
	:math:`\\Delta_{\\rm m}` evolves with redshift, leading to an evolving bias at fixed peak height. 
	
	Parameters
	----------
	nu: array_like
		Peak height; can be a number or a numpy array.
	z: array_like
		Redshift; can be a number or a numpy array.
	mdef: str
		The mass definition. See :doc:`halo_mass` for details.
		
	Returns
	-------
	bias: array_like
		Halo bias; has the same dimensions as ``nu`` or ``z``.
	"""
	
	if z is None:
		raise Exception('The Tinker et al. 2010 model needs a redshift to be passed.')
	if mdef is None:
		raise Exception('The Tinker et al. 2010 model needs a mass definition to be passed.')
	
	cosmo = cosmology.getCurrent()
	Delta = mass_so.densityThreshold(z, mdef) / cosmo.rho_m(z)
	y = np.log10(Delta)

	A = 1.0 + 0.24 * y * np.exp(-1.0 * (4.0 / y)**4)
	a = 0.44 * y - 0.88
	B = 0.183
	b = 1.5
	C = 0.019 + 0.107 * y + 0.19 * np.exp(-1.0 * (4.0 / y)**4)
	c = 2.4
	
	bias = 1.0 - A * nu**a / (nu**a + constants.DELTA_COLLAPSE**a) + B * nu**b + C * nu**c
	
	return bias

###################################################################################################

def modelBhattacharya11(nu, z):
	"""
	A bias model based on a mass function calibration.
	
	This bias model is derived using the peak-background split logic of Sheth & Tormen 1999, but 
	with the updated and z-dependent best-fit parameters derived for the mass function. The 
	authors note that this bias function does not match the numerical results as well as direct
	calibrations.

	Parameters
	----------
	nu: array_like
		Peak height; can be a number or a numpy array.
	z: float
		Redshift.
		
	Returns
	-------
	bias: array_like
		Halo bias; has the same dimensions as ``nu``.
	"""

	delta_c = peaks.collapseOverdensity()

	zp1 = 1.0 + z
	a = 0.788 * zp1**-0.01
	p = 0.807
	q = 1.795

	bias = 1.0 + (a * nu**2 - q) / delta_c + 2.0 * p / delta_c / (1.0 + (a * nu**2)**p)

	return bias

###################################################################################################

def modelComparat17(nu):
	"""
	A bias model based on a mass function calibration.
	
	This model is equivalent to the Bhattacharya et al 2011 model in that it uses the same
	functional form and that its best-fit parameters are derived from a fit to the mass function
	rather than bias itself. However, this model does not depend on redshift. The parameters used 
	here are updated compared to the published version of the paper.

	Parameters
	----------
	nu: array_like
		Peak height; can be a number or a numpy array.
		
	Returns
	-------
	bias: array_like
		Halo bias; has the same dimensions as ``nu``.
	"""

	delta_c = peaks.collapseOverdensity()

	a = 0.897
	p = 0.624
	q = 1.589

	bias = 1.0 + (a * nu**2 - q) / delta_c + 2.0 * p / delta_c / (1.0 + (a * nu**2)**p)

	return bias

###################################################################################################
