###################################################################################################
#
# concentration.py          (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This module implements a range of models for halo concentration as a function of mass, redshift, 
and cosmology. 

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

The main function in this module, :func:`concentration`, is a wrapper for all models::
	
	from colossus.cosmology import cosmology
	from colossus.halo import concentration
	
	cosmology.setCosmology('planck15')
	cvir = concentration.concentration(1E12, 'vir', 0.0, model = 'bullock01')

Alternatively, the user can also call the individual model functions directly. However, there are
two aspects which the :func:`concentration` function handles automatically. First, many 
concentration models are only valid over a certain range of masses, redshifts, and 
cosmologies. If the user requests a mass or redshift outside these ranges, the function returns a 
*soft fail*: the concentration value is computed, but a warning is displayed and/or a False flag
is returned in a boolean array. If a concentration model cannot be computed, this leads to a *hard
fail* and a returned value of ``INVALID_CONCENTRATION`` (-1).

Second, each model was only calibrated for one of a few particular mass definitions, such as 
:math:`c_{200c}`, :math:`c_{vir}`, or :math:`c_{200m}`. The :func:`concentration` function 
automatically converts these definitions to the definition chosen by the user (see :doc:`halo_mass`
for more information on spherical overdensity masses). For the conversion, we necessarily have to
assume a particular form of the density profile (see the documentation of the 
:func:`~halo.mass_defs.changeMassDefinition` function). 

.. note::
	The conversion to other mass definitions can degrade the accuracy of the predicted 
	concentration by up to ~15-20% for certain mass definitions, masses, and redshifts. Using 
	the DK14 profile (see the :mod:`halo.profile_dk14` module) for the mass conversion gives 
	slightly improved results, but the conversion is slower. Please see Appendix C in 
	`Diemer & Kravtsov 2015 <http://adsabs.harvard.edu/abs/2015ApJ...799..108D>`__ for details.

.. note::
	The user must ensure that the cosmology is set consistently. Many concentration models were 
	calibrated only for one particular cosmology (though the default concentration model, 
	``diemer15``, is valid for all masses, redshifts, and cosmologies). Neither the 
	:func:`concentration` function nor the invidual model functions issue warnings if the set 
	cosmology does not match the concentration model (with the exception of the 
	:func:`modelKlypin16fromM` and :func:`modelKlypin16fromNu` functions). For example, it is 
	possible to set a WMAP9 cosmology, and evaluate the Duffy et al. 2008 model which is only 
	valid for a WMAP5 cosmology. When using such models, it is the user's responsibility to ensure 
	consistency with other calculations.

---------------------------------------------------------------------------------------------------
Concentration models
---------------------------------------------------------------------------------------------------

The following models are supported in this module, and their ID can be passed as the ``model`` 
parameter to the :func:`concentration` function:

.. table::
	:widths: auto

	============== ================ ================== =========== =============== ============================================================================
	ID             Native mdefs     M-range (z=0)      z-range     Cosmology       Reference
	============== ================ ================== =========== =============== ============================================================================
	bullock01	   200c             Almost any         Any         Any             `Bullock et al. 2001 <http://adsabs.harvard.edu/abs/2001MNRAS.321..559B>`__
	duffy08        200c, vir, 200m  1E11 < M < 1E15    0 < z < 2   WMAP5           `Duffy et al. 2008 <http://adsabs.harvard.edu/abs/2008MNRAS.390L..64D>`__
	klypin11       vir              3E10 < M < 5E14    0           WMAP7           `Klypin et al. 2011 <http://adsabs.harvard.edu/abs/2011ApJ...740..102K>`__
	prada12        200c             Any                Any         Any             `Prada et al. 2012 <http://adsabs.harvard.edu/abs/2012MNRAS.423.3018P>`__
	bhattacharya13 200c, vir, 200m  2E12 < M < 2E15    0 < z < 2   WMAP7           `Bhattacharya et al. 2013 <http://adsabs.harvard.edu/abs/2013ApJ...766...32B>`__
	dutton14       200c, vir        M > 1E10           0 < z < 5   planck13        `Dutton & Maccio 2014 <http://adsabs.harvard.edu/abs/2014MNRAS.441.3359D>`__
	diemer15_orig  200c             Any                Any         Any             `Diemer & Kravtsov 2015 <http://adsabs.harvard.edu/abs/2015ApJ...799..108D>`__
	diemer15       200c             Any                Any         Any             `Diemer & Joyce 2019 <https://ui.adsabs.harvard.edu//#abs/2018arXiv180907326D/abstract>`__
	klypin16_m     200c, vir        M > 1E10           0 < z < 5   planck13/WMAP7  `Klypin et al. 2016 <http://adsabs.harvard.edu/abs/2016MNRAS.457.4340K>`__
	klypin16_nu    200c, vir        M > 1E10           0 < z < 5   planck13        `Klypin et al. 2016 <http://adsabs.harvard.edu/abs/2016MNRAS.457.4340K>`__
	ludlow16       200c             Any                Any         Any             `Ludlow et al. 2016 <https://ui.adsabs.harvard.edu//#abs/2016MNRAS.460.1214L/abstract>`__
	child18        200c             M > 2.1E11         0 < z < 4   WMAP7           `Child et al. 2016 <https://ui.adsabs.harvard.edu//#abs/2018ApJ...859...55C/abstract>`__
	diemer19       200c             Any                Any         Any             `Diemer & Joyce 2019 <https://ui.adsabs.harvard.edu/abs/2019ApJ...871..168D/abstract>`__
	ishiyama21     500c, 200c, vir  Any                Any         Any             `Ishiyama et al. 2021 <https://ui.adsabs.harvard.edu/abs/2021MNRAS.506.4210I/abstract>`__
	============== ================ ================== =========== =============== ============================================================================

The original version of the ``diemer15`` model suffered from a small numerical error, a corrected
set of parameters is given in 
`Diemer & Joyce 2019 <https://ui.adsabs.harvard.edu//#abs/2018arXiv180907326D/abstract>`__. The differences 
between the models are less than 5%, but the original model should be used only for the purpose 
of backwards compatibility.

---------------------------------------------------------------------------------------------------
Module contents
---------------------------------------------------------------------------------------------------

.. autosummary:: 
	ConcentrationModel
	models
	concentration
	modelBullock01
	modelDuffy08
	modelKlypin11
	modelPrada12
	modelBhattacharya13
	modelDutton14
	modelDiemer15fromM
	modelDiemer15fromNu
	modelKlypin16fromM
	modelKlypin16fromNu
	modelLudlow16
	modelChild18
	modelDiemer19
	modelIshiyama21

---------------------------------------------------------------------------------------------------
Module reference
--------------------------------------------------------------------------------------------------- 
"""

###################################################################################################

import numpy as np
import scipy.interpolate
import scipy.optimize
import scipy.special
import warnings
from collections import OrderedDict

from colossus.utils import utilities
from colossus.utils import constants
from colossus.utils import storage
from colossus import defaults
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.halo import mass_so
from colossus.halo import mass_defs
from colossus.halo import profile_nfw
from colossus.halo import profile_einasto

###################################################################################################

class ConcentrationModel():
	"""
	Characteristics of concentration models.
	
	This object contains certain characteristics of a concentration model, most importantly the 
	mass definitions for which concentration can be output (note that the :func:`concentration` 
	function can automatically convert mass definitions). The :data:`models` dictionary contains 
	one item of this class for each available model.
	"""
	def __init__(self):
		
		self.func = None
		self.mdefs = []
		"""
		The native mass definition(s) of the model.
		"""
		self.universal = False
		"""
		Whether this model is universal in the sense that it can be evaluated at any mass or 
		redshift.
		"""
		self.depends_on_statistic = False
		"""
		Whether this model can predict different statistics such as mean and median concentration.
		"""
		
		return

###################################################################################################

models = OrderedDict()
"""
Dictionary containing a list of models.

An ordered dictionary containing one :class:`ConcentrationModel` entry for each model.
"""

models['bullock01'] = ConcentrationModel()
models['bullock01'].mdefs = ['200c']

models['duffy08'] = ConcentrationModel()
models['duffy08'].mdefs = ['200c', 'vir', '200m']

models['klypin11'] = ConcentrationModel()
models['klypin11'].mdefs = ['vir']

models['prada12'] = ConcentrationModel()
models['prada12'].mdefs = ['200c']
models['prada12'].universal = True

models['bhattacharya13'] = ConcentrationModel()
models['bhattacharya13'].mdefs = ['200c', 'vir', '200m']

models['dutton14'] = ConcentrationModel()
models['dutton14'].mdefs = ['200c', 'vir']

models['diemer15_orig'] = ConcentrationModel()
models['diemer15_orig'].mdefs = ['200c']
models['diemer15_orig'].universal = True
models['diemer15_orig'].depends_on_statistic = True

models['diemer15'] = ConcentrationModel()
models['diemer15'].mdefs = ['200c']
models['diemer15'].universal = True
models['diemer15'].depends_on_statistic = True

models['klypin16_m'] = ConcentrationModel()
models['klypin16_m'].mdefs = ['200c', 'vir']

models['klypin16_nu'] = ConcentrationModel()
models['klypin16_nu'].mdefs = ['200c', 'vir']

models['ludlow16'] = ConcentrationModel()
models['ludlow16'].mdefs = ['200c']
models['ludlow16'].universal = False

models['child18'] = ConcentrationModel()
models['child18'].mdefs = ['200c']

models['diemer19'] = ConcentrationModel()
models['diemer19'].mdefs = ['200c']
models['diemer19'].universal = False
models['diemer19'].depends_on_statistic = True

models['ishiyama21'] = ConcentrationModel()
models['ishiyama21'].mdefs = ['500c', '200c', 'vir']
models['ishiyama21'].universal = False

###################################################################################################

INVALID_CONCENTRATION = -1.0
"""The concentration value returned if the model routine fails to compute."""

###################################################################################################
# STORAGE SYSTEM
###################################################################################################

storageUser = None

def _getName():
	return 'concentration'

def _getHash():
	return 'concentration'

def _getStorageUser():

	global storageUser
	if storageUser is None:
		storageUser = storage.StorageUser('halo.concentration', 'rw', _getName, _getHash, None)
	
	return storageUser

###################################################################################################

def concentration(M, mdef, z,
				model = defaults.HALO_CONCENTRATION_MODEL, 
				statistic = defaults.HALO_CONCENTRATION_STATISTIC,
				conversion_profile = defaults.HALO_MASS_CONVERSION_PROFILE, 
				range_return = False, range_warning = True, **kwargs):
	"""
	Concentration as a function of halo mass and redshift.
	
	This function encapsulates all the concentration models implemented in this module. It
	automatically converts between mass definitions if necessary. For some models, a cosmology 
	must be set.
	
	In some cases, the function cannot return concentrations for the masses, redshift, or cosmology
	requested due to limitations on a particular concentration model. If so, the mask return 
	parameter contains a boolean list indicating which elements are valid. It is highly recommended
	that you switch this functionality on by setting ``range_return = True`` if you are not sure
	about the concentration model used.
	
	Some of the individual concentration model functions take additional parameters, e.g., they
	are calibrated for different halo samples. These parameters can be passed through this function
	as keyword args. Please see the documentations of the individual functions for details.
	
	Parameters
	----------
	M: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	mdef: str
		The mass definition in which the halo mass ``M`` is given, and in which ``c`` is returned. 
		See :doc:`halo_mass` for details.
	z: float
		Redshift
	model: str
		The model of the c-M relation used; see list above.
	statistic: str
		Some models distinguish between the ``mean`` and ``median`` concentration. Note that most 
		models do not, in which case this parameter is ignored.
	conversion_profile: str
		The profile form used to convert from one mass definition to another. See the
		:func:`~halo.mass_defs.changeMassDefinition` function).
	range_return: bool
		If ``True``, the function returns a boolean mask indicating the validty of the returned 
		concentrations.
	range_warning: bool
		If ``True``, a warning is thrown if the user requested a mass or redshift where the model is 
		not calibrated. This warning is suppressed if ``range_return == True``, since it is assumed 
		that the user will evaluate the returned mask array to check the validity of the returned
		concentrations.
	kwargs: kwargs
		Extra arguments passed to the function of the particular model. See the documentation of 
		those functions for valid arguments.
		
	Returns
	-------
	c: array_like
		Halo concentration(s) in the mass definition ``mdef``; has the same dimensions as ``M``.
	mask: array_like
		If ``range_return == True``, the function returns True/False values, where 
		False indicates that the model was not calibrated at the chosen mass or redshift; has the
		same dimensions as ``M``.
	"""
	
	guess_factors = [2.0, 5.0, 10.0, 100.0, 10000.0]
	n_guess_factors = len(guess_factors)

	# ---------------------------------------------------------------------------------------------
	# Evaluate the concentration model

	def evaluateC(func, M, universal, args, kwargs):
		if not universal:
			c, mask = func(M, *args, **kwargs)
		else:
			mask = None
			c = func(M, *args, **kwargs)
		return c, mask
	
	# ---------------------------------------------------------------------------------------------
	# This equation is zero for a mass MDelta (in the mass definition of the c-M model) when the
	# corresponding mass in the user's mass definition is M_desired.
	def eq(MDelta, M_desired, mdef_model, func, universal, args):
		
		cDelta, _ = evaluateC(func, MDelta, universal, args, kwargs)
		if cDelta < 0.0:
			return np.nan
		Mnew, _, _ = mass_defs.changeMassDefinition(MDelta, cDelta, z, mdef_model, mdef)
		
		return Mnew - M_desired

	# ---------------------------------------------------------------------------------------------
	# Distinguish between models
		
	if not model in models.keys():
		raise Exception('Unknown model, %s.' % (model))
	
	mdefs_model = models[model].mdefs
	universal = models[model].universal
	func = models[model].func
	args = (z,)
	if models[model].depends_on_statistic:
		args = args + (statistic,)
	
	# Now check whether the definition the user has requested is the native definition of the model.
	# If yes, we just return the model concentration. If not, the problem is much harder. Without 
	# knowing the concentration, we do not know what mass in the model definition corresponds to 
	# the input mass M. Thus, we need to   find both M and c iteratively.
	if mdef in mdefs_model:
		
		if len(mdefs_model) > 1:
			args = args + (mdef,)
		c, mask = evaluateC(func, M, universal, args, kwargs)
		
		# Generate a mask if the model doesn't return one
		if universal and range_return:
			if utilities.isArray(c):
				mask = np.ones((len(c)), dtype = bool)
			else:
				mask = True
			
	else:
		
		# Convert to array
		M_array, is_array = utilities.getArray(M)
		M_array = M_array.astype(float)
		N = len(M_array)
		mask = np.ones((N), dtype = bool)

		mdef_model = mdefs_model[0]
		if len(mdefs_model) > 1:
			args = args + (mdef_model,)

		# To a good approximation, the relation M2 / M1 = Delta1 / Delta2. We use this mass
		# as a guess around which to look for the solution.
		Delta_ratio = mass_so.densityThreshold(z, mdef) / mass_so.densityThreshold(z, mdef_model)
		M_guess = M_array * Delta_ratio
		c = np.zeros_like(M_array)
		
		for i in range(N):
			
			# Iteratively enlarge the search range, if necessary
			args_solver = M_array[i], mdef_model, func, universal, args
			j = 0
			MDelta = None
			while MDelta is None and j < n_guess_factors:
				M_min = M_guess[i] / guess_factors[j]
				M_max = M_guess[i] * guess_factors[j]
				
				# If we catch an exception at this point, or the model function can't compute c,
				# it's not gonna get better by going to more extreme masses, we might as well give
				# up.
				try:
					eq_min = eq(M_min, *args_solver)
					eq_max = eq(M_max, *args_solver)
					if np.isnan(eq_min) or np.isnan(eq_max):
						break						
					if eq_min * eq_max < 0.0:
						MDelta = scipy.optimize.brentq(eq, M_min, M_max, args = args_solver)
					else:
						j += 1
				except Exception as e:
					print('ERROR')
					print(e)
					break

			if MDelta is None or MDelta < 0.1:
				if range_warning:
					warnings.warn('Could not find concentration for model %s, mass %.2e, mdef %s.' \
						% (model, M_array[i], mdef))
				c[i] = INVALID_CONCENTRATION
				mask[i] = False
			
			else:
				cDelta, mask_element = evaluateC(func, MDelta, universal, args, kwargs)
				_, _, c[i] = mass_defs.changeMassDefinition(MDelta, cDelta, z, mdef_model,
										mdef, profile = conversion_profile)
				if not universal:
					mask[i] = mask_element
	
		# If necessary, convert back to scalars
		if not is_array:
			c = c[0]
			mask = mask[0]

	# Spit out warning if the range was violated
	if range_warning and not range_return and not universal:
		mask_array, _ = utilities.getArray(mask)
		if False in mask_array:
			warnings.warn('Some masses or redshifts are outside the validity of the concentration model.')
	
	if range_return:
		return c, mask
	else:
		return c

###################################################################################################
# BULLOCK ET AL 2001 / MACCIO ET AL 2008 MODEL
###################################################################################################

def modelBullock01(M200c, z):
	"""
	The model of Bullock et al 2001.
	
	This function implements the improved version of 
	`Maccio et al. 2008 <http://adsabs.harvard.edu/abs/2008MNRAS.391.1940M>`__. The model is 
	universal, but limited by the finite growth factor in a given cosmology which means that the 
	model cannot be evaluated for arbitrarily large masses (halos that will never collapse).
	  
	Parameters
	----------
	M200c: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
		
	Returns
	-------
	c200c: array_like
		Halo concentration; has the same dimensions as ``M``.
	mask: array_like
		Boolean, has the same dimensions as ``M``. Where ``False``, one or more input parameters were
		outside the range where the model was calibrated, and the returned concentration may not 
		be reliable.
	"""
	
	K = 3.85
	F = 0.01

	# Get an inverse interpolator to determine D+ from z. This is an advanced use of the internal
	# table system of the cosmology class.
	cosmo = cosmology.getCurrent()
	interp = cosmo._zInterpolator('lnzp1_growthfactor', cosmo._growthFactorExact, 
								inverse = True, future = True)
	Dmin = interp.get_knots()[0]
	Dmax = interp.get_knots()[-1]

	# The math works out such that we are looking for the redshift where the growth factor is
	# equal to the peak height of a halo with mass F * M.
	M_array, is_array = utilities.getArray(M200c)
	D_target = peaks.peakHeight(F * M_array, 0.0)
	mask = (D_target > Dmin) & (D_target < Dmax)
	N = len(M_array)
	c200c = np.zeros((N), dtype = float)
	H0 = cosmo.Hz(z)
	for i in range(N):
		if mask[i]:
			lnzp1 = interp(D_target[i])
			zc = np.exp(lnzp1) - 1.0
			Hc = cosmo.Hz(zc)
			c200c[i] = K * (Hc / H0)**0.6666
		else:
			c200c[i] = INVALID_CONCENTRATION
	
	if not is_array:
		c200c = c200c[0]
		mask = mask[0]
		
	return c200c, mask

###################################################################################################
# DUFFY ET AL 2008 MODEL
###################################################################################################

def modelDuffy08(M, z, mdef):
	"""
	The model of Duffy et al. 2008.
	
	This power-law fit was calibrated for a WMAP5 cosmology.
	  
	Parameters
	----------
	M: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition in which the mass is given, and in which concentration is returned.
		Can be ``200c``, ``vir``, or ``200m`` for this function. See :doc:`halo_mass` for details.
		
	Returns
	-------
	c: array_like
		Halo concentration; has the same dimensions as ``M``.
	mask: array_like
		Boolean, has the same dimensions as ``M``. Where ``False``, one or more input parameters were
		outside the range where the model was calibrated, and the returned concentration may not 
		be reliable.
	"""
	
	if mdef == '200c':
		A = 5.71
		B = -0.084
		C = -0.47
	elif mdef == 'vir':
		A = 7.85
		B = -0.081
		C = -0.71
	elif mdef == '200m':
		A = 10.14
		B = -0.081
		C = -1.01
	else:
		raise Exception('Invalid mass definition for Duffy et al. 2008 model, %s.' % mdef)

	c = A * (M / 2E12)**B * (1.0 + z)**C
	mask = (M >= 1E11) & (M <= 1E15) & (z <= 2.0)
	
	return c, mask

###################################################################################################
# KLYPIN ET AL 2011 MODEL
###################################################################################################

def modelKlypin11(Mvir, z):
	"""
	The model of Klypin et al 2011.
	
	This power-law fit was calibrated for the WMAP7 cosmology of the Bolshoi simulation. Note 
	that this model relies on concentrations that were measured from circular velocities, rather 
	than from a fit to the actual density profiles. Klypin et al. 2011 also give fits at particular 
	redshifts other than zero. However, there is no clear procedure to interpolate between redshifts, 
	particularly since the z = 0 relation has a different functional form than the high-z 
	relations. Thus, we only implement the z = 0 relation here.
	
	Parameters
	----------
	Mvir: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
		
	Returns
	-------
	cvir: array_like
		Halo concentration; has the same dimensions as ``Mvir``.
	mask: array_like
		Boolean, has the same dimensions as ``Mvir``. Where ``False``, one or more input parameters 
		were outside the range where the model was calibrated, and the returned concentration may 
		not be reliable.
	"""

	cvir = 9.6 * (Mvir / 1E12)**-0.075
	mask = (Mvir > 3E10) & (Mvir < 5E14) & (z < 0.01)

	return cvir, mask

###################################################################################################
# PRADA ET AL 2012 MODEL
###################################################################################################

def modelPrada12(M200c, z):
	"""
	The model of Prada et al 2012.
	
	This model predicts :math:`c_{200c}` based on the :math:`c-\\nu` relation. The model was 
	calibrated on the Bolshoi and Multidark simulations, but is in principle applicable to any 
	cosmology. The implementation follows equations 12 to 22 in Prada et al. 2012. This function 
	uses the exact values for peak height rather than their approximation.

	Parameters
	----------
	M200c: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
		
	Returns
	-------
	c200c: array_like
		Halo concentration; has the same dimensions as ``M200c``.
	"""

	def cmin(x):
		return 3.681 + (5.033 - 3.681) * (1.0 / np.pi * np.arctan(6.948 * (x - 0.424)) + 0.5)
	def smin(x):
		return 1.047 + (1.646 - 1.047) * (1.0 / np.pi * np.arctan(7.386 * (x - 0.526)) + 0.5)

	cosmo = cosmology.getCurrent()
	nu = peaks.peakHeight(M200c, z)

	a = 1.0 / (1.0 + z)
	x = (cosmo.Ode0 / cosmo.Om0) ** (1.0 / 3.0) * a
	B0 = cmin(x) / cmin(1.393)
	B1 = smin(x) / smin(1.393)
	temp_sig = 1.686 / nu
	temp_sigp = temp_sig * B1
	temp_C = 2.881 * ((temp_sigp / 1.257) ** 1.022 + 1) * np.exp(0.06 / temp_sigp ** 2)
	c200c = B0 * temp_C

	return c200c

###################################################################################################
# BHATTACHARYA ET AL 2013 MODEL
###################################################################################################

def modelBhattacharya13(M, z, mdef):
	"""
	The model of Bhattacharya et al 2013.
	
	This power-law fit in :math:`c-\\nu` was calibrated for a WMAP7 cosmology.

	Parameters
	----------
	M: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition in which the mass is given, and in which concentration is returned.
		Can be ``200c``, ``vir``, or ``200m``. See :doc:`halo_mass` for details.
		
	Returns
	-------
	c: array_like
		Halo concentration; has the same dimensions as ``M``.
	mask: array_like
		Boolean, has the same dimensions as ``M``. Where ``False``, one or more input parameters 
		were outside the range where the model was calibrated, and the returned concentration may 
		not be reliable.
	"""

	cosmo = cosmology.getCurrent()
	D = cosmo.growthFactor(z)
	
	# Note that peak height in the B13 paper is defined wrt. the mass definition in question, so 
	# we can just use M to evaluate nu. 
	nu = peaks.peakHeight(M, z)

	if mdef == '200c':
		c_fit = 5.9 * D**0.54 * nu**-0.35
	elif mdef == 'vir':
		c_fit = 7.7 * D**0.90 * nu**-0.29
	elif mdef == '200m':
		c_fit = 9.0 * D**1.15 * nu**-0.29
	else:
		raise Exception('Invalid mass definition for Bhattacharya et al. 2013 model, %s.' % mdef)
				
	M_min = 2E12
	M_max = 2E15
	if z > 0.5:
		M_max = 2E14
	if z > 1.5:
		M_max = 1E14
	mask = (M >= M_min) & (M <= M_max) & (z <= 2.0)
	
	return c_fit, mask

###################################################################################################
# DUTTON & MACCIO 2014 MODEL
###################################################################################################

def modelDutton14(M, z, mdef):
	"""
	The model of Dutton & Maccio 2014.
	
	This power-law fit was calibrated for the ``planck13`` cosmology.

	Parameters
	----------
	M: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition in which the mass is given, and in which concentration is returned.
		Can be ``200c`` or ``vir``. See :doc:`halo_mass` for details.
		
	Returns
	-------
	c: array_like
		Halo concentration; has the same dimensions as ``M``.
	mask: array_like
		Boolean, has the same dimensions as ``M``. Where ``False``, one or more input parameters 
		were outside the range where the model was calibrated, and the returned concentration may 
		not be reliable.
	"""

	if mdef == '200c':
		a = 0.520 + (0.905 - 0.520) * np.exp(-0.617 * z**1.21)
		b = -0.101 + 0.026 * z
	elif mdef == 'vir':
		a = 0.537 + (1.025 - 0.537) * np.exp(-0.718 * z**1.08)
		b = -0.097 + 0.024 * z
	else:
		raise Exception('Invalid mass definition for Dutton & Maccio 2014 model, %s.' % mdef)
	
	logc = a + b * np.log10(M / 1E12)
	c = 10**logc

	mask = (M > 1E10) & (z <= 5.0)

	return c, mask

###################################################################################################
# DIEMER & KRAVTSOV 2015 MODEL
###################################################################################################

def modelDiemer15fromM(M200c, z, statistic = 'median', original_params = False):
	"""
	The model of Diemer & Kravtsov 2015.
	
	This universal model in :math:`c_{\\rm 200c}-\\nu` space is a function of peak height and the 
	slope of the power spectrum. A cosmology must be set before executing this function.

	Parameters
	----------
	M200c: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	statistic: str
		Can be ``mean`` or ``median``.
	original_params: bool
		If ``True``, use the parameters given in the original paper. By default, use the updated
		parameters.
		
	Returns
	-------
	c200c: array_like
		Halo concentration; has the same dimensions as ``M200c``.
	
	See also
	--------
	modelDiemer15fromNu: The same function, but with peak height as input.
	"""
	
	cosmo = cosmology.getCurrent()
	
	if cosmo.power_law:
		n = cosmo.power_law_n * M200c / M200c
	else:
		n = _diemer15_n_fromM(M200c, original_params = original_params)
	
	nu = peaks.peakHeight(M200c, z)
	c200c = _diemer15(nu, n, statistic, original_params = original_params)

	return c200c

###################################################################################################

# Wrapper function for using the old DK15 parameters.

def _modelDiemer15fromM_orig(M200c, z, statistic = 'median'):

	return modelDiemer15fromM(M200c, z, statistic = statistic, original_params = True)

###################################################################################################

def modelDiemer15fromNu(nu200c, z, statistic = 'median', original_params = False):
	"""
	The model of Diemer & Kravtsov 2015.
	
	This universal model in :math:`c_{\\rm 200c}-\\nu` space is a function of peak height and the 
	slope of the power spectrum. A cosmology must be set before executing this function.

	Parameters
	----------
	nu200c: array_like
		Halo peak heights; can be a number or a numpy array. The peak heights must correspond to 
		:math:`M_{\\rm 200c}` and a top-hat filter.
	z: float
		Redshift
	statistic: str
		Can be ``mean`` or ``median``.
	original_params: bool
		If ``True``, use the parameters given in the original paper. By default, use the updated
		parameters.
		
	Returns
	-------
	c200c: array_like
		Halo concentration; has the same dimensions as ``nu200c``.
	
	See also
	--------
	modelDiemer15fromM: The same function, but with mass as input.
	"""

	cosmo = cosmology.getCurrent()
	
	if cosmo.power_law:
		n = cosmo.power_law_n * nu200c / nu200c
	else:
		n = _diemer15_n_fromnu(nu200c, z, original_params = original_params)
	
	ret = _diemer15(nu200c, n, statistic, original_params = original_params)

	return ret

###################################################################################################

# The universal prediction of the Diemer & Kravtsov 2014 model for a given peak height, power 
# spectrum slope, and statistic.

def _diemer15(nu, n, statistic = 'median', original_params = False):
	
	if original_params:
		DIEMER15_MEDIAN_PHI_0 = 6.58
		DIEMER15_MEDIAN_PHI_1 = 1.37
		DIEMER15_MEDIAN_ETA_0 = 6.82
		DIEMER15_MEDIAN_ETA_1 = 1.42
		DIEMER15_MEDIAN_ALPHA = 1.12
		DIEMER15_MEDIAN_BETA  = 1.69
		
		DIEMER15_MEAN_PHI_0 = 7.14
		DIEMER15_MEAN_PHI_1 = 1.60
		DIEMER15_MEAN_ETA_0 = 4.10
		DIEMER15_MEAN_ETA_1 = 0.75
		DIEMER15_MEAN_ALPHA = 1.40
		DIEMER15_MEAN_BETA  = 0.67
	else:
		DIEMER15_MEDIAN_PHI_0 = 6.58
		DIEMER15_MEDIAN_PHI_1 = 1.27
		DIEMER15_MEDIAN_ETA_0 = 7.28
		DIEMER15_MEDIAN_ETA_1 = 1.56
		DIEMER15_MEDIAN_ALPHA = 1.08
		DIEMER15_MEDIAN_BETA  = 1.77

		DIEMER15_MEAN_PHI_0 = 6.66
		DIEMER15_MEAN_PHI_1 = 1.37
		DIEMER15_MEAN_ETA_0 = 5.41
		DIEMER15_MEAN_ETA_1 = 1.06
		DIEMER15_MEAN_ALPHA = 1.22
		DIEMER15_MEAN_BETA  = 1.22

	if statistic == 'median':
		floor = DIEMER15_MEDIAN_PHI_0 + n * DIEMER15_MEDIAN_PHI_1
		nu0 = DIEMER15_MEDIAN_ETA_0 + n * DIEMER15_MEDIAN_ETA_1
		alpha = DIEMER15_MEDIAN_ALPHA
		beta = DIEMER15_MEDIAN_BETA
	elif statistic == 'mean':
		floor = DIEMER15_MEAN_PHI_0 + n * DIEMER15_MEAN_PHI_1
		nu0 = DIEMER15_MEAN_ETA_0 + n * DIEMER15_MEAN_ETA_1
		alpha = DIEMER15_MEAN_ALPHA
		beta = DIEMER15_MEAN_BETA
	else:
		raise Exception('Unknown statistic.')
	
	c = 0.5 * floor * ((nu0 / nu)**alpha + (nu / nu0)**beta)
	
	return c

###################################################################################################

# Compute the characteristic wavenumber for a particular halo mass.

def _diemer15_k_R(M, original_params = False):
	
	if original_params:
		DIEMER15_KAPPA = 0.69
	else:
		DIEMER15_KAPPA = 1.00

	R = peaks.lagrangianR(M)
	k_R = 2.0 * np.pi / R * DIEMER15_KAPPA

	return k_R

###################################################################################################

# Get the slope n = d log(P) / d log(k) at a scale k_R and a redshift z. The slope is computed from
# the Eisenstein & Hu 1998 approximation to the power spectrum (without BAO).

def _diemer15_n(k_R):

	if np.min(k_R) < 0:
		raise Exception('k_R < 0.')

	cosmo = cosmology.getCurrent()
	
	# The way we compute the slope depends on the settings in the cosmology module. If interpolation
	# tables are used, we can compute the slope directly from the spline interpolation which is
	# very fast. If not, we need to compute the slope manually.
	if cosmo.interpolation:
		n = cosmo.matterPowerSpectrum(k_R, model = 'eisenstein98_zb', derivative = True)
		
	else:
		# We need coverage to compute the local slope at kR, which can be an array. Thus, central
		# difference derivatives don't make much sense here, and we use a spline instead.
		k_min = np.min(k_R) * 0.9
		k_max = np.max(k_R) * 1.1
		logk = np.arange(np.log10(k_min), np.log10(k_max), 0.01)
		Pk = cosmo.matterPowerSpectrum(10**logk, model = 'eisenstein98_zb')
		interp = scipy.interpolate.InterpolatedUnivariateSpline(logk, np.log10(Pk))
		n = interp(np.log10(k_R), nu = 1)
	
	return n

###################################################################################################

# Wrapper for the function above which accepts M instead of k.

def _diemer15_n_fromM(M, original_params = False):

	k_R = _diemer15_k_R(M, original_params = original_params)
	n = _diemer15_n(k_R)
	
	return n

###################################################################################################

# Wrapper for the function above which accepts nu instead of M.

def _diemer15_n_fromnu(nu, z, original_params = False):

	M = peaks.massFromPeakHeight(nu, z)
	n = _diemer15_n_fromM(M, original_params = original_params)
	
	return n

###################################################################################################
# KLYPIN ET AL 2016 MODELS
###################################################################################################

def modelKlypin16fromM(M, z, mdef):
	"""
	The model of Klypin et al 2016, based on mass.
	
	The paper suggests both peak height-based and mass-based fitting functions for concentration; 
	this function implements the mass-based version. The fits are given for the ``planck13`` and 
	``bolshoi`` cosmologies. Thus, the user must set one of those cosmologies before evaluating 
	this model. The best-fit parameters refer to the mass-selected samples of all halos (as 
	opposed to :math:`v_{max}`-selected samples, or relaxed halos).

	Parameters
	----------
	M: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition in which the mass(es) are given, and in which concentration is returned.
		Can be ``200c`` or ``vir``. See :doc:`halo_mass` for details.
		
	Returns
	-------
	c: array_like
		Halo concentration; has the same dimensions as ``M``.
	mask: array_like
		Boolean, has the same dimensions as ``M``. Where ``False``, one or more input parameters 
		were outside the range where the model was calibrated, and the returned concentration may 
		not be reliable.
	
	See also
	--------
	modelKlypin16fromNu: The model of Klypin et al 2016, based on peak height.
	"""
	
	if not mdef in ['200c', 'vir']:
		raise Exception('Invalid mass definition for Klypin et al 2016 m-based model, %s.' % mdef)

	cosmo = cosmology.getCurrent()

	if cosmo.name == 'planck13':
		z_bins = [0.0, 0.35, 0.5, 1.0, 1.44, 2.15, 2.5, 2.9, 4.1, 5.4]
		if mdef == '200c':
			C0_bins = [7.4, 6.25, 5.65, 4.3, 3.53, 2.7, 2.42, 2.2, 1.92, 1.65]
			gamma_bins = [0.120, 0.117, 0.115, 0.110, 0.095, 0.085, 0.08, 0.08, 0.08, 0.08]
			M0_bins = [5.5E5, 1E5, 2E4, 900.0, 300.0, 42.0, 17.0, 8.5, 2.0, 0.3]
		elif mdef == 'vir':
			C0_bins = [9.75, 7.25, 6.5, 4.75, 3.8, 3.0, 2.65, 2.42, 2.1, 1.86]
			gamma_bins = [0.110, 0.107, 0.105, 0.1, 0.095, 0.085, 0.08, 0.08, 0.08, 0.08]
			M0_bins = [5E5, 2.2E4, 1E4, 1000.0, 210.0, 43.0, 18.0, 9.0, 1.9, 0.42]
			
	elif cosmo.name == 'bolshoi':
		z_bins = [0.0, 0.5, 1.0, 1.44, 2.15, 2.5, 2.9, 4.1]
		if mdef == '200c':
			C0_bins = [6.6, 5.25, 3.85, 3.0, 2.1, 1.8, 1.6, 1.4]
			gamma_bins = [0.110, 0.105, 0.103, 0.097, 0.095, 0.095, 0.095, 0.095]
			M0_bins = [2E6, 6E4, 800.0, 110.0, 13.0, 6.0, 3.0, 1.0]
		elif mdef == 'vir':
			C0_bins = [9.0, 6.0, 4.3, 3.3, 2.3, 2.1, 1.85, 1.7]
			gamma_bins = [0.1, 0.1, 0.1, 0.1, 0.095, 0.095, 0.095, 0.095]
			M0_bins = [2E6, 7E3, 550.0, 90.0, 11.0, 6.0, 2.5, 1.0]
		
	else:
		raise Exception('Invalid cosmology for Klypin et al 2016 m-based model, %s.' % cosmo.name)

	C0 = np.interp(z, z_bins, C0_bins)
	gamma = np.interp(z, z_bins, gamma_bins)
	M0 = np.interp(z, z_bins, M0_bins)
	M0 *= 1E12

	c = C0 * (M / 1E12)**-gamma * (1.0 + (M / M0)**0.4)
	
	mask = (M > 1E10) & (z <= z_bins[-1])

	return c, mask

###################################################################################################

def modelKlypin16fromNu(M, z, mdef):
	"""
	The model of Klypin et al 2016, based on peak height.
	
	The paper suggests both peak height-based and mass-based fitting functions for concentration; 
	this function implements the peak height-based version. The fits are given for the ``planck13`` 
	and ``bolshoi`` cosmologies. Thus, the user must set one of those cosmologies before evaluating 
	this model. The best-fit parameters refer to the mass-selected samples of all halos (as 
	opposed to :math:`v_{max}`-selected samples, or relaxed halos). The fits refer to median 
	concentrations at fixed mass and redshift.

	Parameters
	----------
	M: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition in which the mass is given, and in which concentration is returned.
		Can be ``200c`` or ``vir``. See :doc:`halo_mass` for details.
		
	Returns
	-------
	c: array_like
		Halo concentration; has the same dimensions as ``M``.
	mask: array_like
		Boolean, has the same dimensions as ``M``. Where ``False``, one or more input parameters were
		outside the range where the model was calibrated, and the returned concentration may not 
		be reliable.
	
	See also
	--------
	modelKlypin16fromM: The model of Klypin et al 2016, based on mass.
	"""

	if mdef == '200c':
		z_bins = [0.0, 0.38, 0.5, 1.0, 1.44, 2.5, 2.89, 5.41]
		a0_bins = [0.4, 0.65, 0.82, 1.08, 1.23, 1.6, 1.68, 1.7]
		b0_bins = [0.278, 0.375, 0.411, 0.436, 0.426, 0.375, 0.360, 0.351]
	elif mdef == 'vir':
		z_bins = [0.0, 0.38, 0.5, 1.0, 1.44, 2.5, 5.5]
		a0_bins = [0.75, 0.9, 0.97, 1.12, 1.28, 1.52, 1.62]
		b0_bins = [0.567, 0.541, 0.529, 0.496, 0.474, 0.421, 0.393]
	else:
		raise Exception('Invalid mass definition for Klypin et al 2016 peak height-based model, %s.' % mdef)

	nu = peaks.peakHeight(M, z)
	sigma = constants.DELTA_COLLAPSE / nu
	a0 = np.interp(z, z_bins, a0_bins)
	b0 = np.interp(z, z_bins, b0_bins)

	sigma_a0 = sigma / a0
	c = b0 * (1.0 + 7.37 * sigma_a0**0.75) * (1.0 + 0.14 * sigma_a0**-2.0)
	
	mask = (M > 1E10) & (z <= z_bins[-1])

	return c, mask

###################################################################################################

def modelLudlow16(M200c, z):
	"""
	The model of Ludlow et al 2016.
	
	This function finds the solution by brute-force computation of a large array of concentrations.
	This technique is efficient if M200c is a large array, but inefficient for few values. 
	Moreover, the function assumes a LCDM cosmology and is not strictly valid for wCDM or other
	DE models. The code was adapted from a routine by Steven Murray.

	Parameters
	----------
	M200c: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
		
	Returns
	-------
	c: array_like
		Halo concentration; has the same dimensions as ``M200c``.
	mask: array_like
		Boolean, has the same dimensions as ``M200c``. Where ``False``, one or more input 
		parameters were outside the range where the model was calibrated, and the returned 
		concentration may not be reliable.
	"""
	
	f = 0.02
	C = 650.0
	
	cosmo = cosmology.getCurrent()

	# Make sure we are dealing with an array
	M200c, is_array = utilities.getArray(M200c)
	M200c = M200c.astype(float)

	# We solve this model by computing Equations 6 and 7 in Ludlow+16 for a large range of
	# concentrations. 
	c_array = np.logspace(0, 2, 200)
	
	# Use an Einasto profile with alpha = 0.18
	p_ein = profile_einasto.EinastoProfile(M = 1E12, c = 1.0, z = z, mdef = '200c', alpha = 0.18)
	rs_ein = p_ein.par['rs']
	M_ratio = p_ein.enclosedMassInner(rs_ein) / p_ein.enclosedMassInner(rs_ein * c_array)
	
	# Formation density in units of critical density. We invert this to find the formation redshift
	# as a function.
	rho_f_rho_c = 200.0 * c_array**3 * M_ratio / C
	
	# This equation is a slight of hand. Actually, we should numerically solve for the redshift 
	# where the critical density is rho_f_rho_c * rho_c(now). This is not implemented in the 
	# cosmology module though. Instead, we assume a simple LCDM cosmology with no relativistic
	# species, a case for which we can solve the equation directly for the formation redshift.
	# We also need to cut the c array at this point because not all concentrations will lead to 
	# defined formation redshifts.
	t1 = (rho_f_rho_c * (cosmo.Om0 * (1.0 + z)**3 + cosmo.Ode0) - cosmo.Ode0) / cosmo.Om0
	mask_c_array = (t1 > 0.0)
	zf = t1[mask_c_array]**(1.0 / 3.0) - 1.0
	c_array = c_array[mask_c_array]
	M_ratio = M_ratio[mask_c_array]

	# We can now solve Equation 7 numerically. Note that there are terms which have the length
	# of our artificial c array and terms of the length of M200c. We compute the right hand side
	# for each combination.
	sigma2_fM = cosmo.sigma(peaks.lagrangianR(f * M200c), 0.0)**2
	sigma2_M = cosmo.sigma(peaks.lagrangianR(M200c), 0.0)**2
	delta_z = constants.DELTA_COLLAPSE / cosmo.growthFactor(z)
	delta_zf = constants.DELTA_COLLAPSE / cosmo.growthFactor(zf)
	rhs = scipy.special.erfc(np.outer(delta_zf - delta_z , 1.0 / np.sqrt(2.0 * (sigma2_fM - sigma2_M))))

	# We can now find the solution by interpolation: the concentration in the c_array vector
	# appears where lhs - rhs = 0.
	c200c = np.zeros_like(M200c)
	mask = np.ones_like(M200c, bool)
	for i in range(len(M200c)):
		lhs_rhs = M_ratio - rhs[:, i]
		mask_nan = np.logical_not(np.isnan(lhs_rhs))
		lhs_rhs = lhs_rhs[mask_nan]
		if (np.count_nonzero(lhs_rhs < 0.0) == 0) or (np.count_nonzero(lhs_rhs > 0.0) == 0):
			mask[i] = False
			c200c[i] = INVALID_CONCENTRATION
		else:
			c200c[i] = np.interp(0.0, lhs_rhs, c_array[mask_nan])

	# Convert back to scalar if necessary
	if not is_array:
		c200c = c200c[0]
		mask = mask[0]

	return c200c, mask

###################################################################################################

def modelChild18(M200c, z, halo_sample = 'individual_all'):
	"""
	The model of Child et al 2018.
	
	The authors suggest multiple fitting functions, multiple ways to define the halo sample, and
	concetration measured by multiple profile fits. By default, this function represents Equation 18 
	using the parameters for individual halos (as opposed to stacks) and all halos (as opposed to 
	relaxed halos). Other samples can be selected with the ``halo_sample`` parameter.
	
	The mass definition is 200c for this model. The halo sample used reaches 2.1E11 
	:math:`M_{\\odot}/h`, this function returns a mask indicating this mass range.

	Parameters
	----------
	M200c: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	halo_sample: str
		Can be ``individual_all`` (default), ``individual_relaxed`` (the mean concentration of 
		individual, relaxed halos), ``stacked_nfw`` (the stacked profile with with an NFW profile), 
		and ``stacked_einasto`` (the stacked profile with with an Einasto profile).
		
	Returns
	-------
	c: array_like
		Halo concentration; has the same dimensions as ``M``.
	mask: array_like
		Boolean, has the same dimensions as ``M``. Where ``False``, one or more input parameters were
		outside the range where the model was calibrated, and the returned concentration may not 
		be reliable.
	"""

	if halo_sample == 'individual_all':
		m = -0.10
		A = 3.44
		b = 430.49
		c0 = 3.19
	elif halo_sample == 'individual_relaxed':
		m = -0.09
		A = 2.88
		b = 1644.53
		c0 = 3.54
	elif halo_sample == 'stacked_nfw':
		m = -0.07
		A = 4.61
		b = 638.65
		c0 = 3.59
	elif halo_sample == 'stacked_einasto':
		m = -0.01
		A = 63.2
		b = 431.48
		c0 = 3.36
	else:
		raise Exception('Unknown halo sample for child18 concentration model, %s.' % (halo_sample))

	mask = (M200c >= 2.1E11) & (z >= 0.0) & (z <= 4.0)
	
	Mstar = peaks.nonLinearMass(z)
	M_MT = M200c / (Mstar * b)
	
	c200c = c0 + A * (M_MT**m * (1.0 + M_MT)**-m - 1.0)

	return c200c, mask

###################################################################################################

# The effective exponent of linear growth, the logarithmic derivative of the growth factor.

def _diemer19_alpha_eff(z):

	cosmo = cosmology.getCurrent()
	D = cosmo.growthFactor(z, derivative = 0)
	dDdz = cosmo.growthFactor(z, derivative = 1)
	alpha_eff = -dDdz * (1.0 + z) / D
	
	return alpha_eff

###################################################################################################

# This function is a generalized version of the Diemer & Joyce 2019 model, where the user can 
# pass a set of parameters and give mass in any definition.

def _diemer19_general(M, z, params, ps_args = defaults.PS_ARGS):

	# ---------------------------------------------------------------------------------------------

	# Currently, only NFW predictions are implemented. In principle, the model can predict 
	# c based on other mass profiles, but the parameters would need to be re-tuned. The 
	# implementation of the Einasto profile below is for test purposes only.
	
	profile = 'nfw'

	# ---------------------------------------------------------------------------------------------

	def getTableName(profile):

		if profile == 'nfw':
			table_name = 'diemer19_'
		elif profile == 'einasto':
			table_name = 'diemer19_einasto_'
		else:
			raise Exception('Unknown profile type, %s. Valid are "nfw" or "einasto".' % (profile))
		
		return table_name
	
	# ---------------------------------------------------------------------------------------------
	# The G(c) inverse function that needs to be mumerically inverted is tabulated, the table 
	# inverted to give c(G, n). This is a little tricky because G(c) has a minimum that depends on
	# n.
	
	def computeGcTable(profile):
	
		n_G = 80
		n_n = 40
		n_c = 80

		n = np.linspace(-4.0, 0.0, n_n)
		c = np.linspace(-1.0, 3.0, n_c)
		
		# The left hand side of the equation in DJ19
		lin_c = 10**c
		
		if profile == 'nfw':
			mu = profile_nfw.NFWProfile.mu(lin_c)
		elif profile == 'einasto':
			p_ein = profile_einasto.EinastoProfile(M = 1E12, c = 5.0, z = 0.0, mdef = '200c', alpha = 0.18)
			rs_ein = p_ein.par['rs']
			mu = p_ein.enclosedMassInner(rs_ein * lin_c) / p_ein.enclosedMassInner(rs_ein)
		else:
			raise Exception('Unknown profile type, %s. Valid are "nfw" or "einasto".' % (profile))

		lhs = np.log10(lin_c[:, None] / mu[:, None]**((5.0 + n) / 6.0))
		
		# At very low concentration and shallow slopes, the LHS begins to rise again. This will cause
		# issues with the inversion. We set those parts of the curve to the minimum concentration of 
		# a given n bin.
		mask_ascending = np.ones_like(lhs, bool)
		mask_ascending[:-1, :] = (np.diff(lhs, axis = 0) > 0.0)
		
		# Create a table of c as a function of G and n. First, use the absolute min and max of G as 
		# the table range
		G_min = np.min(lhs)
		G_max = np.max(lhs)
		G = np.linspace(G_min, G_max, n_G)
		
		gc_table = np.ones((n_G, n_n), float) * -10.0
		mins = np.zeros_like(n)
		maxs = np.zeros_like(n)
		for i in range(n_n):
			
			# We interpolate only the ascending values to get c(G)
			mask_ = mask_ascending[:, i]
			lhs_ = lhs[mask_, i]
			mins[i] = np.min(lhs_)
			maxs[i] = np.max(lhs_)
			interp = scipy.interpolate.InterpolatedUnivariateSpline(lhs_, c[mask_])
		
			# Not all G exist for all n
			mask = (G >= mins[i]) & (G <= maxs[i])
			res = interp(G[mask])
			gc_table[mask, i] = res
	
			mask_low = (G < mins[i])
			gc_table[mask_low, i] = np.min(res)
			mask_high = (G > maxs[i])
			gc_table[mask_high, i] = np.max(res)

		# Store the objects using the storage module. An interpolator will automatically be 
		# created.
		storageUser = _getStorageUser()
		table_name = getTableName(profile)
		
		object_data = (G, n, gc_table)
		storageUser.storeObject('%sGc' % table_name, object_data = object_data, persistent = True)
		object_data = np.array([n, mins])
		storageUser.storeObject('%sGmin' % table_name, object_data = object_data, persistent = True)
		object_data = np.array([n, maxs])
		storageUser.storeObject('%sGmax' % table_name, object_data = object_data, persistent = True)

		return

	# ---------------------------------------------------------------------------------------------
	# Try to load the interpolators from storage. If they do not exist, create them.
	
	def getGcTable(profile):
		
		storageUser = _getStorageUser()
		table_name = getTableName(profile)
		
		interp_Gc = storageUser.getStoredObject('%sGc' % table_name, interpolator = True, 
											store_interpolator = True)
		if interp_Gc is None:
			computeGcTable(profile)
			interp_Gc = storageUser.getStoredObject('%sGc' % table_name, interpolator = True, 
											store_interpolator = True)
		
		interp_Gmin = storageUser.getStoredObject('%sGmin' % table_name, interpolator = True, 
											store_interpolator = True)
		interp_Gmax = storageUser.getStoredObject('%sGmax' % table_name, interpolator = True, 
											store_interpolator = True)

		# These interpolators are created at the same time as the Gc interpolator. If they do not
		# exist, something went wrong.
		if interp_Gmin is None or interp_Gmax is None:
			raise Exception('Loading table for diemer19 concentration model failed.')
		
		return interp_Gc, interp_Gmin, interp_Gmax

	# ---------------------------------------------------------------------------------------------

	# Compute peak height, n_eff, and alpha_eff
	nu = peaks.peakHeight(M, z, ps_args = ps_args)
	n_eff = peaks.powerSpectrumSlope(nu, z, slope_type = 'sigma', scale = params['kappa'], ps_args = ps_args)
	alpha_eff = _diemer19_alpha_eff(z)

	is_array = utilities.isArray(nu)
	if not is_array:
		nu = np.array([nu])
		n_eff = np.array([n_eff])
		alpha_eff = np.array([alpha_eff])
	
	# Compute input parameters and the right-hand side of the c-M equation. We use interpolation
	# tables to find the concentration at which the equation gives that RHS.
	A_n = params['a_0'] * (1.0 + params['a_1'] * (n_eff + 3.0))
	B_n = params['b_0'] * (1.0 + params['b_1'] * (n_eff + 3.0))
	C_alpha = 1.0 - params['c_alpha'] * (1.0 - alpha_eff)
	rhs = np.log10(A_n / nu * (1.0 + nu**2 / B_n))
	
	# Get interpolation table
	interp_Gc, interp_Gmin, interp_Gmax = getGcTable(profile)
	
	# Mask out values of rhs for which do G not exist. The interpolator will still work because it
	# is based on a full grid of G values, but we mask out the invalid array elements.
	mask = (rhs >= interp_Gmin(n_eff)) & (rhs <= interp_Gmax(n_eff))
	c = 10**interp_Gc(rhs, n_eff, grid = False) * C_alpha
	c[np.logical_not(mask)] = INVALID_CONCENTRATION

	if not is_array:
		c = c[0]
		mask = mask[0]
		
	return c, mask

###################################################################################################

def modelDiemer19(M200c, z, statistic = 'median', ps_args = defaults.PS_ARGS):
	"""
	The model of Diemer & Joyce 2019.
	
	This model improves on the Diemer & Kravtsov 2015 model in a number of ways. First, it is based
	on a mathematical derivation of the evolution of concentration at the low-mass end. This more
	physically motivated functional form allows fewer free parameters (six instead of seven). 
	Second, because of the improved functional form, the model improves the fit, particularly to
	scale-free cosmologies. Finally, the new model fixed a slight numerical bug in the DK15 model.
	
	The first time this model is ever called, it will compute a lookup table (in three dimensions)
	for c(G, n) where G is the left-hand side of the c-M model equation. This table then serves as
	a lookup and avoids having to numerically solve the equation.
	
	Parameters
	----------
	M200c: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	statistic: str
		Can be ``mean`` or ``median``.
	ps_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` 
		function, and functions that depend on it such as the power spectrum slope.
		
	Returns
	-------
	c200c: array_like
		Halo concentration; has the same dimensions as ``M200c``.
	"""

	if statistic == 'median':
		params = dict( \
		kappa             = 0.41,
		a_0               = 2.45,
		a_1               = 1.82,
		b_0               = 3.20,
		b_1               = 2.30,
		c_alpha           = 0.21)
	elif statistic == 'mean':
		params = dict( \
		kappa             = 0.42,
		a_0               = 2.37,
		a_1               = 1.74,
		b_0               = 3.39,
		b_1               = 1.82,
		c_alpha           = 0.20)
	else:
		raise Exception('Statistic %s not implemented in diemer19 model.' % statistic)
		
	return _diemer19_general(M200c, z, params, ps_args = ps_args)

###################################################################################################

def modelIshiyama21(M, z, mdef, ps_args = defaults.PS_ARGS, c_type = 'fit', halo_sample = 'all'):
	"""
	The model of Ishiyama et al 2021.
	
	This model constitutes a recalibration of the Diemer & Joyce 2019 model based on the Uchuu
	simulation. The model provides median concentrations only but was calibrated for the 500c, 
	200c, and vir mass definitions and for concentrations derived from an NFW fit and estimated 
	from Vmax.
	
	Parameters
	----------
	M: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition in which the mass is given, and in which concentration is returned.
		Can be ``200c`` or ``vir``. See :doc:`halo_mass` for details.
	ps_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` 
		function, and functions that depend on it such as the power spectrum slope.
	c_type: str
		The type of concentration; can be ``fit`` for concentrations derived from an NFW fit or
		``vmax`` for concentrations derived from the ratio of Vmax and V200c. Note that no fit is
		provided for ``mdef = 500c`` and ``c_type = vmax``.
	halo_sample: str
		The halo sample on which the fit is based; can be ``all`` or ``relaxed``.
		
	Returns
	-------
	c: array_like
		Halo concentration; has the same dimensions as ``M``.
	"""

	if not halo_sample in ['all', 'relaxed']:
		raise Exception('Invalid halo sample (%s) for ishiyama21 model, allowed are all and relaxed.' % halo_sample)
	
	if not c_type in ['fit', 'vmax']:
		raise Exception('Invalid concentration type (%s) for ishiyama21 model, allowed are fit and vmax.' % c_type)
	
	if not mdef in ['500c', '200c', 'vir']:
		raise Exception('Invalid mdef (%s) for ishiyama21 model, allowed are 500c, 200c, and vir.' % mdef)
	
	if c_type == 'vmax' and mdef == '500c':
		raise Exception('Invalid parameter combination for ishiyama21 model, mdef cannot be 500c when c_type is vmax.')
	
	if c_type == 'fit':
		
		if mdef == '500c':
			
			if halo_sample == 'all':
				params = dict( \
				kappa             = 1.83,
				a_0               = 1.95,
				a_1               = 1.17,
				b_0               = 3.57,
				b_1               = 0.91,
				c_alpha           = 0.26)
			elif halo_sample == 'relaxed':
				params = dict( \
				kappa             = 0.38,
				a_0               = 1.44,
				a_1               = 3.41,
				b_0               = 2.86,
				b_1               = 2.99,
				c_alpha           = 0.42)
			
		elif mdef == '200c':
			
			if halo_sample == 'all':
				params = dict( \
				kappa             = 1.19,
				a_0               = 2.54,
				a_1               = 1.33,
				b_0               = 4.04,
				b_1               = 1.21,
				c_alpha           = 0.22)
			elif halo_sample == 'relaxed':
				params = dict( \
				kappa             = 0.60,
				a_0               = 2.14,
				a_1               = 2.63,
				b_0               = 1.69,
				b_1               = 6.36,
				c_alpha           = 0.37)
							
		elif mdef == 'vir':
			
			if halo_sample == 'all':
				params = dict( \
				kappa             = 1.64,
				a_0               = 2.67,
				a_1               = 1.23,
				b_0               = 3.92,
				b_1               = 1.30,
				c_alpha           = -0.19)
			elif halo_sample == 'relaxed':
				params = dict( \
				kappa             = 1.22,
				a_0               = 2.52,
				a_1               = 1.87,
				b_0               = 2.13,
				b_1               = 4.19,
				c_alpha           = -0.017)				

	elif c_type == 'vmax':
				
		if mdef == '200c':
			if halo_sample == 'all':
				params = dict( \
				kappa             = 1.10,
				a_0               = 2.30,
				a_1               = 1.64,
				b_0               = 1.72,
				b_1               = 3.60,
				c_alpha           = 0.32)
			elif halo_sample == 'relaxed':
				params = dict( \
				kappa             = 1.79,
				a_0               = 2.15,
				a_1               = 2.06,
				b_0               = 0.88,
				b_1               = 9.24,
				c_alpha           = 0.51)				

		elif mdef == 'vir':
			
			if halo_sample == 'all':
				params = dict( \
				kappa             = 0.76,
				a_0               = 2.34,
				a_1               = 1.82,
				b_0               = 1.83,
				b_1               = 3.52,
				c_alpha           = -0.18)
			elif halo_sample == 'relaxed':
				params = dict( \
				kappa             = 2.40,
				a_0               = 2.27,
				a_1               = 1.80,
				b_0               = 0.56,
				b_1               = 13.24,
				c_alpha           = 0.079)		
		
	return _diemer19_general(M, z, params, ps_args = ps_args)

###################################################################################################
# Pointers to model functions
###################################################################################################

models['bullock01'].func = modelBullock01
models['duffy08'].func = modelDuffy08
models['klypin11'].func = modelKlypin11
models['prada12'].func = modelPrada12
models['bhattacharya13'].func = modelBhattacharya13
models['dutton14'].func = modelDutton14
models['diemer15_orig'].func = _modelDiemer15fromM_orig
models['diemer15'].func = modelDiemer15fromM
models['klypin16_m'].func = modelKlypin16fromM
models['klypin16_nu'].func = modelKlypin16fromNu
models['ludlow16'].func = modelLudlow16
models['child18'].func = modelChild18
models['diemer19'].func = modelDiemer19
models['ishiyama21'].func = modelIshiyama21

###################################################################################################
