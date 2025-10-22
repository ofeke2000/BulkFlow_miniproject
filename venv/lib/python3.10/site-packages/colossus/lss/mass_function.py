###################################################################################################
#
# mass_function.py          (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
The mass function quantifies how many halos of a certain mass exist at a given redshift and in a
given cosmology.

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

This module implements a number of models for the halo mass function. The easiest and recommended
use of the module is through the :func:`massFunction` function, a wrapper for all individual
models::
	
	from colossus.cosmology import cosmology
	from colossus.lss import mass_function
	
	cosmology.setCosmology('WMAP9')
	mfunc_so = mass_function.massFunction(1E12, 0.0, mdef = 'vir', model = 'tinker08')
	mfunc_fof = mass_function.massFunction(1E12, 0.0, mdef = 'fof', model = 'watson13')

Of course, the function accepts numpy arrays for the mass parameter. By default, the mass function 
is returned as :math:`f(\\sigma)`, the natural units in Press-Schechter theory, where

.. math::
	\\frac{dn}{d \\ln(M)} = f(\\sigma) \\frac{\\rho_0}{M} \\frac{d \\ln(\\sigma^{-1})}{d \\ln(M)} .

Here, :math:`\\rho_0` is the matter density at :math:`z = 0` and :math:`\\sigma` is the variance 
on the lagrangian size scale of the halo mass in question (see 
:func:`~cosmology.cosmology.Cosmology.sigma`). The function can also return the mass function 
in other units, namely as the number density per logarithmic interval in mass, :math:`dn/d\\ln(M)` 
(units of comoving :math:`({\\rm Mpc}/h)^{-3}`, indicated by ``q_out = dndlnM``) and as 
:math:`M^2 / \\rho_0 dn/dM` (dimensionless, indicated by ``q_out = M2dndM``). These conversions 
can separately be performed separately using the :func:`convertMassFunction` function. Please see 
the :doc:`tutorials` for more extensive code examples.

---------------------------------------------------------------------------------------------------
Mass function models
---------------------------------------------------------------------------------------------------

The following models are supported, and are listed in the :data:`models` dictionary. Their ID can 
be passed as the ``model`` parameter to the :func:`massFunction` function: 

============== ================ =========== ======================================
ID             mdefs            z-dep.      Reference
============== ================ =========== ======================================
press74        fof  	        delta_c     `Press& Schechter 1974 <http://adsabs.harvard.edu/abs/1974ApJ...187..425P>`__
sheth99	       fof  	        delta_c     `Sheth & Tormen 1999 <http://adsabs.harvard.edu/abs/1999MNRAS.308..119S>`__
jenkins01      fof  	        No	        `Jenkins et al. 2001 <http://adsabs.harvard.edu/abs/2001MNRAS.321..372J>`__
reed03	       fof  	        delta_c     `Reed et al. 2003 <http://adsabs.harvard.edu/abs/2003MNRAS.346..565R>`__
warren06       fof  	        No	        `Warren et al. 2006 <http://adsabs.harvard.edu/abs/2006ApJ...646..881W>`__
reed07	       fof   	        delta_c     `Reed et al. 2007 <http://adsabs.harvard.edu/abs/2007MNRAS.374....2R>`__
tinker08       Any SO 	        Yes	        `Tinker et al. 2008 <http://adsabs.harvard.edu/abs/2008ApJ...688..709T>`__
crocce10       fof   	        No          `Crocce et al. 2010 <http://adsabs.harvard.edu/abs/2010MNRAS.403.1353C>`__
bhattacharya11 fof   	        Yes         `Bhattacharya et al. 2011 <http://adsabs.harvard.edu/abs/2011ApJ...732..122B>`__
courtin11      fof              No          `Courtin et al. 2011 <http://adsabs.harvard.edu/abs/2011MNRAS.410.1911C>`__
angulo12       fof  	        No          `Angulo et al. 2012 <http://adsabs.harvard.edu/abs/2012MNRAS.426.2046A>`__
watson13       fof, any SO      Yes (SO)    `Watson et al. 2013 <http://adsabs.harvard.edu/abs/2013MNRAS.433.1230W>`__
bocquet16      200m,200c,500c   Yes         `Bocquet et al. 2016 <http://adsabs.harvard.edu/abs/2016MNRAS.456.2361B>`__
despali16      Any SO           Yes         `Despali et al. 2016 <http://adsabs.harvard.edu/abs/2016MNRAS.456.2486D>`__
comparat17     vir              No          `Comparat et al. 2017 <https://ui.adsabs.harvard.edu//#abs/2017MNRAS.469.4157C/abstract>`__
diemer20       sp-apr-*         No          `Diemer 2020b <https://ui.adsabs.harvard.edu/abs/2020arXiv200710346D/abstract>`__
seppi20        vir              Yes         `Seppi et al. 2020 <https://ui.adsabs.harvard.edu/abs/2020arXiv200803179S/abstract>`__
============== ================ =========== ======================================

Note that the mass definition (set to ``fof`` by default) needs to match one of the allowed mass 
definitions of the chosen model. For most models, only ``fof`` is allowed, but some SO models are 
calibrated to various mass definitions. The ``tinker08`` model can handle any overdensity between 
200m and 3200m (though they can be expressed as critical and virial overdensities as well).

There are two different types of redshift dependence of :math:`f(\\sigma)` listed above: some 
models explicitly depend on redshift (e.g., ``bhattacharya11``), some models only change through 
the small variation of the collapse overdensity :math:`\\delta_{\\rm c}` 
(see :func:`~lss.peaks.collapseOverdensity`). The ``tinker08`` model depends on redshift only through 
the conversion of the overdensity threshold.

Mass functions based on the Press-Schechter formalism (i.e., parameterizations in terms of f(sigma)
which includes all of the mass functions implemented here) are, in principle, supposed to be valid 
across redshifts and cosmologies. However, it has been shown that this is true only approximately. 
Nevertheless, this function does not check whether the user input is outside the range where the
model was calibrated, be it in halo mass, redshift, or cosmology.

---------------------------------------------------------------------------------------------------
Module contents
--------------------------------------------------------------------------------------------------- 

.. autosummary::
	HaloMassFunctionModel
	models
	massFunction
	convertMassFunction
	modelPress74
	modelSheth99
	modelJenkins01
	modelReed03
	modelWarren06
	modelReed07
	modelTinker08
	modelCrocce10
	modelCourtin11
	modelBhattacharya11
	modelAngulo12
	modelWatson13
	modelBocquet16
	modelDespali16
	modelComparat17
	modelDiemer20
	modelSeppi20
	
---------------------------------------------------------------------------------------------------
Module reference
--------------------------------------------------------------------------------------------------- 
"""

###################################################################################################

import numpy as np
import scipy.integrate
from collections import OrderedDict

from colossus import defaults
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.halo import mass_so

###################################################################################################

class HaloMassFunctionModel():
	"""
	Characteristics of halo mass function models.
	
	This class contains certain characteristics of a mass function model, namely the mass 
	definitions for which it is valid, whether it explicitly depends on the redshift (in some cases
	this dependence arises because of the slight dependence of the collapse overdensity on z), and
	how the collapse overdensity is computed by default (if applicable).

	The :data:`models` dictionary contains one item of this class for each available model.
	"""
	def __init__(self):
		
		self.func = None
		
		self.z_dependence = False
		"""
		Indicates whether :math:`f(\\sigma)` depends on redshift in this model.
		"""
		
		self.ps_dependence = False
		"""
		Indicates whether :math:`f(\\sigma)` depends directly on the power spectrum.
		"""
		
		self.sigma_dependence = False
		"""
		Indicates whether :math:`f(\\sigma)` depends directly on :math:`\\sigma(M)` beyond the
		input :math:`\\sigma`.
		"""
		
		self.deltac_dependence = False
		"""
		Indicates whether :math:`f(\\sigma)` depends directly on the collapse overdensity.
		"""
	
		self.mdef_dependence = False
		"""
		Indicates whether :math:`f(\\sigma)` depends on the mass definition (for SO models).
		"""
		
		self.mdefs = []
		"""
		A list of mass definitions for which this model is valid. See :doc:`halo_mass` for details.
		"""
		
		return

###################################################################################################

models = OrderedDict()
"""
Dictionary containing a list of models.

An ordered dictionary containing one :class:`HaloMassFunctionModel` entry for each model.
"""

models['press74'] = HaloMassFunctionModel()
models['press74'].mdefs = ['fof']
models['press74'].deltac_dependence = True

models['sheth99'] = HaloMassFunctionModel()
models['sheth99'].mdefs = ['fof']
models['sheth99'].deltac_dependence = True

models['jenkins01'] = HaloMassFunctionModel()
models['jenkins01'].mdefs = ['fof']

models['reed03'] = HaloMassFunctionModel()
models['reed03'].mdefs = ['fof']
models['reed03'].deltac_dependence = True

models['warren06'] = HaloMassFunctionModel()
models['warren06'].mdefs = ['fof']

models['reed07'] = HaloMassFunctionModel()
models['reed07'].mdefs = ['fof']
models['reed07'].ps_dependence = True
models['reed07'].sigma_dependence = True
models['reed07'].deltac_dependence = True

models['tinker08'] = HaloMassFunctionModel()
models['tinker08'].mdefs = ['*']
models['tinker08'].z_dependence = True
models['tinker08'].mdef_dependence = True

models['crocce10'] = HaloMassFunctionModel()
models['crocce10'].mdefs = ['fof']
models['crocce10'].z_dependence = True

models['bhattacharya11'] = HaloMassFunctionModel()
models['bhattacharya11'].mdefs = ['fof']
models['bhattacharya11'].deltac_dependence = True

models['courtin11'] = HaloMassFunctionModel()
models['courtin11'].mdefs = ['fof']

models['angulo12'] = HaloMassFunctionModel()
models['angulo12'].mdefs = ['fof']

models['watson13'] = HaloMassFunctionModel()
models['watson13'].mdefs = ['fof', '*']
models['watson13'].z_dependence = True
models['watson13'].mdef_dependence = True

models['bocquet16'] = HaloMassFunctionModel()
models['bocquet16'].mdefs = ['200m', '200c', '500c']
models['bocquet16'].z_dependence = True
models['bocquet16'].ps_dependence = True
models['bocquet16'].sigma_dependence = True
models['bocquet16'].deltac_dependence = True
models['bocquet16'].mdef_dependence = True

models['despali16'] = HaloMassFunctionModel()
models['despali16'].mdefs = ['*']
models['despali16'].z_dependence = True
models['despali16'].mdef_dependence = True

models['comparat17'] = HaloMassFunctionModel()
models['comparat17'].mdefs = ['vir']

models['diemer20'] = HaloMassFunctionModel()
models['diemer20'].mdefs = ['sp-apr-mn', 'sp-apr-p*']
models['diemer20'].z_dependence = True
models['diemer20'].deltac_dependence = True
models['diemer20'].mdef_dependence = True

models['seppi20'] = HaloMassFunctionModel()
models['seppi20'].mdefs = ['vir']
models['seppi20'].z_dependence = True
models['seppi20'].deltac_dependence = True

###################################################################################################

def massFunction(x, z, q_in = 'M', q_out = 'f', mdef = 'fof', 
				model = defaults.HALO_MASS_FUNCTION_MODEL,
				ps_args = defaults.PS_ARGS, sigma_args = defaults.SIGMA_ARGS, 
				deltac_args = defaults.DELTAC_ARGS, **kwargs):
	"""
	The abundance of halos as a function of mass, variance, or peak height.
	
	This function is a wrapper for all individual models implemented in this module. It accepts
	either mass, variance, or peak height as input (controlled by the ``q_in`` parameter, see the 
	:func:`~cosmology.cosmology.Cosmology.sigma` and :func:`~lss.peaks.peakHeight` functions for
	details on those quantities). The output units are controlled by the ``q_out`` parameter, see 
	the basic usage section for details.
	
	This function also deals with different mass definitions. By default, the definition is set to
	``fof``, but some models can also return the mass function for SO definitions (see 
	:data:`models` list). 
	
	Parameters
	----------
	x: array_like
		Either halo mass in :math:`M_{\\odot}/h`, the variance :math:`\\sigma`, or peak height
		:math:`\\nu`, depending on the value of the ``q_in`` parameter; can be a number or a numpy 
		array.
	z: float
		Redshift
	q_in: str
		Either ``M``, ``sigma``, or ``nu`` indicating which is passed for the ``x`` parameter.
	q_out: str
		The units in which the mass function is returned; can be ``f``, ``dndlnM``, or ``M2dndM``.
		See the `Basics`_ section for details on these units.
	mdef: str
		The mass definition in which the halo mass M is given (or from which the variance or peak
		height were computed). The returned mass function refers to this mass definition. Please 
		see the model table for the mass definitions for which each model is valid.
		See :doc:`halo_mass` for details.
	model: str
		The model of the mass function used.
	ps_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` 
		function.
	sigma_args: dict
		Extra arguments to be passed to the :func:`~cosmology.cosmology.Cosmology.sigma` function 
		if mass is converted to a variance.
	deltac_args: dict
		Extra parameters that are passed to the :func:`~lss.peaks.collapseOverdensity` function; see 
		the documentation of the individual models for possible parameters. Note that not all 
		models of the mass function rely on the collapse overdensity.
	kwargs: kwargs
		Extra arguments passed to the function of the particular model. See the documentation of 
		those functions for valid arguments.
		
	Returns
	-------
	mfunc: array_like
		The halo mass function in the desired units.
	"""

	cosmo = cosmology.getCurrent()
	
	# Compute sigma
	M = None
	if q_in == 'M':
		M = x
		R = peaks.lagrangianR(x)
		sigma = cosmo.sigma(R, z, ps_args = ps_args, **sigma_args)
	elif q_in == 'sigma':
		sigma = x
	elif q_in == 'nu':
		delta_c = peaks.collapseOverdensity(z = z, **deltac_args)
		sigma = delta_c / x
	else:
		raise Exception('Unknown input quantity, %s.' % (q_in))

	# Check that the model exists
	if not model in models.keys():
		raise Exception('Unknown model, %s.' % (model))
	model_props = models[model]
	
	# Check that the mass definition and model are compatible.
	found = False
	if '*' in model_props.mdefs:
		found = True
	if mdef in model_props.mdefs:
		found = True
	if mdef.startswith('sp-apr-p') and ('sp-apr-p*' in model_props.mdefs):
		found = True
		
	if not found:
		raise Exception('The mass definition %s is not allowed for model %s. Allowed are: %s.' % \
					(mdef, model, str(model_props.mdefs)))
	
	# Create the argument list depending on the model and evaluate it.
	args = (sigma,)
	if model_props.z_dependence or model_props.ps_dependence \
		or model_props.sigma_dependence or model_props.deltac_dependence:
		args += (z,)
	if model_props.mdef_dependence:
		args += (mdef,)
	if model_props.ps_dependence:
		args += (ps_args,)
	if model_props.sigma_dependence:
		args += (sigma_args,)
	if model_props.deltac_dependence:
		args += (deltac_args,)

	f = model_props.func(*args, **kwargs)

	if q_out == 'f':
		mfunc = f
	else:
		if len(f.shape) > 1:
			raise Exception('Mass function model returned multi-dimensional output; cannot convert from f to %s.' \
						% (q_out))
		
		if M is None:
			R = cosmo.sigma(sigma, z, inverse = True, ps_args = ps_args, **sigma_args)
			M = peaks.lagrangianM(R)
		mfunc = convertMassFunction(f, M, z, 'f', q_out, ps_args = ps_args, sigma_args = sigma_args)

	return mfunc

###################################################################################################

def convertMassFunction(mfunc, M, z, q_in, q_out, 
					ps_args = defaults.PS_ARGS, sigma_args = defaults.SIGMA_ARGS):
	"""
	Convert different units of the mass function.
	
	Virtually all models parameterize the mass function in the natural Press-Schechter units, 
	:math:`f(\\sigma)`. This function convert any allowed units into any other units. See the 
	basic usage section for details on the meaning of the units.
	
	Parameters
	----------
	mfunc: array_like
		The mass function in the input units.
	M: array_like
		Halo mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
	z: float
		Redshift
	q_in: str
		The units in which the mass function is input; can be ``f``, ``dndlnM``, or ``M2dndM``. See
		`Basics`_ section for the meaning of these units.
	q_out: str
		The units in which the mass function is returned; see above.
	ps_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` 
		function.
	sigma_args: dict
		Extra arguments to be passed to the :func:`~cosmology.cosmology.Cosmology.sigma` function 
		when mass is converted to a variance.
		
	Returns
	-------
	mfunc: array_like
		The halo mass function in the desired units.
	"""

	if q_in == q_out:
		return mfunc

	cosmo = cosmology.getCurrent()
	R = peaks.lagrangianR(M)
	d_ln_sigma_d_ln_R = cosmo.sigma(R, z, derivative = True, ps_args = ps_args, **sigma_args)
	rho_Mpc = cosmo.rho_m(0.0) * 1E9
	
	if q_in == 'dndlnM':
		dn_dlnM = mfunc

	elif q_in == 'f':
		dn_dlnM = -(1.0 / 3.0) * mfunc * rho_Mpc / M * d_ln_sigma_d_ln_R
	
	elif q_in == 'M2dndM':
		dn_dlnM = mfunc / M * rho_Mpc
	
	else:
		raise Exception('Cannot handle input quantity %s.' % q_in)
	
	if q_out == 'dndlnM':
		mfunc_out = dn_dlnM
		
	elif q_out == 'M2dndM':
		mfunc_out = dn_dlnM * M / rho_Mpc
		
	elif q_out == 'f':
		mfunc_out = -3.0 * dn_dlnM * M / rho_Mpc / d_ln_sigma_d_ln_R
	
	else:
		raise Exception('Cannot handle output quantity %s.' % q_out)
	
	return mfunc_out

###################################################################################################
# FUNCTIONS FOR INDIVIDUAL MASS FUNCTION MODELS
###################################################################################################

def modelPress74(sigma, z, deltac_args = {'corrections': True}):
	"""
	The mass function model of Press & Schechter 1974.
	
	This model depends on redshift only through the collapse overdensity :math:`\\delta_{\\rm c}`.
	By default, the collapse overdensity is computed including corrections due to cosmology.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
	z: float
		Redshift
	deltac_args: dict
		Arguments passed to the :func:`~lss.peaks.collapseOverdensity` function.
	
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
	
	delta_c = peaks.collapseOverdensity(z = z, **deltac_args)
	nu = delta_c / sigma
	f = np.sqrt(2.0 / np.pi) * nu * np.exp(-0.5 * nu**2)
	
	return f

###################################################################################################

def modelSheth99(sigma, z, deltac_args = {'corrections': True}):
	"""
	The mass function model of Sheth & Tormen 1999.
	
	This model was created to account for the differences between the classic 
	`Press& Schechter 1974 <http://adsabs.harvard.edu/abs/1974ApJ...187..425P>`__ model
	and measurements of the halo abundance in numerical simulations. The model is given in Equation 
	10. Note that, by default, the collapse overdensity is computed including corrections due to 
	cosmology. Compared to the paper, the equation implemented here contains an extra factor of 2 
	because the original equation refers to the A = 1/2 normalization of Press & Schechter.
	This model is sometimes also known as "SMT" for 
	`Sheth, Mo and Tormen 2001 <http://adsabs.harvard.edu/abs/2001MNRAS.323....1S>`__ who use the
	same fitting function.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
	z: float
		Redshift
	deltac_args: dict
		Arguments passed to the :func:`~lss.peaks.collapseOverdensity` function.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
		
	delta_c = peaks.collapseOverdensity(z = z, **deltac_args)
	A = 0.3222
	a = 0.707
	p = 0.3
	
	nu_p = a * delta_c**2 / sigma**2
	f = A * np.sqrt(nu_p * 2.0 / np.pi) * np.exp(-0.5 * nu_p) * (1.0 + nu_p**-p)
	
	return f

###################################################################################################

def modelJenkins01(sigma):
	"""
	The mass function model of Jenkins et al 2001.
	
	The model is given in Equation 9. It does not explicitly rely on the collapse overdensity and
	thus has no redshift evolution.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
	
	f = 0.315 * np.exp(-np.abs(np.log(1.0 / sigma) + 0.61)**3.8)
	
	return f

###################################################################################################

def modelReed03(sigma, z, deltac_args = {'corrections': True}):
	"""
	The mass function model of Reed et al 2003.
	
	This model corrects the 
	`Sheth & Tormen 1999 <http://adsabs.harvard.edu/abs/1999MNRAS.308..119S>`__ model at high 
	masses, the functional form is given in Equation 9.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
	z: float
		Redshift
	deltac_args: dict
		Arguments passed to the :func:`~lss.peaks.collapseOverdensity` function.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
		
	f_ST = modelSheth99(sigma, z, deltac_args = deltac_args)
	f = f_ST * np.exp(-0.7 / (sigma * np.cosh(2.0 * sigma)**5))
	
	return f

###################################################################################################

def modelWarren06(sigma):
	"""
	The mass function model of Warren et al 2006.
	
	This model does not explicitly rely on the collapse overdensity and thus has no redshift 
	dependence. The functional form is given in Equation 5. 
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
		
	A = 0.7234
	a = 1.625
	b = 0.2538
	c = 1.1982
	
	f = A * (sigma**-a + b) * np.exp(-c / sigma**2)
	
	return f

###################################################################################################

def modelReed07(sigma, z, ps_args = defaults.PS_ARGS, sigma_args = defaults.SIGMA_ARGS,
			deltac_args = {'corrections': True}, exact_n = True):
	"""
	The mass function model of Reed et al 2007.
	
	This model takes the changing slope of the power spectrum into account. This slope can be 
	computed numerically using the Colossus interpolation tables, or using the approximation in 
	Equation 14 (the more exact numerical version is the default). The paper gives two expressions 
	for their mass function, this code uses Equation 11.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
	z: float
		Redshift
	ps_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` 
		function.
	sigma_args: dict
		Extra arguments to be passed to the :func:`~cosmology.cosmology.Cosmology.sigma` function 
		when mass is converted to a variance.
	deltac_args: dict
		Arguments passed to the :func:`~lss.peaks.collapseOverdensity` function.
	exact_n: bool
		Compute the slope of the power spectrum numerically or approximate it.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
			
	delta_c = peaks.collapseOverdensity(z = z, **deltac_args)

	if exact_n:
		cosmo = cosmology.getCurrent()
		R = cosmo.sigma(sigma, z, inverse = True, ps_args = ps_args, **sigma_args)
		d_ln_sigma_d_ln_R = cosmo.sigma(R, z, derivative = True, ps_args = ps_args, **sigma_args)
		n_eff = -2.0 * d_ln_sigma_d_ln_R - 3.0
	else:
		mz = 0.55 - 0.32 * (1.0 - (1.0 / (1.0 + z)))**5
		rz = -1.74 - 0.8 * np.abs((np.log(1.0 / (1.0 + z))))**0.8
		n_eff = mz * np.log(1.0 / sigma) + rz
	
	A = 0.3222
	ca = 0.764
	c = 1.08
	a = ca / c
	p = 0.3
	
	log_sig_inv = np.log(1.0 / sigma)
	nu = delta_c / sigma

	G1 = np.exp(-(log_sig_inv - 0.4)**2/(2*0.6**2))
	G2 = np.exp(-(log_sig_inv - 0.75)**2/(2*0.2**2))

	f = A * np.sqrt(2.0 * a / np.pi) * (1.0 + (a * nu**2)**-p + 0.6 * G1 + 0.4 * G2) * nu \
				* np.exp(-ca * nu**2 / 2.0 - 0.03 * nu**0.6 / (n_eff + 3)**2)
	
	return f

###################################################################################################

def modelTinker08(sigma, z, mdef):
	"""
	The mass function model of Tinker et al 2008.
	
	This model was the first calibrated for SO rather than FOF masses, and can predict the mass 
	function for a large range of overdensities (:math:`200 \\leq \\Delta_{\\rm m} \\leq 3200`).
	The authors found that the SO mass function is not universal with redshift and took this 
	dependence into account explicitly (Equations 3-8).
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition to which ``sigma`` corresponds. See :doc:`halo_mass` for details.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
	
	if mdef == 'fof':
		raise Exception('Cannot use mass definition fof for Tinker 08 model, need an SO definition.')
	
	cosmo = cosmology.getCurrent()
	Delta_m = round(mass_so.densityThreshold(z, mdef) / cosmo.rho_m(z))

	fit_Delta = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])
	fit_A0 = np.array([0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260])
	fit_a0 = np.array([1.47, 1.52, 1.56, 1.61, 1.87, 2.13, 2.30, 2.53, 2.66])
	fit_b0 = np.array([2.57, 2.25, 2.05, 1.87, 1.59, 1.51, 1.46, 1.44, 1.41])
	fit_c0 = np.array([1.19, 1.27, 1.34, 1.45, 1.58, 1.80, 1.97, 2.24, 2.44])
		
	# Compute fit parameters and f-function
	if Delta_m < fit_Delta[0]:
		raise Exception('Delta_m %d is too small, minimum %d.' % (Delta_m, fit_Delta[0]))
	if Delta_m > fit_Delta[-1]:
		raise Exception('Delta_m %d is too large, maximum %d.' % (Delta_m, fit_Delta[-1]))
	
	A0 = np.interp(Delta_m, fit_Delta, fit_A0)
	a0 = np.interp(Delta_m, fit_Delta, fit_a0)
	b0 = np.interp(Delta_m, fit_Delta, fit_b0)
	c0 = np.interp(Delta_m, fit_Delta, fit_c0)
	
	alpha = 10**(-(0.75 / np.log10(Delta_m / 75.0))**1.2)
	A = A0 * (1.0 + z)**-0.14
	a = a0 * (1.0 + z)**-0.06
	b = b0 * (1.0 + z)**-alpha
	c = c0
	f = A * ((sigma / b)**-a + 1.0) * np.exp(-c / sigma**2)
	
	return f

###################################################################################################

def modelCrocce10(sigma, z):
	"""
	The mass function model of Crocce et al 2010.
	
	This function was calibrated between z = 0 and 1, and is given in Equation 22.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
		
	zp1 = 1.0 + z
	A = 0.58 * zp1**-0.13
	a = 1.37 * zp1**-0.15
	b = 0.30 * zp1**-0.084
	c = 1.036 * zp1**-0.024
	
	f = A * (sigma**-a + b) * np.exp(-c / sigma**2)
	
	return f

###################################################################################################

def modelCourtin11(sigma):
	"""
	The mass function model of Courtin et al 2011.
	
	The model uses the same functional form as the 
	`Sheth & Tormen 1999 <http://adsabs.harvard.edu/abs/1999MNRAS.308..119S>`__, but with different
	parameters and a fixed collapse overdensity :math:`\\delta_{\\rm c} = 1.673`. Note that there
	appears to be an error in Equations 8 and 22: the factor of :math:`\\sqrt{a}` should be in the 
	numerator rather than denominator in order to reproduce the ST expression. Other authors have 
	taken the formula literally, e.g. 
	`Watson et al. 2013 <http://adsabs.harvard.edu/abs/2013MNRAS.433.1230W>`__. Here, we assume that 
	the intended expression is, indeed, that of ST.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
		
	delta_c = 1.673
	A = 0.348
	a = 0.695
	p = 0.1

	nu_p = a * delta_c**2 / sigma**2
	f = A * np.sqrt(nu_p * 2.0 / np.pi) * np.exp(-0.5 * nu_p) * (1.0 + nu_p**-p)

	return f

###################################################################################################

def modelBhattacharya11(sigma, z, deltac_args = {'corrections': False}):
	"""
	The mass function model of Bhattacharya et al 2011.
	
	This model was calibrated between redshift 0 and 2. The authors found that varying 
	:math:`\\delta_{\\rm c}` does not account for the redshift dependence. Thus, they keep it 
	fixed and added an explicit redshift dependence into the model. The functional form is given 
	in Table 4.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
	z: float
		Redshift
	deltac_args: dict
		Arguments passed to the :func:`~lss.peaks.collapseOverdensity` function.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
		
	delta_c = peaks.collapseOverdensity(z = z, **deltac_args)
	nu = delta_c / sigma
	nu2 = nu**2

	zp1 = 1.0 + z
	A = 0.333 * zp1**-0.11
	a = 0.788 * zp1**-0.01
	p = 0.807
	q = 1.795

	f = A * np.sqrt(2 / np.pi) * np.exp(-a * nu2 * 0.5) * (1.0 + (a * nu2)**-p) * (nu * np.sqrt(a))**q

	return f

###################################################################################################

def modelAngulo12(sigma):
	"""
	The mass function model of Angulo et al 2012.
	
	The model is specified in Equation 2. Note that there is a typo in this equation that was 
	corrected in other implementations of the model.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
	
	f = 0.201 * ((2.08 / sigma)**1.7 + 1.0) * np.exp(-1.172 / sigma**2)

	return f

###################################################################################################

def modelWatson13(sigma, z, mdef):
	"""
	The mass function model of Watson et al 2013.
	
	This function contains multiple models, namely the redshift-independent model for the FOF mass 
	function as given in Equation 12, the redshift-evolving fit to the AHF mass function given in 
	Equations 14-16, as well as the dependence on mass definition given in Equations 17-19. At 
	z = 0 and z > 6 the authors suggest slightly different sets of parameters for the SO mass
	function which are used at those redshifts. Please note that the different parameterizations do
	not agree, as in, the redshift-dependent formula at z ~ 0 does not match the z = 0 expression. 
	To get the redshift-dependent version at z = 0, simply use a very small redshift.
	
	Note that there is a typo in the paper text, where the values of :math:`\\alpha` and 
	:math:`\\beta` are switched. The correct values are in Table 2.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition to which ``sigma`` corresponds. See :doc:`halo_mass` for details.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
	
	if mdef == 'fof':

		A = 0.282
		alpha = 2.163
		beta = 1.406
		gamma = 1.210
		
		f = A * ((beta / sigma)**alpha + 1.0) * np.exp(-gamma / sigma**2)

	else:
	
		cosmo = cosmology.getCurrent()
		Om = cosmo.Om(z)
		Delta_m = mass_so.densityThreshold(z, mdef) / cosmo.rho_m(z)
		Delta_178 = Delta_m / 178.0
		
		if z == 0.0:
			A = 0.194
			alpha = 1.805
			beta = 2.267
			gamma = 1.287
		elif z > 6.0:
			A = 0.563
			alpha = 3.810
			beta = 0.874
			gamma = 1.453
		else:
			A = Om * (1.097 * (1.0 + z)**-3.216 + 0.074)
			alpha = Om * (5.907 * (1.0 + z)**-3.058 + 2.349)
			beta = Om * (3.136 * (1.0 + z)**-3.599 + 2.344)
			gamma = 1.318

		f_178 = A * ((beta / sigma)**alpha + 1.0) * np.exp(-gamma / sigma**2)
		C = np.exp(0.023 * (Delta_178 - 1.0))
		d = -0.456 * Om - 0.139
		Gamma = C * Delta_178**d * np.exp(0.072 * (1.0 - Delta_178) / sigma**2.130)
		f = f_178 * Gamma

	return f

###################################################################################################

def modelBocquet16(sigma, z, mdef, ps_args = defaults.PS_ARGS, sigma_args = defaults.SIGMA_ARGS,
			deltac_args = {'corrections': True}, hydro = True):
	"""
	The mass function model of Bocquet et al 2016.
	
	The parameters were separately fit for dark matter-only and hydrodynamical simulations. The
	fits cover three mass definitions (200m, 200c, and 500c); requesting a different definition
	raises an exception. Note that, beyond different best-fit parameters, the 200c and 500c 
	mass functions rely on a conversion that depends explicitly on redshift, cosmology, and 
	halo mass (rather than peak height).
	
	Due to this conversion to halo mass, the model does implicitly depend on the power spectrum, 
	variance etc parameters beyond the value of sigma.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition to which ``sigma`` corresponds. See :doc:`halo_mass` for details.
	ps_args: dict
		Arguments passed to the :func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` 
		function.
	sigma_args: dict
		Extra arguments to be passed to the :func:`~cosmology.cosmology.Cosmology.sigma` function 
		when mass is converted to a variance.
	deltac_args: dict
		Arguments passed to the :func:`~lss.peaks.collapseOverdensity` function.
	hydro: bool
		If True, return the model for hydro simulations, otherwise DM-only.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
	
	if mdef == '200m':
		if hydro:
			A0 = 0.228
			a0 = 2.15
			b0 = 1.69
			c0 = 1.30
			Az = 0.285
			az = -0.058
			bz = -0.366
			cz = -0.045
		else:
			A0 = 0.175
			a0 = 1.53
			b0 = 2.55
			c0 = 1.19
			Az = -0.012
			az = -0.040
			bz = -0.194
			cz = -0.021
	elif mdef == '200c':
		if hydro:
			A0 = 0.202
			a0 = 2.21
			b0 = 2.00
			c0 = 1.57
			Az = 1.147
			az = 0.375
			bz = -1.074 
			cz = -0.196
		else:
			A0 = 0.222
			a0 = 1.71
			b0 = 2.24
			c0 = 1.46
			Az = 0.269
			az = 0.321
			bz = -0.621
			cz = -0.153
	elif mdef == '500c':
		if hydro:
			A0 = 0.180
			a0 = 2.29
			b0 = 2.44
			c0 = 1.97
			Az = 1.088
			az = 0.150
			bz = -1.008
			cz = -0.322
		else:
			A0 = 0.241
			a0 = 2.18
			b0 = 2.35
			c0 = 2.02
			Az = 0.370
			az = 0.251
			bz = -0.698
			cz = -0.310
	else:
		raise Exception('The mass definition %s is not allowed for Bocquet et al. 2016 model; allowed are 200m, 200c, and 500c.')
	
	zp1 = z + 1.0
	A = A0 * zp1**Az
	a = a0 * zp1**az
	b = b0 * zp1**bz
	c = c0 * zp1**cz
	
	f = A * ((sigma / b)**-a + 1.0) * np.exp(-c / sigma**2)

	if mdef in ['200c', '500c']:
		cosmo = cosmology.getCurrent()
		Omega_m = cosmo.Om(z)
		delta_c = peaks.collapseOverdensity(z = z, **deltac_args)
		nu = delta_c / sigma
		MDelta = peaks.massFromPeakHeight(nu, z, ps_args = ps_args, sigma_args = sigma_args, 
										deltac_args = deltac_args)
		ln_MDelta_Msun = np.log(MDelta / cosmo.h)
		
	if mdef == '200c':
		gamma0 = 3.54E-2 + Omega_m**0.09
		gamma1 = 4.56E-2 + 2.68E-2 / Omega_m
		gamma2 = 0.721 + 3.50E-2 / Omega_m
		gamma3 = 0.628 + 0.164 / Omega_m
		delta0 = -1.67E-2 + 2.18E-2 * Omega_m
		delta1 = 6.52E-3 - 6.86E-3 * Omega_m
		gamma = gamma0 + gamma1 * np.exp(-((gamma2 - z) / gamma3)**2)
		delta = delta0 + delta1 * z
		M200c_M200m = gamma + delta * ln_MDelta_Msun
		f *= M200c_M200m

	elif mdef == '500c':
		alpha0 = 0.880 + 0.329 * Omega_m
		alpha1 = 1.00 + 4.31E-2 / Omega_m
		alpha2 = -0.365 + 0.254 / Omega_m
		alpha = alpha0 * (alpha1 * z + alpha2) / (z + alpha2)
		beta = -1.7E-2 + 3.74E-3 * Omega_m
		M500c_M200m = alpha + beta * ln_MDelta_Msun
		f *= M500c_M200m
	
	return f

###################################################################################################

def modelDespali16(sigma, z, mdef, deltac_args = {'corrections': True}, ellipsoidal = False):
	"""
	The mass function model of Despali et al 2016.
	
	The parameters were fit for a number of different mass definitions, redshifts, and cosmologies.
	Here, we use the most general parameter set using the rescaling formula for redshift and mass 
	definitions given in Equation 12. Furthermore, the user can choose between results based on 
	conventional SO halo finding and an ellipsoidal halo finder.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The mass definition to which ``sigma`` corresponds. See :doc:`halo_mass` for details.
	ellipsoidal: bool
		If True, return the results for an ellipsoidal halo finder, otherwise standard SO.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""

	Delta = mass_so.densityThreshold(z, mdef)
	Delta_vir = mass_so.densityThreshold(z, 'vir')
	x = np.log10(Delta / Delta_vir)
	
	if ellipsoidal:
		A = -0.1768 * x + 0.3953
		a = 0.3268 * x**2 + 0.2125 * x + 0.7057
		p = -0.04570 * x**2 + 0.1937 * x + 0.2206
	else:
		A = -0.1362 * x + 0.3292
		a = 0.4332 * x**2 + 0.2263 * x + 0.7665
		p = -0.1151 * x**2 + 0.2554 * x + 0.2488
		
	delta_c = peaks.collapseOverdensity(z = z, **deltac_args)
	
	nu_p = a * delta_c**2 / sigma**2
	f = 2.0 * A * np.sqrt(nu_p / 2.0 / np.pi) * np.exp(-0.5 * nu_p) * (1.0 + nu_p**-p)
		
	return f

###################################################################################################

def modelComparat17(sigma):
	"""
	The mass function model of Comparat et al 2017.
	
	This model was calibrated only at redshift 0, and for the virial SO mass definition. The 
	cosmology used is the ``multidark-planck`` cosmology (which is very close to ``planck13``). 
	Outside of this redshift and cosmology, the model relies on the universality of the mass
	function. 
	
	The functional form is the same as in the Bhattacharya et al 2011 model, but without their 
	redshift dependence. The parameters used here are updated compared to the published version of 
	the paper.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
		
	delta_c = peaks.collapseOverdensity()
	nu = delta_c / sigma
	nu2 = nu**2

	A = 0.324
	a = 0.897
	p = 0.624
	q = 1.589

	f = A * np.sqrt(2 / np.pi) * np.exp(-a * nu2 * 0.5) * (1.0 + (a * nu2)**-p) * (nu * np.sqrt(a))**q

	return f

###################################################################################################

def modelDiemer20(sigma, z, mdef, deltac_args = {'corrections': True}):
	"""
	The splashback mass function model of Diemer 2020b.
	
	This model represents a universal fitting function for splashback masses measured dynamically,
	that is, as the mean or percentiles of the particle apocenter distribution. The model is 
	nominally valid for any redshift or cosmology, see the corresponding paper for details on its
	accuracy.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
	z: float
		Redshift
	mdef: str
		The splashback mass definition to which ``sigma`` corresponds. This function predicts the 
		mass function for dynamically measured splashback masses, namely the mean (``sp-apr-mn``)
		or higher percentiles (``sp-apr-p75`` etc) of the splashback distribution. Any percentile
		between 50 and 90 is allowed. See :doc:`halo_mass` for details.
	deltac_args: dict
		Arguments passed to the :func:`~lss.peaks.collapseOverdensity` function.
		
	Returns
	-------
	f: array_like
		The halo mass function :math:`f(\\sigma)`, has the same dimensions as ``sigma``.
	"""
	
	delta_c = peaks.collapseOverdensity(z = z, **deltac_args)
	nu = delta_c / sigma
	
	if mdef == 'sp-apr-mn':
		A = 0.124399
		a = 1.191457
		b = 0.337871
		c = 0.431710
		
	elif mdef.startswith('sp-apr-p'):
		A = 0.091878
		a0 = 1.088267
		b = 0.242074
		c0 = 0.445337
		ap = 0.167465
		cp = -0.068969
		alpha = 1.756618
	
		p_int = int(mdef[-2:])
		p = float(p_int) / 100.0
		
		if (p < 0.5) or (p > 0.9):
			raise Exception('Cannot evaluate percentile %d, the allowed range is [50..90].' % p_int)
		
		pprime = p**alpha
		a = a0 + ap * pprime
		c = c0 + cp * pprime

	else:
		raise Exception('Mass definition %s cannot be evaluated with diemer20 model.' % (mdef))

	f = A * ((nu / b)**a + 1.0) * np.exp(-c * nu**2)

	return f

###################################################################################################

def modelSeppi20(sigma, z, deltac_args = {'corrections': True},
				xoff = None, spin = None, int_over_sigma = False, int_over_xoff = True, int_over_spin = True):
	"""
	The mass function model of Seppi et al 2020.
	
	This model constitutes a 3D distribution of halo abundance over the variance, the spatial 
	offset between a halo's center of mass and the peak of its mass profile and the Peebles spin 
	parameter. Depending on the ``int_over_sigma``, ``int_over_xoff``, and ``int_over_spin`` 
	parameters, this function can return 1D, 2D, or 3D results on a grid given by ``sigma``, 
	``xoff``, and ``spin``. If those arrays are not given, a standard set of bins is used (and 
	integrated over depending on the dimensionality of the desired output).
	
	Note that if 2D or 3D arrays are returned, the output units must be ``f`` when using the 
	generalized :func:`massFunction` function because the output cannot be converted.
	
	The model was calibrated only for masses above 4*10^13 solar masses and should not be used
	below this mass scale. The function is given in Equation 21 of the paper.
	
	Parameters
	----------
	sigma: array_like
		Variance; can be a number or a numpy array.
	xoff: array_like
		Offset between cluster peak and center of mass; can be a number or a numpy array.
	spin: array_like
		Peebles spin parameter; can be a number or a numpy array.
	int_over_sigma: bool
		If ``True``, marginalize over the ``sigma`` parameter.
	int_over_xoff: bool
		If ``True``, marginalize over the ``xoff`` parameger.
	int_over_spin: bool
		If ``True``, marginalize over the ``spin`` parameger.
	
	Returns
	-------
	h: array_like
		The 3D halo mass-xoff-spin function :math:`h(\\sigma,xoff,\\lambda)`
	g_sigma_xoff: array_like
		The 2D halo mass-xoff function :math:`g(\\sigma,xoff)`, marginalized over spin
	g_sigma_spin: array_like
		The 2D halo mass-spin function :math:`g(\\sigma,\\lambda)`, marginalized over xoff
	g_xoff_spin: array_like
		The 2D halo xoff-spin function :math:`g(xoff,\\lambda)`, marginalized over mass
	f_xoff: array_like
		The halo xoff function :math:`f(xoff)`, marginalized over mass and spin
	f_spin: array_like
		The halo spin function :math:`f(\\sigma)`, marginalized over mass and xoff
	f: array_like
		The halo mass function :math:`f(\\sigma)`, marginalized over xoff and spin
	"""
	
	zp1 = 1.0 + z
	
	A     = -22.004 * (zp1)**-0.0441
	a     =   0.886 * (zp1)**-0.1611
	q     =   2.285 * (zp1)**0.0409
	mu    =  -3.326 * (zp1)**-0.1286
	alpha =   5.623 * (zp1)**0.1081
	beta  =  -0.391 * (zp1)**-0.3114
	gamma =   3.024 * (zp1)**0.0902
	delta =   1.209 * (zp1)**-0.0768
	e     =  -1.105 * (zp1)**0.6123
	
	if sigma is None:
		sigma = np.linspace(0.25, 1.2, 50)
	if xoff is None:
		xoff = np.logspace(-3.5, -0.3, 50)
	if spin is None:
		spin = np.logspace(-3.5, -0.3, 50)
		
	n_sigma = len(sigma)
	n_xoff = len(xoff)
	n_spin = len(spin)

	# Compute 3D distribution
	delta_c = peaks.collapseOverdensity(z = 0, **deltac_args)    
	sigma_, xoff_, spin_ = np.meshgrid(sigma, xoff, spin, indexing = 'ij')
	
	nu_ = delta_c / sigma_
	t1 = xoff_ / 10**(1.83 * mu)
	ln10 = np.log(10.0)
	
	h_log = A + np.log10(np.sqrt(2.0 / np.pi)) + q * np.log10(np.sqrt(a) * nu_) \
			- a / 2.0 / ln10 * nu_**2 + alpha * np.log10(t1) \
			- 1.0 / ln10 * t1**(0.05 * alpha) + gamma * np.log10(spin_ / 10**(mu)) \
			- 1.0 / ln10 * (t1 / sigma_**e)**beta * (spin_ / (10**mu))**delta
	h = 10**h_log
	
	# Compute 2D distributions
	g_xoff_spin = np.zeros((n_xoff, n_spin))    
	for i in range(n_xoff):
		for j in range(n_spin):
			if n_sigma == 1:
				g_xoff_spin[i,j] = h[:,i,j]
			else:    
				g_xoff_spin[i,j] = scipy.integrate.simpson(h[:,i,j], x=1.0 / sigma)
	
	g_sigma_spin = np.zeros((n_sigma, n_spin))    
	for i in range(n_sigma):
		for j in range(n_spin):
			if n_xoff == 1:
				g_sigma_spin[i,j] = h[i,:,j]
			else:    
				g_sigma_spin[i,j] = scipy.integrate.simpson(h[i,:,j], x=np.log10(xoff))
	
	g_sigma_xoff = np.zeros((n_sigma, n_xoff))    
	for i in range(n_sigma):
		for j in range(n_xoff):
			if n_spin == 1:
				g_sigma_xoff[i,j] = h[i,j,:]
			else:    
				g_sigma_xoff[i,j] = scipy.integrate.simpson(h[i,j,:], x=np.log10(spin))
	
	# Compute 1D distributions
	f_xoff = np.zeros(n_xoff)
	for i in range(n_xoff):
		if n_sigma == 1:
			f_xoff[i] = g_sigma_xoff[:,i]
		else:    
			f_xoff[i] = scipy.integrate.simpson(g_sigma_xoff[:,i], x=1.0 / sigma)
	
	f_spin = np.zeros(n_spin)
	for i in range(n_spin):
		if n_sigma == 1:
			f_spin[i] = g_sigma_spin[:,i]
		else:    
			f_spin[i] = scipy.integrate.simpson(g_sigma_spin[:,i], x=1.0 / sigma)
	
	f_sigma = np.zeros(n_sigma)
	for i in range(n_sigma):
		if n_xoff == 1:
			f_sigma[i] = g_sigma_xoff[i,:]
		else:    
			f_sigma[i] = scipy.integrate.simpson(g_sigma_xoff[i,:], x=np.log10(xoff))
	
	# Return the correct 1D, 2D, or 3D mass function depending on the parameters
	if (not int_over_sigma) & (not int_over_xoff) & (not int_over_spin):
		return h
	
	elif int_over_sigma & (not int_over_xoff) & (not int_over_spin):
		return g_xoff_spin
	
	elif (not int_over_sigma) & int_over_xoff & (not int_over_spin):
		return g_sigma_spin
	
	elif (not int_over_sigma) & (not int_over_xoff) & int_over_spin:
		return g_sigma_xoff
	
	elif int_over_sigma & int_over_xoff & (not int_over_spin):
		return f_spin
	
	elif int_over_sigma & (not int_over_xoff) & int_over_spin:
		return f_xoff
	
	elif (not int_over_sigma) & int_over_xoff & int_over_spin:
		return f_sigma 

	else:
		raise Exception('Invalid combination of int_over_sigma, int_over_xoff, int_over_spin.')

###################################################################################################
# Pointers to model functions
###################################################################################################

models['press74'].func = modelPress74
models['sheth99'].func = modelSheth99
models['jenkins01'].func = modelJenkins01
models['reed03'].func = modelReed03
models['warren06'].func = modelWarren06
models['reed07'].func = modelReed07
models['tinker08'].func = modelTinker08
models['crocce10'].func = modelCrocce10
models['courtin11'].func = modelCourtin11
models['bhattacharya11'].func = modelBhattacharya11
models['angulo12'].func = modelAngulo12
models['watson13'].func = modelWatson13
models['bocquet16'].func = modelBocquet16
models['despali16'].func = modelDespali16
models['comparat17'].func = modelComparat17
models['diemer20'].func = modelDiemer20
models['seppi20'].func = modelSeppi20

###################################################################################################
