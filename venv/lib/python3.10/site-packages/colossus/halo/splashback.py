###################################################################################################
#
# splashback.py             (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This module contains a collection of routines related to the splashback radius.

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

The splashback radius, :math:`R_{\\rm sp}`, corresponds to the apocenter of particles on their 
first orbit after falling into a halo. By analogy with the spherical collapse model, this radius 
represents a physically motivated definition of the halo boundary because it separates infalling 
from orbiting material. The splashback mass, :math:`M_{\\rm sp}`, is defined as the mass within 
:math:`R_{\\rm sp}`.

In practice, not all particles reach apocenter at the same radius, and the resulting drop in the
density profile is less sharp than in spherical models. As a result, there are several different 
definitions of the splashback radius, for example the radius where the logarithmic slope of the 
density profile is steepest, or definitions derived from the actual distribution of particle 
apocenters. For more information, please see the papers that first suggested the splashback 
radius, namely
`Diemer & Kravtsov 2014 <http://adsabs.harvard.edu/abs/2014ApJ...789....1D>`__,
`Adhikari et al. 2014 <http://adsabs.harvard.edu/abs/2014JCAP...11..019A>`__,
and `More et al. 2015 <http://adsabs.harvard.edu/abs/2015ApJ...810...36M>`__, as well as any of the
model papers listed below.

This module implements a number of theoretical models and fitting functions for the splashback
radius, splashback mass, the scatter in those quantities, and the mass accretion rate. The models 
can be evaluated using the generalized function :func:`splashbackModel` as well as a convenient
wrapper, :func:`splashbackRadius` which evaluates :math:`R_{\\rm sp}` given a spherical overdensity
mass. The splashback models can return a number of different quantities, identified by the 
following shorthand:

- ``RspR200m``, the splashback radius in units of :math:`R_{\\rm 200m}`
- ``MspM200m``, the splashback mass in units of :math:`M_{\\rm 200m}`
- ``Deltasp``, the enclosed overdensity wrt. the mean density of the universe,
  :math:`\\Delta_{\\rm sp} = M_{\\rm sp} / (4 \\pi R_{\\rm sp}^3 / 3) / \\rho_{\\rm m}`
- ``RspR200m-1s``, the 68% scatter in :math:`R_{\\rm sp}/R_{\\rm 200m}` in dex
- ``MspM200m-1s``, the 68% scatter in :math:`M_{\\rm sp}/M_{\\rm 200m}` in dex
- ``Deltasp-1s``, the 68% scatter in :math:`\\Delta_{\\rm sp}` in dex

Not all models can return all of these quantities (see table below). Moreover, depending on the 
model the quantities can be computed as a function of different input variables, namely

- ``Gamma``, the mass accretion rate :math:`\\Gamma`
- ``nu200m``, the peak height corresponding to :math:`M_{\\rm 200m}` (see 
  :func:`~lss.peaks.peakHeight`)
- ``z``, the redshift

Note that the mass accretion rate is defined in different ways depending on which model is used.
In theoretical models, :math:`\\Gamma` typically stands for the instantaneous accretion rate
:math:`s = d \\log(M) / d \\log(a)`. In other models, it means the mass accretion rate measured
over either a fixed time interval as in 
`Diemer & Kravtsov 2014 <http://adsabs.harvard.edu/abs/2014ApJ...789....1D>`__, or measured
over a dynamical time. Please consult the model papers for details. The following code example 
shows how to calculate :math:`R_{\\rm sp}/R_{\\rm 200m}` when only the mass of a halo is known::

	from colossus.lss import peaks
	from colossus.halo import splashback

	M200m = 1E12
	z = 0.5
	nu200m = peaks.peakHeight(M200m, z)
	RspR200m, mask = splashback.splashbackModel('RspR200m', nu200m = nu200m, z = z)

If the mass accretion rate is known, that constraint can also be used::

	Gamma = 1.1
	RspR200m, mask = splashback.splashbackModel('RspR200m', Gamma = Gamma, nu200m = nu200m, z = z)
	
All functions take numpy arrays as well as float values, though with certain restrictions (see 
below). The ``mask`` return variable indicates whether the chosen model could be evaluated given 
the input parameters. If not (e.g., because the redshift of peak height were outside a model's 
range), ``mask`` (or certain elements of it) will be False and the corresponding elements of 
``RspR200m`` will be missing. Please see the :doc:`tutorials` for more extensive code examples.

---------------------------------------------------------------------------------------------------
Splashback models
---------------------------------------------------------------------------------------------------

The following models are supported in this module, and their ID can be passed as the ``model`` 
parameter to the :func:`splashbackModel` and :func:`splashbackRadius` functions:

.. table::
	:widths: auto

	============== ==================== =========================== =========================================
	ID             Predicts...          ...as a function of...      Reference
	============== ==================== =========================== =========================================
	adhikari14     Rsp/Msp              (Gamma, z)                  `Adhikari et al. 2014 <http://adsabs.harvard.edu/abs/2014JCAP...11..019A>`__
	more15         Rsp/Msp              (Gamma, z, M) or (z, M)     `More et al. 2015 <http://adsabs.harvard.edu/abs/2015ApJ...810...36M>`__
	shi16          Rsp/Msp              (Gamma, z)                  `Shi 2016 <http://adsabs.harvard.edu/abs/2016MNRAS.459.3711S>`__
	mansfield17    Rsp/Msp, Scatter     (Gamma, z, M)               `Mansfield et al. 2017 <http://adsabs.harvard.edu/abs/2017ApJ...841...34M>`__
	diemer17       Rsp/Msp, Scatter     (Gamma, z, M) or (z, M)     `Diemer et al. 2017 <http://adsabs.harvard.edu/abs/2017ApJ...843..140D>`__
	diemer20       Rsp/Msp, Scatter     (Gamma, z, M) or (z, M)     Diemer 2020
	============== ==================== =========================== =========================================

The individual functions for these models are documented towards the bottom of this page. Note that
the ``diemer20`` model includes a fitting function for the mass accretion rate as a function of 
peak height and redshift.

Note that, in principle, the user can overwrite the model properties and limitations stored in 
the :data:`models` dictionary, for example to evaluate a model outside of the range of parameters
where it was calibrated. However, there is no guarantee that the results will be correct.

---------------------------------------------------------------------------------------------------
Module contents
---------------------------------------------------------------------------------------------------

.. autosummary:: 
	SplashbackModel
	models
	splashbackModel
	splashbackRadius
	modelAdhikari14Deltasp
	modelAdhikari14RspR200m
	modelMore15RspR200m
	modelMore15MspM200m
	modelShi16Delta
	modelShi16RspR200m
	modelMansfield17RspR200m
	modelMansfield17MspM200m
	modelDiemerRspMsp
	modelDiemerScatter
	modelDiemerGamma
	modelDiemer17RspR200m
	modelDiemer17MspM200m
	modelDiemer17RspR200mScatter
	modelDiemer17MspM200mScatter
	modelDiemer17Gamma
	modelDiemer20RspR200m
	modelDiemer20MspM200m
	modelDiemer20RspR200mScatter
	modelDiemer20MspM200mScatter
	modelDiemer20Gamma
	
---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
import scipy.interpolate
from collections import OrderedDict
import warnings

from colossus import defaults
from colossus.utils import utilities
from colossus.cosmology import cosmology
from colossus.lss import peaks
from colossus.halo import mass_so
from colossus.halo import mass_defs
from colossus.halo import mass_adv
from colossus.halo import profile_nfw

###################################################################################################

class SplashbackModel():
	"""
	Characteristics of splashback models.
	
	This object contains certain characteristics of a model, most importantly the input quantities
	``q_in`` and output quantities ``q_out`` the model is capable of processing. Additionally, the 
	``depends_on`` field lists the quantities the model depends on. If, for example, z is 
	among these quantities, then the redshift needs to be passed to the :func:`splashbackModel`
	function.
	
	This object does not contain a function pointer to the model functions because those functions
	do not work in a uniform way, necessitating a somewhat more complex decision tree when 
	evaluating them.

	The :data:`models` dictionary contains one item of this class for each available model.
	"""
	def __init__(self):
		
		self.q_in = []
		"""
		The model can be evaluated as a function of these quantities. Valid entries are 
		``Gamma``, ``z``, and ``nu200m``.
		"""
		self.q_out = []
		"""
		The quantities the model can predict. See table above.
		"""
		self.depends_on = []
		"""
		The quantities the model depends on (which need to be passed to the evaluating function). 
		Valid entries are ``Gamma``, ``z``, ``nu``, and ``rspdef``.
		"""
		self.min_Gamma = -np.inf
		"""
		The minimum mass accretion rate where the model is valid.
		"""
		self.max_Gamma = np.inf
		"""
		The maximum mass accretion rate where the model is valid.
		"""
		self.min_nu200m = 0.0
		"""
		The minimum peak height where the model is valid.
		"""
		self.max_nu200m = np.inf
		"""
		The maximum peak height where the model is valid.
		"""
		self.min_z = 0.0
		"""
		The minimum redshift where the model is valid.
		"""
		self.max_z = np.inf
		"""
		The maximum redshift where the model is valid.
		"""

		self.label = ''
		self.style = {}
		
		return

###################################################################################################

models = OrderedDict()
"""
Dictionary containing a list of models.

An ordered dictionary containing one :class:`SplashbackModel` entry for each model.
"""

models['adhikari14'] = SplashbackModel()
models['adhikari14'].q_in = ['Gamma']
models['adhikari14'].q_out = ['RspR200m', 'MspM200m', 'Deltasp']
models['adhikari14'].depends_on = ['Gamma', 'z']
models['adhikari14'].min_Gamma = 0.2
models['adhikari14'].max_Gamma = 5.9

models['more15'] = SplashbackModel()
models['more15'].q_in = ['Gamma', 'nu200m']
models['more15'].q_out = ['RspR200m', 'MspM200m', 'Deltasp']
models['more15'].depends_on = ['Gamma', 'z']

models['shi16'] = SplashbackModel()
models['shi16'].q_in = ['Gamma']
models['shi16'].q_out = ['RspR200m', 'MspM200m', 'Deltasp']
models['shi16'].depends_on = ['Gamma', 'z']
models['shi16'].min_Gamma = 0.5
models['shi16'].max_Gamma = 5.0

models['mansfield17'] = SplashbackModel()
models['mansfield17'].q_in = ['Gamma']
models['mansfield17'].q_out = ['RspR200m', 'MspM200m', 'Deltasp', 'RspR200m-1s', 'MspM200m-1s']
models['mansfield17'].depends_on = ['Gamma', 'z', 'nu']
models['mansfield17'].min_Gamma = 0.5
models['mansfield17'].max_Gamma = 7.0

models['diemer17'] = SplashbackModel()
models['diemer17'].q_in = ['Gamma', 'nu200m']
models['diemer17'].q_out = ['RspR200m', 'MspM200m', 'Deltasp', 'RspR200m-1s', 'MspM200m-1s', 'Deltasp-1s']
models['diemer17'].depends_on = ['Gamma', 'z', 'nu', 'rspdef']
models['diemer17'].min_nu200m = 0.0
models['diemer17'].max_nu200m = 5.0
models['diemer17'].min_z = 0.0
models['diemer17'].max_z = 8.0

models['diemer20'] = SplashbackModel()
models['diemer20'].q_in = ['Gamma', 'nu200m']
models['diemer20'].q_out = ['RspR200m', 'MspM200m', 'Deltasp', 'RspR200m-1s', 'MspM200m-1s', 'Deltasp-1s']
models['diemer20'].depends_on = ['Gamma', 'z', 'nu', 'rspdef']
models['diemer20'].min_nu200m = 0.0
models['diemer20'].max_nu200m = 5.0
models['diemer20'].min_z = 0.0
models['diemer20'].max_z = 8.0

###################################################################################################

def splashbackModel(q_out, Gamma = None, nu200m = None, z = None,
				model = defaults.HALO_SPLASHBACK_MODEL,
				statistic = defaults.HALO_SPLASHBACK_STATISTIC,
				rspdef = None):
	"""
	The splashback radius, mass, and scatter.
	
	Depending on the model, this function can return quantities such as :math:`R_{sp} / R_{\\rm 200m}`,
	:math:`M_{\\rm sp} / M_{\\rm 200m}`, the splashback overdensity, or the scatter in these 
	quantities (see the table of models above). The primary input variable can be mass or accretion 
	rate, and other quantities such as redshift may be required depending on which model is chosen
	(see the properties listed in :data:`models` for details).
	
	The function returns only valid predictions for the desired quantity. If any of the parameters
	lie outside the range of validity of the chosen model, no output is returned for that input,
	as indicated by the returned boolean mask. No warning is output in such cases unless there is
	no valid output for any input.
	
	Parameters
	----------
	q_out: str
		Identifier of the output quantity (see listing above). 
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This quantity only needs to be 
		passed for models that depend on the mass accretion rate.
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array. This
		quantity only needs to be passed for models that depend on peak height.
	z: array_like
		Redshift; can be a number or a numpy array. This quantity only needs to be passed for 
		models that depend on redshift.
	model: str
		The splashback model to use for the prediction (see table above).
	statistic: str
		Can be ``mean`` or ``median``, determining whether the function returns the best fit to the 
		mean or median profile of a halo sample. This parameter is ignored by most models. Not
		to be mixed up with the definition ``rspdef`` used in the ``diemer20`` model.
	rspdef: str
		The definition of the splashback radius. This parameter is ignored by most models, but 
		used by the ``diemer17`` and ``diemer20`` models to distinguish the ``mean`` of the 
		apocenter distribution or higher percentiles (e.g. ``percentile75``). The function also 
		accepts the newer notation used in the SPARTA code, namely ``sp-apr-mn`` for the mean and 
		``sp-apr-p75`` and so on for percentiles. For models that use this parameter (``diemer17`` 
		and ``diemer20``), it must be given, otherwise the function throws an error. The acceptable
		range of percentiles is between 50 and 90.
	
	Returns
	-------
	y: array_like
		The desired quantity; if the input (``Gamma`` or ``nu200m``) is a number, this is a number, 
		otherwise a numpy array.
	mask: array_like
		A boolean mask of the same dimensions as the input, indicating whether a valid input was 
		returned for each input element (``Gamma`` or ``nu200m``). Note that only the valid values 
		are returned, meaning ``y`` can contain fewer items than ``mask``.
	"""

	# Check that this model exists
	if not model in models:
		raise Exception('Unknown model, %s.' % model)
	m = models[model]

	# Check that this model can perform the requested operation in principle. For this purpose we
	# determine the primary input quantity, either Gamma or nu200m
	if Gamma is None and nu200m is None:
		raise Exception('Either Gamma or nu200m must be passed.')
	if Gamma is not None:
		q_in = 'Gamma'
	else:
		q_in = 'nu200m'
	if not q_in in m.q_in:
		raise Exception('Model %s cannot handle input quantity %s.' % (model, q_in))
	if not q_out in m.q_out:
		raise Exception('Model %s cannot output quantity %s.' % (model, q_out))

	# Check for wrong input of parameters. If multiple fields are given, they must either be 
	# numbers of arrays of the same dimension as the primary input. Create a mask indicating
	# where the results are valid.
	if q_in == 'Gamma':
		if utilities.isArray(Gamma):
			if z is not None and utilities.isArray(z) and len(z) != len(Gamma):
				raise Exception('If z is an array, it must have the same dimensions as Gamma.')
			if nu200m is not None and utilities.isArray(nu200m) and len(nu200m) != len(Gamma):
				raise Exception('If nu200m is an array, it must have the same dimensions as Gamma.')
		elif utilities.isArray(nu200m):
			Gamma = np.ones_like(nu200m) * Gamma
		mask = (Gamma >= m.min_Gamma) & (Gamma <= m.max_Gamma)
		if nu200m is not None:
			mask = mask & (nu200m >= m.min_nu200m) & (nu200m <= m.max_nu200m)
	elif q_in == 'nu200m':
		if utilities.isArray(nu200m):
			if z is not None and utilities.isArray(z) and len(z) != len(nu200m):
				raise Exception('If z is an array, it must have the same dimensions as nu200m.')
		mask = (nu200m >= m.min_nu200m) & (nu200m <= m.max_nu200m)
	else:
		raise Exception('Unknown input quantity, %s.' % q_in)
	if z is not None:
		mask = mask & (z >= m.min_z) & (z <= m.max_z) 
	
	# Now apply the mask to the input array, and return if there are no valid entries.
	if q_in == 'Gamma':
		x = Gamma
	elif q_in == 'nu200m':
		x = nu200m
	x, is_array = utilities.getArray(x)
	x = x.astype(float)
	mask, _ = utilities.getArray(mask)
	x = x[mask]
	if np.count_nonzero(mask) == 0:
		warnings.warn('Found no input values within the limits of model %s.' % model)
		return np.array([]), mask
	if q_in == 'Gamma':
		Gamma = x
		if nu200m is not None and utilities.isArray(nu200m):
			nu200m = nu200m[mask]
	elif q_in == 'nu200m':
		nu200m = x
	
	# Compute Om from z, but only if z is given. Some models use Om rather than z.
	if z is None:
		Om = None
	else:
		cosmo = cosmology.getCurrent()
		Om = cosmo.Om(z)
	ret = None

	if model == 'adhikari14':
		
		Delta, c = modelAdhikari14Deltasp(Gamma, Om)
		if q_out == 'Deltasp':
			ret = Delta
		elif q_out == 'RspR200m':
			ret, _ = modelAdhikari14RspR200m(Delta, c, z)
		elif q_out == 'MspM200m':
			_, ret = modelAdhikari14RspR200m(Delta, c, z)
			
	elif model == 'more15':

		if q_out == 'RspR200m':
			ret = modelMore15RspR200m(z = z, Gamma = Gamma, nu200m = nu200m, statistic = statistic)
		elif q_out == 'MspM200m':
			ret = modelMore15MspM200m(z = z, Gamma = Gamma, nu200m = nu200m, statistic = statistic)
		elif q_out == 'Deltasp':
			msp200m = modelMore15MspM200m(z = z, Gamma = Gamma, nu200m = nu200m, statistic = statistic)
			rsp200m = modelMore15RspR200m(z = z, Gamma = Gamma, nu200m = nu200m, statistic = statistic)
			ret = 200.0 * msp200m / rsp200m**3

	elif model == 'shi16':
		
		if q_out == 'RspR200m':
			ret = modelShi16RspR200m(Gamma, Om)
		elif q_out == 'MspM200m':
			delta = modelShi16Delta(Gamma, Om)
			rspr200m = modelShi16RspR200m(Gamma, Om)
			ret = delta / 200.0 * rspr200m**3
		elif q_out == 'Deltasp':
			ret = modelShi16Delta(Gamma, Om)
	
	elif model == 'mansfield17':
		
		if z < 2.0:
			mask_new = (x <= 5.0)
			mask[mask] = mask_new
			x = x[mask_new]
			if utilities.isArray(nu200m):
				nu200m = nu200m[mask_new]
			if utilities.isArray(Om):
				Om = Om[mask_new]
			
		if q_out == 'RspR200m':
			ret = modelMansfield17RspR200m(x, Om, nu200m)
		elif q_out == 'MspM200m':
			ret = modelMansfield17MspM200m(x, Om)
		elif q_out == 'Deltasp':
			rspr200m = modelMansfield17RspR200m(x, Om, nu200m)
			mspm200m = modelMansfield17MspM200m(x, Om)
			ret = mspm200m * 200.0 / rspr200m**3
		elif q_out == 'RspR200m-1s':
			ret = np.ones((len(x)), float) * 0.046
		elif q_out == 'MspM200m-1s':
			ret = np.ones((len(x)), float) * 0.054
	
	elif model in ['diemer17', 'diemer20']:

		# Check that a definition was given.
		if rspdef is None:
			raise Exception('Need Rsp/Msp definition such as sp-apr-mn or sp-apr-p75, none given.')
		
		# The model is only valid between the 50th and 90th percentile
		p = _modelDiemerPercentileValue(rspdef)
		if (p > 0.0 and p < 0.5) or p > 0.901:
			mask[:] = False
			ret = np.array([])
			return ret, mask
		
		# If only nu200m is given, we compute Gamma from nu and z.
		if q_in == 'nu200m':
			pars = _modelDiemerGetPars(model, 'Gamma', None)
			Gamma = modelDiemerGamma(nu200m, z, pars)
		
		if q_out in ['RspR200m', 'MspM200m']:
			pars = _modelDiemerGetPars(model, q_out, rspdef)
			ret = modelDiemerRspMsp(Gamma, nu200m, z, rspdef, pars)
		
		elif q_out == 'Deltasp':
			pars = _modelDiemerGetPars(model, 'RspR200m', rspdef)
			rsp200m = modelDiemerRspMsp(Gamma, nu200m, z, rspdef, pars)
			pars = _modelDiemerGetPars(model, 'MspM200m', rspdef)
			msp200m = modelDiemerRspMsp(Gamma, nu200m, z, rspdef, pars)
			ret = 200.0 * msp200m / rsp200m**3
		
		else:
			if q_in == 'nu200m':
				if q_out == 'RspR200m-1s':
					ret = np.ones((len(mask)), float) * 0.07
				elif q_out == 'MspM200m-1s':
					ret = np.ones((len(mask)), float) * 0.07
				elif q_out == 'Deltasp-1s':
					ret = np.ones((len(mask)), float) * 0.15
				else:
					raise Exception('Unknown quantity, %s.' % (q_out))
			else:
				if q_out in ['RspR200m-1s', 'MspM200m-1s']:
					pars = _modelDiemerGetPars(model, q_out, rspdef)
					ret = modelDiemerScatter(Gamma, nu200m, rspdef, pars)
				elif q_out == 'Deltasp-1s':
					pars = _modelDiemerGetPars(model, 'RspR200m-1s', rspdef)
					rsp_1s = modelDiemerScatter(Gamma, nu200m, rspdef, pars)
					pars = _modelDiemerGetPars(model, 'MspM200m-1s', rspdef)
					msp_1s = modelDiemerScatter(Gamma, nu200m, rspdef, pars)
					ret = np.sqrt(msp_1s**2 + 3.0 * rsp_1s**2)
				else:
					raise Exception('Unknown quantity, %s.' % (q_out))

				min_scatter = 0.02
				if utilities.isArray(ret):
					ret[ret < min_scatter] = min_scatter
				else:
					ret = max(ret, min_scatter)
	
	else:
		raise Exception('Unknown model, %s.' % model)

	if not is_array:
		ret = ret[0]
		mask = mask[0]

	return ret, mask

###################################################################################################

def splashbackRadius(z, mdef, R = None, M = None, c = None, Gamma = None,
		model = defaults.HALO_SPLASHBACK_MODEL, 
		statistic = defaults.HALO_SPLASHBACK_STATISTIC,
		rspdef = None,
		c_model = defaults.HALO_CONCENTRATION_MODEL,
		profile = defaults.HALO_MASS_CONVERSION_PROFILE):
	"""
	:math:`R_{\\rm sp}` and :math:`M_{\\rm sp}` as a function of spherical overdensity radius or mass.
	
	This function represents a convenient wrapper for the :func:`splashbackModel` function, where
	only a spherical overdensity mass or radius needs to be passed. Additionally, the user can 
	pass any of the parameters to the :func:`splashbackModel` function. Note that the redshift
	must be a number, not an array.
	
	Parameters
	----------
	z: float
		Redshift
	mdef: str
		Mass definition in which any combination of ``R``, ``M``, and ``c`` is given. See 
		:doc:`halo_mass` for details.
	R: array_like
		Spherical overdensity radius in physical :math:`kpc/h`; can be a number or a numpy array.
		Either ``R`` or ``M`` need to be passed, not both.
	M: array_like
		Spherical overdensity mass in :math:`M_{\\odot}/h`; can be a number or a numpy array.
		Either ``R`` or ``M`` need to be passed, not both.
	c: array_like
		Halo concentration; must have the same dimensions as ``R`` or ``M``, or be ``None`` in 
		which case the concentration is computed automatically.
	Gamma: array_like
		The mass accretion rate that can be optionally passed to the splashback model. If this 
		field is set, the splashback model is evaluated with Gamma as the primary input, otherwise
		peak height is the primary input. If ``Gamma`` is an array, it must have the same 
		dimensions as the input ``R`` or ``M``.
	model: str
		The splashback model to use for the prediction (see table above).
	statistic: str
		Can be ``mean`` or ``median``, determining whether the function returns the best fit to the 
		mean or median profile of a halo sample. This parameter is ignored by most models.
	rspdef: str
		The definition of the splashback radius. This parameter is ignored by most models, but 
		used by the ``diemer17`` model to distinguish the ``mean`` of the apocenter distribution
		or higher percentiles (e.g. ``percentile75``). The function also accepts the newer notation
		used in the SPARTA code, namely ``sp-apr-mn`` for the mean and ``sp-apr-p75`` and so on
		for percentiles.
	c_model: str
		The concentration model used to compute c if it is not passed by the user. See the
		:doc:`halo_concentration` module for details.
	profile: str
		The functional form of the profile assumed in the conversion between mass definitions; 
		can be ``nfw`` or ``dk14``.

	Returns
	-------
	Rsp: array_like
		:math:`R_{\\rm sp}` in physical :math:`kpc/h`
	Msp: array_like
		:math:`M_{\\rm sp}` in :math:`M_{\\odot}/h`
	mask: array_like
		A boolean mask of the same dimensions as ``R`` or ``M``, indicating whether a valid input 
		was returned for each input. Note that only the valid outputs are returned, meaning Rsp 
		and Msp can have fewer items than mask.
	"""
	
	if R is None and M is None:
		raise Exception('Either R or M must be passed.')
	if R is not None and M is not None:
		raise Exception('Only R or M can be passed, not both.')
	
	if R is not None:
		R, is_array = utilities.getArray(R)
		R = R.astype(float)
	else:
		M, is_array = utilities.getArray(M)
		M = M.astype(float)
	
	if mdef == '200m':
		if R is None:
			M200m = M
			R200m = mass_so.M_to_R(M200m, z, '200m')
		else:
			R200m = R
			M200m = mass_so.R_to_M(R200m, z, '200m')
	else:
		if M is None:
			M = mass_so.R_to_M(R, z, mdef)
		if c is None:
			M200m, R200m, _ = mass_adv.changeMassDefinitionCModel(M, z, mdef, '200m', 
										profile = profile, c_model = c_model)
		else:
			M200m, R200m, _ = mass_defs.changeMassDefinition(M, c, z, mdef, '200m', 
										profile = profile)
	
	# Final parameter check: if Gamma is given, it must have the same dimensions as R/M.
	if Gamma is not None:
		gamma_is_array = utilities.isArray(Gamma)
		if gamma_is_array:
			if not is_array:
				raise Exception('Gamma is an array, but the given R/M is not. They must agree in dimensions.')
			if len(Gamma) != len(M200m):
				raise Exception('The Gamma array has length %d, the R/M array %d; they must agree in dimensions.' \
							% (len(Gamma), len(M200m)))

	# Perform the actual computations. We first convert mass to peak height and then evaluate the
	# splashback model.			
	nu200m = peaks.peakHeight(M200m, z)
	
	RspR200m, mask1 = splashbackModel('RspR200m', Gamma = Gamma, nu200m = nu200m, z = z, model = model,
				statistic = statistic, rspdef = rspdef)
	MspM200m, mask2 = splashbackModel('MspM200m', Gamma = Gamma, nu200m = nu200m, z = z, model = model,
				statistic = statistic, rspdef = rspdef)
	
	# The masks should be the same but we require both to be true to be safe.
	mask = mask1 & mask2
	
	# Special case: if there is only one element and its mask is False, we need to return some
	# value for R and M.
	if (np.count_nonzero(mask) == 0) and (not is_array):
		Rsp = None
		Msp = None
		mask = False
		
	else:
		Rsp = R200m[mask] * RspR200m
		Msp = M200m[mask] * MspM200m
		if not is_array:
			Rsp = Rsp[0]
			Msp = Msp[0]
			mask = mask[0]
		
	return Rsp, Msp, mask

###################################################################################################
# FUNCTIONS FOR INDIVIDUAL SPLASHBACK MODELS
###################################################################################################

def modelAdhikari14Deltasp(s, Om):
	"""
	:math:`\\Delta_{\\rm sp}` and concentration for the Adhikari et al 2014 model.
	
	The model is evaluated numerically, this function returns interpolated results as a function of
	mass accretion rate ``s`` and :math:`\\Omega_{\\rm m}(z)`.
	
	Parameters
	----------
	s: array_like
		Instantaneous mass accretion rate
	Om: array_like
		Matter density of the universe in units of the critical density.

	Returns
	-------
	Delta: array_like
		The overdensity enclosed by the predicted splashback radius, in units of the mean density.
	c: array_like
		The concentration of the predicted NFW profile.
	"""
	
	bins_Om = np.array([1.000e-02, 6.210e-02, 1.142e-01, 1.663e-01, 2.184e-01, 2.705e-01, 3.226e-01, 3.747e-01, 4.268e-01, 4.789e-01, 5.310e-01, 5.831e-01, 6.352e-01, 6.873e-01, 7.394e-01, 7.915e-01, 8.436e-01, 8.957e-01, 9.478e-01, 9.999e-01])
	bins_s = np.array([2.000e-01, 2.392e-01, 2.860e-01, 3.421e-01, 4.091e-01, 4.893e-01, 5.851e-01, 6.998e-01, 8.369e-01, 1.001e+00, 1.197e+00, 1.431e+00, 1.712e+00, 2.047e+00, 2.449e+00, 2.928e+00, 3.502e+00, 4.188e+00, 5.009e+00, 5.990e+00])
	c = np.array([5.503e+02, 2.370e+02, 1.152e+02, 6.164e+01, 3.562e+01, 2.188e+01, 1.410e+01, 9.431e+00, 6.495e+00, 4.571e+00, 3.266e+00, 2.354e+00, 1.701e+00, 1.223e+00, 8.671e-01, 5.971e-01, 3.896e-01, 2.283e-01, 1.014e-01, 8.349e-04])
	Deltasp = np.array([
		[8.131e+02, 8.277e+02, 8.450e+02, 8.673e+02, 8.967e+02, 9.357e+02, 9.860e+02, 1.049e+03, 1.098e+03, 1.191e+03, 1.295e+03, 1.410e+03, 1.532e+03, 1.662e+03, 1.799e+03, 1.947e+03, 2.114e+03, 2.315e+03, 2.573e+03, 2.923e+03],
		[2.342e+02, 2.396e+02, 2.460e+02, 2.540e+02, 2.640e+02, 2.767e+02, 2.924e+02, 3.117e+02, 3.289e+02, 3.567e+02, 3.883e+02, 4.233e+02, 4.612e+02, 5.015e+02, 5.437e+02, 5.875e+02, 6.336e+02, 6.839e+02, 7.418e+02, 8.133e+02],
		[1.633e+02, 1.672e+02, 1.719e+02, 1.777e+02, 1.849e+02, 1.940e+02, 2.051e+02, 2.187e+02, 2.312e+02, 2.507e+02, 2.728e+02, 2.975e+02, 3.243e+02, 3.528e+02, 3.827e+02, 4.136e+02, 4.456e+02, 4.796e+02, 5.173e+02, 5.619e+02],
		[1.328e+02, 1.361e+02, 1.400e+02, 1.448e+02, 1.508e+02, 1.582e+02, 1.674e+02, 1.785e+02, 1.888e+02, 2.047e+02, 2.227e+02, 2.428e+02, 2.647e+02, 2.881e+02, 3.127e+02, 3.380e+02, 3.640e+02, 3.912e+02, 4.207e+02, 4.548e+02],
		[1.152e+02, 1.182e+02, 1.216e+02, 1.258e+02, 1.311e+02, 1.376e+02, 1.455e+02, 1.552e+02, 1.643e+02, 1.780e+02, 1.937e+02, 2.111e+02, 2.302e+02, 2.506e+02, 2.720e+02, 2.941e+02, 3.167e+02, 3.401e+02, 3.651e+02, 3.934e+02],
		[1.036e+02, 1.063e+02, 1.094e+02, 1.132e+02, 1.180e+02, 1.238e+02, 1.310e+02, 1.397e+02, 1.480e+02, 1.603e+02, 1.744e+02, 1.901e+02, 2.073e+02, 2.257e+02, 2.450e+02, 2.648e+02, 2.852e+02, 3.061e+02, 3.282e+02, 3.528e+02],
		[9.521e+01, 9.768e+01, 1.006e+02, 1.042e+02, 1.085e+02, 1.139e+02, 1.205e+02, 1.285e+02, 1.362e+02, 1.475e+02, 1.604e+02, 1.749e+02, 1.907e+02, 2.076e+02, 2.254e+02, 2.437e+02, 2.624e+02, 2.816e+02, 3.016e+02, 3.237e+02],
		[8.884e+01, 9.115e+01, 9.390e+01, 9.723e+01, 1.013e+02, 1.064e+02, 1.126e+02, 1.200e+02, 1.272e+02, 1.377e+02, 1.498e+02, 1.632e+02, 1.780e+02, 1.938e+02, 2.104e+02, 2.276e+02, 2.450e+02, 2.628e+02, 2.814e+02, 3.016e+02],
		[8.378e+01, 8.598e+01, 8.858e+01, 9.174e+01, 9.562e+01, 1.004e+02, 1.062e+02, 1.133e+02, 1.201e+02, 1.300e+02, 1.413e+02, 1.540e+02, 1.680e+02, 1.829e+02, 1.986e+02, 2.148e+02, 2.313e+02, 2.480e+02, 2.653e+02, 2.841e+02],
		[7.967e+01, 8.177e+01, 8.425e+01, 8.727e+01, 9.096e+01, 9.551e+01, 1.011e+02, 1.077e+02, 1.142e+02, 1.237e+02, 1.345e+02, 1.465e+02, 1.597e+02, 1.740e+02, 1.889e+02, 2.043e+02, 2.200e+02, 2.359e+02, 2.523e+02, 2.698e+02],
		[7.623e+01, 7.825e+01, 8.064e+01, 8.353e+01, 8.708e+01, 9.143e+01, 9.674e+01, 1.031e+02, 1.094e+02, 1.184e+02, 1.287e+02, 1.402e+02, 1.529e+02, 1.665e+02, 1.808e+02, 1.956e+02, 2.106e+02, 2.258e+02, 2.414e+02, 2.580e+02],
		[7.331e+01, 7.526e+01, 7.756e+01, 8.035e+01, 8.377e+01, 8.796e+01, 9.307e+01, 9.922e+01, 1.052e+02, 1.139e+02, 1.238e+02, 1.349e+02, 1.471e+02, 1.602e+02, 1.739e+02, 1.881e+02, 2.026e+02, 2.172e+02, 2.321e+02, 2.479e+02],
		[7.080e+01, 7.268e+01, 7.491e+01, 7.761e+01, 8.092e+01, 8.497e+01, 8.990e+01, 9.584e+01, 1.017e+02, 1.100e+02, 1.196e+02, 1.303e+02, 1.420e+02, 1.547e+02, 1.680e+02, 1.817e+02, 1.957e+02, 2.097e+02, 2.241e+02, 2.392e+02],
		[6.860e+01, 7.043e+01, 7.260e+01, 7.522e+01, 7.843e+01, 8.236e+01, 8.714e+01, 9.289e+01, 9.854e+01, 1.066e+02, 1.159e+02, 1.262e+02, 1.376e+02, 1.499e+02, 1.628e+02, 1.761e+02, 1.896e+02, 2.033e+02, 2.171e+02, 2.316e+02],
		[6.666e+01, 6.844e+01, 7.055e+01, 7.310e+01, 7.623e+01, 8.005e+01, 8.469e+01, 9.028e+01, 9.578e+01, 1.036e+02, 1.126e+02, 1.227e+02, 1.337e+02, 1.456e+02, 1.582e+02, 1.711e+02, 1.843e+02, 1.975e+02, 2.109e+02, 2.249e+02],
		[6.493e+01, 6.667e+01, 6.873e+01, 7.122e+01, 7.427e+01, 7.799e+01, 8.252e+01, 8.796e+01, 9.333e+01, 1.010e+02, 1.097e+02, 1.195e+02, 1.303e+02, 1.419e+02, 1.541e+02, 1.667e+02, 1.795e+02, 1.924e+02, 2.054e+02, 2.190e+02],
		[6.338e+01, 6.508e+01, 6.710e+01, 6.953e+01, 7.250e+01, 7.614e+01, 8.056e+01, 8.587e+01, 9.112e+01, 9.858e+01, 1.071e+02, 1.167e+02, 1.272e+02, 1.385e+02, 1.504e+02, 1.628e+02, 1.753e+02, 1.878e+02, 2.005e+02, 2.137e+02],
		[6.197e+01, 6.364e+01, 6.562e+01, 6.800e+01, 7.091e+01, 7.447e+01, 7.879e+01, 8.398e+01, 8.912e+01, 9.642e+01, 1.047e+02, 1.141e+02, 1.243e+02, 1.354e+02, 1.471e+02, 1.592e+02, 1.714e+02, 1.837e+02, 1.961e+02, 2.088e+02],
		[6.070e+01, 6.233e+01, 6.427e+01, 6.661e+01, 6.947e+01, 7.295e+01, 7.719e+01, 8.227e+01, 8.731e+01, 9.444e+01, 1.026e+02, 1.117e+02, 1.218e+02, 1.326e+02, 1.441e+02, 1.559e+02, 1.679e+02, 1.799e+02, 1.920e+02, 2.045e+02],
		[5.953e+01, 6.114e+01, 6.304e+01, 6.534e+01, 6.814e+01, 7.156e+01, 7.572e+01, 8.070e+01, 8.565e+01, 9.264e+01, 1.006e+02, 1.096e+02, 1.194e+02, 1.301e+02, 1.413e+02, 1.529e+02, 1.647e+02, 1.764e+02, 1.883e+02, 2.005e+02]])

	Om_ = np.ones((len(s)), float) * Om
	interp = scipy.interpolate.RectBivariateSpline(bins_Om, bins_s, Deltasp)
	Delta = interp(Om_, s, grid = False)
	
	if np.max(s) > np.max(bins_s):
		raise Exception('Found s = %.2f, greater than max %.2f.' % (np.max(s), np.max(bins_s)))
	if np.min(s) < np.min(bins_s):
		raise Exception('Found s = %.2f, smaller than min %.2f.' % (np.min(s), np.min(bins_s)))
	
	c = np.interp(s, bins_s, c)
	if np.count_nonzero(c < 0.0) > 0:
		raise Exception('Found negative concentration values.')
	
	return Delta, c

###################################################################################################

# This function converts the Adhikari et al model overdensity into Rsp. The concentration
# given by the Adhikari refers to R = r_ta / 2 = Rvir, so we create an NFW profile with 
# that concentration. The mass doesn't matter since we're only after the ratio of Rsp and
# R200m, i.e. the model does not depend on the absolute mass scale.

def modelAdhikari14RspR200m(Delta, c, z):
	"""
	:math:`R_{\\rm sp}/R_{\\rm 200m}` and :math:`M_{\\rm sp}/M_{\\rm 200m}` for the Adhikari et al 2014 model.

	Parameters
	----------
	Delta: array_like
		Overdensity; should be computed using the :func:`modelAdhikari14Deltasp` function.
	c: array_like
		Concentration; should be computed using the :func:`modelAdhikari14Deltasp` function.
	z: float
		Redshift	

	Returns
	-------
	RspR200m: array_like
		:math:`R_{\\rm sp}/R_{\\rm 200m}`; has the same dimensions as ``Delta`` and ``c``.
	MspM200m: array_like
		:math:`M_{\\rm sp}/M_{\\rm 200m}`; has the same dimensions as ``Delta`` and ``c``.
	"""

	cosmo = cosmology.getCurrent()
	rho_m = cosmo.rho_m(z)
	rhos, rs = profile_nfw.NFWProfile.nativeParameters(1E10, c, z, 'vir')
	xsp = profile_nfw.NFWProfile.xDelta(rhos, Delta * rho_m)
	Rsp = xsp * rs
	x200m = profile_nfw.NFWProfile.xDelta(rhos, 200.0 * rho_m)
	R200m = x200m * rs
	RspR200m = Rsp / R200m
	MspM200m = Delta * RspR200m**3 / 200.0
	
	return RspR200m, MspM200m

###################################################################################################

def modelMore15RspR200m(nu200m = None, z = None, Gamma = None, statistic = 'median'):
	"""
	:math:`R_{\\rm sp}/R_{\\rm 200m}` for the More et al 2015 model.
	
	If the mass accretion rate is given, the prediction is based on that. If not, the peak height 
	is used instead, giving a less accurate fit.

	Parameters
	----------
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array.
	z: float
		Redshift	
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This model was calibrated with 
		Gamma computed according to the definition of 
		`Diemer & Kravtsov 2014 <http://adsabs.harvard.edu/abs/2014ApJ...789....1D>`__.
	statistic: str
		Can be ``mean`` or ``median``, determining whether the fit was performed on the mean or 
		median density profile of a halo sample.

	Returns
	-------
	RspR200m: array_like
		:math:`R_{\\rm sp}/R_{\\rm 200m}`; has the same dimensions as ``Gamma`` or ``nu200m``.
	"""

	if (Gamma is not None) and (z is not None):
		cosmo = cosmology.getCurrent()
		if statistic == 'median':
			ratio = 0.54 * (1.0 + 0.53 * cosmo.Om(z)) * (1 + 1.36 * np.exp(-Gamma / 3.04))
		elif statistic == 'mean':
			ratio = 0.58 * (1.0 + 0.63 * cosmo.Om(z)) * (1 + 1.08 * np.exp(-Gamma / 2.26))
		else:
			raise Exception('Unknown statistic, %s.' % statistic)
	elif nu200m is not None:
		if statistic == 'median':
			ratio = 0.81 * (1.0 + 0.97 * np.exp(-nu200m / 2.44))
		elif statistic == 'mean':
			ratio = 0.88 * (1.0 + 0.77 * np.exp(-nu200m / 1.95))
		else:
			raise Exception('Unknown statistic, %s.' % statistic)
	else:
		raise Exception('Need either Gamma and z, or nu.')

	return ratio

###################################################################################################

def modelMore15MspM200m(nu200m = None, z = None, Gamma = None, statistic = 'median'):
	"""
	:math:`M_{\\rm sp}/M_{\\rm 200m}` for the More et al 2015 model.

	If the mass accretion rate is given, the prediction is based on that. If not, the peak height 
	is used instead, giving a less accurate fit.

	Parameters
	----------
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array.
	z: float
		Redshift	
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This model was calibrated with 
		Gamma computed according to the definition of 
		`Diemer & Kravtsov 2014 <http://adsabs.harvard.edu/abs/2014ApJ...789....1D>`__.
	statistic: str
		Can be ``mean`` or ``median``, determining whether the fit was performed on the mean or 
		median density profile of a halo sample.
		
	Returns
	-------
	MspM200m: array_like
		:math:`M_{\\rm sp}/M_{\\rm 200m}`; has the same dimensions as ``Gamma`` or ``nu200m``.
	"""

	if (Gamma is not None) and (z is not None):
		cosmo = cosmology.getCurrent()
		if statistic == 'median':
			ratio = 0.59 * (1.0 + 0.35 * cosmo.Om(z)) * (1 + 0.92 * np.exp(-Gamma / 4.54))
		elif statistic == 'mean':
			ratio = 0.70 * (1.0 + 0.37 * cosmo.Om(z)) * (1 + 0.62 * np.exp(-Gamma / 2.69))
		else:
			raise Exception('Unknown statistic, %s.' % statistic)
	elif nu200m is not None:
		if statistic == 'median':
			ratio = 0.82 * (1.0 + 0.63 * np.exp(-nu200m / 3.52))
		elif statistic == 'mean':
			ratio = 0.92 * (1.0 + 0.45 * np.exp(-nu200m / 2.26))
		else:
			raise Exception('Unknown statistic, %s.' % statistic)
	else:
		raise Exception('Need either Gamma and z, or nu.')
	
	return ratio

###################################################################################################

def modelShi16Delta(s, Om):
	"""
	:math:`\\Delta_{\\rm sp}` for the Shi 2016 model.

	Parameters
	----------
	s: array_like
		Instantaneous mass accretion rate; can be a number or a numpy array.
	Om: array_like
		Matter density of the universe in units of the critical density; can be a number or a 
		numpy array.

	Returns
	-------
	Delta: array_like
		The overdensity enclosed by the predicted splashback radius, in units of the mean density.
		Has the same dimensions as ``s`` and/or ``Om``.
	"""

	return 33.0 * Om**-0.45 * np.exp((0.88 - 0.14 * np.log(Om)) * s**0.6)

###################################################################################################

def modelShi16RspR200m(s, Om):
	"""
	:math:`R_{\\rm sp}/R_{\\rm 200m}` for the Shi 2016 model.

	Parameters
	----------
	s: array_like
		Instantaneous mass accretion rate; can be a number or a numpy array.
	Om: array_like
		Matter density of the universe in units of the critical density; can be a number or a 
		numpy array.

	Returns
	-------
	RspR200m: array_like
		:math:`R_{\\rm sp}/R_{\\rm 200m}`; has the same dimensions as ``Gamma`` or ``nu200m``.
	"""

	return np.exp((0.24 + 0.074 * np.log(s)) * np.log(Om) + 0.55 - 0.15 * s)

###################################################################################################

def modelMansfield17RspR200m(Gamma, Om, nu200m):
	"""
	:math:`R_{\\rm sp}/R_{\\rm 200m}` for the Mansfield et al 2017 model.

	Parameters
	----------
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This model was calibrated with 
		Gamma computed according to the definition of 
		`Diemer & Kravtsov 2014 <http://adsabs.harvard.edu/abs/2014ApJ...789....1D>`__.
	Om: array_like
		Matter density of the universe in units of the critical density; can be a number or a 
		numpy array.
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array.		

	Returns
	-------
	RspR200m: array_like
		:math:`R_{\\rm sp}/R_{\\rm 200m}`; has the same dimensions as ``Gamma`` and/or ``nu200m``
		and/or ``Om``.
	"""

	M0 = 0.2181
	M1 = 0.4996
	A = 0.8533
	eta0 = -0.1742
	eta1 = 0.3386
	eta2 = -0.1929
	xi = -0.04668
	
	M_Om = M0 * Om + M1
	eta_Om = eta0 * Om**2 + eta1 * Om + eta2
	ret = M_Om * np.exp(Gamma * (eta_Om + nu200m * xi)) + A
	
	return ret

###################################################################################################

def modelMansfield17MspM200m(Gamma, Om):
	"""
	:math:`M_{\\rm sp}/M_{\\rm 200m}` for the Mansfield et al 2017 model.
	
	Note that, unlike the radius function for this model, the mass does not depend on peak height.

	Parameters
	----------
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This model was calibrated with 
		Gamma computed according to the definition of 
		`Diemer & Kravtsov 2014 <http://adsabs.harvard.edu/abs/2014ApJ...789....1D>`__.
	Om: array_like
		Matter density of the universe in units of the critical density; can be a number or a 
		numpy array.		

	Returns
	-------
	MspM200m: array_like
		:math:`M_{\\rm sp}/M_{\\rm 200m}`; has the same dimensions as ``Gamma`` and/or ``Om``.
	"""

	A0 = 0.1925
	A1 = 1.072
	a0 = -0.0781
	a1 = -0.02842
	Gamma_pivot = 3.0
	ret = (A0 * Om + A1) * (Gamma / Gamma_pivot)**(a0 * Om + a1)

	return ret

###################################################################################################

# Create a generalized structure for the parameters of the diemer17 and diemer20 models.
mdp = {}

mdp['diemer17'] = {}
mdp['diemer20'] = {}

for k in mdp.keys():
	mdp[k]['RspR200m'] = {}
	mdp[k]['MspM200m'] = {}
	mdp[k]['RspR200m-1s'] = {}
	mdp[k]['MspM200m-1s'] = {}
	for j in mdp[k].keys():
		mdp[k][j]['mean'] = {}
		mdp[k][j]['perc'] = {}
	mdp[k]['Gamma'] = {}

mdp['diemer17']['RspR200m']['mean']['a0'] = 0.649783
mdp['diemer17']['RspR200m']['mean']['b0'] = 0.600362
mdp['diemer17']['RspR200m']['mean']['b_om'] = 0.091996
mdp['diemer17']['RspR200m']['mean']['b_nu'] = 0.061557
mdp['diemer17']['RspR200m']['mean']['c0'] = -0.806288
mdp['diemer17']['RspR200m']['mean']['c_om'] = 17.520522
mdp['diemer17']['RspR200m']['mean']['c_nu'] = -0.293465
mdp['diemer17']['RspR200m']['mean']['c_om2'] = -9.624342
mdp['diemer17']['RspR200m']['mean']['c_nu2'] = 0.039196
mdp['diemer17']['RspR200m']['mean']['a0_p'] = 0.000000
mdp['diemer17']['RspR200m']['mean']['b0_p'] = 0.000000
mdp['diemer17']['RspR200m']['mean']['b_om_p'] = 0.000000
mdp['diemer17']['RspR200m']['mean']['b_om_p2'] = 0.000000
mdp['diemer17']['RspR200m']['mean']['b_nu_p'] = 0.000000
mdp['diemer17']['RspR200m']['mean']['c_om_p'] = 0.000000
mdp['diemer17']['RspR200m']['mean']['c_om_p2'] = 0.000000
mdp['diemer17']['RspR200m']['mean']['c_om2_p'] = 0.000000
mdp['diemer17']['RspR200m']['mean']['c_om2_p2'] = 0.000000
mdp['diemer17']['RspR200m']['mean']['c_nu_p'] = 0.000000
mdp['diemer17']['RspR200m']['mean']['c_nu2_p'] = 0.000000

mdp['diemer17']['MspM200m']['mean']['a0'] = 0.679244
mdp['diemer17']['MspM200m']['mean']['b0'] = 0.405083
mdp['diemer17']['MspM200m']['mean']['b_om'] = 0.291925
mdp['diemer17']['MspM200m']['mean']['b_nu'] = 0.000000
mdp['diemer17']['MspM200m']['mean']['c0'] = 3.365943
mdp['diemer17']['MspM200m']['mean']['c_om'] = 1.469818
mdp['diemer17']['MspM200m']['mean']['c_nu'] = -0.075635
mdp['diemer17']['MspM200m']['mean']['c_om2'] = 0.000000
mdp['diemer17']['MspM200m']['mean']['c_nu2'] = 0.000000
mdp['diemer17']['MspM200m']['mean']['a0_p'] = 0.000000
mdp['diemer17']['MspM200m']['mean']['b0_p'] = 0.000000
mdp['diemer17']['MspM200m']['mean']['b_om_p'] = 0.000000
mdp['diemer17']['MspM200m']['mean']['b_om_p2'] = 0.000000
mdp['diemer17']['MspM200m']['mean']['b_nu_p'] = 0.000000
mdp['diemer17']['MspM200m']['mean']['c_om_p'] = 0.000000
mdp['diemer17']['MspM200m']['mean']['c_om_p2'] = 0.000000
mdp['diemer17']['MspM200m']['mean']['c_om2_p'] = 0.000000
mdp['diemer17']['MspM200m']['mean']['c_om2_p2'] = 0.000000
mdp['diemer17']['MspM200m']['mean']['c_nu_p'] = 0.000000
mdp['diemer17']['MspM200m']['mean']['c_nu2_p'] = 0.000000
		
mdp['diemer17']['RspR200m-1s']['mean']['sigma_0'] = 0.052645
mdp['diemer17']['RspR200m-1s']['mean']['sigma_Gamma'] = 0.003846
mdp['diemer17']['RspR200m-1s']['mean']['sigma_nu'] = -0.012054
mdp['diemer17']['RspR200m-1s']['mean']['sigma_p'] = 0.000000

mdp['diemer17']['MspM200m-1s']['mean']['sigma_0'] = 0.052815
mdp['diemer17']['MspM200m-1s']['mean']['sigma_Gamma'] = 0.002456
mdp['diemer17']['MspM200m-1s']['mean']['sigma_nu'] = -0.011182
mdp['diemer17']['MspM200m-1s']['mean']['sigma_p'] = 0.000000

mdp['diemer17']['RspR200m']['perc']['a0'] = 0.320332
mdp['diemer17']['RspR200m']['perc']['b0'] = 0.267433
mdp['diemer17']['RspR200m']['perc']['b_om'] = 0.113389
mdp['diemer17']['RspR200m']['perc']['b_nu'] = 0.207989
mdp['diemer17']['RspR200m']['perc']['c0'] = -0.959629
mdp['diemer17']['RspR200m']['perc']['c_om'] = 16.245894
mdp['diemer17']['RspR200m']['perc']['c_nu'] = 0.000000
mdp['diemer17']['RspR200m']['perc']['c_om2'] = -9.497861
mdp['diemer17']['RspR200m']['perc']['c_nu2'] = -0.018484
mdp['diemer17']['RspR200m']['perc']['a0_p'] = 0.614807
mdp['diemer17']['RspR200m']['perc']['b0_p'] = 0.545238
mdp['diemer17']['RspR200m']['perc']['b_om_p'] = 0.000000
mdp['diemer17']['RspR200m']['perc']['b_om_p2'] = 0.000000
mdp['diemer17']['RspR200m']['perc']['b_nu_p'] = -0.223282
mdp['diemer17']['RspR200m']['perc']['c_om_p'] = 0.003941
mdp['diemer17']['RspR200m']['perc']['c_om_p2'] = 8.969094
mdp['diemer17']['RspR200m']['perc']['c_om2_p'] = -0.000485
mdp['diemer17']['RspR200m']['perc']['c_om2_p2'] = 10.613168
mdp['diemer17']['RspR200m']['perc']['c_nu_p'] = -0.451066
mdp['diemer17']['RspR200m']['perc']['c_nu2_p'] = 0.088029
			
mdp['diemer17']['MspM200m']['perc']['a0'] = 0.264765
mdp['diemer17']['MspM200m']['perc']['b0'] = 0.666040
mdp['diemer17']['MspM200m']['perc']['b_om'] = 0.168814
mdp['diemer17']['MspM200m']['perc']['b_nu'] = 0.000000
mdp['diemer17']['MspM200m']['perc']['c0'] = 4.728709
mdp['diemer17']['MspM200m']['perc']['c_om'] = 2.388866
mdp['diemer17']['MspM200m']['perc']['c_nu'] = -0.084108
mdp['diemer17']['MspM200m']['perc']['c_om2'] = 0.000000
mdp['diemer17']['MspM200m']['perc']['c_nu2'] = 0.000000
mdp['diemer17']['MspM200m']['perc']['a0_p'] = 0.843509
mdp['diemer17']['MspM200m']['perc']['b0_p'] = -0.639169
mdp['diemer17']['MspM200m']['perc']['b_om_p'] = 0.003195
mdp['diemer17']['MspM200m']['perc']['b_om_p2'] = 4.939266
mdp['diemer17']['MspM200m']['perc']['b_nu_p'] = 0.225399
mdp['diemer17']['MspM200m']['perc']['c_om_p'] = -0.705712
mdp['diemer17']['MspM200m']['perc']['c_om_p2'] = -1.241920
mdp['diemer17']['MspM200m']['perc']['c_om2_p'] = 0.000000
mdp['diemer17']['MspM200m']['perc']['c_om2_p2'] = 0.000000
mdp['diemer17']['MspM200m']['perc']['c_nu_p'] = -0.391103
mdp['diemer17']['MspM200m']['perc']['c_nu2_p'] = 0.074216

mdp['diemer17']['RspR200m-1s']['perc']['sigma_0'] = 0.044548
mdp['diemer17']['RspR200m-1s']['perc']['sigma_Gamma'] = 0.004404
mdp['diemer17']['RspR200m-1s']['perc']['sigma_nu'] = -0.014636
mdp['diemer17']['RspR200m-1s']['perc']['sigma_p'] = 0.022637

mdp['diemer17']['MspM200m-1s']['perc']['sigma_0'] = 0.027594
mdp['diemer17']['MspM200m-1s']['perc']['sigma_Gamma'] = 0.002330
mdp['diemer17']['MspM200m-1s']['perc']['sigma_nu'] = -0.012491
mdp['diemer17']['MspM200m-1s']['perc']['sigma_p'] = 0.047344

mdp['diemer17']['Gamma']['a0'] = 1.222190
mdp['diemer17']['Gamma']['a1'] = 0.351460
mdp['diemer17']['Gamma']['b0'] = -0.286441
mdp['diemer17']['Gamma']['b1'] = 0.077767
mdp['diemer17']['Gamma']['b2'] = -0.056228
mdp['diemer17']['Gamma']['b3'] = 0.004100

mdp['diemer20']['RspR200m']['mean']['a0'] = 0.659745
mdp['diemer20']['RspR200m']['mean']['b0'] = 0.556171
mdp['diemer20']['RspR200m']['mean']['b_om'] = 0.114053
mdp['diemer20']['RspR200m']['mean']['b_nu'] = 0.069775
mdp['diemer20']['RspR200m']['mean']['c0'] = -0.850819
mdp['diemer20']['RspR200m']['mean']['c_om'] = 18.446356
mdp['diemer20']['RspR200m']['mean']['c_nu'] = -0.333245
mdp['diemer20']['RspR200m']['mean']['c_om2'] = -10.059591
mdp['diemer20']['RspR200m']['mean']['c_nu2'] = 0.047432
mdp['diemer20']['RspR200m']['mean']['a0_p'] = 0.000000
mdp['diemer20']['RspR200m']['mean']['b0_p'] = 0.000000
mdp['diemer20']['RspR200m']['mean']['b_om_p'] = 0.000000
mdp['diemer20']['RspR200m']['mean']['b_om_p2'] = 0.000000
mdp['diemer20']['RspR200m']['mean']['b_nu_p'] = 0.000000
mdp['diemer20']['RspR200m']['mean']['c_om_p'] = 0.000000
mdp['diemer20']['RspR200m']['mean']['c_om_p2'] = 0.000000
mdp['diemer20']['RspR200m']['mean']['c_om2_p'] = 0.000000
mdp['diemer20']['RspR200m']['mean']['c_om2_p2'] = 0.000000
mdp['diemer20']['RspR200m']['mean']['c_nu_p'] = 0.000000
mdp['diemer20']['RspR200m']['mean']['c_nu2_p'] = 0.000000

mdp['diemer20']['MspM200m']['mean']['a0'] = 0.696229
mdp['diemer20']['MspM200m']['mean']['b0'] = 0.373627
mdp['diemer20']['MspM200m']['mean']['b_om'] = 0.300490
mdp['diemer20']['MspM200m']['mean']['b_nu'] = 0.000000
mdp['diemer20']['MspM200m']['mean']['c0'] = 3.344518
mdp['diemer20']['MspM200m']['mean']['c_om'] = 1.371785
mdp['diemer20']['MspM200m']['mean']['c_nu'] = -0.082529
mdp['diemer20']['MspM200m']['mean']['c_om2'] = 0.000000
mdp['diemer20']['MspM200m']['mean']['c_nu2'] = 0.000000
mdp['diemer20']['MspM200m']['mean']['a0_p'] = 0.000000
mdp['diemer20']['MspM200m']['mean']['b0_p'] = 0.000000
mdp['diemer20']['MspM200m']['mean']['b_om_p'] = 0.000000
mdp['diemer20']['MspM200m']['mean']['b_om_p2'] = 0.000000
mdp['diemer20']['MspM200m']['mean']['b_nu_p'] = 0.000000
mdp['diemer20']['MspM200m']['mean']['c_om_p'] = 0.000000
mdp['diemer20']['MspM200m']['mean']['c_om_p2'] = 0.000000
mdp['diemer20']['MspM200m']['mean']['c_om2_p'] = 0.000000
mdp['diemer20']['MspM200m']['mean']['c_om2_p2'] = 0.000000
mdp['diemer20']['MspM200m']['mean']['c_nu_p'] = 0.000000
mdp['diemer20']['MspM200m']['mean']['c_nu2_p'] = 0.000000

mdp['diemer20']['RspR200m-1s']['mean']['sigma_0'] = 0.050135
mdp['diemer20']['RspR200m-1s']['mean']['sigma_Gamma'] = 0.003548
mdp['diemer20']['RspR200m-1s']['mean']['sigma_nu'] = -0.010766
mdp['diemer20']['RspR200m-1s']['mean']['sigma_p'] = 0.000000

mdp['diemer20']['MspM200m-1s']['mean']['sigma_0'] = 0.045588
mdp['diemer20']['MspM200m-1s']['mean']['sigma_Gamma'] = 0.001679
mdp['diemer20']['MspM200m-1s']['mean']['sigma_nu'] = -0.007945
mdp['diemer20']['MspM200m-1s']['mean']['sigma_p'] = 0.000000

mdp['diemer20']['RspR200m']['perc']['a0'] = 0.307107
mdp['diemer20']['RspR200m']['perc']['b0'] = 0.250844
mdp['diemer20']['RspR200m']['perc']['b_om'] = 0.152731
mdp['diemer20']['RspR200m']['perc']['b_nu'] = 0.195627
mdp['diemer20']['RspR200m']['perc']['c0'] = -1.221448
mdp['diemer20']['RspR200m']['perc']['c_om'] = 17.537448
mdp['diemer20']['RspR200m']['perc']['c_nu'] = 0.000000
mdp['diemer20']['RspR200m']['perc']['c_om2'] = -10.315817
mdp['diemer20']['RspR200m']['perc']['c_nu2'] = -0.018938
mdp['diemer20']['RspR200m']['perc']['a0_p'] = 0.642779
mdp['diemer20']['RspR200m']['perc']['b0_p'] = 0.507395
mdp['diemer20']['RspR200m']['perc']['b_om_p'] = 0.000000
mdp['diemer20']['RspR200m']['perc']['b_om_p2'] = 0.000000
mdp['diemer20']['RspR200m']['perc']['b_nu_p'] = -0.212788
mdp['diemer20']['RspR200m']['perc']['c_om_p'] = 0.002358
mdp['diemer20']['RspR200m']['perc']['c_om_p2'] = 9.711469
mdp['diemer20']['RspR200m']['perc']['c_om2_p'] = -0.000550
mdp['diemer20']['RspR200m']['perc']['c_om2_p2'] = 10.762591
mdp['diemer20']['RspR200m']['perc']['c_nu_p'] = -0.473487
mdp['diemer20']['RspR200m']['perc']['c_nu2_p'] = 0.094019

mdp['diemer20']['MspM200m']['perc']['a0'] = 0.287428
mdp['diemer20']['MspM200m']['perc']['b0'] = 0.661551
mdp['diemer20']['MspM200m']['perc']['b_om'] = 0.132142
mdp['diemer20']['MspM200m']['perc']['b_nu'] = 0.000000
mdp['diemer20']['MspM200m']['perc']['c0'] = 4.591265
mdp['diemer20']['MspM200m']['perc']['c_om'] = 3.092819
mdp['diemer20']['MspM200m']['perc']['c_nu'] = -0.115458
mdp['diemer20']['MspM200m']['perc']['c_om2'] = 0.000000
mdp['diemer20']['MspM200m']['perc']['c_nu2'] = 0.000000
mdp['diemer20']['MspM200m']['perc']['a0_p'] = 0.822820
mdp['diemer20']['MspM200m']['perc']['b0_p'] = -0.656698
mdp['diemer20']['MspM200m']['perc']['b_om_p'] = 0.003182
mdp['diemer20']['MspM200m']['perc']['b_om_p2'] = 4.953590
mdp['diemer20']['MspM200m']['perc']['b_nu_p'] = 0.288168
mdp['diemer20']['MspM200m']['perc']['c_om_p'] = -0.660871
mdp['diemer20']['MspM200m']['perc']['c_om_p2'] = -1.105000
mdp['diemer20']['MspM200m']['perc']['c_om2_p'] = 0.000000
mdp['diemer20']['MspM200m']['perc']['c_om2_p2'] = 0.000000
mdp['diemer20']['MspM200m']['perc']['c_nu_p'] = -0.376065
mdp['diemer20']['MspM200m']['perc']['c_nu2_p'] = 0.078376

mdp['diemer20']['RspR200m-1s']['perc']['sigma_0'] = 0.041872
mdp['diemer20']['RspR200m-1s']['perc']['sigma_Gamma'] = 0.004279
mdp['diemer20']['RspR200m-1s']['perc']['sigma_nu'] = -0.014068
mdp['diemer20']['RspR200m-1s']['perc']['sigma_p'] = 0.023470

mdp['diemer20']['MspM200m-1s']['perc']['sigma_0'] = 0.022368
mdp['diemer20']['MspM200m-1s']['perc']['sigma_Gamma'] = 0.000916
mdp['diemer20']['MspM200m-1s']['perc']['sigma_nu'] = -0.009101
mdp['diemer20']['MspM200m-1s']['perc']['sigma_p'] = 0.045386

mdp['diemer20']['Gamma']['a0'] = 1.172092
mdp['diemer20']['Gamma']['a1'] = 0.325463
mdp['diemer20']['Gamma']['b0'] = -0.256528
mdp['diemer20']['Gamma']['b1'] = 0.093206
mdp['diemer20']['Gamma']['b2'] = -0.057122
mdp['diemer20']['Gamma']['b3'] = 0.004207

###################################################################################################

def _modelDiemerPercentileValue(rspdef):

	if rspdef is None:
		raise Exception('Need Rsp/Msp definition such as sp-apr-mn or sp-apr-p75, none given.')

	if rspdef == 'sp-apr-mn' or rspdef == 'mean':
		p = -1.0
	else:
		p = float(rspdef[-2:]) / 100.0

	return p

###################################################################################################

def _modelDiemerGetPars(model, q, rspdef):
	
	if q == 'Gamma':
		pars = mdp[model][q]
	
	else:
		if not q in mdp[model].keys():
			raise Exception('Unknown output quantity, %s.' % (q))
		
		if rspdef in ['mean', 'sp-apr-mn']:
			def_str = 'mean'
		elif rspdef.startswith('percentile') or rspdef.startswith('sp-apr-p'):
			def_str = 'perc'
		else:
			raise Exception('Unknown splashback definition, %s.' % (rspdef))
			
		pars = mdp[model][q][def_str]
		
	return pars

###################################################################################################

def modelDiemerRspMsp(Gamma, nu200m, z, rspdef, pars):
	"""
	:math:`R_{\\rm sp}/R_{\\rm 200m}` and :math:`M_{\\rm sp}/M_{\\rm 200m}` for the Diemer 2017/2020 models.

	This is a general function that takes a set of parameters for the general function. Thus, the
	user can either use the parameters given in the model defaults or a different set of 
	parameters. See convenience functions below, though the :func:`splashbackModel` function is
	the preferred way to evaluate the model. To evaluate the overdensity, use the 
	:func:`splashbackModel` function.

	Parameters
	----------
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This model was calibrated using 
		:math:`\\Gamma_{\\rm dyn}`.
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array
		with the same dimensions as ``Gamma``.
	z: float
		Redshift	
	rspdef: str
		The definition of the splashback radius. This parameter distinguishes the ``mean`` of 
		the apocenter distribution or higher percentiles (e.g. ``percentile75``). The function 
		also accepts the newer notation used in the SPARTA code, namely ``sp-apr-mn`` for the mean 
		and ``sp-apr-p75`` and so on for percentiles, which is preferred.
	pars: dict
		A dictionary of parameters; see ``mdp`` dictionary for the necessary contents.
	
	Returns
	-------
	q: array_like
		:math:`R_{\\rm sp}/R_{\\rm 200m}` or :math:`M_{\\rm sp}/M_{\\rm 200m}` depending on the 
		given parameters; has the same dimensions as ``Gamma``.
	"""

	if rspdef is None:
		raise Exception('Need Rsp/Msp definition such as sp-apr-mn or sp-apr-p75, none given.')

	cosmo = cosmology.getCurrent()
	Om = cosmo.Om(z)
	p = _modelDiemerPercentileValue(rspdef)
		
	A0 = pars['a0'] + p * pars['a0_p']
	B0 = pars['b0'] + p * pars['b0_p']
	B_om = pars['b_om'] + pars['b_om_p'] * np.exp(p * pars['b_om_p2'])
	B_nu = pars['b_nu'] + p * pars['b_nu_p']
	C0 = pars['c0']
	C_om = pars['c_om'] + pars['c_om_p'] * np.exp(p * pars['c_om_p2'])
	C_om2 = pars['c_om2'] + pars['c_om2_p'] * np.exp(p * pars['c_om2_p2'])
	C_nu = pars['c_nu'] + p * pars['c_nu_p']
	C_nu2 = pars['c_nu2'] + p * pars['c_nu2_p']
	
	A = A0
	B = (B0 + B_om * Om) * (1.0 + B_nu * nu200m)
	C = (C0 + C_om * Om + C_om2 * Om**2) * (1.0 + C_nu * nu200m + C_nu2 * nu200m**2)
	
	if utilities.isArray(C):
		C[C < 1E-4] = 1E-4
	
	ret = A + B * np.exp(-Gamma / C)
	
	return ret

###################################################################################################

def modelDiemerScatter(Gamma, nu200m, rspdef, pars):
	"""
	The 68% scatter for the Diemer 2017/2020 models.

	This is a general function that takes a set of parameters for the general function. Thus, the
	user can either use the parameters given in the model defaults or a different set of 
	parameters. See convenience functions below, though the :func:`splashbackModel` function is
	the preferred way to evaluate the model. To evaluate the scatter in the overdensity, use the 
	:func:`splashbackModel` function.

	Parameters
	----------
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This model was calibrated using 
		:math:`\\Gamma_{\\rm dyn}`.
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array
		with the same dimensions as ``Gamma``.
	rspdef: str
		The definition of the splashback radius. This parameter distinguishes the ``mean`` of 
		the apocenter distribution or higher percentiles (e.g. ``percentile75``). The function 
		also accepts the newer notation used in the SPARTA code, namely ``sp-apr-mn`` for the mean 
		and ``sp-apr-p75`` and so on for percentiles, which is preferred.
	pars: dict
		A dictionary of parameters; see ``mdp`` dictionary for the necessary contents.

	Returns
	-------
	scatter: array_like
		The scatter in either :math:`R_{\\rm sp}/R_{\\rm 200m}` or 
		:math:`M_{\\rm sp}/M_{\\rm 200m}`, depending on the parameters; has the same dimensions as 
		``Gamma``.
	"""

	if rspdef is None:
		raise Exception('Need Rsp/Msp definition such as sp-apr-mn or sp-apr-p75, none given.')

	p = _modelDiemerPercentileValue(rspdef)	
	ret = pars['sigma_0'] + pars['sigma_Gamma'] * Gamma + pars['sigma_nu'] * nu200m + pars['sigma_p'] * p

	return ret

###################################################################################################

def modelDiemerGamma(nu200m, z, pars):
	"""
	:math:`\\Gamma_{\\rm dyn}` as a function of peak height for the Diemer 2017/2020 models.
	
	A fit to the median accretion rate of halos as a function of peak height and redshift. This 
	accretion rate can be used as a guess when the real accretion rate of halos is not known.

	This is a general function that takes a set of parameters for the general function. Thus, the
	user can either use the parameters given in the model defaults or a different set of 
	parameters. See convenience functions below.
	
	Parameters
	----------
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array.
	z: array_like
		Redshift	
	pars: dict
		A dictionary of parameters; see ``mdp`` dictionary for the necessary contents.

	Returns
	-------
	Gamma: array_like
		The accretion rate measured over one dynamical time; has the same dimensions as ``nu200m`` 
		and/or ``z``.
	"""
	
	A = pars['a0'] + pars['a1'] * z
	B = pars['b0'] + pars['b1'] * z + pars['b2'] * z**2 + pars['b3'] * z**3
	Gamma = A * nu200m + B * nu200m**1.5
	
	return Gamma

###################################################################################################

def modelDiemer17RspR200m(Gamma, nu200m, z, rspdef):
	"""
	:math:`R_{\\rm sp}/R_{\\rm 200m}` for the Diemer+2017 model.

	Parameters
	----------
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This model was calibrated using 
		:math:`\\Gamma_{\\rm dyn}`.
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array
		with the same dimensions as ``Gamma``.
	z: float
		Redshift	
	rspdef: str
		The definition of the splashback radius. This parameter distinguishes the ``mean`` of 
		the apocenter distribution or higher percentiles (e.g. ``percentile75``). The function 
		also accepts the newer notation used in the SPARTA code, namely ``sp-apr-mn`` for the mean 
		and ``sp-apr-p75`` and so on for percentiles, which is preferred.
		
	Returns
	-------
	q: array_like
		:math:`R_{\\rm sp}/R_{\\rm 200m}`, has the same dimensions as ``Gamma``.
	"""

	return modelDiemerRspMsp(Gamma, nu200m, z, rspdef, _modelDiemerGetPars('diemer17', 'RspR200m', rspdef))

###################################################################################################

def modelDiemer17MspM200m(Gamma, nu200m, z, rspdef):
	"""
	:math:`M_{\\rm sp}/M_{\\rm 200m}` for the Diemer+2017 model.

	Parameters
	----------
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This model was calibrated using 
		:math:`\\Gamma_{\\rm dyn}`.
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array
		with the same dimensions as ``Gamma``.
	z: float
		Redshift	
	rspdef: str
		The definition of the splashback radius. This parameter distinguishes the ``mean`` of 
		the apocenter distribution or higher percentiles (e.g. ``percentile75``). The function 
		also accepts the newer notation used in the SPARTA code, namely ``sp-apr-mn`` for the mean 
		and ``sp-apr-p75`` and so on for percentiles, which is preferred.
		
	Returns
	-------
	q: array_like
		:math:`M_{\\rm sp}/M_{\\rm 200m}`, has the same dimensions as ``Gamma``.
	"""
		
	return modelDiemerRspMsp(Gamma, nu200m, z, rspdef, _modelDiemerGetPars('diemer17', 'MspM200m', rspdef))

###################################################################################################

def modelDiemer17RspR200mScatter(Gamma, nu200m, z, rspdef):
	"""
	The 68% scatter in :math:`R_{\\rm sp}/R_{\\rm 200m}` for the Diemer+2017 model.

	Parameters
	----------
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This model was calibrated using 
		:math:`\\Gamma_{\\rm dyn}`.
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array
		with the same dimensions as ``Gamma``.
	z: float
		Redshift	
	rspdef: str
		The definition of the splashback radius. This parameter distinguishes the ``mean`` of 
		the apocenter distribution or higher percentiles (e.g. ``percentile75``). The function 
		also accepts the newer notation used in the SPARTA code, namely ``sp-apr-mn`` for the mean 
		and ``sp-apr-p75`` and so on for percentiles, which is preferred.
		
	Returns
	-------
	scatter: array_like
		Scatter in :math:`R_{\\rm sp}/R_{\\rm 200m}`, has the same dimensions as ``Gamma``.
	"""
		
	return modelDiemerScatter(Gamma, nu200m, z, rspdef, _modelDiemerGetPars('diemer17', 'RspR200m-1s', rspdef))

###################################################################################################

def modelDiemer17MspM200mScatter(Gamma, nu200m, z, rspdef):
	"""
	The 68% scatter in :math:`M_{\\rm sp}/M_{\\rm 200m}` for the Diemer+2017 model.

	Parameters
	----------
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This model was calibrated using 
		:math:`\\Gamma_{\\rm dyn}`.
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array
		with the same dimensions as ``Gamma``.
	z: float
		Redshift	
	rspdef: str
		The definition of the splashback radius. This parameter distinguishes the ``mean`` of 
		the apocenter distribution or higher percentiles (e.g. ``percentile75``). The function 
		also accepts the newer notation used in the SPARTA code, namely ``sp-apr-mn`` for the mean 
		and ``sp-apr-p75`` and so on for percentiles, which is preferred.
		
	Returns
	-------
	scatter: array_like
		Scatter in :math:`M_{\\rm sp}/M_{\\rm 200m}`, has the same dimensions as ``Gamma``.
	"""
		
	return modelDiemerScatter(Gamma, nu200m, z, rspdef, _modelDiemerGetPars('diemer17', 'MspM200m-1s', rspdef))

###################################################################################################

def modelDiemer17Gamma(nu200m, z):
	"""
	:math:`\\Gamma_{\\rm dyn}` as a function of peak height for the Diemer+2017 model.
	
	A fit to the median accretion rate of halos as a function of peak height and redshift. This 
	accretion rate can be used as a guess when the real accretion rate of halos is not known.
	
	Parameters
	----------
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array.
	z: array_like
		Redshift	

	Returns
	-------
	Gamma: array_like
		The accretion rate measured over one dynamical time; has the same dimensions as ``nu200m`` 
		and/or ``z``.
	"""

	return modelDiemerGamma(nu200m, z, _modelDiemerGetPars('diemer17', 'Gamma', None))

###################################################################################################

def modelDiemer20RspR200m(Gamma, nu200m, z, rspdef):
	"""
	:math:`R_{\\rm sp}/R_{\\rm 200m}` for the Diemer 2020 model.

	Parameters
	----------
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This model was calibrated using 
		:math:`\\Gamma_{\\rm dyn}`.
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array
		with the same dimensions as ``Gamma``.
	z: float
		Redshift	
	rspdef: str
		The definition of the splashback radius. This parameter distinguishes the ``mean`` of 
		the apocenter distribution or higher percentiles (e.g. ``percentile75``). The function 
		also accepts the newer notation used in the SPARTA code, namely ``sp-apr-mn`` for the mean 
		and ``sp-apr-p75`` and so on for percentiles, which is preferred.
		
	Returns
	-------
	q: array_like
		:math:`R_{\\rm sp}/R_{\\rm 200m}`, has the same dimensions as ``Gamma``.
	"""
		
	return modelDiemerRspMsp(Gamma, nu200m, z, rspdef, _modelDiemerGetPars('diemer20', 'RspR200m', rspdef))

###################################################################################################

def modelDiemer20MspM200m(Gamma, nu200m, z, rspdef):
	"""
	:math:`M_{\\rm sp}/M_{\\rm 200m}` for the Diemer 2020 model.

	Parameters
	----------
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This model was calibrated using 
		:math:`\\Gamma_{\\rm dyn}`.
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array
		with the same dimensions as ``Gamma``.
	z: float
		Redshift	
	rspdef: str
		The definition of the splashback radius. This parameter distinguishes the ``mean`` of 
		the apocenter distribution or higher percentiles (e.g. ``percentile75``). The function 
		also accepts the newer notation used in the SPARTA code, namely ``sp-apr-mn`` for the mean 
		and ``sp-apr-p75`` and so on for percentiles, which is preferred.
		
	Returns
	-------
	q: array_like
		:math:`M_{\\rm sp}/M_{\\rm 200m}`, has the same dimensions as ``Gamma``.
	"""
		
	return modelDiemerRspMsp(Gamma, nu200m, z, rspdef, _modelDiemerGetPars('diemer20', 'MspM200m', rspdef))

###################################################################################################

def modelDiemer20RspR200mScatter(Gamma, nu200m, z, rspdef):
	"""
	The 68% scatter in :math:`R_{\\rm sp}/R_{\\rm 200m}` for the Diemer 2020 model.

	Parameters
	----------
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This model was calibrated using 
		:math:`\\Gamma_{\\rm dyn}`.
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array
		with the same dimensions as ``Gamma``.
	z: float
		Redshift	
	rspdef: str
		The definition of the splashback radius. This parameter distinguishes the ``mean`` of 
		the apocenter distribution or higher percentiles (e.g. ``percentile75``). The function 
		also accepts the newer notation used in the SPARTA code, namely ``sp-apr-mn`` for the mean 
		and ``sp-apr-p75`` and so on for percentiles, which is preferred.
		
	Returns
	-------
	scatter: array_like
		Scatter in :math:`R_{\\rm sp}/R_{\\rm 200m}`, has the same dimensions as ``Gamma``.
	"""
		
	return modelDiemerScatter(Gamma, nu200m, z, rspdef, _modelDiemerGetPars('diemer20', 'RspR200m-1s', rspdef))

###################################################################################################

def modelDiemer20MspM200mScatter(Gamma, nu200m, z, rspdef):
	"""
	The 68% scatter in :math:`M_{\\rm sp}/M_{\\rm 200m}` for the Diemer 2020 model.

	Parameters
	----------
	Gamma: array_like
		Mass accretion rate; can be a number or a numpy array. This model was calibrated using 
		:math:`\\Gamma_{\\rm dyn}`.
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array
		with the same dimensions as ``Gamma``.
	z: float
		Redshift	
	rspdef: str
		The definition of the splashback radius. This parameter distinguishes the ``mean`` of 
		the apocenter distribution or higher percentiles (e.g. ``percentile75``). The function 
		also accepts the newer notation used in the SPARTA code, namely ``sp-apr-mn`` for the mean 
		and ``sp-apr-p75`` and so on for percentiles, which is preferred.
		
	Returns
	-------
	scatter: array_like
		Scatter in :math:`M_{\\rm sp}/M_{\\rm 200m}`, has the same dimensions as ``Gamma``.
	"""
		
	return modelDiemerScatter(Gamma, nu200m, z, rspdef, _modelDiemerGetPars('diemer20', 'MspM200m-1s', rspdef))

###################################################################################################

def modelDiemer20Gamma(nu200m, z):
	"""
	:math:`\\Gamma_{\\rm dyn}` as a function of peak height for the Diemer 2020 model.
	
	A fit to the median accretion rate of halos as a function of peak height and redshift. This 
	accretion rate can be used as a guess when the real accretion rate of halos is not known.
	
	Parameters
	----------
	nu200m: array_like
		The peak height as computed from :math:`M_{\\rm 200m}`; can be a number or a numpy array.
	z: array_like
		Redshift	

	Returns
	-------
	Gamma: array_like
		The accretion rate measured over one dynamical time; has the same dimensions as ``nu200m`` 
		and/or ``z``.
	"""

	return modelDiemerGamma(nu200m, z, _modelDiemerGetPars('diemer20', 'Gamma', None))

###################################################################################################
