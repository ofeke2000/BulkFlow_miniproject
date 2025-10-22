###################################################################################################
#
# power_spectrum.py         (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This module implements models for the matter power spectrum, which can be evaluated using the 
:func:`~cosmology.cosmology.Cosmology.matterPowerSpectrum` function. This module is automatically
imported with the cosmology module.

---------------------------------------------------------------------------------------------------
Module contents
---------------------------------------------------------------------------------------------------

.. autosummary::
	PowerSpectrumModel
	models
	powerSpectrum
	modelSugiyama95
	modelEisenstein98
	modelEisenstein98ZeroBaryon
	modelCamb

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

###################################################################################################

import numpy as np
from collections import OrderedDict
import warnings

from colossus import defaults
from colossus.utils import utilities

###################################################################################################
# CONSTANTS
###################################################################################################

CAMB_KMIN = 1E-4
"""The minimum wavenumber for which P(k) can be evaluated by CAMB."""

CAMB_KMAX = 1E3
"""The default maximum wavenumber for which P(k) can be evaluated by CAMB. The user can set a 
different upper limit, but that may increase the runtime significantly."""

###################################################################################################

class PowerSpectrumModel():
	"""
	Characteristics of power spectrum models.
	
	The :data:`models` dictionary contains one item of this class for each available model.

	Parameters
	----------
	output: str
		Indicates the quantity a given model outputs, which can be the transfer function (``tf``)
		or the power spectrum (``ps``).
	allowed_types: array_like
		List of strings that indicate different types of power spectra that can be returned. By
		default, the ``total`` PS is returned, but some models (i.e., Boltzmann codes) can also
		compute the spectra of components such as CDM only.
	"""
		
	def __init__(self, output = None, allowed_types = None, long_name = ''):
		
		self.output = output
		self.allowed_types = allowed_types
		self.long_name = long_name
		
		return

###################################################################################################

models = OrderedDict()
"""
Dictionary containing a list of models.

An ordered dictionary containing one :class:`PowerSpectrumModel` entry for each model.
"""

models['sugiyama95']      = PowerSpectrumModel(output = 'tf', allowed_types = ['total'],        long_name = 'Sugiyama 1995')
models['eisenstein98']    = PowerSpectrumModel(output = 'tf', allowed_types = ['total'],        long_name = 'Eisenstein & Hu 1998')
models['eisenstein98_zb'] = PowerSpectrumModel(output = 'tf', allowed_types = ['total'],        long_name = 'Eisenstein & Hu 1998 (no BAO)')
models['camb']            = PowerSpectrumModel(output = 'ps', allowed_types = ['total', 'cdm'], long_name = 'CAMB')

###################################################################################################

def powerSpectrum(k, model, cosmo, output = 'ps', **kwargs):
	"""
	The power spectrum (or transfer function) as a function of wavenumber.
	
	The transfer function transforms the spectrum of primordial fluctuations into the
	linear power spectrum of the matter density fluctuations. The primordial power spectrum is 
	usually described as a power law, leading to a power spectrum of
	
	.. math::
		P(k) = T(k)^2 k^{n_s}
		
	where P(k) is the matter power spectrum, T(k) is the transfer function, and :math:`n_s` is 
	the tilt of the primordial power spectrum. See the :class:`~cosmology.cosmology.Cosmology` 
	class for further  details on the cosmological parameters.

	Parameters
	----------
	k: array_like
		The wavenumber k (in comoving h/Mpc); can be a number or a numpy array.
	model: str
		The power spectrum model (see table above).
	cosmo: cosmology
		A :class:`~cosmology.cosmology.Cosmology` object.
	output: str
		Indicates which quantity should be returned, namely the transfer function (``tf``)
		or the power spectrum (``ps``).
	kwargs: kwargs
		Keyword arguments that are passed to the function evaluating the given model.
		
	Returns
	-------
	Pk: array_like
		The power spectrum; has the same dimensions as ``k``.
	"""

	if model == 'sugiyama95':
		ret = modelSugiyama95(k, cosmo.h, cosmo.Om0, cosmo.Ob0, cosmo.Tcmb0)
	elif model == 'eisenstein98':
		ret = modelEisenstein98(k, cosmo.h, cosmo.Om0, cosmo.Ob0, cosmo.Tcmb0)
	elif model == 'eisenstein98_zb':
		ret = modelEisenstein98ZeroBaryon(k, cosmo.h, cosmo.Om0, cosmo.Ob0, cosmo.Tcmb0)
	elif model == 'camb':
		ret = modelCamb(k, cosmo, **kwargs)
	else:
		raise Exception('Unknown model, %s.' % model)
	
	if output != models[model].output:
		if (models[model].output == 'tf') and (output == 'ps'):
			ret = ret**2 * k**cosmo.ns
		elif (models[model].output == 'ps') and (output == 'tf'):
			ret = np.sqrt(ret / k**cosmo.ns)
		else:
			raise Exception('Unrecognized combination of model (%s) and output (%s) quantities.' \
						% (models[model].output, output))
			
	return ret

###################################################################################################

def powerSpectrumModelName(model, **ps_args):
	"""
	A unique internal name for the given power spectrum model (and parameters).
	
	By default, the internal name used for power spectrum models is just the name of the model 
	itself, but there are certain parameters that alter the spectrum so much that Colossus keeps
	track of separate spectra. In particular, the user can request the spectra of components such
	as dark matter and baryons (see :func:`powerSpectrum`), which are folded into the unique 
	model name.	

	Parameters
	----------
	model: str
		The power spectrum model (see table above).
	ps_args: kwargs
		Keyword arguments that are passed to the function evaluating the given model. These 
		arguments need to be consistent with those passed when the model is evaluated.
		
	Returns
	-------
	name: str
		A unique name for this power spectrum model.
	"""

	name = model

	if ('ps_type' in ps_args) and (ps_args['ps_type'] != 'tot'):
		name += '-%s' % (ps_args['ps_type'])
		
	if ('kmax' in ps_args) and (ps_args['kmax'] is not None):
		name += '-kmax%.4e' % (ps_args['kmax'])
	
	return name

###################################################################################################

def powerSpectrumLimits(model, **ps_args):
	"""
	The lower and upper wavenumbers between which a model can be evaluated.
	
	This function returns (None, None) for fitting functions that do not have a k-limit, and two
	floats for tabulated power spectra, Boltzmann codes, or other models that do have defined
	limits.

	Parameters
	----------
	model: str
		The power spectrum model (see table above).
	ps_args: kwargs
		Keyword arguments that are passed to the function evaluating the given model. These 
		arguments need to be consistent with those passed when the model is evaluated.
		
	Returns
	-------
	kmin: float
		The lowest wavenumber (in comoving h/Mpc) where the model can be evaluated.
	kmax: float
		The highest wavenumber (in comoving h/Mpc) where the model can be evaluated.
	"""

	kmin = None
	kmax = None

	if model in ['sugiyama95', 'eisenstein98', 'eisenstein98_zb']:
		pass
	elif model == 'camb':
		kmin = CAMB_KMIN
		if 'kmax' in ps_args:
			kmax = ps_args['kmax']
		else:
			kmax = CAMB_KMAX
	else:
		raise Exception('Unknown model, %s.' % model)
	
	return kmin, kmax

###################################################################################################

def modelCamb(k, cosmo, ps_type = 'tot', kmax = CAMB_KMAX, **kwargs):
	"""
	The power spectrum as computed by the CAMB Boltzmann solver.
	
	This function translates a Colossus Cosmology object into parameters for the CAMB code and 
	computes the power spectrum. See the 
	`CAMB documentation <https://camb.readthedocs.io/en/latest/index.html>`__ for information
	on possible keyword arguments and details about the calculations. We deliberately turn off
	the reionization component of the power spectrum, since that is usually not desired for 
	large-scale structure and halo calculations. 
	
	In general, we leave as many parameters as possible to their default values in order to take
	advantage of future optimizations in the CAMB code. This means, on the other hand, that the
	results depend slightly on the code version. This function was tested with versions up to 
	CAMB 1.3.5.
	
	Important note: Colossus automatically keeps track of multiple versions of the power spectrum
	if different ``ps_type`` and ``kmax`` parameters are passed, but NOT for different keyword
	arguments (see ``kwargs`` below).

	Parameters
	----------
	k: array_like
		The wavenumber k (in comoving h/Mpc); can be a number or a numpy array, but the input must
		be larger than 1E-4 (a fixed limit in CAMB) and smaller than :func:`CAMB_KMAX` unless a 
		larger upper limit is passed as a ``kmax`` keyword argument.
	cosmo: cosmology
		A :class:`~cosmology.cosmology.Cosmology` object.
	ps_type: str
		CAMB can evaluate the ``tot`` (total) power spectrum (the default) or that of components, e.g.
		``cdm`` (only dark matter) or ``baryon`` (only baryons). Other options can be passed by 
		setting ``var1 = 'delta_XXX`` in the keyword arguments, but such a choice is not taken into
		account when internally naming the power spectrum model and will thus likely lead to 
		inconsistent behavior. Conversely, Colossus can handle multiple power spectrum types
		within the same cosmology if ``ps_type`` is set.
	kmax: float
		The maximum wavenumber for which the CAMB calculation will be set up. This parameter is
		separate from the ``k`` array because kmax is fixed after CAMB has been initialized when
		this function is called for the first time. Note that increasing ``kmax`` significantly
		increases the runtime. On the other hand, it is up to the user to make sure that an 
		insufficient ``kmax`` does not lead to inaccurate results. For example, when evaluating
		the variance :func:`~cosmology.cosmology.Cosmology.sigma` for radii close to kmax, the
		variance will be underestimated due to the missing high-k modes in the power spectrum.
	kwargs: kwargs
		Arguments that are passed to the set_params() function in CAMB (see documentation). Note
		that Colossus does not keep track of different versions of the power spectra created with
		different kwargs. If multiple spectra are to be computed, it is easiest to create multiple
		cosmology objects.

	Returns
	-------
	P: array_like
		The power spectrum in units of :math:`({\\rm Mpc}/h)^3`; has the same dimensions as ``k``.
	"""

	try:
		import camb
	except:
		raise Exception('Could not find CAMB python unit. Please make sure it is installed and in the PYTHON_PATH.')

	# Check k input; CAMB can only evaluate evenly spaced arrays. Even if the user has requested
	# only a single wavenumber, we need to pass at least two.
	k_array, is_array = utilities.getArray(k)
	if not is_array:
		kmin_eval = k
		kmax_eval = k * 1.001
		nk_eval = 2
		k_array = np.array([kmin_eval, kmax_eval])
	else:
		kmin_eval = k_array[0]
		kmax_eval = k_array[-1]
		nk_eval = len(k_array)

	# Get camb_results from storage of cosmology object. This way, we know that the object is 
	# deleted when the cosmology changes, and that we are not dealing with multiple cosmologies.
	object_name = powerSpectrumModelName('camb', ps_type = ps_type, kmax = kmax) + '_results'
	camb_results = cosmo.storageUser.getStoredObject(object_name)

	if camb_results is None:
	
		# Warn if interpolation is turned off, as this will be very slow for the CAMB model.
		if (not cosmo.interpolation):
			warnings.warn('When using the CAMB power spectrum model, it is recommended to set interpolation = True for efficiency.')
		
		# Set basic cosmology and user-defined parameters
		h = cosmo.h
		h2 = h**2
		if cosmo.relspecies:
			num_nu_massless = cosmo.Neff
		else:
			num_nu_massless = 0.0
		camb_args = dict(ns = cosmo.ns, H0 = h * 100.0, ombh2 = cosmo.Ob0 * h2, 
						omch2 = (cosmo.Om0 - cosmo.Ob0) * h2, omnuh2 = 0.0,
						num_nu_massless = num_nu_massless, num_nu_massive = 0, 
						nu_mass_numbers = [0], nu_mass_degeneracies = [0], nu_mass_fractions = [0],
						Reion = camb.reionization.TanhReionization(Reionization = False),
						WantTensors = False, WantVectors = False, WantDerivedParameters = False,
						Want_cl_2D_array = False, Want_CMB_lensing = False, 
						DoLensing = False, NonLinear = False, WantTransfer = True)
		camb_args.update(kwargs)
		cp = camb.set_params(**camb_args)
	
		# Set dark energy model
		cp.DarkEnergy = camb.dark_energy.DarkEnergyFluid()
		if cosmo.de_model == 'lambda':
			cp.DarkEnergy.set_params(w = -1.0)
		elif cosmo.de_model == 'w0':
			cp.DarkEnergy.set_params(w = cosmo.w0)
		elif cosmo.de_model == 'w0wa':
			cp.DarkEnergy.set_params(w = cosmo.w0, wa = cosmo.wa)
		elif cosmo.de_model == 'user':
			a = 10**np.linspace(-2.0, 0.0, 50)
			z = 1.0 / a - 1.0
			w = cosmo.wz_function(z)
			cp.DarkEnergy.set_w_a_table(a, w)
		else:
			raise Exception('Unknown de_model, %s.' % (cosmo.de_model))
	
		# Set parameters for transfer function
		cp.set_matter_power(redshifts = [0.0], kmax = kmax, accurate_massive_neutrino_transfers = False)
		cp.Transfer.high_precision = True
		
		# Initialize CAMB calculations
		camb_results = camb.get_results(cp)
		
		# Store the results in the cosmology's storage system, but do not write it to disk between
		# runs (because the user might make different choices about the CAMB parameters, which 
		# would be hard to keep track of).
		cosmo.storageUser.storeObject(object_name, camb_results, persistent = False)
	
	# Evaluate power spectrum
	k_camb, _, P = camb_results.get_matter_power_spectrum(kmin_eval, kmax_eval, npoints = nk_eval, 
														var1 = 'delta_%s' % (ps_type))
	P = P[0]
	
	# We check that the wavenumbers returned by CAMB match the input.
	if np.max(np.abs((k_camb - k_array) / k_array)) > 1E-4:
		print(np.diff(np.log10(k_array)))
		raise Exception('CAMB power spectrum can only be evaluated for wavenumbers evenly spaced in log10(k). Differences are shown above.')

	if not is_array:
		P = P[0]
	
	return P

###################################################################################################

def transferFunction(k, h, Om0, Ob0, Tcmb0, model = defaults.POWER_SPECTRUM_MODEL):
	"""
	The transfer function (deprecated).
	
	This function is deprecated, :func:`powerSpectrum` should be used instead.
	
	The transfer function transforms the spectrum of primordial fluctuations into the
	linear power spectrum of the matter density fluctuations. The primordial power spectrum is 
	usually described as a power law, leading to a power spectrum
	
	.. math::
		P(k) = T(k)^2 k^{n_s}
		
	where P(k) is the matter power spectrum, T(k) is the transfer function, and :math:`n_s` is 
	the tilt of the primordial power spectrum. See the :class:`~cosmology.cosmology.Cosmology` 
	class for further  details on the cosmological parameters.

	Parameters
	----------
	k: array_like
		The wavenumber k (in comoving h/Mpc); can be a number or a numpy array.
	h: float
		The Hubble constant in units of 100 km/s/Mpc.
	Om0: float
		:math:`\\Omega_{\\rm m}`, the matter density in units of the critical density at z = 0.
	Ob0: float
		:math:`\\Omega_{\\rm b}`, the baryon density in units of the critical density at z = 0.
	Tcmb0: float
		The temperature of the CMB at z = 0 in Kelvin.

	Returns
	-------
	Tk: array_like
		The transfer function; has the same dimensions as ``k``.
	"""
	
	warnings.warn('transferFunction() is deprecated and will be removed in a future version. Please use powerSpectrum() instead.')
	
	if model == 'sugiyama95':
		T = modelSugiyama95(k, h, Om0, Ob0, Tcmb0)
	elif model == 'eisenstein98':
		T = modelEisenstein98(k, h, Om0, Ob0, Tcmb0)
	elif model == 'eisenstein98_zb':
		T = modelEisenstein98ZeroBaryon(k, h, Om0, Ob0, Tcmb0)
	else:
		raise Exception('Unknown model, %s.' % model)
	
	return T

###################################################################################################

def modelSugiyama95(k, h, Om0, Ob0, Tcmb0):
	"""
	The transfer function according to Sugiyama 1995.
	
	This function computes the 
	`Sugiyama 1995 <https://ui.adsabs.harvard.edu/abs/1995ApJS..100..281S/abstract>`__ 
	approximation to the transfer function at a scale k, which is based on the 
	`Bardeen et al. 1986 <https://ui.adsabs.harvard.edu/abs/1986ApJ...304...15B/abstract>`__
	formulation. Note that this approximation is not as accurate as the ``eisenstein98`` model,
	with deviations of about 10-20% in the power spectrum, variance, and correlation function.

	Parameters
	----------
	k: array_like
		The wavenumber k (in comoving h/Mpc); can be a number or a numpy array.
	h: float
		The Hubble constant in units of 100 km/s/Mpc.
	Om0: float
		:math:`\\Omega_{\\rm m}`, the matter density in units of the critical density at z = 0.
	Ob0: float
		:math:`\\Omega_{\\rm b}`, the baryon density in units of the critical density at z = 0.
	Tcmb0: float
		The temperature of the CMB at z = 0 in Kelvin.

	Returns
	-------
	Tk: array_like
		The transfer function; has the same dimensions as ``k``.
	"""

	k, is_array = utilities.getArray(k)
	
	# The input is k/h rather than k, so one h cancels out
	q = (Tcmb0 / 2.7)**2 * k / (Om0 * h * np.exp(-Ob0 * (1.0 + np.sqrt(2 * h) / Om0)))
	
	Tk = np.log(1.0 + 2.34 * q) / (2.34 * q) \
		* (1.0 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4)**-0.25

	# Numerically, very small values of q lead to issues with T become zero rather than one.
	Tk[q < 1E-9] = 1.0

	if not is_array:
		Tk = Tk[0]
	
	return Tk

###################################################################################################

def modelEisenstein98(k, h, Om0, Ob0, Tcmb0):
	"""
	The transfer function according to Eisenstein & Hu 1998.
	
	This function computes the 
	`Eisenstein & Hu 1998 <http://adsabs.harvard.edu/abs/1998ApJ...496..605E>`__ approximation 
	to the transfer function at a scale k. The code was adapted from Matt Becker's cosmocalc 
	code.
	
	This function was tested against numerical calculations based on the CAMB code 
	(`Lewis et al. 2000 <http://adsabs.harvard.edu/abs/2000ApJ...538..473L>`__) and found to be
	accurate to 5\\% or better up to k of about 100 h/Mpc (see the Colossus code paper for 
	details). 

	Parameters
	----------
	k: array_like
		The wavenumber k (in comoving h/Mpc); can be a number or a numpy array.
	h: float
		The Hubble constant in units of 100 km/s/Mpc.
	Om0: float
		:math:`\\Omega_{\\rm m}`, the matter density in units of the critical density at z = 0.
	Ob0: float
		:math:`\\Omega_{\\rm b}`, the baryon density in units of the critical density at z = 0.
	Tcmb0: float
		The temperature of the CMB at z = 0 in Kelvin.

	Returns
	-------
	Tk: array_like
		The transfer function; has the same dimensions as ``k``.

	See also
	--------
	modelEisenstein98ZeroBaryon: The zero-baryon transfer function according to Eisenstein & Hu 1998.
	"""

	if np.abs(np.min(Ob0)) < 1E-20:
		raise Exception('The Eisenstein & Hu 98 transfer function cannot be computed for Ob0 = 0.')

	# Define shorter expressions
	omc = Om0 - Ob0
	ombom0 = Ob0 / Om0
	h2 = h**2
	om0h2 = Om0 * h2
	ombh2 = Ob0 * h2
	theta2p7 = Tcmb0 / 2.7
	theta2p72 = theta2p7**2
	theta2p74 = theta2p72**2
	
	# Convert kh from h/Mpc to 1/Mpc
	kh = k * h

	# Equation 2
	zeq = 2.50e4 * om0h2 / theta2p74

	# Equation 3
	keq = 7.46e-2 * om0h2 / theta2p72

	# Equation 4
	b1d = 0.313 * om0h2**-0.419 * (1.0 + 0.607 * om0h2**0.674)
	b2d = 0.238 * om0h2**0.223
	zd = 1291.0 * om0h2**0.251 / (1.0 + 0.659 * om0h2**0.828) * (1.0 + b1d * ombh2**b2d)

	# Equation 5
	Rd = 31.5 * ombh2 / theta2p74 / (zd / 1e3)
	Req = 31.5 * ombh2 / theta2p74 / (zeq / 1e3)

	# Equation 6
	s = 2.0 / 3.0 / keq * np.sqrt(6.0 / Req) * np.log((np.sqrt(1.0 + Rd) + \
		np.sqrt(Rd + Req)) / (1.0 + np.sqrt(Req)))

	# Equation 7
	ksilk = 1.6 * ombh2**0.52 * om0h2**0.73 * (1.0 + (10.4 * om0h2)**-0.95)

	# Equation 10
	q = kh / 13.41 / keq

	# Equation 11
	a1 = (46.9 * om0h2)**0.670 * (1.0 + (32.1 * om0h2)**-0.532)
	a2 = (12.0 * om0h2)**0.424 * (1.0 + (45.0 * om0h2)**-0.582)
	ac = a1**(-ombom0) * a2**(-ombom0**3)

	# Equation 12
	b1 = 0.944 / (1.0 + (458.0 * om0h2)**-0.708)
	b2 = (0.395 * om0h2)**-0.0266
	bc = 1.0 / (1.0 + b1 * ((omc / Om0)**b2 - 1.0))

	# Equation 15
	y = (1.0 + zeq) / (1.0 + zd)
	Gy = y * (-6.0 * np.sqrt(1.0 + y) + (2.0 + 3.0 * y) \
		* np.log((np.sqrt(1.0 + y) + 1.0) / (np.sqrt(1.0 + y) - 1.0)))

	# Equation 14
	ab = 2.07 * keq * s * (1.0 + Rd)**(-3.0 / 4.0) * Gy

	# Get CDM part of transfer function

	# Equation 18
	f = 1.0 / (1.0 + (kh * s / 5.4)**4)

	# Equation 20
	C = 14.2 / ac + 386.0 / (1.0 + 69.9 * q**1.08)

	# Equation 19
	T0t = np.log(np.e + 1.8 * bc * q) / (np.log(np.e + 1.8 * bc * q) + C * q * q)

	# Equation 17
	C1bc = 14.2 + 386.0 / (1.0 + 69.9 * q**1.08)
	T0t1bc = np.log(np.e + 1.8 * bc * q) / (np.log(np.e + 1.8 * bc * q) + C1bc * q * q)
	Tc = f * T0t1bc + (1.0 - f) * T0t

	# Get baryon part of transfer function

	# Equation 24
	bb = 0.5 + ombom0 + (3.0 - 2.0 * ombom0) * np.sqrt((17.2 * om0h2) * (17.2 * om0h2) + 1.0)

	# Equation 23
	bnode = 8.41 * om0h2**0.435

	# Equation 22
	st = s / (1.0 + (bnode / kh / s) * (bnode / kh / s) * (bnode / kh / s))**(1.0 / 3.0)

	# Equation 21
	C11 = 14.2 + 386.0 / (1.0 + 69.9 * q**1.08)
	T0t11 = np.log(np.e + 1.8 * q) / (np.log(np.e + 1.8 * q) + C11 * q * q)
	Tb = (T0t11 / (1.0 + (kh * s / 5.2)**2) + ab / (1.0 + (bb / kh / s)**3) * np.exp(-(kh / ksilk)**1.4)) \
		* np.sin(kh * st) / (kh * st)

	# Total transfer function
	Tk = ombom0 * Tb + omc / Om0 * Tc

	return Tk

###################################################################################################

def modelEisenstein98ZeroBaryon(k, h, Om0, Ob0, Tcmb0):
	"""
	The zero-baryon transfer function according to Eisenstein & Hu 1998.
	
	This fitting function is significantly simpler than the full 
	:func:`modelEisenstein98` version, and still approximates numerical calculations from a 
	Boltzmann code to better than 10\\%, and almost as accurate when computing the variance or
	correlation function (see the Colossus code paper for details).
	
	If Ob > 0, the assumptions of zero baryons is obviously inconsistent. However, the function
	executes without a warning because it is most commonly intended as a simplification of the 
	power spectrum (e.g., to avoid the BAO peaks), not as an actual model for a Universe without
	baryons.

	Parameters
	----------
	k: array_like
		The wavenumber k (in comoving h/Mpc); can be a number or a numpy array.
	h: float
		The Hubble constant in units of 100 km/s/Mpc.
	Om0: float
		:math:`\\Omega_{\\rm m}`, the matter density in units of the critical density at z = 0.
	Ob0: float
		:math:`\\Omega_{\\rm b}`, the baryon density in units of the critical density at z = 0.
	Tcmb0: float
		The temperature of the CMB at z = 0 in Kelvin.

	Returns
	-------
	Tk: array_like
		The transfer function; has the same dimensions as ``k``.

	See also
	--------
	modelEisenstein98: The transfer function according to Eisenstein & Hu 1998.
	"""
	
	ombom0 = Ob0 / Om0
	h2 = h**2
	om0h2 = Om0 * h2
	ombh2 = Ob0 * h2
	theta2p7 = Tcmb0 / 2.7
	
	# Convert kh from hMpc^-1 to Mpc^-1
	kh = k * h

	# Equation 26
	s = 44.5 * np.log(9.83 / om0h2) / np.sqrt(1.0 + 10.0 * ombh2**0.75)

	# Equation 31
	alphaGamma = 1.0 - 0.328 * np.log(431.0 * om0h2) * ombom0 + 0.38 * np.log(22.3 * om0h2) * ombom0**2

	# Equation 30
	Gamma = Om0 * h * (alphaGamma + (1.0 - alphaGamma) / (1.0 + (0.43 * kh * s)**4))

	# Equation 28
	q = k * theta2p7 * theta2p7 / Gamma

	# Equation 29
	C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
	L0 = np.log(2.0 * np.exp(1.0) + 1.8 * q)
	Tk = L0 / (L0 + C0 * q * q)

	return Tk

###################################################################################################
