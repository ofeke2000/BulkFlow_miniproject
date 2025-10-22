###################################################################################################
#
# profile_composite.py      (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

"""
This unit implements a constructor for profiles that consist of an inner (orbiting, or 1-halo) term
and one ore more outer (infalling, 2-halo) terms. Please see 
:doc:`halo_profile` for a general introduction and a list of implemented profile forms, as well
as :doc:`tutorials` for coding examples.

---------------------------------------------------------------------------------------------------
Module contents
---------------------------------------------------------------------------------------------------

.. autosummary:: 

	compositeProfile
	getProfileClass

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

from colossus.halo import profile_outer
from colossus.halo import profile_nfw
from colossus.halo import profile_hernquist
from colossus.halo import profile_einasto
from colossus.halo import profile_dk14
from colossus.halo import profile_diemer23

###################################################################################################

def compositeProfile(inner_name = None, outer_names = ['mean', 'pl'], **kwargs):
	"""
	A wrapper function to create a profile with or without outer profile term(s).
	
	At large radii, fitting functions for halo density profiles only make sense if they are 
	combined with a description of the profile of infalling matter and/or the two-halo term, that is,
	the statistical contribution from other halos. This function provides a convenient way to 
	construct such profiles without having to set the properties of the outer terms manually. Valid 
	short codes for the inner and outer terms are listed in :doc:`halo_profile`.
	
	The function can take any combination of keyword arguments that is accepted by the constructors
	of the various profile terms. Note that some parameters, such as ``z``, can be accepted by
	multiple constructors; this is by design. 
	
	Parameters
	----------
	inner_name: str
		A shortcode for a density profile class (see :doc:`halo_profile` for a list).
	outer_names: array_like
		A list of shortcodes for one or more outer (infalling) terms (see :doc:`halo_profile` for 
		a list).
	kwargs: kwargs
		The arguments passed to the profile constructors.
	"""

	outer_terms = []
	for i in range(len(outer_names)):
		outer_cls = getProfileClass(outer_names[i])
		outer_obj = outer_cls(**kwargs)
		outer_terms.append(outer_obj)

	inner_cls = getProfileClass(inner_name)
	inner_obj = inner_cls(outer_terms = outer_terms, **kwargs)
	
	return inner_obj

###################################################################################################

def getProfileClass(name):
	"""
	Utility function that translates the name of a profile model into its class.
	
	This function does not distinguish between inner and outer profiles.
	
	Parameters
	----------
	name: str
		A shortcode for a density profile class (see :doc:`halo_profile` for a list).

	Returns
	-------
	cls: class
		The profile class.
	"""

	if name == 'nfw':
		cls = profile_nfw.NFWProfile
	elif name == 'hernquist':
		cls = profile_hernquist.HernquistProfile
	elif name == 'einasto':
		cls = profile_einasto.EinastoProfile
	elif name == 'dk14':
		cls = profile_dk14.DK14Profile
	elif name == 'diemer23':
		cls = profile_diemer23.ModelAProfile
	elif name == 'diemer23b':
		cls = profile_diemer23.ModelBProfile
	elif name == 'mean':
		cls = profile_outer.OuterTermMeanDensity
	elif name == 'cf':
		cls = profile_outer.OuterTermCorrelationFunction
	elif name == 'pl':
		cls = profile_outer.OuterTermPowerLaw
	elif name == 'infalling':
		cls = profile_outer.OuterTermInfalling
	else:
		raise Exception('Unknown profile model name, %s.' % (name))

	return cls

###################################################################################################
