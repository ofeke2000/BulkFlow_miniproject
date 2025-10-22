###################################################################################################
#
# utilities.py 	        (c) Benedikt Diemer
#     				    	diemer@umd.edu
#
###################################################################################################

"""
Common routines for Colossus modules.

---------------------------------------------------------------------------------------------------
Module contents
---------------------------------------------------------------------------------------------------

.. autosummary::
	printLine
	getHomeDir
	getCodeDir
	isArray
	getArray
	safeLog
	safeExp

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

import os
import numpy as np
import sys
import six

###################################################################################################
# CONSTANTS
###################################################################################################

max_float_exp = 100.0
"""
Highest exponent that is safe for the log/exp functions in numpy. Overflows can occur when higher
exponentials are called.
"""

min_float_log = np.exp(-max_float_exp)
max_float_log = np.exp(max_float_exp)

###################################################################################################

def printLine():
	"""
	Print a line to the console.
	"""

	print('-------------------------------------------------------------------------------------')

	return

###################################################################################################

def parseVersionString(version_str):
	"""
	Parse a version string into numbers.
	
	There are more official functions to parse version numbers that use regular expressions and 
	can handle more formats according to PEP 440. Since Colossus versions are always composed of 
	three numbers and no letters, we implement a comparison manually to avoid needing to include 
	non-standard libraries.

	Parameters
	----------
	version_str: str
		The version string to be converted.
	
	Returns
	-------
	nums: array_like
		A list of the three version numbers.
	"""
	
	w = version_str.split('.')
	if len(w) != 3:
		raise Exception('Version string invalid (%s), expected three numbers separated by dots.' \
					% version_str)

	nums = []
	for i in range(3):
		try:
			n = int(w[i])
		except Exception:
			raise Exception('Could not parse version element %s, expected a number.' % w[i])
		nums.append(n)
		
	return nums

###################################################################################################

def versionIsOlder(v1, v2):
	"""
	Compare two version strings.

	Parameters
	----------
	v1: str
		A version string.
	v2: str
		A second version string.
	
	Returns
	-------
	is_older: bool
		``True`` if v2 is older than v1, ``False`` otherwise.
	"""

	n1 = parseVersionString(v1)
	n2 = parseVersionString(v2)

	is_older = False
	for i in range(3):
		if n2[i] == n1[i]:
			continue
		is_older = (n2[i] < n1[i])
		break
	
	return is_older

###################################################################################################

def getHomeDir():
	""" 
	Finds the home directory on this system.

	Returns
	-------
	path : string
		The home directory, or ``None`` if home cannot be found.
	"""
	
	def decodePath(path):
		if six.PY2:
			return path.decode(sys.getfilesystemencoding())
		else:
			return path

	# There are basically two options for the operating system, either it's POSIX compatible of 
	# windows. POSIX includes UNIX, LINUX, Mac OS etc. The following choices were inspired by the 
	# astropy routine for finding a home directory.
	
	if os.name == 'posix':
		
		if 'HOME' in os.environ:
			home_dir = decodePath(os.environ['HOME'])
		else:
			raise Warning('Could not find HOME variable on POSIX-compatible operating system.')
			home_dir = None
	
	elif os.name == 'nt':
	
		if 'MSYSTEM' in os.environ and os.environ.get('HOME'):
			home_dir = decodePath(os.environ['HOME'])
		elif 'HOMESHARE' in os.environ:
			home_dir = decodePath(os.environ['HOMESHARE'])
		elif 'HOMEDRIVE' in os.environ and 'HOMEPATH' in os.environ:
			home_dir = os.path.join(os.environ['HOMEDRIVE'], os.environ['HOMEPATH'])
			home_dir = decodePath(home_dir)
		elif 'USERPROFILE' in os.environ:
			home_dir = decodePath(os.path.join(os.environ['USERPROFILE']))
		elif 'HOME' in os.environ:
			home_dir = decodePath(os.environ['HOME'])
		else:
			raise Warning('Could not find HOME directory on Windows system.')
			home_dir = None

	else:
	
		raise Warning('Unknown operating system type, %s. Cannot find home directory.' % os.name)
		home_dir = None
	
	return home_dir

###################################################################################################

def getCodeDir():
	"""
	Returns the path to this code file.
	
	Returns
	-------
	path: string
		The code directory.
	"""
	
	# Strip /utils from the end of the path
	path = os.path.dirname(os.path.realpath(__file__))
	path = path[:-6]

	return path

###################################################################################################

def isArray(var):
	"""
	Tests whether a variable is iterable or not.

	Parameters
	----------
	var: array_like
		Variable to be tested.
	
	Returns
	-------
	is_array: boolean
		Whether ``var`` is a numpy array or not.
	"""
	
	try:
		dummy = iter(var)
	except TypeError:
		is_array = False
	else:
		is_array = (not isinstance(var, dict))
		
	return is_array

###################################################################################################

def getArray(var):
	"""
	Convert a variable to a numpy array, whether it already is one or not.

	Parameters
	----------
	var: array_like
		Variable to be converted to an array.
	
	Returns
	-------
	var_ret: numpy array
		A numpy array with one or more entries.
	is_array: boolean
		Whether ``var`` is a numpy array or not.
	"""
		
	is_array = isArray(var)
	if is_array:
		var_ret = var
	else:
		var_ret = np.array([var])
		
	return var_ret, is_array 

###################################################################################################

# An exponential function that avoids overruns. If the exponent is smaller than -100, the result
# is zero; if it is larger than 100, it is set to e^100.

def safeLog(x):
	"""
	Natural log with check for overrun.
	
	This function checks whether any elements in a given numpy array are larger than the maximum
	(positive or negative) exponent. If so, the corresponding elements are set to the min and max
	floats that are allowed.

	Parameters
	----------
	x: array_like
		Variable to be taken the log of.
	
	Returns
	-------
	log: array_like
		Log of x
	"""
	
	if np.any(x > max_float_log) or np.any(x < min_float_log):

		f = np.zeros_like(x)
		mask_lo = (x < min_float_log)
		mask_hi = (x > max_float_log)
		mask_val = np.logical_not(mask_lo) & np.logical_not(mask_hi)
		
		f[mask_lo] = -max_float_exp
		f[mask_hi] = max_float_exp
		f[mask_val] = np.log(x[mask_val])
		
		return f
	
	else:
		return np.log(x)

###################################################################################################

def safeExp(x):
	"""
	Natural exp with check for overrun.
	
	This function checks whether any elements in a given numpy array are larger than the maximum
	(positive or negative) exponent. If so, the corresponding elements are set to the min and max
	floats that are allowed.

	Parameters
	----------
	x: array_like
		Variable to be taken the exponential of.
	
	Returns
	-------
	log: array_like
		Exponential of x
	"""
	
	if np.any(x > max_float_exp) or np.any(x < -max_float_exp):

		f = np.zeros_like(x)
		mask_lo = (x > -max_float_exp)
		mask_hi = (x > max_float_exp)
		mask_val = mask_lo & np.logical_not(mask_hi)
		
		f[mask_hi] = np.exp(max_float_exp)
		f[mask_val] = np.exp(x[mask_val])
		
		return f
	
	else:
		return np.exp(x)

###################################################################################################
