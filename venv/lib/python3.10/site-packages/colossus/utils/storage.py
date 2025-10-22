###################################################################################################
#
# storage.py 	        (c) Benedikt Diemer
#     				    	diemer@umd.edu
#
###################################################################################################

"""
This module provides both non-persistent and persistent storage and interpolation to be used by any
module in Colossus. 

---------------------------------------------------------------------------------------------------
Basics
---------------------------------------------------------------------------------------------------

There are two levels of storage: all stored fields are stored in dictionaries in memory. The data
can be of any type, if persistent storage is used the data must be 
`pickleable <https://docs.python.org/3/library/pickle.html#what-can-be-pickled-and-unpickled>`__. 
The persistent storage can be turned on and off by the user, both generally and for each field 
individually. 

Each "user" of the storage module receives their own storage space and a uniquely identifying hash 
code that can be used to detect changes that make it necessary to reset the storage, for example 
changes in physical parameters to a model. The code example below shows how to set up a storage
user within a class::

	from colossus.utils import storage
	
	class DemoClass():
		
		def __init__(self, some_parameter = 1.5):
			self.some_parameter = some_parameter
			self.su = storage.StorageUser('myModule', 'rw', self.getName, self.getHashableString, 
							self.reportChanges)
			self.some_data = [2.6, 9.5]
			self.su.storeObject('test_data', self.some_data, persistent = False)
			return
	
		def getName(self):
			return 'MyModule'
	
		def getHashableString(self):
			param_string = 'MyModule_%.4f' % (self.some_parameter)
			return param_string
			
		def reportChanges(self):
			print('Changes in this module detected, storage has been reset.')
			return
		
		def loadData(self):
			data = self.su.getStoredObject('test_data')
			return data
	
In the constructor, we have given the storage module pointers to three functions that return a
unique name for this module, a hashable string, and one that should be called when changes are
detected. The hashable string must change if the previously stored data is to be discarded upon a
parameter change. In the class above, we discard the field ``some_data`` when a change in 
``some_parameter`` is detected. The ``persistent`` parameter determines whether this object will 
be written to disk as part of a pickle and loaded next time the same user class (same name and 
same hash code) is instantiated. Let us now try loading the data::

	dc = DemoClass()
	print(dc.loadData())
	>>> [2.6, 9.5]
	
Let us now change ``some_parameter``. The data object is discarded and the load function returns
``None``, indicating that no valid ``some_data`` for the new hash string was found::

	dc.some_parameter = 1.2
	print(dc.loadData())
	>>> 'Changes in this module detected, storage has been reset.'
	>>> None

The storage module offers native support for interpolation tables. For example, if we have 
stored a table of variables x and y, we can get a spline interpolator for y(x) or even a reverse
interpolator for x(y) by calling::

	interp_y_of_x = su.getStoredObject('xy', interpolator = True)
	interp_x_of_y = su.getStoredObject('xy', interpolator = True, inverse = True)

where ``su`` is a ``StorageUser`` object. The 
:func:`~utils.storage.StorageUser.getStoredObject` function returns ``None`` if no object is 
found.

---------------------------------------------------------------------------------------------------
Module reference
---------------------------------------------------------------------------------------------------
"""

import os
import hashlib
import pickle
import warnings
import numpy as np
import scipy.interpolate

from colossus import version
from colossus import settings
from colossus.utils import utilities

###################################################################################################

class StorageUser():
	"""
	A storage user object allows access to persistent and non-persistent storage.
	
	Parameters
	----------
	module: str
		The name of the module to which this user belongs. This name determines the cache sub-
		directory where files will be stored.
	persistence: str
		A combination of ``'r'`` and ``'w'``, e.g. ``'rw'`` or ``''``, indicating whether the 
		storage is read and/or written from and to disk.
	func_name: function
		A function that takes no parameters and returns the name of the user class.
	func_hashstring: function
		A function that takes no parameters and returns a unique string identifying the user class
		and any of its properties that, if changed, should trigger a resetting of the storage. If 
		the hash string changes, the storage is emptied.
	func_changed: function
		A function that takes no parameters and will be called if the hash string has been found to
		have changed (see above).
	"""
	
	def __init__(self, module, persistence, func_name, func_hashstring, func_changed):

		self.module = module
		self.func_name = func_name
		self.func_hashstring = func_hashstring
		self.func_changed = func_changed

		if persistence in [True, False]:
			raise DeprecationWarning('The persistence parameter is not boolean but a combination of r and w, such as "rw".')

		for l in persistence:
			if not l in ['r', 'w']:
				raise Exception('The persistence parameter contains an unknown letter %c.' % l)
		self.persistence_read = ('r' in persistence)
		self.persistence_write = ('w' in persistence)

		if self.persistence_read or self.persistence_write:
			self.cache_dir = getCacheDir(module = self.module)
		
		self.resetStorage()
	
		return

	###############################################################################################

	def getHash(self):
		"""
		Get a unique string from the user class and convert it to a hash.
		
		Returns
		-------
		hash: str
			A string that changes if the input string is changed, but can be much shorter than the 
			input string.
		"""
			
		hashable_string = self.func_hashstring()
		hash_ret = hashlib.md5(hashable_string.encode()).hexdigest()
		
		return hash_ret

	###############################################################################################

	def getUniqueFilename(self):
		"""
		Create a unique filename for this storage user.
		
		Returns
		-------
		filename: str
			A filename that is unique to this module, storage user name, and the properties of the 
			user as encapsulated in its hashable string.
		"""
					
		return self.cache_dir + self.func_name() + '_' + self.getHash()
	
	###############################################################################################
	
	def checkForChangedHash(self):
		"""
		Check whether the properties of the user class have changed.
		
		Returns
		-------
		has_changed: bool
			Returns ``True`` if the hash has changed compared to the last stored hash.
		"""
			
		hash_new = self.getHash()
		has_changed = (hash_new != self.hash_current)
		
		return has_changed
	
	###############################################################################################

	def resetStorage(self):
		"""
		Reset the storage arrays and load persistent storage from file.
		"""
			
		# Reset the test hash and storage containers. There are two containes, one for objects
		# that are stored in a pickle file, and one for those that will be discarded when the 
		# class is destroyed.
		self.hash_current = self.getHash()
		self.storage_pers = {}
		self.storage_temp = {}
		
		# Check if there is a persistent object storage file. If so, load its contents into the
		# storage dictionary. We only load from file if the user has not switched of storage, and
		# if the user has not switched off interpolation.
		#
		# Loading the pickle can go wrong due to python version differences, so we generously
		# catch any exceptions that may occur and simply delete the file in that case.
		if self.persistence_read:
			filename_pickle = self.getUniqueFilename()
			if os.path.exists(filename_pickle):
				try:
					input_file = open(filename_pickle, 'rb')
					self.storage_pers = pickle.load(input_file)
					
					# Check if a version was stored with this file. If not, assume it is old.
					if 'colossus_version' in self.storage_pers:
						persistence_version = self.storage_pers['colossus_version']
					else:
						persistence_version = '1.0.0'
					input_file.close()
					
					# If the file version is below the allowed version limit, throw away this file.
					if utilities.versionIsOlder(settings.PERSISTENCE_OLDEST_VERSION, persistence_version):
						try:
							os.remove(filename_pickle)
							print('Deleted outdated persistence file, no further action needed (%s, versions %s/%s).' \
								% (filename_pickle, settings.PERSISTENCE_OLDEST_VERSION, persistence_version))
						except Exception:
							warnings.warn('Could not delete outdated persistence file %s. Please delete manually.' \
										% (filename_pickle))
						self.storage_pers = {}
						self.storage_pers['colossus_version'] = version.__version__
					
				except Exception:
					warnings.warn('Encountered file error while reading cache file. This usually \
						happens when switching between python 2 and 3. Deleting cache file.')
					try:
						os.remove(filename_pickle)
					except Exception:
						pass
			
			else:
				self.storage_pers['colossus_version'] = version.__version__
		
		return
	
	###############################################################################################

	def storeObject(self, object_name, object_data, persistent = True):
		"""
		Save an object in memory and/or file storage.

		The object is written to a dictionary in memory, and also to file if ``persistent == True``
		(unless persistence does not contain ``'w'``). 

		Parameters
		----------
		object_name: str
			The name of the object by which it can be retrieved later.
		object_data: any
			The object; can be any picklable data type.
		persistent: bool
			If ``True``, store this object on disk (if persistence is activated globally).
		"""
	
		if persistent:
			self.storage_pers[object_name] = object_data
			
			if self.persistence_write:
				filename_pickle = self.getUniqueFilename()
				output_file = open(filename_pickle, 'wb')
				pickle.dump(self.storage_pers, output_file, pickle.HIGHEST_PROTOCOL)
				output_file.close()  

		else:
			self.storage_temp[object_name] = object_data

		return
		
	###############################################################################################

	def getStoredObject(self, object_name, interpolator = False, inverse = False, path = None,
					store_interpolator = True, store_path_data = True):
		"""
		Retrieve a stored object from memory or file.

		If an object is already stored in memory, return it. If not, try to load it from file, 
		otherwise return None. If the object is a 2-dimensional table, this function can also 
		return an interpolator. If the ``path`` parameter is passed, the file is loaded from that 
		file path.
		
		Parameters
		----------
		object_name: str
			The name of the object to be loaded.
		interpolator: bool
			If ``True``, return a spline interpolator instead of the underlying table. For this to
			work, the object data must either be an array of dimensionality ``[2, n]`` or a tuple
			with three entries of the format ``(x, y, z[x, y])`` where ``x`` and ``y`` are 
			ascending arrays and ``z`` is of dimensionality ``len(x), len(y)``.
		inverse: bool
			Return an interpolator that gives x(y) instead of y(x). This parameter only works for
			a 1-dimensional interpolator (see ``interpolator`` above).
		path: str
			If not ``None``, data is loaded from this file path (unless it has already been loaded, 
			in which case it is found in memory).
		store_interpolator: bool
			If ``True`` (the default), an interpolator that has been created is temporarily stored 
			so that it does not need to be created again.
		store_path_data: bool
			If ``True`` (the default), data loaded from a file defined by path is stored 
			temporarily so that it does not need to be loaded again.
	
		Returns
		-------
		object_data: any
			Returns the loaded object (any pickleable data type), or a 
			scipy.interpolate.InterpolatedUnivariateSpline interpolator object, or ``None`` if no 
			object was found.
		"""
		
		# -----------------------------------------------------------------------------------------
		
		def tryTxtLoad(self, read_path):
			
			object_data = None
			if not self.persistence_read:
				return None
	
			if read_path is not None:
				if os.path.exists(read_path):
					object_data = np.loadtxt(read_path, usecols = (0, 1),
										skiprows = 0, comments = '#', unpack = True)
				else:
					raise Exception('File %s not found.' % (read_path))
							
			return object_data

		# -----------------------------------------------------------------------------------------
		
		# First, check for changes in the hash. If changes are detected, first call the user's 
		# change callback function and then reset the storage.
		if self.checkForChangedHash():
			if self.func_changed is not None:
				self.func_changed()
			self.resetStorage()
			
		# Compute object name. If the object contains a file path, we need to isolate 
		object_id = object_name
		if interpolator:
			object_id += '_interpolator'
		if inverse:
			object_id += '_inverse'

		# Find the object. There are multiple possibilities:
		# - Check for the exact object the user requested (the object_id)
		#   - Check in persistent storage
		#   - Check in temporary storage (where interpolator / inverse objects live)
		#   - Check in user text file (where the path was given)
		# - Check for the raw object (the object_name)
		#   - Check in persistent storage
		#   - Check in temporary storage (where user-defined, pre-loaded objects live)
		#   - Check in user text files (where the path was given)
		#  - Convert to the exact object, store in temporary storage
		# - If all fail, return None

		object_data = None
		if object_id in self.storage_pers:	
			object_data = self.storage_pers[object_id]
		
		elif object_id in self.storage_temp:
			object_data = self.storage_temp[object_id]

		elif not interpolator:
			object_data = tryTxtLoad(self, path)
			if (object_data is not None) and store_path_data:
				self.storage_temp[object_id] = object_data

		# We could not find the object ID anywhere. This can have two reasons: the object does
		# not exist, or we must transform an existing object.
		if interpolator and object_data is None:
			
			# Try to find the object to transform.
			object_raw = None
			
			if object_name in self.storage_pers:	
				object_raw = self.storage_pers[object_name]
			elif object_name in self.storage_temp:
				object_raw = self.storage_temp[object_name]
			else:
				object_raw = tryTxtLoad(self, path)

			if object_raw is None:
				
				# We cannot find an object to convert, return none.
				object_data = None
			
			else:
				
				# Guess the type of interpolator we are trying to create. 
				if isinstance(object_raw, tuple) and len(object_raw) == 3:
					
					if inverse:
						raise Exception('Inverse 2D interpolator is not possible.')
					
					object_data = scipy.interpolate.RectBivariateSpline(object_raw[0], object_raw[1], 
												object_raw[2], kx = 3, ky = 3)

				elif isinstance(object_raw, np.ndarray) and len(object_raw.shape) == 2 \
					and object_raw.shape[0] == 2:
				
					# Convert and store in temporary storage.
					if inverse: 
						
						# There is a subtlety: the spline interpolator can't deal with decreasing 
						# x-values, so if the y-values are decreasing, we reverse their order.
						# If they are still not monotonically ascending after switching the order,
						# the interpolator will fail. In the case where the values are flat, we
						# use a bit of a hack and remove the flat values; this is OK but the user
						# should check the actual limits of the interpolator before using it.
						if object_raw[1][-1] < object_raw[1][0]:
							object_raw = object_raw[:, ::-1]
						
						if (np.min(np.diff(object_raw[1])) < 0.0):
							raise Exception('While inverting interpolator %s, found not monotonically increasing values.' \
										% (object_name))
						elif (np.min(np.diff(object_raw[1])) == 0.0):
							mask = np.ones(len(object_raw[1]), bool)
							mask[1:] = (np.diff(object_raw[1]) > 0.0)
							n_removed = len(mask) - np.count_nonzero(mask)
							warnings.warn('While inverting interpolator %s, removed %d flat values from the data.' \
										% (object_name, n_removed))
							object_data = scipy.interpolate.InterpolatedUnivariateSpline(object_raw[1][mask],
																					object_raw[0][mask])
							
						else:
							object_data = scipy.interpolate.InterpolatedUnivariateSpline(object_raw[1],
																					object_raw[0])
					else:
						object_data = scipy.interpolate.InterpolatedUnivariateSpline(object_raw[0],
																					object_raw[1])
				else:
					raise Exception('Trying to generate interpolator from mismatched data.')
						
				if store_interpolator:
					self.storage_temp[object_id] = object_data

		return object_data

###################################################################################################

def getCacheDir(module = None):
	"""
	Get a directory for the persistent caching of data. The function attempts to locate the home 
	directory and (if necessary) create a '.colossus' sub-directory. In the rare case where that 
	fails, the location of this code file is used as a base directory.

	Parameters
	----------
	module: string
		The name of the module that is requesting this cache directory. Each module has its own
		directory in order to avoid name conflicts.
	
	Returns
	-------
	path: string
		The cache directory.
	"""
	
	if settings.BASE_DIR is None:
		base_dir = utilities.getHomeDir()
		if base_dir is None:
			base_dir = utilities.getCodeDir()
	else:
		base_dir = settings.BASE_DIR
		
	cache_dir = base_dir + '/.colossus/cache/'
	
	if module is not None:
		cache_dir += module + '/'

	if not os.path.exists(cache_dir):
		os.makedirs(cache_dir)
	
	return cache_dir

###################################################################################################
