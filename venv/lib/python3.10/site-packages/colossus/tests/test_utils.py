###################################################################################################
#
# test_utils.py         (c) Benedikt Diemer
#     				    	diemer@umd.edu
#
###################################################################################################

import unittest
import numpy as np

from colossus import version
from colossus.tests import test_colossus
from colossus.utils import storage
from colossus.utils import utilities
from colossus.utils import constants

###################################################################################################
# TEST CASE: VERSIONS
###################################################################################################

class TCVersions(test_colossus.ColosssusTestCase):

	def setUp(self):
		pass
	
	def test_versions(self):
		self.assertEqual(utilities.versionIsOlder('1.0.1', '1.0.0') , True)
		self.assertEqual(utilities.versionIsOlder('1.1.0', '1.0.0') , True)
		self.assertEqual(utilities.versionIsOlder('2.0.0', '1.0.0') , True)
		self.assertEqual(utilities.versionIsOlder('1.0.0', '1.0.0') , False)
		self.assertEqual(utilities.versionIsOlder('1.0.0', '2.0.0') , False)
		self.assertEqual(utilities.versionIsOlder('1.0.0', '1.1.0') , False)
		self.assertEqual(utilities.versionIsOlder('1.0.0', '1.0.1') , False)
		
	def test_get_version(self):
		v = version.__version__
		is_older = utilities.versionIsOlder('1.0.0', v)
		self.assertEqual(is_older, False)

###################################################################################################
# TEST CASE: CONSTANTS
###################################################################################################

class TCConstants(test_colossus.ColosssusTestCase):

	def setUp(self):
		pass
	
	def test_gravitational_constant(self):
		G_const = constants.G / 1000.0
		G_deriv = constants.G_CGS / constants.MPC * constants.MSUN / 1E10
		self.assertAlmostEqual(G_const, G_deriv, places = 9)

		G_const = constants.G
		G_deriv = constants.G_CGS / constants.KPC * constants.MSUN / 1E10
		self.assertAlmostEqual(G_const, G_deriv, places = 9)

	def test_critical_density(self):
		G_Mpc = constants.G / 1000.0
		rhoc_deriv = 3 * 100.0**2 / (8 * np.pi * G_Mpc)
		rhoc_const = constants.RHO_CRIT_0_MPC3
		self.assertAlmostEqual(rhoc_const, rhoc_deriv, places = 9)

		H0_kpc = 100.0 / 1000.0
		rhoc_deriv = 3 * H0_kpc**2 / (8 * np.pi * constants.G)
		rhoc_const = constants.RHO_CRIT_0_KPC3
		self.assertAlmostEqual(rhoc_const, rhoc_deriv, places = 9)

	def test_deltac(self):
		deltac_deriv = 3.0 / 5.0 * (3.0 * np.pi / 2.0)**(2.0 / 3.0)
		deltac_const = constants.DELTA_COLLAPSE
		self.assertAlmostEqual(deltac_const, deltac_deriv, places = 5)

###################################################################################################
# TEST CASE: STORAGE
###################################################################################################

class TestStorageClass():
	
	def __init__(self, some_parameter = 1.5, persistent = False):
		self.some_parameter = some_parameter
		self.some_data = [2.6, 9.5]
		self.su = storage.StorageUser('unit_test', 'rw', self.getName, self.getHashableString, self.reportChanges)
		self.su.storeObject('test_data', self.some_data, persistent = persistent)
		return

	def getName(self):
		return 'unit_test'

	def getHashableString(self):
		param_string = 'unit_test_%.4f' % (self.some_parameter)
		return param_string
		
	def reportChanges(self):
		return
	
	def loadData(self):
		data = self.su.getStoredObject('test_data')
		return data

class TCStorageNonPersistent(test_colossus.ColosssusTestCase):
	
	def setUp(self):
		self.dc = TestStorageClass(persistent = False)
	
	def test_storage_dir(self):
		self.assertNotEqual(storage.getCacheDir(), None)

	def test_storage(self):
		d = self.dc.loadData()
		self.assertAlmostEqual(d[0], 2.6)
		self.assertAlmostEqual(d[1], 9.5)
		self.dc.some_parameter = 1.2
		d = self.dc.loadData()
		self.assertEqual(d, None)
		
class TCStoragePersistent(test_colossus.ColosssusTestCase):
	
	def setUp(self):
		self.dc = TestStorageClass(persistent = True)
	
	def test_storage_dir(self):
		self.assertNotEqual(storage.getCacheDir(), None)

	def test_storage(self):
		d = self.dc.loadData()
		self.assertAlmostEqual(d[0], 2.6)
		self.assertAlmostEqual(d[1], 9.5)
		self.dc.some_parameter = 1.2
		d = self.dc.loadData()
		self.assertEqual(d, None)
		
###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
