###################################################################################################
#
# test_halo_concentration.py (c) Benedikt Diemer
#     				    	     diemer@umd.edu
#
###################################################################################################

import unittest
import numpy as np

from colossus.tests import test_colossus
from colossus.cosmology import cosmology
from colossus.halo import concentration

###################################################################################################
# TEST CASES
###################################################################################################

class TCConcentration(test_colossus.ColosssusTestCase):

	def setUp(self):

		pass

	###############################################################################################

	def test_model_returns(self):

		cosmology.setCosmology('bolshoi', {'persistence': ''})
		M_one = 1E12
		M_one_array = np.array([1E12])
		M_many = np.array([1E10, 1E12, 1E15])
		N_array = len(M_many)
		mdefs = ['200c', 'vir', '200m', '345m']
		z = 0.0
		models = concentration.models
		
		for k in models.keys():
			for j in range(len(mdefs)):

				c, mask = concentration.concentration(M_one, mdefs[j], z = z, model = k, range_return = True, range_warning = False)
				self.assertNotIsInstance(c, np.ndarray, 'Concentration should be scalar float, model %s, mdef %s.' % (k, mdefs[j]))
				self.assertNotIsInstance(mask, np.ndarray, 'Mask should be scalar bool, model %s, mdef %s.' % (k, mdefs[j]))
			
				c, mask = concentration.concentration(M_one_array, mdefs[j], z = z, model = k, range_return = True, range_warning = False)
				self.assertIsInstance(c, np.ndarray, 'Concentration should be an array with one element, model %s, mdef %s.' % (k, mdefs[j]))
				self.assertIsInstance(mask, np.ndarray, 'Mask should be an array with one element, model %s, mdef %s.' % (k, mdefs[j]))
				self.assertEqual(len(c), 1, 'Concentration should be an array with one element, model %s, mdef %s.' % (k, mdefs[j]))
				self.assertEqual(len(mask), 1, 'Mask should be an array with one element, model %s, mdef %s.' % (k, mdefs[j]))

				c, mask = concentration.concentration(M_many, mdefs[j], z = z, model = k, range_return = True, range_warning = False)
				self.assertIsInstance(c, np.ndarray, 'Concentration should be an array with multiple elements, model %s, mdef %s.' % (k, mdefs[j]))
				self.assertIsInstance(mask, np.ndarray, 'Mask should be an array with multiple elements, model %s, mdef %s.' % (k, mdefs[j]))
				self.assertEqual(len(c), N_array, 'Concentration should be an array with multiple elements, model %s, mdef %s.' % (k, mdefs[j]))
				self.assertEqual(len(mask), N_array, 'Mask should be an array with multiple elements, model %s, mdef %s.' % (k, mdefs[j]))

				c = concentration.concentration(M_one, mdefs[j], z = z, model = k, range_return = False, range_warning = False)
				self.assertNotIsInstance(c, np.ndarray, 'Concentration should be scalar float, model %s, mdef %s.' % (k, mdefs[j]))
			
				c = concentration.concentration(M_one_array, mdefs[j], z = z, model = k, range_return = False, range_warning = False)
				self.assertIsInstance(c, np.ndarray, 'Concentration should be an array with one element, model %s, mdef %s.' % (k, mdefs[j]))
				self.assertEqual(len(c), 1, 'Concentration should be an array with one element, model %s, mdef %s.' % (k, mdefs[j]))
				
				c = concentration.concentration(M_many, mdefs[j], z = z, model = k, range_return = False, range_warning = False)
				self.assertIsInstance(c, np.ndarray, 'Concentration should be an array with multiple elements, model %s, mdef %s.' % (k, mdefs[j]))
				self.assertEqual(len(c), N_array, 'Concentration should be an array with multiple elements, model %s, mdef %s.' % (k, mdefs[j]))

	###############################################################################################

	def test_model_values(self):
		cosmology.setCosmology('bolshoi', {'persistence': ''})
		M = 1E12
		z = 0.5
		mdef = '257m'
		models = concentration.models
		for k in models.keys():
			msg = 'Failure in model = %s' % (k)
			c = concentration.concentration(M, mdef, z = z, model = k, range_return = False, 
										range_warning = False)
			if k == 'bullock01':
				self.assertAlmostEqual(c, 6.838152607826e+00, msg = msg)
			elif k == 'duffy08':
				self.assertAlmostEqual(c, 5.844133619011e+00, msg = msg)
			elif k == 'klypin11':
				self.assertAlmostEqual(c, 9.328978854869e+00, msg = msg)
			elif k == 'prada12':
				self.assertAlmostEqual(c, 7.553654417697e+00, msg = msg)
			elif k == 'bhattacharya13':
				self.assertAlmostEqual(c, 5.862285902415e+00, msg = msg)
			elif k == 'dutton14':
				self.assertAlmostEqual(c, 7.590718273862e+00, msg = msg)
			elif k == 'diemer15':
				self.assertAlmostEqual(c, 6.818310486858e+00, msg = msg)
			elif k == 'diemer15_orig':
				self.assertAlmostEqual(c, 6.650158878433e+00, msg = msg)
			elif k == 'klypin16_nu':
				self.assertAlmostEqual(c, 6.458765763781e+00, msg = msg)
			elif k == 'klypin16_m':
				self.assertAlmostEqual(c, 6.210791674864e+00, msg = msg)
			elif k == 'ludlow16':
				self.assertAlmostEqual(c, 7.627295171886e+00, msg = msg)
			elif k == 'child18':
				self.assertAlmostEqual(c, 6.694980847317e+00, msg = msg)
			elif k == 'diemer19':
				self.assertAlmostEqual(c, 6.804756380278e+00, msg = msg)
			elif k == 'ishiyama21':
				self.assertAlmostEqual(c, 6.979742163903e+00, msg = msg)
			else:
				msg = 'Unknown model, %s.' % k
				raise Exception(msg)

	###############################################################################################
	
	def test_hard_fail(self):
		c = concentration.concentration(1E16, 'vir', z = 0.0, model = 'bullock01', range_return = False, range_warning = False)
		self.assertEqual(c, concentration.INVALID_CONCENTRATION)

	###############################################################################################

	# If interpolation = False, the slope is computed "manually" in the concentration routine. This 
	# function tests how different the result is from the derivative function in the Cosmology module.

	def test_PkSlopeComputation(self):
		
		M = 1E1
		z = 30.0
		cosmo = cosmology.setCosmology('bolshoi', {'persistence': ''})
		k_R = concentration._diemer15_k_R(M)
		cosmo.interpolation = True
		n1 = concentration._diemer15_n(k_R)
		c1 = concentration.modelDiemer15fromM(M, z, statistic = 'median')
		cosmo.interpolation = False
		n2 = concentration._diemer15_n(k_R)
		c2 = concentration.modelDiemer15fromM(M, z, statistic = 'median')
		err1 = abs(n2 / n1 - 1.0)
		err2 = abs(c2 / c1 - 1.0)
		self.assertLess(err1, 1E-3)
		self.assertLess(err2, 1E-3)
			
###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
