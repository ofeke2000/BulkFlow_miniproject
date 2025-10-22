###################################################################################################
#
# test_lss_bias.py      (c) Benedikt Diemer
#     				    	diemer@umd.edu
#
###################################################################################################

import unittest
import numpy as np

from colossus.tests import test_colossus
from colossus.cosmology import cosmology
from colossus.lss import bias

###################################################################################################
# TEST CASES
###################################################################################################

class TCBias(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15', {'persistence': ''})
		pass
	
	def test_haloBiasFromNu(self):
		self.assertAlmostEqual(bias.haloBiasFromNu(3.0, 1.0, '200c'), 5.290627697104e+00)
		self.assertAlmostEqual(bias.haloBiasFromNu(3.0, 1.0, '200c', model = 'cole89'), 5.743636115674e+00)
		self.assertAlmostEqual(bias.haloBiasFromNu(3.0, 1.0, '200c', model = 'sheth01'), 4.720384850956e+00)
	
	def test_haloBias(self):
		self.assertAlmostEqual(bias.haloBias(2.3E12, 1.0, '200c'), 1.552809676995e+00)
		
	def test_twoHaloTerm(self):
		r = np.array([0.012, 0.18, 2.05])
		correct = np.array([2.622957711484e+04, 7.224685475659e+03, 1.089771755305e+03])
		self.assertAlmostEqualArray(bias.twoHaloTerm(r, 2.3E12, 1.0, '200c'), correct)
	
	def test_biasModels(self):
		models = bias.models
		for k in models.keys():
			msg = 'Failure in model = %s' % (k)
			if k == 'cole89':
				correct = 5.743636115674e+00
			elif k == 'jing98':
				correct = 5.747378724450e+00
			elif k == 'sheth01':
				correct = 4.720384850956e+00
			elif k == 'seljak04':
				correct = 1.874832517421e+01
			elif k == 'pillepich10':
				correct = 4.793683473192e+00
			elif k == 'tinker10':
				correct = 5.290627697104e+00
			elif k == 'bhattacharya11':
				correct = 4.275927523958e+00
			elif k == 'comparat17':
				correct = 5.002797628469e+00
			else:
				msg = 'Unknown model, %s.' % k
				raise Exception(msg)
			self.assertAlmostEqual(bias.haloBiasFromNu(3.0, z = 1.0, mdef = '200c', model = k), 
								correct, msg = msg)
		
###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
