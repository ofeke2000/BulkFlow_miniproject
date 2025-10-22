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
from colossus.cosmology import power_spectrum

###################################################################################################
# TEST PARAMETERS
###################################################################################################

TEST_K = np.array([1.2E-3, 1.1E3])

###################################################################################################
# TEST CASES
###################################################################################################

class TCTransferFunction(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15', {'persistence': ''})
		pass
	
	def test_transferFunction(self):
		models = power_spectrum.models
		cosmo = cosmology.getCurrent()
		for m in models.keys():
			if m in ['camb']:
				continue
			msg = 'Failure in model = %s' % (m)
			if m == 'sugiyama95':
				correct = [9.811438043156e-01, 1.349997806341e-08]
			elif m == 'eisenstein98':
				correct = [9.892256929454e-01, 1.490479340442e-08]
			elif m == 'eisenstein98_zb':
				correct = [9.890484789741e-01, 1.471045424612e-08]
			else:
				msg = 'Unknown model, %s.' % m
				raise Exception(msg)
			T = power_spectrum.powerSpectrum(TEST_K, m, cosmo, output = 'tf')
			self.assertAlmostEqualArray(T, correct, msg = msg)

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
