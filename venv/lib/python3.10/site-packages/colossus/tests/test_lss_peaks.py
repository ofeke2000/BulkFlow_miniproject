###################################################################################################
#
# test_lss_peaks.py     (c) Benedikt Diemer
#     				    	diemer@umd.edu
#
###################################################################################################

import unittest

from colossus.tests import test_colossus
from colossus.cosmology import cosmology
from colossus.lss import peaks

###################################################################################################
# TEST PARAMETERS
###################################################################################################

TEST_Z2 = 5.4
TEST_M = 3E12
TEST_R = 1.245
TEST_NU = 0.89
TEST_SLOPE_SCALE = 2.2

###################################################################################################
# TEST CASE 1: STANDARD PEAK FUNCTIONS
###################################################################################################

class TCPeaks(test_colossus.ColosssusTestCase):

	def setUp(self):
		self.cosmo_name = 'planck15'
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 
															'persistence': ''})
		pass
	
	def test_lagrangianR(self):
		self.assertAlmostEqual(peaks.lagrangianR(TEST_M), 2.029075203793e+00)

	def test_lagrangianM(self):
		self.assertAlmostEqual(peaks.lagrangianM(TEST_R), 6.930026225629e+11)

	def test_peakHeight(self):
		self.assertAlmostEqual(peaks.peakHeight(TEST_M, 0.0), 9.433577339095e-01)
		self.assertAlmostEqual(peaks.peakHeight(TEST_M, TEST_Z2), 4.741662779773e+00)

	def test_peakCurvature(self):
		correct = [[1.402929413737e+00, 6.308470244050e-01, 2.286791514807e+00, 
					1.401757668700e+00, -6.592042734768e-02], 
				[7.051638996162e+00, 6.308470244050e-01, 4.862419061151e+00, 
					4.139135832436e-01, 6.617914312826e+00]]
		for j in range(2):
			z = [0.0, TEST_Z2][j]
			res = peaks.peakCurvature(TEST_M, z, exact = False)
			for i in range(5):
				self.assertAlmostEqual(res[i], correct[j][i])

###################################################################################################
# TEST CASE 2: PEAK FUNCTIONS THAT NEED COSMOLOGICAL INTERPOLATION
###################################################################################################

class TCPeaksInterp(test_colossus.ColosssusTestCase):

	def setUp(self):
		self.cosmo_name = 'planck15'
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': True, 
															'persistence': ''})
		pass

	def test_massFromPeakHeight(self):
		self.assertAlmostEqual(peaks.massFromPeakHeight(TEST_NU, 0.0), 2.073792914980e+12)
		self.assertAlmostEqual(peaks.massFromPeakHeight(TEST_NU, TEST_Z2), 5.958212682072e+04)

	def test_nonLinearMass(self):
		self.assertAlmostEqual(peaks.nonLinearMass(1.1), 9.825929418905e+10)

	def test_powerSpectrumSlope(self):
		self.assertAlmostEqual(peaks.powerSpectrumSlope(TEST_NU, 0.0, slope_type = 'P', 
								scale = 1.0), -2.492969198392e+00)
		self.assertAlmostEqual(peaks.powerSpectrumSlope(TEST_NU, 0.0, slope_type = 'sigma', 
								scale = 1.0), -2.069620231275e+00)
		self.assertAlmostEqual(peaks.powerSpectrumSlope(TEST_NU, 0.0, slope_type = 'P', 
								scale = TEST_SLOPE_SCALE), -2.581359321370e+00)
		self.assertAlmostEqual(peaks.powerSpectrumSlope(TEST_NU, 0.0, slope_type = 'sigma', 
								scale = TEST_SLOPE_SCALE), -1.872913021264e+00)

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
