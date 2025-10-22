###################################################################################################
#
# test_halo_splashback.py  (c) Benedikt Diemer
#     				    	   diemer@umd.edu
#
###################################################################################################

import unittest
import numpy as np

from colossus.tests import test_colossus
from colossus.cosmology import cosmology
from colossus.halo import splashback

###################################################################################################
# TEST CASE: SPLASHBACK MODELS
###################################################################################################

class TCSplashbackModel(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15', {'persistence': ''})
		pass

	def test_modelGamma(self):
		rsp, mask = splashback.splashbackModel('RspR200m', Gamma = 1.2, z = 0.1, model = 'more15')
		self.assertEqual(mask, True)
		self.assertAlmostEqual(rsp, 1.239462644843)
	
	def test_modelNu(self):
		rsp, mask = splashback.splashbackModel('RspR200m', nu200m = 0.6, z = 0.1, model = 'more15')
		self.assertEqual(mask, True)
		self.assertAlmostEqual(rsp, 1.424416723584)

	def test_modelGammaArray(self):
		Gamma = np.array([0.5, 4.1])
		z = 0.1
		mdef = '200m'
		R200m_test = 900.0
		R200m = np.ones_like(Gamma)* R200m_test
		models = splashback.models
		for k in models.keys():
			msg = 'Failure in model = %s' % (k)
			Rsp, _, mask = splashback.splashbackRadius(z, mdef, Gamma = Gamma, R = R200m, 
													model = k, rspdef = 'sp-apr-mn')
			RspR200m = Rsp / R200m_test
			
			if k == 'adhikari14':
				correct_rsp = [1.269417774113e+00, 8.167315805978e-01]
			elif k == 'more15':
				correct_rsp = [1.392934316796e+00, 8.750736226357e-01]
			elif k == 'shi16':
				correct_rsp = [1.334955458442e+00, 6.672045990854e-01]
			elif k == 'mansfield17':
				correct_rsp = [1.386073423900e+00, 1.138961025552e+00]
			elif k == 'diemer17':
				correct_rsp = [1.232502896365e+00, 7.998312439193e-01]
			elif k == 'diemer20':
				correct_rsp = [1.214166952231e+00, 7.980230003679e-01]
			else:
				raise Exception('No test case defined for model %s.' % k)
			
			for i in range(len(Gamma)):
				self.assertEqual(mask[i], True, msg = msg)
				self.assertAlmostEqual(RspR200m[i], correct_rsp[i], msg = msg)
	
###################################################################################################
# TEST CASE: SPLASHBACK RADIUS
###################################################################################################

class TCSplashbackRadius(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15', {'persistence': ''})
	
	def test_rspR200m(self):
		R = np.array([900.0, 1100.0])
		z = 0.1
		mdef = '200m'
		Rsp, Msp, mask = splashback.splashbackRadius(z, mdef, R = R, model = 'more15')
		correct_rsp = [1.072325264941e+03, 1.270966525596e+03]
		correct_msp = [7.896203193991e+13, 1.415192318977e+14]
		for i in range(len(R)):
			self.assertEqual(mask[i], True)
			self.assertAlmostEqual(Rsp[i], correct_rsp[i])
			self.assertAlmostEqual(Msp[i], correct_msp[i])

	def test_rspRvir(self):
		R = np.array([900.0, 1100.0])
		z = 0.1
		mdef = 'vir'
		Rsp, Msp, mask = splashback.splashbackRadius(z, mdef, R = R, 
									model = 'more15', c_model = 'diemer15')
		correct_rsp = [1.238595733861e+03, 1.464913446611e+03]
		correct_msp = [1.294406412531e+14, 2.322614981111e+14]
		for i in range(len(R)):
			self.assertEqual(mask[i], True)
			self.assertAlmostEqual(Rsp[i], correct_rsp[i])
			self.assertAlmostEqual(Msp[i], correct_msp[i])

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
