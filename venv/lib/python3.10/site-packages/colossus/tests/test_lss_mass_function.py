###################################################################################################
#
# test_lss_mass_function.py (c) Benedikt Diemer
#     				    	    diemer@umd.edu
#
###################################################################################################

import unittest
import numpy as np

from colossus.tests import test_colossus
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from colossus.lss import peaks
from colossus.tests.test_colossus import TEST_N_DIGITS

###################################################################################################
# TEST CASES
###################################################################################################

class TCMassFunction(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15', {'persistence': ''})
		pass
		
	def test_hmfInput(self):
		
		M = 1E12
		z = 1.0
		nu = peaks.peakHeight(M, z)
		delta_c = peaks.collapseOverdensity()
		sigma = delta_c / nu
		
		correct = 4.432195287965e-01
		
		mf = mass_function.massFunction(M, z, q_in = 'M', mdef = 'fof', model = 'press74')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity M.')			

		mf = mass_function.massFunction(sigma, z, q_in = 'sigma', mdef = 'fof', model = 'press74')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity sigma.')			

		mf = mass_function.massFunction(nu, z, q_in = 'nu', mdef = 'fof', model = 'press74')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity nu.')			

	def test_hmfConvert(self):
		
		M = 1E13
		z = 0.2
		
		correct = 4.496534812146e-01
		mf = mass_function.massFunction(M, z, q_in = 'M', mdef = 'fof', model = 'press74', q_out = 'f')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity f.')			

		correct = 6.783447915168e-04
		mf = mass_function.massFunction(M, z, q_in = 'M', mdef = 'fof', model = 'press74', q_out = 'dndlnM')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity dndlnM.')			

		correct = 7.912473712923e-02
		mf = mass_function.massFunction(M, z, q_in = 'M', mdef = 'fof', model = 'press74', q_out = 'M2dndM')
		self.assertAlmostEqual(mf, correct, msg = 'Quantity M2dndM.')			
				
	def test_hmfModelsFOF(self):
		models = mass_function.models
		for k in models.keys():
			msg = 'Failure in model = %s.' % (k)
			
			if not 'fof' in models[k].mdefs:
				continue
			
			if k == 'press74':
				correct = [2.236951589613e-01, 1.794240874352e-02]
			elif k == 'sheth99':
				correct = [2.037079488516e-01, 3.220604186726e-02]
			elif k == 'jenkins01':
				correct = [6.028246283415e-02, 3.442133684742e-02]
			elif k == 'reed03':
				correct = [2.037079488516e-01, 2.878504861884e-02]
			elif k == 'warren06':
				correct = [2.176098033649e-01, 3.384132788182e-02]
			elif k == 'reed07':
				correct = [1.913045961113e-01, 3.727770524449e-02]
			elif k == 'crocce10':
				correct = [2.196797134176e-01, 4.199167929524e-02]
			elif k == 'bhattacharya11':
				correct = [2.241161031162e-01, 4.069787585277e-02]
			elif k == 'courtin11':
				correct = [1.519241923997e-01, 4.493372336068e-02]
			elif k == 'angulo12':
				correct = [2.283431994155e-01, 3.774020705480e-02]
			elif k == 'watson13':
				correct = [2.847704926977e-01, 3.807992631589e-02]
			else:
				msg = 'Unknown model, %s.' % k
				raise Exception(msg)
			
			self.assertAlmostEqualArray(mass_function.massFunction(np.array([1E8, 1E15]), 0.0, 
								q_in = 'M', mdef = 'fof', model = k), correct, msg = msg)

	def test_hmfModelsSO_200m(self):
		models = mass_function.models
		for k in models.keys():
			msg = 'Failure in model = %s.' % (k)
			
			mdef = '200m'
			z = 1.0
			
			if not (('*' in models[k].mdefs) or (mdef in models[k].mdefs)):
				continue
			
			if k == 'tinker08':
				correct = [2.510155734496e-01, 4.627851852759e-05]
			elif k == 'watson13':
				correct = [1.621439113725e-01, 4.444441298941e-05]
			elif k == 'bocquet16':
				correct = [2.836183350625e-01, 3.846532915653e-05]
			elif k == 'despali16':
				correct = [2.566946159920e-01, 6.664190195668e-05]
			else:
				msg = 'Unknown model, %s.' % k
				raise Exception(msg)
			
			self.assertAlmostEqualArray(mass_function.massFunction(np.array([1E8, 1E15]), z, 
								q_in = 'M', mdef = mdef, model = k), correct, msg = msg)

	def test_hmfModelsSO_vir(self):
		models = mass_function.models
		for k in models.keys():
			msg = 'Failure in model = %s.' % (k)
			
			mdef = 'vir'
			z = 1.0
			
			if not (('*' in models[k].mdefs) or (mdef in models[k].mdefs)):
				continue
			
			digits = TEST_N_DIGITS
			if k == 'tinker08':
				correct = [2.509298990290e-01, 4.556608550623e-05]
			elif k == 'watson13':
				correct = [1.613552783063e-01, 4.383224392763e-05]
			elif k == 'despali16':
				correct = [2.566169917705e-01, 6.560381212926e-05]
			elif k == 'comparat17':
				correct = [2.449612008069e-01, 2.351693095425e-05]
			elif k == 'seppi20':
				correct = [2.359726326775e-01, 5.108267257217e-05]
				digits = 4
			else:
				msg = 'Unknown model, %s.' % k
				raise Exception(msg)
			
			self.assertAlmostEqualArray(mass_function.massFunction(np.array([1E8, 1E15]), z, 
								q_in = 'M', mdef = mdef, model = k), correct, msg = msg, places = digits)

	def test_hmfModelsSplashbackMean(self):
		models = mass_function.models
		for k in models.keys():
			msg = 'Failure in model = %s.' % (k)
			
			mdef = 'sp-apr-mn'
			z = 1.0
			
			if (not 'sp-apr-mn' in models[k].mdefs):
				continue
			
			if k == 'diemer20':
				correct = [2.839481935931e-01, 3.342581497124e-05]
			else:
				msg = 'Unknown model, %s.' % k
				raise Exception(msg)
			
			self.assertAlmostEqualArray(mass_function.massFunction(np.array([1E8, 1E15]), z, 
								q_in = 'M', mdef = mdef, model = k), correct, msg = msg)

	def test_hmfModelsSplashbackPercentiles(self):
		models = mass_function.models
		for k in models.keys():
			msg = 'Failure in model = %s.' % (k)
			
			mdef = 'sp-apr-p75'
			z = 1.0
			
			if (not 'sp-apr-p*' in models[k].mdefs):
				continue
			
			if k == 'diemer20':
				correct = [2.729257516844e-01, 7.599996379257e-05]
			else:
				msg = 'Unknown model, %s.' % k
				raise Exception(msg)
			
			self.assertAlmostEqualArray(mass_function.massFunction(np.array([1E8, 1E15]), z, 
								q_in = 'M', mdef = mdef, model = k), correct, msg = msg)

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
