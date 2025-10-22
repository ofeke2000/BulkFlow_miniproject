###################################################################################################
#
# test_halo_mass.py     (c) Benedikt Diemer
#     				    	diemer@umd.edu
#
###################################################################################################

import unittest

from colossus.tests import test_colossus
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.halo import mass_defs
from colossus.halo import mass_adv
from colossus.halo import profile_dk14
from colossus.halo import profile_outer

###################################################################################################
# TEST CASE: SPHERICAL OVERDENSITY
###################################################################################################

class TCMassSO(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15', {'persistence': ''})
		pass

	def test_parseMassDefinition(self):
		t, d = mass_so.parseMassDefinition('200m')
		self.assertEqual(t, 'm')
		self.assertEqual(d, 200)
		t, d = mass_so.parseMassDefinition('500c')
		self.assertEqual(t, 'c')
		self.assertEqual(d, 500)
		t, d = mass_so.parseMassDefinition('vir')
		self.assertEqual(t, 'vir')
		self.assertEqual(d, None)
		with self.assertRaises(Exception):
			mass_so.parseMassDefinition('100r')
			mass_so.parseMassDefinition('e')
			mass_so.parseMassDefinition('79.6c')

	def test_parseRadiusMassDefinition(self):
		rm, _, t, d = mass_so.parseRadiusMassDefinition('R200m')
		self.assertEqual(rm, 'R')
		self.assertEqual(t, 'm')
		self.assertEqual(d, 200)
		rm, _, t, d = mass_so.parseRadiusMassDefinition('r200m')
		self.assertEqual(rm, 'R')
		self.assertEqual(t, 'm')
		self.assertEqual(d, 200)
		rm, _, t, d = mass_so.parseRadiusMassDefinition('M500c')
		self.assertEqual(rm, 'M')
		self.assertEqual(t, 'c')
		self.assertEqual(d, 500)
		rm, _, t, d = mass_so.parseRadiusMassDefinition('Mvir')
		self.assertEqual(rm, 'M')
		self.assertEqual(t, 'vir')
		self.assertEqual(d, None)
		with self.assertRaises(Exception):
			mass_so.parseRadiusMassDefinition('e500c')
			mass_so.parseRadiusMassDefinition('e')
			mass_so.parseRadiusMassDefinition('79.6c')

	def test_densityThreshold(self):
		self.assertAlmostEqual(mass_so.densityThreshold(0.7, '200m'), 8.423934363988e+04)
		self.assertAlmostEqual(mass_so.densityThreshold(6.1, '400c'), 1.237606836324e+07)
		self.assertAlmostEqual(mass_so.densityThreshold(1.2, 'vir'), 1.792681534674e+05)
		with self.assertRaises(Exception):
			mass_so.densityThreshold('100t')

	def test_deltaVir(self):
		self.assertAlmostEqual(mass_so.deltaVir(0.7), 1.481550413377e+02)
	
	def test_M_to_R(self):
		self.assertAlmostEqual(mass_so.M_to_R(1.1E12, 0.7, '200m'), 1.460818859860e+02)
		self.assertAlmostEqual(mass_so.M_to_R(1.1E12, 0.7, 'vir'), 1.424507010731e+02)

	def test_R_to_M(self):
		self.assertAlmostEqual(mass_so.R_to_M(212.0, 0.7, '200m'), 3.362104178665e+12)
		self.assertAlmostEqual(mass_so.R_to_M(150.0, 0.7, 'vir'), 1.284318355150e+12)

###################################################################################################
# TEST CASE: DEFINITIONS
###################################################################################################

class TCMassDefs(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15', {'persistence': ''})
	
	def test_pseudoEvolve(self):
		z1 = 0.68
		z2 = 3.1
		M1 = [1.5E8, 1.1E15]
		c1 = 4.6
		correct_M = [4.458465807339e+07, 3.269541592049e+14]
		correct_R = [2.151683787142e+00, 4.180346261028e+02]
		correct_c = [1.300870547291e+00, 1.300870547291e+00]
		for i in range(len(M1)):
			M, R, c = mass_defs.evolveSO(M1[i], c1, z1, '200m', z2, 'vir')
			self.assertAlmostEqual(M, correct_M[i])
			self.assertAlmostEqual(R, correct_R[i])
			self.assertAlmostEqual(c, correct_c[i])

	def test_pseudoEvolveWithDk14(self):
		z1 = 0.68
		z2 = 3.1
		M1 = [1.5E8, 1.1E15]
		c1 = 4.6
		correct_M = [4.107574772320e+07, 4.332736277298e+14]
		correct_R = [2.093687177765e+00, 4.591670540319e+02]
		correct_c = [1.265806807242e+00, 1.428869427504e+00]
		for i in range(len(M1)):
			t = profile_outer.OuterTermPowerLaw(norm = 1.0, slope = 1.5, pivot = 'R200m', 
											pivot_factor = 5.0, z = z1)
			M, R, c = mass_defs.evolveSO(M1[i], c1, z1, '200m', z2, 'vir',
						profile = profile_dk14.DK14Profile, profile_args = {'outer_terms': [t]})
			self.assertAlmostEqual(M, correct_M[i])
			self.assertAlmostEqual(R, correct_R[i])
			self.assertAlmostEqual(c, correct_c[i])

	def test_changeMassDefinition(self):
		z1 = 0.98
		M1 = [1.5E8, 1.1E15]
		c1 = 4.6
		correct_M = [1.189464763847e+08, 8.722741601541e+14]
		correct_R = [4.796719616508e+00, 9.319189480303e+02]
		correct_c = [3.433472297235e+00, 3.433472297235e+00]
		for i in range(len(M1)):
			M, R, c = mass_defs.changeMassDefinition(M1[i], c1, z1, 'vir', '300c')
			self.assertAlmostEqual(M, correct_M[i])
			self.assertAlmostEqual(R, correct_R[i])
			self.assertAlmostEqual(c, correct_c[i])
	
###################################################################################################
# TEST CASE: ADVANCED
###################################################################################################

class TCMassAdv(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck15', {'persistence': ''})
		pass

	def test_changeMassDefinitionCModel(self):
		z1 = 0.98
		M1 = [1.5E8, 1.1E15]
		correct_M = [1.299211300160e+08, 8.814260889278e+14]
		correct_R = [4.939925538406e+00, 9.351668565967e+02]
		correct_c = [9.166087251441e+00, 3.752755631233e+00]
		for i in range(len(M1)):
			M, R, c = mass_adv.changeMassDefinitionCModel(M1[i], z1, 'vir', '300c')
			self.assertAlmostEqual(M, correct_M[i])
			self.assertAlmostEqual(R, correct_R[i])
			self.assertAlmostEqual(c, correct_c[i])

		return
	
	def test_M4rs(self):
		self.assertAlmostEqual(mass_adv.M4rs(1E12, 0.7, '500c', 3.8), 1041815679897.7153)
	
###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
