###################################################################################################
#
# test_cosmology.py     (c) Benedikt Diemer
#     				    	diemer@umd.edu
#
###################################################################################################

import unittest
import numpy as np

from colossus.tests import test_colossus
from colossus import defaults
from colossus.cosmology import cosmology

###################################################################################################
# TEST PARAMETERS
###################################################################################################

TEST_Z = np.array([0.0, 1.283, 20.0])
TEST_Z2 = 5.4
TEST_K = np.array([1.2E-3, 1.1E3])
TEST_RR = np.array([1.2E-3, 1.4, 1.1E3])
TEST_RR_interp = np.array([1.2E-3, 1.4, 0.9E3])
TEST_AGE = np.array([13.7, 0.1])

###################################################################################################
# GENERAL CLASS FOR COSMOLOGY TEST CASES
###################################################################################################

class CosmologyTestCase(test_colossus.ColosssusTestCase):

	def _testRedshiftArray(self, f, correct):
		self.assertAlmostEqualArray(f(TEST_Z), correct)		

	def _testKArray(self, f, correct):
		self.assertAlmostEqualArray(f(TEST_K), correct)

	def _testRZArray(self, f, z, correct):
		self.assertAlmostEqualArray(f(TEST_RR, z), correct)

###################################################################################################
# TEST CASE 1: COMPUTATIONS WITHOUT INTERPOLATION
###################################################################################################

class TCComp(CosmologyTestCase):

	def setUp(self):
		self.cosmo_name = 'planck15'
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 
															'persistence': ''})

	###############################################################################################
	# BASICS
	###############################################################################################
	
	def test_init(self):
		c_dict = cosmology.cosmologies[self.cosmo_name]
		self.assertEqual(self.cosmo.name, self.cosmo_name)
		self.assertAlmostEqual(self.cosmo.Om0, c_dict['Om0'])
		self.assertAlmostEqual(self.cosmo.Ob0, c_dict['Ob0'])
		self.assertAlmostEqual(self.cosmo.sigma8, c_dict['sigma8'])
		self.assertAlmostEqual(self.cosmo.ns, c_dict['ns'])
		if 'Tcmb0' in c_dict:
			self.assertAlmostEqual(self.cosmo.Tcmb0, c_dict['Tcmb0'])
		else:
			self.assertAlmostEqual(self.cosmo.Tcmb0, defaults.COSMOLOGY_TCMB0)
		if 'Neff' in c_dict:
			self.assertAlmostEqual(self.cosmo.Neff, c_dict['Neff'])
		else:
			self.assertAlmostEqual(self.cosmo.Neff, defaults.COSMOLOGY_NEFF)
		self.assertAlmostEqual(self.cosmo.Ogamma0, 5.389078568217e-05)
		self.assertAlmostEqual(self.cosmo.Onu0, 3.727996897061e-05)
		self.assertAlmostEqual(self.cosmo.Or0, 9.117075465278e-05)
	
	def test_initNoRel(self):
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 
												'persistence': '', 'relspecies': False})
		c_dict = cosmology.cosmologies[self.cosmo_name]
		self.assertAlmostEqual(self.cosmo.Om0, c_dict['Om0'])
		self.assertAlmostEqual(self.cosmo.Ode0, 1.0 - c_dict['Om0'])
		self.assertAlmostEqual(self.cosmo.Ob0, c_dict['Ob0'])
		self.assertAlmostEqual(self.cosmo.sigma8, c_dict['sigma8'])
		self.assertAlmostEqual(self.cosmo.ns, c_dict['ns'])
		self.assertAlmostEqual(self.cosmo.Tcmb0, defaults.COSMOLOGY_TCMB0)
		self.assertAlmostEqual(self.cosmo.Neff, defaults.COSMOLOGY_NEFF)
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 
															'persistence': ''})

	def test_changing(self):
		cosmo1 = cosmology.setCosmology('planck15', {'interpolation': False, 
												'persistence': '', 'print_warnings': False})
		cosmo2 = cosmology.setCosmology('planck18', {'interpolation': False, 
												'persistence': ''})
		c_dict = cosmology.cosmologies['planck18']
		cosmo1.flat = c_dict['flat']
		cosmo1.H0 = c_dict['H0']
		cosmo1.Om0 = c_dict['Om0']
		cosmo1.Ob0 = c_dict['Ob0']
		cosmo1.sigma8 = c_dict['sigma8']
		cosmo1.ns = c_dict['ns']
		
		cosmo1.checkForChangedCosmology()

		self.assertAlmostEqual(cosmo1.h, cosmo2.h)
		self.assertAlmostEqual(cosmo1.h2, cosmo2.h2)
		self.assertAlmostEqual(cosmo1.Omh2, cosmo2.Omh2)
		self.assertAlmostEqual(cosmo1.Ombh2, cosmo2.Ombh2)
		self.assertAlmostEqual(cosmo1.Ogamma0, cosmo2.Ogamma0)
		self.assertAlmostEqual(cosmo1.Onu0, cosmo2.Onu0)
		self.assertAlmostEqual(cosmo1.Or0, cosmo2.Or0)
		self.assertAlmostEqual(cosmo1.a_eq, cosmo2.a_eq)
		self.assertAlmostEqual(cosmo1.Ode0, cosmo2.Ode0)
		self.assertAlmostEqual(cosmo1.Ok0, cosmo2.Ok0)
	
	###############################################################################################
	# Basic cosmology calculations
	###############################################################################################
	
	def test_Ez(self):
		correct = [1.0, 2.090250748388e+00, 5.365766383626e+01]
		self._testRedshiftArray(self.cosmo.Ez, correct)
		
	def test_Hz(self):
		correct = [67.74, 1.415935856958e+02, 3.634770148268e+03]
		self._testRedshiftArray(self.cosmo.Hz, correct)

	###############################################################################################
	# Times & distances
	###############################################################################################

	def test_hubbleTime(self):
		correct = [1.443479672834e+01, 6.905772783226e+00, 2.690164963645e-01]
		self._testRedshiftArray(self.cosmo.hubbleTime, correct)
	
	def test_lookbackTime(self):
		correct = [0.0, 8.928012349881e+00, 1.361899507180e+01]
		self._testRedshiftArray(self.cosmo.lookbackTime, correct)

	def test_age(self):
		correct = [1.379740385589e+01, 4.869391506008e+00, 1.784087840836e-01]
		self._testRedshiftArray(self.cosmo.age, correct)
	
	def test_comovingDistance(self):
		correct = [0.0, 2.740512775683e+03, 7.432211484621e+03]
		self.assertAlmostEqualArray(self.cosmo.comovingDistance(z_max = TEST_Z), correct)		

	def test_luminosityDistance(self):
		correct = [0.0, 6.256590666884e+03, 1.560764411770e+05]
		self._testRedshiftArray(self.cosmo.luminosityDistance, correct)

	def test_angularDiameterDistance(self):
		correct = [0.0, 1.200399814141e+03, 3.539148326010e+02]
		self._testRedshiftArray(self.cosmo.angularDiameterDistance, correct)

	def test_distanceModulus(self):
		correct = [44.827462759550897, 51.81246085652802]
		self.assertAlmostEqualArray(self.cosmo.distanceModulus(TEST_Z[1:]), correct)		
	
	def test_soundHorizon(self):
		self.assertAlmostEqual(self.cosmo.soundHorizon(), 1.017552548264e+02)

	###############################################################################################
	# Densities and overdensities
	###############################################################################################

	def test_rho_c(self):
		correct = [2.775366272457e+02, 1.212598652905e+03, 7.990681616685e+05]
		self._testRedshiftArray(self.cosmo.rho_c, correct)
	
	def test_rho_m(self):
		correct = [8.573106415620e+01, 1.020131008908e+03, 7.939553851506e+05]
		self._testRedshiftArray(self.cosmo.rho_m, correct)
	
	def test_rho_de(self):
		self.assertAlmostEqual(self.cosmo.rho_de(0.0), 1.917802598658e+02)
	
	def test_rho_gamma(self):
		correct = [1.495666689785e-02, 4.063108946635e-01, 2.908787534961e+03]
		self._testRedshiftArray(self.cosmo.rho_gamma, correct)
	
	def test_rho_nu(self):
		correct = [1.034655685193e-02, 2.810732364307e-01, 2.012208723120e+03]
		self._testRedshiftArray(self.cosmo.rho_nu, correct)
	
	def test_rho_r(self):
		correct = [2.530322374978e-02, 6.873841310942e-01, 4.920996258081e+03]
		self._testRedshiftArray(self.cosmo.rho_r, correct)
	
	def test_Om(self):
		correct = [0.3089, 8.412767130033e-01, 9.936015764822e-01]
		self._testRedshiftArray(self.cosmo.Om, correct)
	
	def test_Ob(self):
		correct = [4.860000000000e-02, 1.323601432566e-01, 1.563257902785e-01]
		self._testRedshiftArray(self.cosmo.Ob, correct)
	
	def test_Ode(self):
		correct = [6.910088292453e-01, 1.581564183717e-01, 2.400048820182e-04]
		self._testRedshiftArray(self.cosmo.Ode, correct)
	
	def test_Ok(self):
		correct = [0.0, 0.0, 0.0]
		self._testRedshiftArray(self.cosmo.Ok, correct)
	
	def test_Ogamma(self):
		correct = [5.389078568217e-05, 3.350745060537e-04, 3.640224544658e-03]
		self._testRedshiftArray(self.cosmo.Ogamma, correct)
	
	def test_Onu(self):
		correct = [3.727996897061e-05, 2.317941189835e-04, 2.518194091125e-03]
		self._testRedshiftArray(self.cosmo.Onu, correct)
	
	def test_Or(self):
		correct = [9.117075465278e-05, 5.668686250372e-04, 6.158418635783e-03]
		self._testRedshiftArray(self.cosmo.Or, correct)

	###############################################################################################
	# Structure growth, power spectrum etc.
	###############################################################################################

	def test_growthFactor(self):
		correct = [1.0, 0.54093225419799251, 6.096861032873e-02]
		self._testRedshiftArray(self.cosmo.growthFactor, correct)

	def test_matterPowerSpectrum(self):
		correct = [4.503707661983e+03, 5.933365972297e-07]
		self._testKArray(self.cosmo.matterPowerSpectrum, correct)

	def test_matterPowerSpectrumZ(self):
		correct = [1.824063681536e+03, 2.403095003409e-07]
		Pk = self.cosmo.matterPowerSpectrum(TEST_K, z = 0.9)
		self.assertAlmostEqualArray(Pk, correct)

	def test_sigma(self):
		correct = [1.207139845473e+01, 2.119458716015e+00, 1.280388195458e-03]
		self._testRZArray(self.cosmo.sigma, 0.0, correct)
		correct = [2.401614712027e+00, 4.216680654693e-01, 2.547342910474e-04]
		self._testRZArray(self.cosmo.sigma, TEST_Z2, correct)

	def test_sigma_klimits(self):
		correct = 1.899394142143e-02
		self.assertAlmostEqual(self.cosmo.sigma(1.4, 0.0, kmin = 1E1), correct)
		correct = 2.119380060824e+00
		self.assertAlmostEqual(self.cosmo.sigma(1.4, 0.0, kmin = 1E-2), correct)
		correct = 2.739312270004e-04
		self.assertAlmostEqual(self.cosmo.sigma(1.4, 0.0, kmin = 1E2), correct)
		correct = 2.119372865864e+00
		self.assertAlmostEqual(self.cosmo.sigma(1.4, 0.0, kmax = 1E1), correct)
		correct = 1.770084498979e-02
		self.assertAlmostEqual(self.cosmo.sigma(1.4, 0.0, kmax = 1E-2), correct)
		correct = 2.119459036230e+00
		self.assertAlmostEqual(self.cosmo.sigma(1.4, 0.0, kmax = 1E2), correct)
		correct = 2.082170576816e+00
		self.assertAlmostEqual(self.cosmo.sigma(1.4, 0.0, kmin = 1E-1, kmax = 1E1), correct)

	def test_correlationFunction(self):
		correct = [1.426323791531e+02, 3.998980796003e+00, -2.794706595155e-07]
		self._testRZArray(self.cosmo.correlationFunction, 0.0, correct)
		correct = [5.645593784402e+00, 1.582853855478e-01, -1.106184884282e-08]
		self._testRZArray(self.cosmo.correlationFunction, TEST_Z2, correct)

###################################################################################################
# TEST CASE 2: INTERPOLATION, DERIVATIVES, INVERSES
###################################################################################################

class TCInterp(CosmologyTestCase):

	def setUp(self):
		self.cosmo_name = 'planck15'
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': True, 
															'persistence': ''})

	###############################################################################################
	# Function tests
	###############################################################################################

	def test_sigma(self):
		self.assertAlmostEqual(self.cosmo.sigma(12.5, 0.0), 5.892735283989e-01)

	def test_ZDerivative(self):
		correct = [-1.443560476858e+01, -3.025533589239e+00, -1.281100859272e-02]
		self.assertAlmostEqualArray(self.cosmo.age(TEST_Z, derivative = 1), correct)

	def test_ZDerivative2(self):
		correct = [2.112575913079e+01, 2.994337001869e+00, 1.532141374826e-03]
		self.assertAlmostEqualArray(self.cosmo.age(TEST_Z, derivative = 2), correct)

	def test_ZInverse(self):
		correct = [6.737712833304e-03, 2.981579253668e+01]
		self.assertAlmostEqualArray(self.cosmo.age(TEST_AGE, inverse = True), correct)

	def test_ZInverseDerivative(self):
		correct = [-6.998048234536e-02, -2.036513944918e+02]
		self.assertAlmostEqualArray(self.cosmo.age(TEST_AGE, inverse = True, derivative = 1), correct)		

	def test_ps_derivative(self):
		correct = [9.283892624857e-01, -2.819106688708e+00]
		ps_der_z0 = self.cosmo.matterPowerSpectrum(TEST_K, 0.0, derivative = True)
		self.assertAlmostEqualArray(ps_der_z0, correct)
		ps_der_z2 = self.cosmo.matterPowerSpectrum(TEST_K, 2.0, derivative = True)
		self.assertAlmostEqualArray(ps_der_z2, correct)

	def test_sigma_derivative(self):
		correct = [-1.425800765884e-01, -4.392008200148e-01, -1.794761655074e+00]
		sigma_der_z0 = self.cosmo.sigma(TEST_RR_interp, 0.0, derivative = True)
		self.assertAlmostEqualArray(sigma_der_z0, correct)
		sigma_der_z2 = self.cosmo.sigma(TEST_RR_interp, 2.0, derivative = True)
		self.assertAlmostEqualArray(sigma_der_z2, correct)
		
###################################################################################################
# TEST CASE 3: NON-FLAT COSMOLOGY WITH POSITIVE CURVATURE
###################################################################################################

class TCNotFlat1(CosmologyTestCase):

	def setUp(self):
		c = {'flat': False, 'H0': 70.00, 'Om0': 0.2700, 'Ode0': 0.7, 'Ob0': 0.0469, 'sigma8': 0.8200, 
				'ns': 0.9500, 'relspecies': True, 'persistence': ''}
		cosmology.addCosmology('myCosmo', c)
		self.assertTrue('myCosmo' in cosmology.cosmologies)
		cosmology.setCosmology('myCosmo')
		self.cosmo = cosmology.getCurrent()

	def test_nonFlat(self):
		self.assertAlmostEqual(self.cosmo.Ok0, 2.991462123780e-02)
		self.assertAlmostEqual(self.cosmo.Ok(4.5), 1.941703673565e-02)

	def test_distanceNonFlat(self):
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 1.0, transverse = True), 2.340299029513e+03)
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 10.0, transverse = True), 6.959070744985e+03)
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 1.0, transverse = False), 2.333246157596e+03)
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 10.0, transverse = False), 6.784500313347e+03)

###################################################################################################
# TEST CASE 4: NON-FLAT COSMOLOGY WITH NEGATIVE CURVATURE
###################################################################################################

class TCNotFlat2(CosmologyTestCase):

	def setUp(self):
		c = {'flat': False, 'H0': 70.00, 'Om0': 0.2700, 'Ode0': 0.8, 'Ob0': 0.0469, 'sigma8': 0.8200, 
			'ns': 0.9500, 'relspecies': True, 'persistence': ''}
		cosmology.addCosmology('myCosmo', c)
		self.assertTrue('myCosmo' in cosmology.cosmologies)
		cosmology.setCosmology('myCosmo')
		self.cosmo = cosmology.getCurrent()

	def test_nonFlat(self):
		self.assertAlmostEqual(self.cosmo.Ok0, -7.008537876220e-02)
		self.assertAlmostEqual(self.cosmo.Ok(4.5), -4.853747631639e-02)

	def test_distanceNonFlat(self):
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 1.0, transverse = True), 2.391423790129e+03)
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 10.0, transverse = True), 6.597189591596e+03)
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 1.0, transverse = False), 2.409565053915e+03)
		self.assertAlmostEqual(self.cosmo.comovingDistance(0.0, 10.0, transverse = False), 7.042437309730e+03)

###################################################################################################
# TEST CASE 5: VARYING DARK ENERGY EQUATION OF STATE 1
###################################################################################################

class TCDarkEnergy1(CosmologyTestCase):

	def setUp(self):
		c = {'flat': True, 'H0': 70.00, 'Om0': 0.2700, 'Ob0': 0.0469, 'sigma8': 0.8200, 
			'ns': 0.9500, 'relspecies': True, 'de_model': 'w0wa', 'w0': -0.7, 'wa': 0.2, 'persistence': ''}
		cosmology.addCosmology('myCosmo', c)
		self.assertTrue('myCosmo' in cosmology.cosmologies)
		cosmology.setCosmology('myCosmo')
		self.cosmo = cosmology.getCurrent()

	def test_Ez(self):
		self.assertAlmostEqual(self.cosmo.wz(0.5), -6.333333333333e-01)
		self.assertAlmostEqual(self.cosmo.Ez(1.2), 2.143355338332e+00)

###################################################################################################
# TEST CASE 6: VARYING DARK ENERGY EQUATION OF STATE 2
###################################################################################################

class TCDarkEnergy2(CosmologyTestCase):

	def setUp(self):
		c = {'flat': True, 'H0': 70.00, 'Om0': 0.2700, 'Ob0': 0.0469, 'sigma8': 0.8200, 
			'ns': 0.9500, 'relspecies': True, 'de_model': 'w0', 'w0': -0.7, 'persistence': ''}
		cosmology.addCosmology('myCosmo', c)
		self.assertTrue('myCosmo' in cosmology.cosmologies)
		cosmology.setCosmology('myCosmo')
		self.cosmo = cosmology.getCurrent()

	def test_Ez(self):
		self.assertAlmostEqual(self.cosmo.wz(0.5), -0.7)
		self.assertAlmostEqual(self.cosmo.Ez(1.2), 2.088306376421e+00)

###################################################################################################
# TEST CASE 7: VARYING DARK ENERGY EQUATION OF STATE 2
###################################################################################################

# Dark energy equation of state test function
def wz_func(z):
	return -0.7 + 0.2 * (1.0 - 1.0 / (1.0 + z))

class TCDarkEnergy3(CosmologyTestCase):

	def setUp(self):
		c = {'flat': True, 'H0': 70.00, 'Om0': 0.2700, 'Ob0': 0.0469, 'sigma8': 0.8200, 
			'ns': 0.9500, 'relspecies': True, 'de_model': 'user', 'wz_function': wz_func,
			'persistence': ''}
		cosmology.addCosmology('myCosmo', c)
		self.assertTrue('myCosmo' in cosmology.cosmologies)
		cosmology.setCosmology('myCosmo')
		self.cosmo = cosmology.getCurrent()

	def test_Ez(self):
		self.assertAlmostEqual(self.cosmo.wz(0.5), -6.333333333333e-01)
		self.assertAlmostEqual(self.cosmo.Ez(1.2), 2.143355338332e+00)

###################################################################################################
# TEST CASE 8: GROWTH FACTOR IN w0CDM
###################################################################################################

class TCDarkEnergyGrowthFactor(CosmologyTestCase):

	def setUp(self):
		pass
	
	def test_growthFactorFromODE(self):
		z = np.array([1.0, 0.5, 2.0, -0.9, 3.0, 120.0, 0.0])
		for k in range(2):
			interpolation = (k == 1)
			my_cosmo_1 = {'flat': True, 'H0': 100 * 0.693, 'Om0': 0.287, 'Ob0': 0.043, 'sigma8': 0.820, 'ns': 1, 
						'persistence': '', 'interpolation': interpolation}
			my_cosmo_2 = {'flat': True, 'H0': 100 * 0.693, 'Om0': 0.287, 'Ob0': 0.043, 'sigma8': 0.820, 'ns': 1, 
						'persistence': '', 'interpolation': interpolation, 'de_model': 'w0', 'w0': -1.0}
			cosmo1 = cosmology.setCosmology('test_1', my_cosmo_1)
			D1 = cosmo1.growthFactor(z)
			cosmo2 = cosmology.setCosmology('test_2', my_cosmo_2)
			D2 = cosmo2.growthFactor(z)
			self.assertAlmostEqualArray(D1, D2, places = 4)

###################################################################################################
# TEST CASE 9: SELF-SIMILAR COSMOLOGIES
###################################################################################################

class TCSelfSimilar(CosmologyTestCase):

	def setUp(self):
		self.cosmo_name = 'powerlaw_-2.23'
		self.cosmo = cosmology.setCosmology(self.cosmo_name, {'interpolation': False, 
															'persistence': ''})
	
	def test_sigma_tophat(self):
		sigma_nn = self.cosmo._sigmaExact(8.0, 0.0, filt = 'tophat', ignore_norm = True)
		self.assertAlmostEqual(sigma_nn, 0.1452984341044695)
		sigma = self.cosmo._sigmaExact(8.0, 0.0, filt = 'tophat', ignore_norm = False)
		self.assertAlmostEqual(sigma, self.cosmo.sigma8, places = 5)

		# The normalization of the power spectrum at k=1 must be derived from sigma8 and the 
		# variance of the PS
		Pk = self.cosmo.matterPowerSpectrum(1.0, 0.0)
		self.assertAlmostEqual(Pk, (self.cosmo.sigma8 / sigma_nn)**2, places = 5)
		
	def test_sigma_gaussian(self):
		# Test Gaussian filter; the result should not be the same as for tophat.
		sigma_g = self.cosmo._sigmaExact(8.0, 0.0, filt = 'gaussian', ignore_norm = False)
		self.assertAlmostEqual(sigma_g, 6.126388566928e-01, places = 5)
		
	def test_cf(self):
		# Here, we test to high accuracy to make sure the function uses the analytical expressions
		# for the PL CF.
		cf = self.cosmo._correlationFunctionExact(0.1)
		self.assertAlmostEqual(cf, 1.751768324780e+01, places = 10)
		cf = self.cosmo._correlationFunctionExact(100.0)
		self.assertAlmostEqual(cf, 8.579790219177e-02, places = 10)
		
###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
