###################################################################################################
#
# test_halo_profile.py  (c) Benedikt Diemer
#     				    	diemer@umd.edu
#
###################################################################################################

import numpy as np
import unittest

from colossus.tests import test_colossus
from colossus import defaults
from colossus.utils import utilities
from colossus.cosmology import cosmology
from colossus.halo import mass_so
from colossus.halo import profile_base
from colossus.halo import profile_outer
from colossus.halo import profile_composite
from colossus.halo import profile_spline
from colossus.halo import profile_nfw
from colossus.halo import concentration

###################################################################################################
# CONSTANTS
###################################################################################################

# For some test cases in the profile unit, we cannot expect the results to agree to very high 
# precision because numerical approximations are made.

TEST_N_DIGITS_LOW = 4

all_profs_inner = ['nfw', 'einasto', 'hernquist', 'dk14', 'diemer23', 'diemer23b']

###################################################################################################
# TEST CASE: CREATE PROFILES WITH OUTER PROFILES MANUALLY AND AS COMPOSITE
###################################################################################################

# Create profiles in different ways and compare the parameters, which should be the same. Also
# make sure that M/c are reproduced when outer profiles are added, and that the composite profile
# function does the same thing as a manually constructed composite profile.

class TCCreation(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('planck18', {'persistence': ''})
		self.M = 1E14
		self.c = 7.0
		self.z = 1.0
		self.mdef = '200c'
		cosmology.setCosmology('planck18')
		self.ot_sets = [['pl'], ['infalling'], ['infalling', 'mean']]
		self.p_objs = []
		for pname in all_profs_inner:
			p_obj = profile_composite.getProfileClass(pname)
			self.p_objs.append(p_obj)
	
	def test_creation_inner(self):

		for i in range(len(self.p_objs)):		
			p1 = self.p_objs[i](M = self.M, c = self.c, z = self.z, mdef = self.mdef)
			pars = p1.par.copy()
			pars.update(p1.opt)
			p2 = self.p_objs[i](z = self.z, **pars)
			for k in p1.par:
				self.assertAlmostEqual(p1.par[k], p2.par[k], places = TEST_N_DIGITS_LOW)

	def test_creation_outer(self):
		
		for j in range(len(self.p_objs)):
			for i in range(len(self.ot_sets)):
				
				ot_set = self.ot_sets[i]		
				outer_terms = []
				ot_par_all = {}
				for j in range(len(ot_set)):
					if ot_set[j] == 'mean':
						ot_cls = profile_outer.OuterTermMeanDensity
						ot_par = dict(z = self.z)
					elif ot_set[j] == 'cf':
						ot_cls = profile_outer.OuterTermCorrelationFunction
						ot_par = dict(derive_bias_from = 'R200m', z = self.z)
					elif ot_set[j] == 'pl':
						ot_cls = profile_outer.OuterTermPowerLaw
						ot_par = dict(norm = defaults.HALO_PROFILE_DK14_PL_NORM, 
								slope = defaults.HALO_PROFILE_DK14_PL_SLOPE, 
								pivot = 'R200m', pivot_factor = 5.0, z = self.z)
					elif ot_set[j] == 'infalling':
						ot_cls = profile_outer.OuterTermInfalling
						ot_par = dict(pl_delta_1 = 10.0, pl_s = 1.5, z = self.z)
					else:
						raise Exception('Unknown outer term, %s.' % (ot_set[j]))			
					ot_obj = ot_cls(**ot_par)
					ot_par_all.update(ot_par)
					outer_terms.append(ot_obj)
				
				p1 = self.p_objs[j](M = self.M, c = self.c, z = self.z, mdef = self.mdef, outer_terms = outer_terms)
				pars = p1.par.copy()
				pars.update(p1.opt)
				p2 = self.p_objs[j](outer_terms = outer_terms, **pars)
				for k in p1.par:
					self.assertAlmostEqual(p1.par[k], p2.par[k], places = TEST_N_DIGITS_LOW)
				M200m_1 = p1.MDelta(self.z, '200m')
				M200m_2 = p2.MDelta(self.z, '200m')
				self.assertAlmostEqual(M200m_1, M200m_2, places = TEST_N_DIGITS_LOW)

				# Use wrapper	
				pname = all_profs_inner[j]		
				p1 = profile_composite.compositeProfile(inner_name = pname, outer_names = ot_set, 
							M = self.M, c = self.c, mdef = self.mdef, **ot_par_all)
				pars = p1.par.copy()
				pars.update(p1.opt)
				p2 = profile_composite.compositeProfile(inner_name = pname, outer_names = ot_set, 
							**pars)
				for k in p1.par:
					self.assertAlmostEqual(p1.par[k], p2.par[k], places = TEST_N_DIGITS_LOW)

				# Check returned mass and concentration
				R_out, M_out = p1.RMDelta(self.z, self.mdef)
				R_in = mass_so.M_to_R(self.M, self.z, self.mdef)
				self.assertAlmostEqual(M_out, self.M, places = TEST_N_DIGITS_LOW)
				self.assertAlmostEqual(R_out, R_in, places = TEST_N_DIGITS_LOW)
				if 'rs' in p1.par:
					c_out = R_out / p1.par['rs']
					self.assertAlmostEqual(c_out, self.c, places = TEST_N_DIGITS_LOW)

###################################################################################################
# TEST CASE: PROFILE VALUES FOR INNER PROFILES
###################################################################################################

class TCInnerRoutines(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('WMAP9', {'persistence': ''})
		
		M = 4E14
		c = 5.7
		mdef = '200c'
		z = 0.2
		
		self.p = []
		for pname in all_profs_inner:
			self.p.append(profile_composite.compositeProfile(pname, outer_names = [],
													M = M, c = c, mdef = mdef, z = z))
	
	def test_inner(self, verbose = False):
		
		r = 576.2

		correct_rho        = [ 8.781156022850e+04,  8.848655923451e+04,  6.508442621097e+04,  8.937952887979e+04,  8.970741078041e+04,  8.984157864964e+04]
		correct_Menc       = [ 2.363195068242e+14,  2.429904593149e+14,  3.066430903109e+14,  2.472947091540e+14,  2.420824778583e+14,  2.416121908814e+14]
		correct_Sigma      = [ 1.146341922088e+08,  1.071236351793e+08,  6.723110548665e+07,  1.007139029621e+08,  1.031419685910e+08,  1.034057813596e+08]
		correct_DeltaSigma = [ 1.857620896175e+08,  1.923774224862e+08,  2.658087731973e+08,  1.964939959965e+08,  1.910168003712e+08,  1.904880671894e+08]
		correct_derLin     = [-3.794338964922e+02, -3.945881359017e+02, -3.653675818059e+02, -4.043399435091e+02, -3.921603673917e+02, -3.920898434231e+02]
		correct_derLog     = [-2.489761149783e+00, -2.569448805258e+00, -3.234641724491e+00, -2.606644702315e+00, -2.518886697602e+00, -2.514672729221e+00]
		correct_vcirc      = [ 1.328139530786e+03,  1.346754788545e+03,  1.512900996528e+03,  1.358630405956e+03,  1.344236227010e+03,  1.342929886866e+03]
		correct_vmax       = [ 1.338948895668e+03,  1.360299204142e+03,  1.735263473078e+03,  1.373151750154e+03,  1.355718897847e+03,  1.353746005809e+03]
		correct_rdelta     = [ 1.010859075063e+03,  1.013948476519e+03,  1.026753136588e+03,  1.016405547922e+03,  1.014857297856e+03,  1.014738141595e+03]

		for i in range(len(self.p)):
			
			q = self.p[i].density(r)
			self.assertAlmostEqual(q, correct_rho[i], places = TEST_N_DIGITS_LOW)
			
			q = self.p[i].enclosedMass(r)
			self.assertAlmostEqual(q, correct_Menc[i], places = TEST_N_DIGITS_LOW)

			q = self.p[i].surfaceDensity(r)
			self.assertAlmostEqual(q, correct_Sigma[i], places = TEST_N_DIGITS_LOW)

			q = self.p[i].deltaSigma(r)
			self.assertAlmostEqual(q, correct_DeltaSigma[i], places = TEST_N_DIGITS_LOW)

			q = self.p[i].densityDerivativeLin(r)
			self.assertAlmostEqual(q, correct_derLin[i], places = TEST_N_DIGITS_LOW)

			q = self.p[i].densityDerivativeLog(r)
			self.assertAlmostEqual(q, correct_derLog[i], places = TEST_N_DIGITS_LOW)

			q = self.p[i].circularVelocity(r)
			self.assertAlmostEqual(q, correct_vcirc[i], places = TEST_N_DIGITS_LOW)

			q, _ = self.p[i].Vmax()
			self.assertAlmostEqual(q, correct_vmax[i], places = TEST_N_DIGITS_LOW)

			q = self.p[i].RDelta(0.7, mdef = 'vir')
			self.assertAlmostEqual(q, correct_rdelta[i], places = TEST_N_DIGITS_LOW)

###################################################################################################
# TEST CASE: OUTER PROFILES
###################################################################################################

class TCOuterRoutines(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('WMAP9', {'persistence': ''})

		z = 0.2
		M = 4E12
		c = 5.7
		mdef = '200c'

		self.t = []
		self.t.append(profile_outer.OuterTermMeanDensity(z = z))
		self.t.append(profile_outer.OuterTermCorrelationFunction(z = z, bias = 2.2))
		self.t.append(profile_outer.OuterTermPowerLaw(z = z, norm = 2.0, slope = 1.4, 
										max_rho = 1200.0, pivot = 'fixed', pivot_factor = 257.0))
		self.t.append(profile_outer.OuterTermInfalling(z = z, pl_delta_1 = 10.0, pl_s = 1.4, 
										pl_delta_max = 1200.0))
		
		self.p = []
		for i in range(len(self.t)):
			self.p.append(profile_nfw.NFWProfile(M = M, c = c, mdef = mdef, z = z, outer_terms = [self.t[i]]))
	
	def test_outer(self, verbose = False):
		
		r = 980.2

		correct_rho = [4.321019370749e+02, 1.405026434021e+03, 3.350273919041e+02, 6.115469636994e+02]
		correct_der = [-8.769648316169e-01, -1.827787576477e+00, -9.316967625779e-01, -1.300382848541e+00]

		for i in range(len(self.p)):
			
			q = self.p[i].density(r)
			self.assertAlmostEqual(q, correct_rho[i])

			q = self.p[i].densityDerivativeLin(r)
			self.assertAlmostEqual(q, correct_der[i], places = 7)

###################################################################################################
# TEST CASE: NUMERICAL ROUTINES IN BASE CLASS
###################################################################################################

# This test case compares three different implementations of the NFW density profile: 
# - the exact, analytic form
# - the generic implementation of the HaloDensityProfile base class, where only the density is 
#   computed analytically, but all other functions numerically ('Numerical')
# - a discrete profile where density and/or mass are given as arrays. Three cases are tested, with
#   only rho, only M, and both ('ArrayRho', 'ArrayM', and 'ArrayRhoM').

class TCNumerical(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('WMAP9', {'persistence': ''})
		self.MAX_DIFF_RHO = 1E-8
		self.MAX_DIFF_M = 1E-8
		self.MAX_DIFF_DER = 1E-2
		self.MAX_DIFF_SIGMA = 1E-5
		self.MAX_DIFF_VCIRC = 1E-8
		self.MAX_DIFF_RMAX = 1E-3
		self.MAX_DIFF_VMAX = 1E-8
		self.MAX_DIFF_SO_R = 1E-7
		self.MAX_DIFF_SO_M = 1E-7
	
	def test_base_nfw(self, verbose = False):

		class TestProfile(profile_base.HaloDensityProfile):
			
			def __init__(self, rhos, rs):
				
				self.par_names = ['rhos', 'rs']
				self.opt_names = []
				profile_base.HaloDensityProfile.__init__(self, rhos = rhos, rs = rs)
				
				return
			
			def densityInner(self, r):
			
				x = r / self.par['rs']
				density = self.par['rhos'] / x / (1.0 + x)**2
				
				return density
		
			def setNativeParameters(self, M, c, z, mdef, **kwargs):
				
				return

		# Properties of the test halo
		M = 1E12
		c = 10.0
		mdef = 'vir'
		z = 0.0
		
		# Radii and reshifts where to test
		r_test = np.array([0.011, 1.13, 10.12, 102.3, 505.0])
		z_test = 1.0
		mdef_test = '200c'
	
		# Parameters for the finite-resolution NFW profile; here we want to test whether this method
		# converges to the correct solution, so the resolution is high.
		r_min = 1E-2
		r_max = 1E4
		N = 1000
	
		# PROFILE 1: Analytical NFW profile
		prof1 = profile_nfw.NFWProfile(M = M, c = c, z = z, mdef = mdef)
		rs = prof1.par['rs']
		rhos = prof1.par['rhos']
	
		# PROFILE 2: Only the density is analytical, the rest numerical
		prof2 = TestProfile(rhos = rhos, rs = rs)
		
		# PROFILES 3/4/5: User-defined NFW with finite resolution
		log_min = np.log10(r_min)
		log_max = np.log10(r_max)
		bin_width = (log_max - log_min) / N
		r_ = 10**np.arange(log_min, log_max + bin_width, bin_width)
		rho_ = prof1.density(r_)
		M_ = prof1.enclosedMass(r_)
		prof3 = profile_spline.SplineProfile(r = r_, rho = rho_)
		prof4 = profile_spline.SplineProfile(r = r_, M = M_)
		prof5 = profile_spline.SplineProfile(r = r_, rho = rho_, M = M_)
	
		# Test for all profiles
		profs = [prof1, prof2, prof3, prof4, prof5]
		prof_names = ['Reference', 'Numerical', 'ArrayRho', 'ArrayM', 'ArrayRhoM']
	
		if verbose:
			utilities.printLine()
			print(('Profile properties as a function of radius'))
			utilities.printLine()
			print(('Density'))
		
		for i in range(len(profs)):
			res = profs[i].density(r_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_RHO, 'Difference in density too large.')
				if verbose:
					print('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff))
							
		if verbose:
			utilities.printLine()
			print(('Density Linear Derivative'))
		
		for i in range(len(profs)):
			res = profs[i].densityDerivativeLin(r_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_DER, 'Difference in density derivative too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))
	
		if verbose:
			utilities.printLine()
			print(('Density Logarithmic Derivative'))
		
		for i in range(len(profs)):
			res = profs[i].densityDerivativeLog(r_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_DER, 'Difference in density log derivative too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))
		
		if verbose:
			utilities.printLine()
			print(('Enclosed mass'))
		
		for i in range(len(profs)):
			res = profs[i].enclosedMass(r_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_M, 'Difference in enclosed mass too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))
	
		if verbose:
			utilities.printLine()
			print(('Surface density'))
		
		for i in range(len(profs)):
			res = profs[i].surfaceDensity(r_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_SIGMA, 'Difference in surface density too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))
	
		if verbose:
			utilities.printLine()
			print(('Circular velocity'))
		
		for i in range(len(profs)):
			res = profs[i].circularVelocity(r_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_VCIRC, 'Difference in circular velocity too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))
		
		if verbose:
			utilities.printLine()
			print(('Rmax'))
		
		for i in range(len(profs)):
			_, res = profs[i].Vmax()
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_RMAX, 'Difference in Rmax too large.')
				if verbose:
					print('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff))
		
		if verbose:
			utilities.printLine()
			print(('Vmax'))
		
		for i in range(len(profs)):
			res, _ = profs[i].Vmax()
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_VMAX, 'Difference in Vmax too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))
	
		if verbose:
			utilities.printLine()
			print(('Spherical overdensity radii and masses'))
			utilities.printLine()
			print(('Spherical overdensity radius'))
		
		for i in range(len(profs)):
			res = profs[i].RDelta(z_test, mdef_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_SO_R, 'Difference in SO radius too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))
	
		if verbose:
			utilities.printLine()
			print(('Spherical overdensity mass'))
		
		for i in range(len(profs)):
			res = profs[i].MDelta(z_test, mdef_test)
			if i == 0:
				ref = res
			else:
				max_diff = np.abs(np.max((res - ref) / ref))
				self.assertLess(max_diff, self.MAX_DIFF_SO_M, 'Difference in SO mass too large.')
				if verbose:
					print(('Profile: %12s    Max diff: %9.2e' % (prof_names[i], max_diff)))

###################################################################################################
# TEST CASE: FITTING
###################################################################################################

class TCFitting(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('WMAP9', {'persistence': ''})
		M = 1E12
		c = 6.0
		mdef = 'vir'
		z = 0.0
		self.p = profile_nfw.NFWProfile(M = M, c = c, z = z, mdef = mdef)
	
	def test_leastsq(self, verbose = False):

		scatter = 0.001
		r = 10**np.arange(0.1, 3.6, 0.1)
		mask = np.array([True, True])
		q_true = self.p.density(r)
		scatter_sigma = scatter * 0.3
		np.random.seed(157)
		q_err = np.abs(np.random.normal(scatter, scatter_sigma, (len(r)))) * q_true
		q = q_true.copy()
		for i in range(len(r)):
			q[i] += np.random.normal(0.0, q_err[i])
		x_true = self.p.getParameterArray(mask)
		ini_guess = x_true * 1.5
		self.p.setParameterArray(ini_guess, mask = mask)
		dummy = self.p.fit(r, q, 'rho', q_err = q_err, verbose = False, mask = mask, tolerance = 1E-6)
		x = self.p.getParameterArray(mask = mask)
		acc = abs(x / x_true - 1.0)
		
		self.assertLess(acc[0], 1E-2)
		self.assertLess(acc[1], 1E-2)

###################################################################################################
# TEST CASE: NFW SPECIAL FUNCTIONS
###################################################################################################
	
class TCNFW(test_colossus.ColosssusTestCase):

	def setUp(self):
		cosmology.setCosmology('WMAP9', {'persistence': ''})
				
	def test_pdf(self):
		
		M = 10**np.arange(9.0, 15.5, 0.2)
		mdef = 'vir'
		z = 0.0
		c = concentration.concentration(M, mdef, z)
		N = len(M)
		p = np.random.uniform(0.0, 1.0, (N))
		r1 = profile_nfw.radiusFromPdf(M, c, z, mdef, p, interpolate = False)
		r2 = profile_nfw.radiusFromPdf(M, c, z, mdef, p, interpolate = True)
		R = mass_so.M_to_R(M, z, mdef)
		rs = R / c
		p1 = profile_nfw.NFWProfile.mu(r1 / rs) / profile_nfw.NFWProfile.mu(c)
		p2 = profile_nfw.NFWProfile.mu(r2 / rs) / profile_nfw.NFWProfile.mu(c)
		diff1 = np.max(np.abs(p1 / p - 1.0))
		diff2 = np.max(np.abs(p2 / p - 1.0))
		
		self.assertLess(diff1, 1E-8)
		self.assertLess(diff2, 1E-2)

###################################################################################################
# TRIGGER
###################################################################################################

if __name__ == '__main__':
	unittest.main()
