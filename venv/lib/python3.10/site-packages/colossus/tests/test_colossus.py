###################################################################################################
#
# test_colossus.py      (c) Benedikt Diemer
#     				    	diemer@umd.edu
#
###################################################################################################

import unittest

###################################################################################################
# CONSTANTS FOR ALL TESTS
###################################################################################################

TEST_N_DIGITS = 9

###################################################################################################
# UNIT TEST CLASS FOR ALL COLOSSUS TESTS
###################################################################################################

class ColosssusTestCase(unittest.TestCase):
	
	# The places keyword strictly compares decimal digits, which isn't helpful when it comes to 
	# large numbers. 
	
	def assertAlmostEqual(self, first, second, places = TEST_N_DIGITS, msg = None):
		
		if abs(second) < 1E-15:
			diff = abs(first)
			msg_ = 'Got %.12e, expected %.12e, absolute difference %.6e. ' % (first, second, diff)
		else:
			diff = abs((first - second) / second)
			msg_ = 'Got %.12e, expected %.12e, relative difference %.2e. ' % (first, second, diff)
		
		if msg is not None:
			msg_ += msg
		
		unittest.TestCase.assertLess(self, diff, 10**-places, msg = msg_)
	
	def assertAlmostEqualArray(self, first, second, places = TEST_N_DIGITS, msg = None):
		N1 = len(first)
		if N1 != len(second):
			raise Exception('Length of arrays must be the same.')
		for i in range(N1):
			msg_ = ' Array element %d/%d' % (i + 1, N1)
			if msg is not None:
				msg = msg + msg_
			self.assertAlmostEqual(first[i], second[i], places = places, msg = msg)

###################################################################################################
