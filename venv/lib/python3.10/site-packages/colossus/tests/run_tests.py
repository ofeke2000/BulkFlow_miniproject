###################################################################################################
#
# run_tests.py          (c) Benedikt Diemer
#     				    	diemer@umd.edu
#
###################################################################################################

import unittest

from colossus.tests import test_utils
from colossus.tests import test_cosmology
from colossus.tests import test_cosmology_power_spectrum
from colossus.tests import test_lss_bias
from colossus.tests import test_lss_mass_function
from colossus.tests import test_lss_peaks
from colossus.tests import test_halo_concentration
from colossus.tests import test_halo_mass
from colossus.tests import test_halo_profile
from colossus.tests import test_halo_splashback

###################################################################################################

suites = []

# Utils
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_utils.TCVersions))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_utils.TCConstants))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_utils.TCStorageNonPersistent))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_utils.TCStoragePersistent))

# Cosmology
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCComp))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCInterp))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCNotFlat1))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCNotFlat2))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCDarkEnergy1))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCDarkEnergy2))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCDarkEnergy3))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCDarkEnergyGrowthFactor))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_cosmology.TCSelfSimilar))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_cosmology_power_spectrum.TCTransferFunction))

# Large-scale structure
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_lss_peaks.TCPeaks))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_lss_peaks.TCPeaksInterp))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_lss_mass_function.TCMassFunction))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_lss_bias.TCBias))

# Halo
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_concentration.TCConcentration))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_mass.TCMassSO))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_mass.TCMassDefs))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_mass.TCMassAdv))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_splashback.TCSplashbackModel))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_splashback.TCSplashbackRadius))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_profile.TCCreation))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_profile.TCInnerRoutines))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_profile.TCOuterRoutines))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_profile.TCNumerical))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_profile.TCFitting))
suites.append(unittest.TestLoader().loadTestsFromTestCase(test_halo_profile.TCNFW))

# Run tests
suite = unittest.TestSuite(suites)
unittest.TextTestRunner(verbosity = 2).run(suite)
