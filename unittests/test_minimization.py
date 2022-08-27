import unittest
import numpy as np
import dsft.setfunctions as sf

class MinimizationTest(unittest.TestCase):
    def test_random_minimization_MIP_ft3(self):
        for i in range(10):
            rand_coefs = np.random.rand(64).tolist()
            s = sf.WrapSignal(rand_coefs)
            opt_min = s.min()
            ft3 = s.transform_sparse(model = '3')
            found_min, _ = ft3.minimize_MIP()
            np.testing.assert_almost_equal(opt_min,found_min,decimal=8)
    def test_random_minimization_MIP_ft4(self):
        for i in range(10):
            rand_coefs = np.random.rand(64).tolist()
            s = sf.WrapSignal(rand_coefs)
            opt_min = s.min()
            ft4 = s.transform_sparse(model = '4')
            found_min, _ = ft4.minimize_MIP()
            np.testing.assert_almost_equal(opt_min,found_min,decimal=8)
    def test_cardinality_of_min_MIP_ft3(self):
        for i in range(6):
            rand_coefs = np.random.rand(64).tolist()
            s = sf.WrapSignal(rand_coefs)
            ft3 = s.transform_sparse(model = '3')
            card_cons = lambda x: x == i
            found_min , _ = ft3.minimize_MIP(cardinality_constraint = card_cons)
            self.assertTrue(found_min.sum() == i)
    def test_cardinality_of_min_MIP_ft4(self):
        for i in range(6):
            rand_coefs = np.random.rand(64).tolist()
            s = sf.WrapSignal(rand_coefs)
            ft4 = s.transform_sparse(model = '4')
            card_cons = lambda x: x == i
            found_min , _ = ft4.minimize_MIP(cardinality_constraint = card_cons)
            self.assertTrue(found_min.sum() == i)
