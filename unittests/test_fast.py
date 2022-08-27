import unittest
import numpy as np
import dsft.setfunctions as sf
import dsft.transformations as tf
import dsft.utils as utils


class FastTest(unittest.TestCase):
    def test_dsft3_random(self):
        for i in range(10):
            rand_coefs = np.random.rand(64).tolist()
            s = sf.WrapSignal(rand_coefs)
            ft3 = s.transform_fast(model = '3')
            np.testing.assert_almost_equal(ft3(utils.get_indicator_set(6)),rand_coefs,decimal=8)
    def test_dsft4_random(self):
        for i in range(10):
            rand_coefs = np.random.rand(64).tolist()
            s = sf.WrapSignal(rand_coefs)
            ft4 = s.transform_fast(model ='4')
            np.testing.assert_almost_equal(ft4(utils.get_indicator_set(6)),rand_coefs,decimal=8)
    def test_dsfwht_random(self):
        for i in range(10):
            rand_coefs = np.random.rand(64).tolist()
            s = sf.WrapSignal(rand_coefs)
            fwht = s.transform_fast(model = '5')
            np.testing.assert_almost_equal(fwht(utils.get_indicator_set(6)),rand_coefs,decimal=8)
    