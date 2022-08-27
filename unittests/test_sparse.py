import unittest
import numpy as np
import dsft.setfunctions as sf
import dsft.utils as utils


class SparseTest(unittest.TestCase):
    def test_Sparsesft3_randomSignal(self):
        for i in range(10):
            rand_coefs = np.random.rand(64).tolist()
            s = sf.WrapSignal(rand_coefs)
            ft3 = s.transform_sparse(model = '3')
            np.testing.assert_almost_equal(ft3(utils.get_indicator_set(6)),rand_coefs,decimal=8)
    def test_Sparsesft4_randomSignal(self):
        for i in range(10):
            rand_coefs = np.random.rand(64).tolist()
            s = sf.WrapSignal(rand_coefs)
            ft3 = s.transform_sparse(model = '4')
            np.testing.assert_almost_equal(ft3(utils.get_indicator_set(6)),rand_coefs,decimal=8)
    def test_SparseWeightedsft3_randomSignal(self):
        for i in range(10):
            rand_coefs = np.random.rand(64).tolist()
            s = sf.WrapSignal(rand_coefs)
            ftw3 = s.transform_sparse(model = 'W3')
            np.testing.assert_almost_equal(ftw3(utils.get_indicator_set(6)),rand_coefs,decimal=8)