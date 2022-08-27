import unittest
import numpy as np
import dsft.setfunctions as sf


class shapleyTest(unittest.TestCase):
    def test_equality_models(self):
        for i in range(10):
            rand_coefs = np.random.rand(64).tolist()
            rand_sf = sf.WrapSignal(rand_coefs)
            ft3 = rand_sf.transform_fast(model = '3')
            ft4 = rand_sf.transform_fast(model = '4')
            ftWHT = rand_sf.transform_fast(model = '5')

            np.testing.assert_almost_equal(ft3.shapley_values(),ft4.shapley_values(),decimal = 8)
            np.testing.assert_almost_equal(ft4.shapley_values(),ftWHT.shapley_values(),decimal = 8)

