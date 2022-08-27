import unittest

import test_fast
import test_minimization
import test_shapley
import test_sparse

loader = unittest.TestLoader()
suite  = unittest.TestSuite()

suite.addTests(loader.loadTestsFromModule(test_fast))
suite.addTests(loader.loadTestsFromModule(test_sparse))
suite.addTests(loader.loadTestsFromModule(test_shapley))
suite.addTests(loader.loadTestsFromModule(test_minimization))


runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)
