import unittest
import numpy as np
import pandas as pd
from metrics_miscellany.estimators import fwl_regression, reconstruct_coefficients_from_fwl
import datamat as dm

class TestFWLRegression(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        np.random.seed(42)

        N = 1000

        D = {}
        D['Constant'] = dm.DataMat(np.ones(N))
        D['a'] = dm.DataMat(np.random.randn(N,2))
        D['b'] = D['a']*2 + np.random.randn(N,2)
        D['y'] = dm.DataMat(D['a']@np.array((1,1)) + + D['b']@np.array((2,2)) + np.random.randn(N,)/1000,name='y')

        # Reverse order of dict
        D = dict([D.popitem() for i in range(len(D))])

        self.D = D

        # Direct OLS for comparison
        YX = dm.concat(D,levelnames=True,axis=1)
        Y = YX.xs('y',level='v',axis=1,drop_level=False)
        X = YX.iloc[:,1:]

        self.direct_coefs = X.lstsq(Y)

    def test_fwl_regression(self):
        # Run FWL regression
        U, B = fwl_regression(self.D.copy())

        # Check if U & B contains entries for all variables
        self.assertEqual(set(B.keys()), set(U.keys()))

        # Test that the last stage regression gives correct coefficients
        # The last variable (X3) should have the coefficient of Y on X3 after controlling for X1, X2
        # Reconstruct coefficients
        reconstructed = reconstruct_coefficients_from_fwl(B,as_dict=False)

        np.testing.assert_allclose(reconstructed-self.direct_coefs,0,atol=1e-2)

    def test_reconstruct_coefficients(self):
        # Run FWL regression and ask reconstruct_coefficients_from_fwl for
        # both shapes of output; the two views must agree.  (The previous
        # body checked for variable names ['X1','X2','X3'] that do not
        # appear in this fixture, so the loop never executed.)
        U, B = fwl_regression(self.D.copy())

        reconstructed_dict = reconstruct_coefficients_from_fwl(B, as_dict=True)
        reconstructed_concat = reconstruct_coefficients_from_fwl(B, as_dict=False)

        # The dict output's keys are the regressors (everything in D
        # except the dependent variable).
        self.assertEqual(set(reconstructed_dict.keys()),
                         set(self.D.keys()) - {'y'})

        # Total parameter count is preserved across the two output forms.
        total = sum(len(v) for v in reconstructed_dict.values())
        self.assertEqual(total, len(reconstructed_concat))

    def test_empty_dict(self):
        # Test with empty dictionary
        U, B = fwl_regression({})
        self.assertEqual(U, {})
        self.assertEqual(B, {})

        # Test reconstruction with empty results
        reconstructed = reconstruct_coefficients_from_fwl({})
        self.assertEqual(reconstructed, {})

    def test_single_variable(self):
        # Test with a single variable
        D_single = {'y': self.D['y']}
        U, B = fwl_regression(D_single)
        self.assertEqual(U, {})
        self.assertEqual(B, {})

if __name__ == '__main__':
    foo = TestFWLRegression()
    foo.setUp()
    #foo.test_fwl_regression()
    unittest.main()
