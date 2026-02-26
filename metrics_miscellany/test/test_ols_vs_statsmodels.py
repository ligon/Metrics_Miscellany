"""Cross-validate metrics_miscellany.ols against statsmodels.OLS."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from metrics_miscellany.estimators import ols


@pytest.fixture
def ols_data():
    rng = np.random.default_rng(42)
    N, k = 500, 3
    X = pd.DataFrame(rng.standard_normal((N, k)), columns=[f"x{i}" for i in range(k)])
    X["Constant"] = 1.0
    beta = np.array([1.0, -0.5, 0.3, 2.0])
    # Heteroskedastic errors so HC estimators differ from classical OLS
    scale = 1 + np.abs(X.values[:, 0])
    y = X.values @ beta + rng.standard_normal(N) * scale
    y = pd.DataFrame(y, columns=["y"])
    return X, y


@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
def test_coefficients_and_hc_covariance(ols_data, cov_type):
    X, y = ols_data

    b, V = ols(X, y, cov_type=cov_type)

    est = sm.OLS(y, X).fit()
    sm_b = est.params.values
    sm_V = getattr(est, f"cov_{cov_type}")

    np.testing.assert_allclose(b.values.ravel(), sm_b, rtol=1e-10)
    np.testing.assert_allclose(V.values, sm_V, rtol=1e-10)


def test_ols_classical_covariance(ols_data):
    X, y = ols_data

    b, V = ols(X, y, cov_type="OLS")

    est = sm.OLS(y, X).fit()
    sm_b = est.params.values
    # Classical OLS variance: s^2 (X'X)^-1, where s^2 = e'e/n (biased)
    e = y.values.ravel() - X.values @ sm_b
    s2 = np.var(e, ddof=0)
    sm_V = s2 * np.linalg.inv(X.values.T @ X.values)

    np.testing.assert_allclose(b.values.ravel(), sm_b, rtol=1e-10)
    np.testing.assert_allclose(V.values, sm_V, rtol=1e-10)
