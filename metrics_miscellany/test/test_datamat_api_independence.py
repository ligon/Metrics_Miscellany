import functools

import numpy as np
import pytest
from datamat import DataMat, DataVec

import datamat.core as datamat_core
from metrics_miscellany.estimators import tsls, restricted_tsls, linear_gmm
from metrics_miscellany.tests import cragg_donald

CONVERTED = ("inv", "pinv", "det", "trace", "leverage")


@pytest.fixture
def methods_only(monkeypatch):
    """Force DataMat's converted derivations to be methods, not properties.

    On datamat <= 0.2.1 these are cached_property; on >= 0.2.4 they are
    already methods and this is a no-op.  Either way the estimators must
    not care.
    """
    for name in CONVERTED:
        attr = datamat_core.DataMat.__dict__.get(name)
        if isinstance(attr, functools.cached_property):
            monkeypatch.setattr(datamat_core.DataMat, name, attr.func,
                                raising=False)
    return datamat_core.DataMat


def _iv_dgp(N=2000, seed=0):
    rng = np.random.default_rng(seed)
    z = DataMat({'z': rng.standard_normal(N)})
    u = DataVec(rng.standard_normal(N))
    x = DataMat({'x': z.squeeze() + u})
    x['Constant'] = 1
    z['Constant'] = 1
    beta = DataMat({'Coefficients': [1., 0.]}, index=['x', 'Constant'])
    y = (x @ beta).squeeze() + u
    return y, x, z


def test_fixture_really_installs_methods(methods_only):
    X = DataMat(np.eye(2) + np.ones((2, 2)), idxnames='i')
    assert callable(X.inv), "fixture did not convert inv to a method"


@pytest.mark.parametrize("cov", ["HC2", "HC3"])
def test_tsls_runs_with_method_spelling(methods_only, cov):
    """Covers the leverage path too: HC2/HC3 are what consume it."""
    y, x, z = _iv_dgp()
    b, Omega, V = restricted_tsls(y, x, Z=z, cov=cov)
    assert np.allclose(np.asarray(b).ravel(), [1., 0.], atol=1e-1)


def test_tsls_wrapper_runs_with_method_spelling(methods_only):
    y, x, z = _iv_dgp()
    b, V = tsls(x, y, z)
    assert np.allclose(np.asarray(b).ravel(), [1., 0.], atol=1e-1)


def test_linear_gmm_runs_with_method_spelling(methods_only):
    y, x, z = _iv_dgp()
    b = linear_gmm(x, y, z)[0]
    assert np.allclose(np.asarray(b).ravel(), [1., 0.], atol=1e-1)


def test_cragg_donald_runs_with_method_spelling(methods_only):
    rng = np.random.default_rng(1)
    n, l, m = 500, 4, 2
    Z = DataMat(rng.standard_normal((n, l)))
    Pi = DataMat(rng.standard_normal((l, m)))
    X = Z @ Pi + DataMat(rng.standard_normal((n, m)))
    s, pval = cragg_donald(X, Z)
    assert np.isfinite(s) and 0.0 <= float(pval) <= 1.0


if __name__ == '__main__':
    pytest.main([__file__])
