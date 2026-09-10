import pandas as pd
from metrics_miscellany.estimators import tsls, ols, restricted_tsls
from datamat import DataMat, DataVec
import numpy as np

def test_tsls(N=500000,tol=1e-2):

    z = DataMat({'z':np.random.standard_normal((N,))})
    u = DataVec(np.random.standard_normal((N,)))
    x = DataMat({'x':z.squeeze() + u})

    x['Constant'] = 1
    z['Constant'] = 1

    beta = DataMat({'Coefficients':[1,0]},index=['x','Constant'])

    y = (x@beta).squeeze() + u

    b,Omega,V = restricted_tsls(y,x,Z=z)
    b_,V_ = tsls(x,y,z)
    #b,V = ols(x,y)

    assert np.allclose(b,beta.squeeze(),atol=tol)

    #return b,V,b_,V_


def _make_iv_dgp(N, seed):
    rng = np.random.default_rng(seed)
    z = DataMat({'z': rng.standard_normal(N)})
    u = DataVec(rng.standard_normal(N))
    x = DataMat({'x': z.squeeze() + u})
    x['Constant'] = 1
    z['Constant'] = 1
    beta = DataMat({'Coefficients': [1., 0.]}, index=['x', 'Constant'])
    y = (x @ beta).squeeze() + u
    return y, x, z


def test_restricted_tsls_satisfying_constraint():
    """When the population satisfies the restriction (Constant = 0), the
    restricted estimator should recover the true coefficient, hold the
    restriction exactly, and report (essentially) zero variance for the
    restricted direction."""
    y, x, z = _make_iv_dgp(N=200_000, seed=0)

    R = DataMat([[0., 1.]], columns=['x', 'Constant'], index=['c0'])
    r = DataVec([0.], index=['c0'])
    b, lm, Omega, V = restricted_tsls(y, x, R=R, r=r, Z=z)

    # The restriction is satisfied exactly:
    np.testing.assert_allclose(b['Constant'], 0., atol=1e-12)
    # The free coefficient recovers the truth (sampling noise only):
    np.testing.assert_allclose(b['x'], 1., atol=2e-2)
    # Output shapes / labels survive the restricted branch (DataMat may
    # wrap flat labels as 1-tuples; flatten before comparing).
    def _flat(idx):
        return [t[0] if isinstance(t, tuple) and len(t) == 1 else t
                for t in idx]
    assert _flat(b.index) == ['x', 'Constant']
    assert _flat(lm.index) == ['c0']
    assert _flat(V.index) == ['x', 'Constant']
    assert _flat(V.columns) == ['x', 'Constant']
    # Variance along the restricted direction is zero (up to FP noise):
    np.testing.assert_allclose(V.loc['Constant', 'Constant'], 0., atol=1e-12)


def test_restricted_tsls_binding_constraint():
    """A non-trivial restriction must bind: the restricted coefficient
    equals the constraint value exactly."""
    y, x, z = _make_iv_dgp(N=1_000, seed=1)

    R = DataMat([[1., 0.]], columns=['x', 'Constant'], index=['c0'])
    r = DataVec([0.5], index=['c0'])
    b, lm, Omega, V = restricted_tsls(y, x, R=R, r=r, Z=z)

    np.testing.assert_allclose(b['x'], 0.5, atol=1e-12)


def test_restricted_tsls_matches_unrestricted_when_R_None():
    """The R=None path must continue to agree with tsls()."""
    y, x, z = _make_iv_dgp(N=10_000, seed=2)

    b1, Omega1, V1 = restricted_tsls(y, x, Z=z)
    b2, V2 = tsls(x, y, z)

    np.testing.assert_allclose(b1.values, b2.values, atol=1e-12)


if __name__=='__main__':
    test_tsls()
    test_restricted_tsls_satisfying_constraint()
    test_restricted_tsls_binding_constraint()
    test_restricted_tsls_matches_unrestricted_when_R_None()
