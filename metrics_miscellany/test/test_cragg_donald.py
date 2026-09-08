"""Check size of cragg_donald reduced rank test.
"""
import datamat as dm
from metrics_miscellany.tests import cragg_donald
from scipy.stats.distributions import norm
import numpy as np
import pytest


def _rank_deficient_pi(rng, l, m, rank):
    """Random (l, m) matrix of exactly the requested rank."""
    A = rng.standard_normal((l, rank))
    B = rng.standard_normal((rank, m))
    return A @ B


def test_cragg_donald_rejects_under_full_rank():
    """Under the alternative (full-rank Pi with strong signal), the
    test should reject at every conventional level."""
    rng = np.random.default_rng(0)
    n, m, l = 1_000, 3, 5
    Pi = rng.standard_normal((l, m))
    Z = dm.DataMat(rng.standard_normal((n, l)))
    U = dm.DataMat(rng.standard_normal((n, m)))
    X = Z @ Pi + U

    s, p = cragg_donald(X, Z)
    assert np.isfinite(s)
    assert 0.0 <= float(p) <= 1.0
    assert float(p) < 1e-3, (
        f"With a strong full-rank Pi (rank=m=3, l=5, n=1000) the test "
        f"should reject decisively; got p={p}.  This catches the previous "
        f"=df = n - X.shape[1] + 1= bug, which left the test never able to "
        f"reject.")


@pytest.mark.slow
def test_cragg_donald_size_under_null():
    """Under the null (rank(Pi) = m - 1) the p-values should be
    (approximately) uniform.  Permissive KS bound to avoid CI flakiness."""
    from scipy.stats import kstest
    rng = np.random.default_rng(1)
    n, m, l = 1_000, 3, 5
    Z = dm.DataMat(rng.standard_normal((n, l)))

    Ps = []
    for _ in range(300):
        Pi = _rank_deficient_pi(rng, l, m, rank=m - 1)
        U = dm.DataMat(rng.standard_normal((n, m)))
        X = Z @ Pi + U
        _, p = cragg_donald(X, Z)
        Ps.append(float(p))
    avg = np.mean(Ps)
    ks_p = kstest(Ps, 'uniform').pvalue
    assert abs(avg - 0.5) < 0.1, (
        f"Average p-value {avg:.3f} suspiciously far from 0.5.")
    assert ks_p > 0.01, (
        f"KS p={ks_p:.4f}: under H_0 the p-values look non-uniform.")


def test_cragg_donald_asserts_overid():
    """l < m is under-identified and the test is undefined."""
    rng = np.random.default_rng(2)
    n, m, l = 100, 4, 2  # only 2 instruments for 4 endogenous regressors
    Pi = rng.standard_normal((l, m))
    Z = dm.DataMat(rng.standard_normal((n, l)))
    U = dm.DataMat(rng.standard_normal((n, m)))
    X = Z @ Pi + U
    with pytest.raises(AssertionError, match="at least as many instruments"):
        cragg_donald(X, Z)


if __name__ == '__main__':
    test_cragg_donald_rejects_under_full_rank()
    test_cragg_donald_size_under_null()
    test_cragg_donald_asserts_overid()
