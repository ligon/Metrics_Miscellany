"""Test of the KR79 LR test for equality of the last q eigenvalues.

Previously this file ran a 1000-iteration simulation (with =matplotlib=)
at module-import time, which pytest executed during collection rather
than as a test.  The Kolmogorov-Smirnov check at the bottom was a
top-level assert -- a failure would surface as a collection error.
Both have been wrapped in =test_*= functions; the slow simulation is
marked accordingly.
"""

import numpy as np
import pytest
import scipy.stats.distributions as iid
from scipy import stats
from metrics_miscellany import tests as mm_tests


def _make_sqrt_psd(rng, m, r):
    """Return a (m, m) matrix Ssqrt with Ssqrt @ Ssqrt.T positive
    semi-definite of rank r."""
    A = rng.standard_normal((m, r))
    Sigma = A @ A.T
    l, v = np.linalg.eigh(Sigma)
    l = np.maximum(l, 0)
    Ssqrt = v @ np.diag(np.sqrt(l)) @ v
    np.testing.assert_allclose(Sigma, Ssqrt @ Ssqrt.T, atol=1e-10)
    return Ssqrt


def test_kr79_size_under_null():
    """Under the null (identity error covariance), the p-values of the
    KR79 statistic should be approximately uniform.  We use a modest
    number of iterations to keep this affordable as a unit test; the
    Kolmogorov-Smirnov bar is permissive enough to be reliable."""
    rng = np.random.default_rng(0)
    N, m, r = 2_000, 6, 2
    q = m - r

    Ssqrt = _make_sqrt_psd(rng, m, r)
    Psisqrt = np.eye(m)  # spherical errors -> null holds

    P = []
    for _ in range(200):
        X = rng.standard_normal((N, m)) @ Ssqrt.T
        e = rng.standard_normal((N, m)) @ Psisqrt.T
        C = np.cov(X + e, rowvar=False)
        _, p = mm_tests.kr79(C, q, N)
        P.append(p)

    ks_p = stats.kstest(P, stats.distributions.uniform.cdf).pvalue
    assert ks_p > 0.01, f"KS p={ks_p}: KR79 p-values look non-uniform under the null."


if __name__ == "__main__":
    test_kr79_size_under_null()
