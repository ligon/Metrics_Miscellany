"""Smoke tests for =randomization_inference=.

The previous file ran the simulation at import time (with matplotlib),
which pytest picked up during collection rather than as a real test.
Each scenario is now a proper =test_*= function with a seeded RNG and
a numeric assertion about the returned p-value.
"""
import numpy as np
import pandas as pd
import pytest
import scipy.stats.distributions as dists
from metrics_miscellany import tests as mm_tests


def _build_design(seed, n=400, p=0.5):
    """Construct a small (n, X, Y_no_effect, Y_with_effect) fixture."""
    rng = np.random.default_rng(seed)
    C = pd.DataFrame({'Female': rng.binomial(1, p, size=n)})
    C['Male'] = 1 - C['Female']
    delta = pd.Series({"Female": 1.0, "Male": 0.5})

    T1 = pd.Series(rng.standard_normal(n), name='Treatment1')
    T2 = pd.Series(rng.standard_normal(n), name='Treatment2')

    TC = C.multiply(T1, axis=0)
    TC.columns = ['TxFemale', 'TxMale']

    X = pd.concat([T1, T2, C, TC], axis=1).iloc[:, :-1]
    dC = C @ delta
    eps = pd.Series(rng.standard_normal(n), name='epsilon')

    Y_null = (dC + eps).rename('outcome')
    Y_alt = (T1 + T2 + dC + eps).rename('outcome')
    return X, Y_null, Y_alt


@pytest.mark.slow
def test_randomization_inference_returns_valid_pvalue():
    """No treatment effect -> p in [0, 1]; smoke test only."""
    X, Y_null, _ = _build_design(seed=0)
    np.random.seed(0)
    p = float(mm_tests.randomization_inference(
        ['Treatment1'], X, Y_null, VERBOSE=False))
    assert 0.0 <= p <= 1.0


@pytest.mark.slow
def test_randomization_inference_rejects_strong_effect():
    """A unit treatment effect at n=400 should reject at the 5% level."""
    X, _, Y_alt = _build_design(seed=1)
    np.random.seed(1)
    p = float(mm_tests.randomization_inference(
        ['Treatment1'], X, Y_alt, VERBOSE=False))
    assert p < 0.05


@pytest.mark.slow
def test_randomization_inference_with_linear_restriction():
    """Smoke-test the R-supplied path (Treatment1 == Treatment2)."""
    X, _, Y_alt = _build_design(seed=2)
    R = pd.DataFrame({'Coefficients': [1, -1]},
                     index=['Treatment1', 'Treatment2'])
    np.random.seed(2)
    p = float(mm_tests.randomization_inference(
        ['Treatment1', 'Treatment2'],
        X.drop('TxFemale', axis=1), Y_alt, R=R, VERBOSE=False))
    assert 0.0 <= p <= 1.0


if __name__ == '__main__':
    test_randomization_inference_returns_valid_pvalue()
    test_randomization_inference_rejects_strong_effect()
    test_randomization_inference_with_linear_restriction()
