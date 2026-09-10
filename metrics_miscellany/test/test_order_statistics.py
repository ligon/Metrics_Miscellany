import numpy as np
import pandas as pd
from metrics_miscellany.order_statistics import quantile_confidence_intervals


def test_qci_returns_interval_and_coverage():
    """Smoke test: returns ((lo, hi), coverage) with lo <= hi and the
    achieved coverage at least the requested minimum."""
    rng = np.random.default_rng(0)
    x = pd.Series(rng.standard_normal(101))  # odd n -> median sits on a point
    (lo, hi), coverage = quantile_confidence_intervals(
        x, q=0.5, minimum_coverage=0.95)
    assert lo <= hi
    assert coverage >= 0.95


def test_qci_endpoints_are_order_statistics():
    """The returned endpoints must be order statistics of the input
    sample (not interpolated)."""
    rng = np.random.default_rng(1)
    x = pd.Series(rng.standard_normal(200))
    (lo, hi), _ = quantile_confidence_intervals(x, q=0.5)
    sorted_vals = set(x.sort_values().values)
    assert lo in sorted_vals
    assert hi in sorted_vals


def test_qci_brackets_population_median():
    """For a moderately large iid sample from a symmetric distribution
    the binomial CI should almost always bracket the population
    median (0).  We assert this for one fixed seed -- failure would
    signal a real bug, not Monte Carlo noise."""
    rng = np.random.default_rng(2)
    x = pd.Series(rng.standard_normal(500))
    (lo, hi), _ = quantile_confidence_intervals(
        x, q=0.5, minimum_coverage=0.99)
    assert lo <= 0.0 <= hi


def test_qci_quartile_call_still_works():
    """The default q=0.5 is the median; q=0.25 should also produce a
    valid interval."""
    rng = np.random.default_rng(3)
    x = pd.Series(rng.standard_normal(200))
    (lo, hi), coverage = quantile_confidence_intervals(x, q=0.25)
    assert lo <= hi
    assert coverage >= 0.95


if __name__ == '__main__':
    test_qci_returns_interval_and_coverage()
    test_qci_endpoints_are_order_statistics()
    test_qci_brackets_population_median()
    test_qci_quartile_call_still_works()
