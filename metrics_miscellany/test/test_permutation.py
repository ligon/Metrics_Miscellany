import numpy as np
import pandas as pd
from metrics_miscellany import random as mm_random


def test_permutation_invariant_across_fixed_level():
    """When the values are constant across the fixed level =t=, the
    permuted output must also be constant across =t=."""
    np.random.seed(0)
    T = pd.Series(np.random.rand(10) > 0.5)
    df = pd.DataFrame({"a": T, "b": T}).stack()
    df = df + 0
    df.index.names = ["i", "t"]

    p = mm_random.permutation(df, permute_levels=["i"])

    assert np.all(p.unstack("t").std(axis=1) == 0)


def test_permutation_actually_permutes():
    """The within-fixed-level reassignment must be a real permutation:
    the multiset of values along the permuted level is preserved, the
    pairing across fixed levels stays intact, and a non-identity
    permutation occurs with high probability for n=50 distinct values.
    """
    n = 50
    T = np.arange(n, dtype=float)
    df = pd.DataFrame({"a": T, "b": T + 1000.0}).stack().to_frame("v")
    df.index.names = ["i", "t"]

    np.random.seed(2)
    p = mm_random.permutation(df, permute_levels=["i"])

    mapping_a = p.xs("a", level="t")["v"].values
    mapping_b = p.xs("b", level="t")["v"].values - 1000.0

    # The block across fixed levels stays paired:
    np.testing.assert_array_equal(mapping_a, mapping_b)
    # And the assignment is a permutation of arange(n):
    assert sorted(mapping_a.tolist()) == list(range(n))
    # Identity has probability 1/n!, so with this seed we expect non-identity.
    assert not np.array_equal(mapping_a, np.arange(n))


def test_permutation_full_axis():
    """With permute_levels=None we get a full permutation of rows."""
    np.random.seed(3)
    df = pd.DataFrame({"v": np.arange(20.0)})
    p = mm_random.permutation(df)
    assert sorted(p["v"].tolist()) == list(range(20))
    # And it actually shuffles (with overwhelming probability).
    assert not np.array_equal(p["v"].values, np.arange(20.0))


if __name__ == "__main__":
    test_permutation_invariant_across_fixed_level()
    test_permutation_actually_permutes()
    test_permutation_full_axis()
