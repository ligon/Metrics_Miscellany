import pandas as pd
from scipy import stats
from metrics_miscellany import tests
import numpy as np


def main():

    labels = ["a", "b"]
    D = pd.DataFrame([[2, 1], [2, 2]], index=labels, columns=labels)
    D.index.name = "Variable"
    D.columns.name = "Variable"

    V = D.T @ D

    b = pd.DataFrame(stats.multivariate_normal(cov=V).rvs(), index=labels)
    b.index.name = "Variable"

    return tests.chi2_test(b, V, "Variable in ['a']")


def test_chi2():
    p = []
    m = 1000
    for _ in range(m):
        p.append(main()[1])

    p = pd.Series([x[0][0] for x in p]).squeeze()

    X = np.linspace(0.05, 0.95, 10)
    assert np.linalg.norm(p.quantile(X) - X) / len(X) < 1e-1


def _as_scalar(x):
    """chi2_test returns (chi2, p) as 1x1 ndarrays / DataFrames in some
    code paths; extract a plain float for comparison."""
    return float(np.asarray(x).reshape(-1)[0])


def test_chi2_test_var_selection_and_R_together():
    """When =var_selection= and =R= are both supplied, =R= is now
    auto-restricted to the selected subset; the previous code raised
    =ValueError: matrices are not aligned= because =R.T @ b= was
    formed against the *unrestricted* coefficient vector while
    =myV= had already been sliced down.
    """
    labels = ["a", "b", "c"]
    V = pd.DataFrame(np.eye(3), index=labels, columns=labels)
    np.random.seed(0)
    b = pd.DataFrame(
        stats.multivariate_normal(cov=V).rvs(), index=labels, columns=["Coefficients"]
    )

    # R indexed over the full label set, but the caller asks to test
    # only the (a, b) subset.  Should run, not raise.
    R = pd.DataFrame(np.eye(3)[:, :2], index=labels, columns=["c1", "c2"])
    chi2_stat, p = tests.chi2_test(b, V, var_selection=["a", "b"], R=R)
    assert 0.0 <= _as_scalar(p) <= 1.0
    assert _as_scalar(chi2_stat) >= 0.0


def test_chi2_test_var_selection_and_identity_R_match_selection_only():
    """An identity =R= over the selected subset must produce the same
    test as supplying =var_selection= alone."""
    labels = ["a", "b", "c"]
    V = pd.DataFrame(np.eye(3), index=labels, columns=labels)
    np.random.seed(1)
    b = pd.DataFrame(
        stats.multivariate_normal(cov=V).rvs(), index=labels, columns=["Coefficients"]
    )

    chi2_a, p_a = tests.chi2_test(b, V, var_selection=["a", "b"])
    R_id = pd.DataFrame(np.eye(2), index=["a", "b"], columns=["c1", "c2"])
    chi2_b, p_b = tests.chi2_test(b, V, var_selection=["a", "b"], R=R_id)
    np.testing.assert_allclose(_as_scalar(chi2_a), _as_scalar(chi2_b), rtol=1e-10)
    np.testing.assert_allclose(_as_scalar(p_a), _as_scalar(p_b), rtol=1e-10)


if __name__ == "__main__":
    test_chi2()
    test_chi2_test_var_selection_and_R_together()
    test_chi2_test_var_selection_and_identity_R_match_selection_only()
