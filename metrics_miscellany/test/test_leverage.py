import numpy as np
import pandas as pd
import pytest

from metrics_miscellany.utils import leverage


def _projection_diagonal(X):
    """diag(X X^+), computed independently of utils.leverage."""
    Xv = np.asarray(X, dtype=float)
    return np.diag(Xv @ np.linalg.pinv(Xv))


def _frame(a):
    a = np.asarray(a, dtype=float)
    return pd.DataFrame(
        a,
        index=[f"r{i}" for i in range(a.shape[0])],
        columns=[f"c{j}" for j in range(a.shape[1])],
    )


def test_leverage_matches_projection_diagonal():
    X = _frame(np.random.default_rng(0).standard_normal((8, 3)))
    assert np.allclose(leverage(X), _projection_diagonal(X))


def test_leverage_sums_to_rank_not_column_count():
    """Rank-deficient X: sum(h) is rank(X), not X.shape[1].

    The QR form -- (Q**2).sum(axis=1) -- returns k here, because
    np.linalg.qr still yields k orthonormal columns when X is rank
    deficient even though they no longer span col(X).
    """
    A = np.random.default_rng(1).standard_normal((8, 3))
    A[:, 2] = A[:, 0]  # rank 2, three columns
    X = _frame(A)

    assert np.linalg.matrix_rank(X) == 2
    assert np.isclose(float(np.sum(leverage(X))), 2.0)
    assert np.allclose(leverage(X), _projection_diagonal(X))


def test_leverage_accepts_wide_X():
    """n < k must give an answer, not an assertion."""
    X = _frame(np.random.default_rng(2).standard_normal((3, 5)))
    h = leverage(X)
    assert np.allclose(h, 1.0)  # rows of a wide X are fully levered
    assert np.allclose(h, _projection_diagonal(X))


def test_leverage_is_bounded_and_labelled():
    X = _frame(np.random.default_rng(3).standard_normal((10, 4)))
    h = leverage(X)
    assert list(h.index) == list(X.index)
    assert ((h >= -1e-12) & (h <= 1 + 1e-12)).all()


@pytest.mark.parametrize("n,k", [(6, 2), (12, 5), (4, 4)])
def test_leverage_sums_to_rank_full_rank_cases(n, k):
    X = _frame(np.random.default_rng(n * k).standard_normal((n, k)))
    assert np.isclose(float(np.sum(leverage(X))), float(np.linalg.matrix_rank(X)))


if __name__ == "__main__":
    test_leverage_matches_projection_diagonal()
    test_leverage_sums_to_rank_not_column_count()
    print("smoke OK")
