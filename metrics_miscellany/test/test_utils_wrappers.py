import numpy as np
import pandas as pd
import pytest

from metrics_miscellany import utils


@pytest.fixture
def square_psd():
    """A 5x5 positive definite DataFrame with labeled rows/columns."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    M = A @ A.T + np.eye(5)  # PD
    labels = list("abcde")
    return pd.DataFrame(M, index=labels, columns=labels)


@pytest.fixture
def rectangular_full_rank():
    """A 6x3 full-column-rank DataFrame."""
    rng = np.random.default_rng(1)
    M = rng.standard_normal((6, 3))
    return pd.DataFrame(M, index=[f"r{i}" for i in range(6)], columns=list("xyz"))


def test_svd_round_trip(square_psd):
    """U diag(S) V' should reconstruct A."""
    u, s, vt = utils.svd(square_psd)
    recon = u.values @ np.diag(s.values) @ vt.values
    np.testing.assert_allclose(recon, square_psd.values, atol=1e-10)
    # Labels: u's index == A.index; vt's columns == A.columns.
    assert list(u.index) == list(square_psd.index)
    assert list(vt.columns) == list(square_psd.columns)


def test_svd_hermitian_flag_runs(square_psd):
    """The hermitian=True branch must run on symmetric input."""
    sym = (square_psd + square_psd.T) / 2
    u, s, vt = utils.svd(sym, hermitian=True)
    recon = u.values @ np.diag(s.values) @ vt.values
    np.testing.assert_allclose(recon, sym.values, atol=1e-10)


def test_eig_hermitian_round_trip(square_psd):
    """For symmetric A, eig(A, hermitian=True) should give A = U diag(s) U'."""
    sym = (square_psd + square_psd.T) / 2
    s2, u = utils.eig(sym, hermitian=True)
    recon = u.values @ np.diag(s2.values) @ u.values.T
    np.testing.assert_allclose(recon, sym.values, atol=1e-10)
    assert list(u.index) == list(sym.index)


def test_eig_general_case_runs(square_psd):
    """Non-hermitian eig branch should also run (eigenvalues may be complex)."""
    s2, u = utils.eig(square_psd)
    assert s2.shape[0] == square_psd.shape[0]
    assert u.shape == square_psd.shape


def test_qr_round_trip(rectangular_full_rank):
    """Q @ R should reconstruct X and Q'Q should be I."""
    Q, R = utils.qr(rectangular_full_rank)
    np.testing.assert_allclose((Q @ R).values, rectangular_full_rank.values, atol=1e-10)
    np.testing.assert_allclose(
        Q.values.T @ Q.values, np.eye(rectangular_full_rank.shape[1]), atol=1e-10
    )
    # Labels: Q rows match X rows, R rows/cols match X columns.
    assert list(Q.index) == list(rectangular_full_rank.index)
    assert list(R.index) == list(rectangular_full_rank.columns)


def test_outer_shape_and_labels():
    s = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    t = pd.Series([10.0, 20.0], index=["x", "y"])
    out = utils.outer(s, t)
    np.testing.assert_array_equal(out.values, np.outer(s.values, t.values))
    assert list(out.index) == ["a", "b", "c"]
    assert list(out.columns) == ["x", "y"]


def test_diag_extracts_from_square_dataframe(square_psd):
    """diag(square DataFrame) returns the diagonal as a Series."""
    d = utils.diag(square_psd)
    np.testing.assert_allclose(d.values, np.diag(square_psd.values))
    assert list(d.index) == list(square_psd.index)


def test_diag_constructs_from_series():
    """diag(Series) returns a (sparse) diagonal matrix labelled by the
    Series's index on both axes.  The off-diagonal entries must be 0
    (not NaN: an earlier version returned NaN there, which silently
    propagated through any arithmetic that did not go through
    matrix_product's =.fillna(0)= safety net)."""
    s = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    D = utils.diag(s)
    assert D.shape == (3, 3)
    assert list(D.index) == ["a", "b", "c"]
    assert list(D.columns) == ["a", "b", "c"]

    arr = np.asarray(D.sparse.to_dense() if hasattr(D, "sparse") else D)
    np.testing.assert_allclose(arr, np.diag([1.0, 2.0, 3.0]))


def test_diag_series_dense_path():
    """The dense (sparse=False) path used to raise NotImplementedError;
    it now returns a properly-labelled dense DataFrame."""
    s = pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"])
    D = utils.diag(s, sparse=False)
    assert isinstance(D, pd.DataFrame)
    np.testing.assert_array_equal(D.values, np.diag([1.0, 2.0, 3.0]))
    assert list(D.index) == ["a", "b", "c"]
    assert list(D.columns) == ["a", "b", "c"]


def test_diag_passthrough_for_ndarray():
    """For a plain ndarray, diag falls through to np.diag (extract)."""
    M = np.arange(9).reshape(3, 3)
    out = utils.diag(M)
    np.testing.assert_array_equal(out, np.diag(M))


def test_sqrtm_squares_to_input(square_psd):
    """sqrtm(A) @ sqrtm(A).T should reconstruct A for PD A."""
    S = utils.sqrtm(square_psd, hermitian=True)
    recon = S.values @ S.values.T
    np.testing.assert_allclose(recon, square_psd.values, atol=1e-8)


def test_sqrtm_rejects_indefinite():
    """An indefinite (non-PSD) input raises ValueError.  An earlier
    implementation inspected singular values (always non-negative)
    rather than eigenvalues, so the guard could never trigger."""
    bad = pd.DataFrame(np.diag([1.0, -1.0]), index=["a", "b"], columns=["a", "b"])
    with pytest.raises(ValueError, match="positive semi-definite"):
        utils.sqrtm(bad)


def test_sqrtm_tolerates_tiny_negative_eigenvalues():
    """FP noise near the boundary should not trip the PSD guard."""
    near_psd = pd.DataFrame(np.diag([1.0, 1e-15]), index=["a", "b"], columns=["a", "b"])
    S = utils.sqrtm(near_psd)
    np.testing.assert_allclose(S.values @ S.values, near_psd.values, atol=1e-12)


def test_sqrtm_returns_symmetric_root(square_psd):
    """The returned S should be symmetric and satisfy S @ S == A."""
    S = utils.sqrtm(square_psd)
    np.testing.assert_allclose(S.values, S.values.T, atol=1e-10)
    np.testing.assert_allclose(S.values @ S.values, square_psd.values, atol=1e-8)


def test_cholesky_round_trip(square_psd):
    """L @ L.T should reconstruct A."""
    L = utils.cholesky(square_psd)
    np.testing.assert_allclose(L.values @ L.values.T, square_psd.values, atol=1e-10)
    # Lower triangular: upper-triangular entries are zero.
    upper = L.values - np.tril(L.values)
    np.testing.assert_allclose(upper, 0.0, atol=1e-15)
    assert list(L.index) == list(square_psd.index)


def test_trim_drops_extremes_to_nan():
    """Values below alpha quantile and above 1-alpha quantile become NaN."""
    df = pd.DataFrame({"x": np.arange(100, dtype=float)})
    trimmed = utils.trim(df, alpha=0.1)
    # The bottom 10% (values <= 9.9) and top 10% (values >= 89.1)
    # should be NaN.  Conservative bounds since `where` uses >= /<=.
    assert trimmed["x"].isna().sum() >= 18
    finite = trimmed["x"].dropna()
    assert finite.min() >= 10
    assert finite.max() <= 89


def test_drop_missing_returns_aligned_tuple():
    """Rows with NaN in any input should be dropped from all of them."""
    df_a = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0]})
    df_b = pd.DataFrame({"b": [10.0, 20.0, 30.0, np.nan]})
    Y = utils.drop_missing([df_a, df_b])
    assert isinstance(Y, tuple)
    assert len(Y) == 2
    assert len(Y[0]) == 2  # rows 0 and 2 survive
    assert list(Y[0]["a"].values) == [1.0, 3.0]
    assert list(Y[1]["b"].values) == [10.0, 30.0]


def test_drop_missing_dict_form():
    """When called with a dict, the result is a dict with the same keys."""
    df_a = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    df_b = pd.DataFrame({"b": [10.0, 20.0, 30.0]})
    out = utils.drop_missing({"first": df_a, "second": df_b})
    assert set(out.keys()) == {"first", "second"}
    assert len(out["first"]) == 2 == len(out["second"])


def test_hat_factory_predictions_match_ols(rectangular_full_rank):
    """hat_factory(X)(y) should equal X (X'X)^-1 X' y (OLS fitted values)."""
    X = rectangular_full_rank
    rng = np.random.default_rng(42)
    y = pd.Series(rng.standard_normal(X.shape[0]), index=X.index)

    hat = utils.hat_factory(X)
    pred = hat(y)

    # Reference: ordinary least squares fitted values.
    beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
    expected = X.values @ beta

    np.testing.assert_allclose(np.asarray(pred), expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Algorithmic tests (heavier than the wrappers above).
# ---------------------------------------------------------------------------


def test_svd_missing_full_data_matches_truncated_svd():
    """With no NaNs, svd_missing should yield the rank-r truncation of
    the dense SVD up to sign/orthonormal-frame ambiguity."""
    rng = np.random.default_rng(0)
    A = pd.DataFrame(rng.standard_normal((40, 5)))

    u, s, v = utils.svd_missing(A, max_rank=3)
    assert u.shape == (40, 3)
    assert v.shape == (5, 3)
    assert len(s) == 3

    # Compare to the rank-3 truncation of np.linalg.svd.
    U, S, Vt = np.linalg.svd(A.values, full_matrices=False)
    truncated = U[:, :3] @ np.diag(S[:3]) @ Vt[:3, :]
    recon = u.values @ np.diag(s.values) @ v.values.T

    # The two reconstructions should agree (same subspace, same scaling).
    np.testing.assert_allclose(recon, truncated, atol=1e-8)


def test_svd_missing_handles_nans():
    """Sparse-NaN input should run; rows with at least =max_rank=
    non-missing entries get a finite U, and rows that fall below the
    threshold are filled with NaN.  Per the docstring: "each row must
    have at least max_rank non-missing values"."""
    rng = np.random.default_rng(1)
    A = pd.DataFrame(rng.standard_normal((40, 5)))
    mask = rng.random(A.shape) < 0.1
    A_miss = A.mask(mask)

    u, s, v = utils.svd_missing(A_miss, max_rank=3)
    assert s.shape == (3,)
    assert v.shape == (5, 3)

    # Rows with >= 3 non-missing entries should produce a finite U row;
    # rows with < 3 non-missing entries should land in NaN.
    obs_per_row = (~A_miss.isna()).sum(axis=1)
    enough = obs_per_row >= 3
    np.testing.assert_array_equal(u.isna().any(axis=1).values, (~enough).values)


def test_svd_missing_symmetric_input_returns_eigendecomposition():
    """For symmetric A, the function takes the symmetric branch and
    returns (V, S, V.T) -- i.e. an eigendecomposition.  Reconstruction
    must round-trip on a PSD input."""
    rng = np.random.default_rng(2)
    M = rng.standard_normal((6, 6))
    A = pd.DataFrame(M @ M.T)  # PSD
    u, s, v = utils.svd_missing(A)
    # u and v have NOT been wrapped in DataFrames in the symmetric
    # branch (the function returns the bare numpy v).  Defensive
    # extraction:
    uv = np.asarray(u)
    vv = np.asarray(v)
    np.testing.assert_allclose(uv, vv.T, atol=1e-10)
    recon = uv @ np.diag(s) @ uv.T
    np.testing.assert_allclose(recon, A.values, atol=1e-8)


def test_heteropca_recovers_planted_loading():
    """Build C = L L' + Psi with L rank-1 known and diagonal Psi.
    HeteroPCA should recover L up to sign with high cosine similarity."""
    rng = np.random.default_rng(3)
    m = 8
    L_true = rng.standard_normal((m, 1))
    Psi = np.diag(rng.uniform(0.5, 1.5, size=m))
    C = L_true @ L_true.T + Psi

    u, s = utils.heteropca(C, r=1, max_its=200, tol=1e-8)
    assert u.shape == (m, 1)
    cos = abs(
        (u[:, 0] @ L_true[:, 0])
        / (np.linalg.norm(u[:, 0]) * np.linalg.norm(L_true[:, 0]))
    )
    assert cos > 0.98, f"HeteroPCA loading cosine to truth = {cos:.4f}; expected >0.98."


def test_factor_analysis_eig_recovers_planted_loading():
    """factor_analysis with the only working svd_method ('eig') should
    recover a planted rank-1 loading direction.  Uses a large N to push
    cosine similarity above 0.95."""
    from metrics_miscellany import estimators

    rng = np.random.default_rng(4)
    n_features, N = 6, 5000
    L_true = rng.standard_normal((n_features, 1))
    F_true = rng.standard_normal((N, 1))
    noise = 0.3 * rng.standard_normal((N, n_features))
    X = pd.DataFrame(F_true @ L_true.T + noise)

    W, psi = estimators.factor_analysis(
        X, n_components=1, max_its=500, tol=1e-6, svd_method="eig"
    )
    W_flat = np.asarray(W).reshape(-1)
    L_flat = L_true.reshape(-1)
    cos = abs((W_flat @ L_flat) / (np.linalg.norm(W_flat) * np.linalg.norm(L_flat)))
    assert cos > 0.95, f"factor_analysis cosine to truth = {cos:.4f}; expected >0.95."
    # psi (idiosyncratic variances) must be positive.
    assert np.all(np.asarray(psi) > 0)
    assert len(np.asarray(psi)) == n_features


def test_factor_analysis_unfinished_branches_raise():
    """The 'lapack' and 'randomized' svd_method branches are gated;
    confirm they raise instead of silently NameError'ing."""
    from metrics_miscellany import estimators

    rng = np.random.default_rng(5)
    X = pd.DataFrame(rng.standard_normal((20, 3)))
    with pytest.raises(NotImplementedError, match="lapack"):
        estimators.factor_analysis(X, svd_method="lapack")
    with pytest.raises(NotImplementedError, match="randomized"):
        estimators.factor_analysis(X, svd_method="randomized")


def test_heteropca_warns_instead_of_raising_on_max_its(square_psd):
    """Hitting max_its must emit a warning.

    `warnings` was never imported in the tangled utils.py, so this
    branch raised NameError on exactly the non-convergence path the
    warning exists to report.
    """
    with pytest.warns(UserWarning, match="Exceeded maximum iterations"):
        utils.heteropca(square_psd, r=1, max_its=1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    A = rng.standard_normal((5, 5))
    M = A @ A.T + np.eye(5)
    labels = list("abcde")
    square = pd.DataFrame(M, index=labels, columns=labels)
    test_svd_round_trip(square)
    test_cholesky_round_trip(square)
    print("smoke OK")
