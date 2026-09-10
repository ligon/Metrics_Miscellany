import numpy as np
from scipy import sparse as scipy_sparse
import pandas as pd


def inv(A):
    """Inverse of square pandas DataFrame.

    Type preserving: a DataMat in yields a DataMat out, so the result still
    carries DataMat's methods (.eig(), .inv(), ...).  Estimators here call
    this rather than DataMat's own .inv so that they do not depend on
    whether that is spelled as a property or a method (GH #6).

    Labelled like pinv, with the index and columns of A exchanged: if A maps
    the space labelled by A.columns to the one labelled by A.index, its
    inverse maps back.  For the square, equally-labelled matrices this is
    used on the two conventions coincide.
    """
    if np.isscalar(A):
        A = pd.DataFrame(np.array([[A]]))

    B = np.linalg.inv(A)
    return A._constructor(B, index=A.columns, columns=A.index)


def pinv(A):
    """Moore-Penrose pseudo-inverse of A.

    >>> A = pd.DataFrame([[1,2,3],[4,5,6]])
    >>> np.allclose(A@pinv(A),np.eye(2))
    True
    """
    if np.isscalar(A):
        A = pd.DataFrame(np.array([[A]]))

    B = np.linalg.pinv(A)
    return A._constructor(B, columns=A.index, index=A.columns)


def leverage(X):
    """Leverage of matrix X; i.e., diagonal of the projection matrix.

    Uses the pseudo-inverse, so this is the diagonal of the projection
    onto col(X) whatever the rank of X: the leverages sum to rank(X),
    not to X.shape[1].  See the Leverage section for why the QR
    shortcut is not used here.
    """
    return (X * pinv(X).T).sum(axis=1)


def svd(A, hermitian=False):
    """Singular value composition into U@S.dg@V.T."""
    idx = A.index
    cols = A.columns
    u, s, vt = np.linalg.svd(
        A, compute_uv=True, full_matrices=False, hermitian=hermitian
    )
    u = pd.DataFrame(u, index=idx)
    vt = pd.DataFrame(vt, columns=cols)
    s = pd.Series(s)

    return u, s, vt


def eig(A, hermitian=False):
    """Singular value composition into U@S.dg@V.T."""
    idx = A.index
    cols = A.columns
    if hermitian:
        s2, u = np.linalg.eigh(A)
    else:
        s2, u = np.linalg.eig(A)

    s2 = np.flip(s2)
    u = np.fliplr(u)

    u = pd.DataFrame(u, index=idx, columns=cols)
    s2 = pd.Series(s2.squeeze())

    return s2, u


def diag(X, sparse=True):
    """Build or extract a diagonal matrix.

    * Square DataFrame/ndarray input: returns the diagonal as a
      pandas Series (DataFrame input) or 1-d ndarray (ndarray input).
    * 1-d Series input: returns a diagonal matrix whose rows and
      columns are labelled by =X.index=.  =sparse=True= (the default)
      keeps memory linear in =len(X)= by routing through
      =scipy.sparse=; =sparse=False= builds a dense DataFrame.

    The sparse path used to leave the off-diagonal entries as NaN --
    =pd.DataFrame.sparse.from_spmatrix= picks a fill value of NaN
    for float dtypes, and the source =scipy.sparse.diags= matrix
    has no entries off the diagonal -- which silently propagated
    through any arithmetic that did not go through =matrix_product=
    (the latter calls =.fillna(0)= on its inputs and so masked the
    bug).  We now =fillna(0)= the resulting frame and re-cast it to
    =SparseDtype(float64, fill_value=0)= so the off-diagonal is
    actually zero while the sparse memory benefit is preserved.
    """
    try:
        assert X.shape[0] == X.shape[1]
        d = pd.Series(np.diag(X), index=X.index)
    except IndexError:  # X is a Series-like 1-d
        if sparse:
            spmat = scipy_sparse.diags(X.values)
            d = pd.DataFrame.sparse.from_spmatrix(spmat, index=X.index, columns=X.index)
            d = d.fillna(0).astype(pd.SparseDtype("float64", fill_value=0.0))
        else:
            d = pd.DataFrame(np.diag(X.values), index=X.index, columns=X.index)
    except AttributeError:  # Not a pandas object
        d = np.diag(X)

    return d


def outer(S, T):
    """Outer product of two series (vectors) S & T."""
    return pd.DataFrame(np.outer(S, T), index=S.index, columns=T.index)


def matrix_product(X, Y, strict=False, fillmiss=True):
    """Compute matrix product X@Y, allowing for possibility of missing data.

    The "strict" flag if set requires that the names of levels of indices that vary for columns of X be in the intersection of names of levels of indices that vary for rows of Y.
    """

    if strict and not all(X.columns == Y.index):  # Columns and Indices don't match.
        X.columns = drop_vestigial_levels(X.columns)
        Y.index = drop_vestigial_levels(Y.index)

    if fillmiss:
        X = X.fillna(0)
        Y = Y.fillna(0)

    prod = np.dot(X, Y)  # .squeeze()

    if len(prod.shape) == 1 or prod.shape[1] == 1:
        out = pd.Series(prod.squeeze(), index=X.index)
    else:
        try:
            cols = Y.columns
        except AttributeError:
            cols = None
        out = pd.DataFrame(prod, index=X.index, columns=cols)

    return out


def self_inner(X, min_obs=None):
    """Compute inner product X.T@X, allowing for possibility of missing data."""
    n, m = X.shape

    if n < m:
        axis = 1
        N = m
    else:
        axis = 0
        N = n

    xbar = X.mean(axis=axis)

    if axis:
        C = (N - 1) * X.T.cov(min_periods=min_obs)
    else:
        C = (N - 1) * X.cov(min_periods=min_obs)

    return C + N * np.outer(xbar, xbar)


def _kron_axis(left, right):
    """Build the row or column labels of a Kronecker product.

    Handles flat Index x flat Index, MultiIndex x MultiIndex, and the
    mixed cases uniformly by promoting flat-Index entries to 1-tuples
    before taking the cartesian product.  Avoids ``(*i, *j)``, which
    silently unpacks string labels character-by-character and crashes
    on numeric labels.
    """

    def as_tuples(idx):
        if isinstance(idx, pd.MultiIndex):
            return list(idx)
        return [(v,) for v in idx]

    tuples = [(*i, *j) for i in as_tuples(left) for j in as_tuples(right)]
    names = list(left.names) + list(right.names)
    return pd.MultiIndex.from_tuples(tuples, names=names)


def kron(A, B, sparse=False):
    """Kronecker product A (x) B that preserves pandas axis labels.

    Each operand may be a DataFrame, a Series (treated as a column
    vector), or a NumPy array (1-d arrays are treated as column
    vectors).  The result's row index is the cartesian product
    A.index x B.index (with concatenated level names), and likewise
    for the columns; flat Indices are promoted to 1-level MultiIndex
    entries so the bookkeeping is uniform.

    If =sparse= is True the product is computed via
    =scipy.sparse.kron= and returned as a sparse DataFrame.
    """

    def to_frame(M):
        if isinstance(M, pd.DataFrame):
            return M
        if isinstance(M, pd.Series):
            return M.to_frame()
        arr = np.asarray(M)
        if arr.ndim == 1:
            arr = arr.reshape((-1, 1))
        elif arr.ndim != 2:
            raise ValueError(f"kron operands must be 1-d or 2-d; got ndim={arr.ndim}.")
        return pd.DataFrame(arr)

    Af = to_frame(A)
    Bf = to_frame(B)

    index = _kron_axis(Af.index, Bf.index)
    columns = _kron_axis(Af.columns, Bf.columns)

    if sparse:
        from scipy.sparse import kron as sp_kron

        k = sp_kron(Af.values, Bf.values)
        return pd.DataFrame.sparse.from_spmatrix(k, index=index, columns=columns)

    return pd.DataFrame(np.kron(Af.values, Bf.values), index=index, columns=columns)


import warnings


def heteropca(C, r=1, max_its=50, tol=1e-3, verbose=False):
    """Estimate r factors and factor weights of covariance matrix C."""
    from scipy.spatial import procrustes

    N = C - np.diag(np.diag(C))

    ulast = np.zeros((N.shape[1], r))
    u = np.zeros((N.shape[1], r))
    u[0, 0] = 1
    ulast[-1, 0] = 1

    t = 0

    while procrustes(u, ulast)[-1] > tol and t < max_its:
        ulast = u

        u, s, vt = np.linalg.svd(N, full_matrices=False, hermitian=True)

        s = s[:r]
        u = u[:, :r]

        Ntilde = u[:, :r] @ np.diag(s[:r]) @ vt[:r, :]

        N = N - np.diag(np.diag(N)) + np.diag(np.diag(Ntilde))

        t += 1

        if t == max_its:
            warnings.warn("Exceeded maximum iterations (%d)" % max_its)
        if verbose:
            print(f"Iteration {t}, u[0,:r]={u[0,:r]}.")

    return u, s


def svd_missing(A, max_rank=None, min_obs=None, heteroskedastic=False, verbose=False):
    """Singular Value Decomposition with missing values

    Returns matrices U,S,V.T, where A~=U*S*V.T.

    Inputs:
        - A :: matrix or pd.DataFrame, with NaNs for missing data.

        - max_rank :: Truncates the rank of the representation.  Note
                      that this impacts which rows of V will be
                      computed; each row must have at least max_rank
                      non-missing values.  If not supplied rank may be
                      truncated using the Kaiser criterion.

        - min_obs :: Smallest number of non-missing observations for a
                     row of U to be computed.

        - heteroskedastic :: If true, use the "heteroPCA" algorithm
                       developed by Zhang-Cai-Wu (2018) which offers a
                       correction to the svd in the case of
                       heteroskedastic errors.  If supplied as a pair,
                       heteroskedastic[0] gives a maximum number of
                       iterations, while heteroskedastic[1] gives a
                       tolerance for convergence of the algorithm.

    Ethan Ligon                                        September 2021

    """
    # Defaults; modify by passing a tuple to heteroskedastic argument.
    max_its = 50
    tol = 1e-3

    P = self_inner(A, min_obs=min_obs)  # P = A.T@A

    sigmas, v = np.linalg.eigh(P)

    order = np.argsort(-sigmas)
    sigmas = sigmas[order]

    # Truncate rank of representation using Kaiser criterion (positive eigenvalues)
    v = v[:, order]
    v = v[:, sigmas > 0]
    s = np.sqrt(sigmas[sigmas > 0])

    if max_rank is not None and len(s) > max_rank:
        v = v[:, :max_rank]
        s = s[:max_rank]

    r = len(s)

    if heteroskedastic:  # Interpret tuple
        try:
            max_its, tol = heteroskedastic
        except TypeError:
            pass
        Pbar = P.mean()
        v, s = heteropca(P - Pbar, r=r, max_its=max_its, tol=tol, verbose=verbose)

    if A.shape[0] == A.shape[1]:  # Symmetric; v=u
        return v, s, v.T
    else:
        vs = v @ np.diag(s)

        u = np.zeros((A.shape[0], len(s)))
        for j in range(A.shape[0]):
            a = A.iloc[j, :].values.reshape((-1, 1))
            x = np.nonzero(~np.isnan(a))[0]  # non-missing elements of vector a
            if len(x) >= r:
                u[j, :] = (np.linalg.pinv(vs[x, :]) @ a[x]).reshape(-1)
            else:
                u[j, :] = np.nan

    s = pd.Series(s)
    u = pd.DataFrame(u, index=A.index)
    v = pd.DataFrame(v, index=A.columns)

    return u, s, v


def sqrtm(A, hermitian=False, tol=None):
    """
    Return a positive semi-definite square root S of A satisfying
    S @ S == A.

    A must be (approximately) symmetric positive semi-definite.  The
    function symmetrizes A internally and computes an eigendecomposition
    via =np.linalg.eigh=; negative eigenvalues that lie within =tol= of
    zero are clipped to zero (FP noise from a well-formed PSD input),
    while any eigenvalue more negative than =tol= raises =ValueError=.
    The default =tol= is =1e-10 * max(1, |eigval|.max)=.

    The previous implementation inspected =np.linalg.svd='s singular
    values for negativity -- but singular values are non-negative by
    definition, so the guard could never trigger and an indefinite
    input silently received a wrong "square root" via U @ diag(sqrt(s))
    @ V.T (a polar-factor-like object that does not satisfy S @ S = A
    when A is indefinite).

    The =hermitian= flag is retained for backward compatibility; it has
    no effect now that the function always symmetrizes A before
    decomposing.
    """
    del hermitian  # retained only for backward compatibility

    A_arr = np.asarray(A)
    A_sym = (A_arr + A_arr.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(A_sym)

    if tol is None:
        tol = 1e-10 * max(1.0, float(np.abs(eigvals).max()))

    if eigvals.min() < -tol:
        raise ValueError(
            "Matrix must be positive semi-definite; smallest eigenvalue "
            f"is {eigvals.min():.4g} (tol = {tol:.4g})."
        )

    eigvals_clipped = np.maximum(eigvals, 0.0)
    S = eigvecs @ np.diag(np.sqrt(eigvals_clipped)) @ eigvecs.T

    if isinstance(A, pd.DataFrame):
        return pd.DataFrame(S, index=A.index, columns=A.columns)
    return S


def cholesky(A):
    """
    Cholesky decomposition A = L@L.T; return lower-triangular L.
    """
    L = np.linalg.cholesky(A)
    return pd.DataFrame(L, index=A.index, columns=A.columns)


from pandas import concat, get_dummies, MultiIndex


def drop_missing(X, infinities=False):
    """
    Return tuple of pd.DataFrames in X with any
    missing observations dropped.  Assumes common index.

    If infinities is false values of plus or minus infinity are
    treated as missing values.
    """

    if isinstance(X, dict):
        return dict(zip(X.keys(), drop_missing(list(X.values()), infinities=False)))

    for i, x in enumerate(X):
        if type(x) == pd.Series and x.name is None:
            x.name = i

    foo = pd.concat(X, axis=1)
    if not infinities:
        foo.replace(np.inf, np.nan)
        foo.replace(-np.inf, np.nan)

    foo = foo.dropna(how="any")

    assert len(set(foo.columns)) == len(foo.columns)  # Column names must be unique!

    Y = []
    for x in X:
        Y.append(foo.loc[:, pd.DataFrame(x).columns])

    return tuple(Y)


def dummies(df, cols, suffix=False):
    """From a dataframe df, construct an array of indicator (dummy) variables,
    with a column for every unique tuple of values in df[cols].  The list
    =cols= can mix names of regular columns of df with names of levels of a
    MultiIndex on df.index.

    The returned DataFrame has a MultiIndex on its columns whose level order
    matches the order of names supplied in =cols=.  (Prior versions used
    =set= operations internally, which leaked Python's hash randomization
    into the MultiIndex level order, so the output could change from one
    process to the next.)

    The optional argument =suffix=, if provided as a non-empty string, is
    appended to each component of the column labels.  If =suffix=True= the
    string '_d' is used.
    """
    if isinstance(cols, str):
        cols = [cols]
    cols = list(cols)

    # Partition cols into "from the row index" vs. "regular columns" while
    # preserving the user-supplied order.
    index_names = {n for n in (df.index.names or ()) if n is not None}
    idxcols = [c for c in cols if c in index_names]
    colcols = [c for c in cols if c not in index_names]

    if idxcols:
        idx = use_indices(df, idxcols)
        v = concat([idx, df[colcols]], axis=1) if colcols else idx
    else:
        v = df[colcols]

    usecols = [v[c].squeeze() for c in cols]

    tuples = pd.Series(list(zip(*usecols)), index=v.index)

    v = get_dummies(tuples).astype(int)

    if suffix is True:
        suffix = "_d"
    add_suffix = isinstance(suffix, str) and len(suffix) > 0
    if add_suffix:
        columns = [tuple(str(c) + suffix for c in t) for t in v.columns]
    else:
        columns = list(v.columns)

    v.columns = MultiIndex.from_tuples(columns, names=list(cols))

    return v


import pandas as pd
from pandas.errors import InvalidIndexError


def use_indices(df, idxnames):
    if len(set(idxnames).intersection(df.index.names)) == 0:
        return pd.DataFrame(index=df.index)

    try:
        idx = df.index
        df = df.reset_index()[idxnames]
        df.index = idx
        return df
    except InvalidIndexError:
        return df


def drop_vestigial_levels(idx, axis=0, both=False, multiindex=False):
    """
    Drop levels that don't vary across the index.

    Accepts a plain Index, a MultiIndex, or a Series/DataFrame; in the
    latter case the index (or columns, when =axis=1=) is collapsed and
    the original container is returned.

    When =multiindex= is True the result is always returned as a
    MultiIndex; if all but one level survived, the single-level result
    is wrapped via =MultiIndex.from_arrays=.  The default is False (the
    previous default tried to re-parse pipe-separated string labels,
    which crashed on numeric indices and is inappropriate as a generic
    helper -- callers needing that behaviour should do it explicitly).

    >>> idx = pd.MultiIndex.from_tuples([(1,1),(1,2)],names=['i','j'])
    >>> drop_vestigial_levels(idx)
    Index([1, 2], dtype='int64', name='j')
    >>> drop_vestigial_levels(idx, multiindex=True).names
    FrozenList(['j'])
    """
    if both:
        return drop_vestigial_levels(drop_vestigial_levels(idx, axis=1))

    if axis == 1:
        idx = idx.T

    if isinstance(idx, (pd.DataFrame, pd.Series)):
        df = idx
        idx = df.index
        HumptyDumpty = True
    else:
        HumptyDumpty = False

    if isinstance(idx, pd.MultiIndex):
        l = 0
        L = len(idx.levels)
        while l < L and isinstance(idx, pd.MultiIndex):
            # A level is "vestigial" if every code points at the same value
            # (or no value at all -- code == -1).
            if len(set(idx.codes[l])) <= 1:
                idx = idx.droplevel(l)
                L -= 1
            else:
                l += 1

    if multiindex and not isinstance(idx, pd.MultiIndex):
        # Wrap the surviving plain Index as a one-level MultiIndex so that
        # downstream code can still reach for .levels / .names / .codes.
        idx = pd.MultiIndex.from_arrays([idx], names=[idx.name])

    if HumptyDumpty:
        df.index = idx
        idx = df
        if axis == 1:
            idx = idx.T

    return idx


import numpy as np
import pandas as pd


def qr(X):
    """
    Pandas-friendly QR decomposition.
    """
    assert X.shape[0] >= X.shape[1]

    Q, R = np.linalg.qr(X)
    Q = pd.DataFrame(Q, index=X.index, columns=X.columns)
    R = pd.DataFrame(R, index=X.columns, columns=X.columns)

    return Q, R


def hat_factory(X):
    """
    Return a function hat(y) that returns X(X'X)^{-1}X'y.

    This is the least squares prediction of y given X.

    We use the fact that  the hat matrix is equal to QQ',
    where Q comes from the QR decomposition of X.
    """
    Q = qr(X)[0]

    def hat(y):
        return Q @ (Q.T @ y)

    return hat


import pandas as pd


def cov_nearest(V, threshold=1e-12):
    """
    Return a positive definite matrix which is "nearest" to the symmetric matrix V,
    with the smallest eigenvalue not less than threshold.
    """
    s, U = np.linalg.eigh((V + V.T) / 2)  # Eigenvalue decomposition of symmetric matrix

    s = np.maximum(s, threshold)

    return V * 0 + U @ np.diag(s) @ U.T  # Trick preserves attributes of dataframe V


import pandas as pd
import numpy as np


def trim(df, alpha):
    """Trim values below alpha quantile and above (1-alpha) quantile.

    This maps individual extreme elements of df to NaN.
    """
    xmin = df.quantile(alpha)
    xmax = df.quantile(1 - alpha)
    return df.where((df >= xmin) * (df <= xmax), np.nan)
