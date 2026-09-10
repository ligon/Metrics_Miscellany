import numpy as np
from numpy.linalg import lstsq
import warnings
import pandas as pd
from . import utils
import datamat as dm

# Re-exported: metrics_miscellany.test.test_gmm imports `gmm` from here.
# The redundant alias marks that as deliberate rather than an unused import.
from . import gmm as gmm


def ols(X, y, cov_type="HC3", PSD_COV=False):
    """OLS estimator of b in y = Xb + u.

    Returns both estimate b as well as an estimate of Var(b).

    The estimator used for the covariance matrix depends on the
    optional argument =cov_type=.

    If optional flag PSD_COV is set, then an effort is made to ensure that
    the estimated covariance matrix is positive semi-definite.  If PSD_COV is
    set to a positive float, then this will be taken to be the smallest eigenvalue
    of the 'corrected' matrix.
    """
    n, k = X.shape

    Xv = np.asarray(X)
    yv = np.asarray(y).ravel()

    b_vals, _, _, _ = np.linalg.lstsq(Xv, yv, rcond=None)
    b = pd.DataFrame({"Coefficients": b_vals}, index=X.columns)
    e = yv - Xv @ b_vals

    XX = Xv.T @ Xv
    XXinv = np.linalg.inv(XX)

    if cov_type == "OLS":
        if np.linalg.eigh(XX)[0].min() < 0:
            s_xx, U_xx = np.linalg.eigh((XX + XX.T) / 2)
            XX = U_xx @ np.diag(np.maximum(s_xx, 1e-12)) @ U_xx.T
            XXinv = np.linalg.inv(XX)
            warnings.warn(
                "X'X not positive (semi-) definite.  Correcting!  "
                "Estimated variances should not be affected.",
                stacklevel=2,
            )
        V = np.var(e, ddof=0) * XXinv
    elif cov_type in ("HC0", "HC1", "HC2", "HC3"):
        h = np.einsum("ij,jk,ik->i", Xv, XXinv, Xv)  # hat matrix diagonal
        if cov_type == "HC0":
            w = e**2
        elif cov_type == "HC1":
            w = e**2 * (n / (n - k))
        elif cov_type == "HC2":
            w = e**2 / (1 - h)
        elif cov_type == "HC3":
            w = e**2 / (1 - h) ** 2
        V = XXinv @ (Xv.T * w) @ Xv @ XXinv
    else:
        raise ValueError("Unknown type of covariance matrix.")

    if PSD_COV:
        if PSD_COV is True:
            PSD_COV = (b**2).min()
        s, U = np.linalg.eigh((V + V.T) / 2)
        if s.min() < PSD_COV:
            oldV = V
            V = U @ np.diag(np.maximum(s, PSD_COV)) @ U.T
            warnings.warn(
                "Estimated covariance matrix not positive (semi-) definite.\n"
                f"Correcting! Norm of difference is {np.linalg.norm(oldV - V):g}.",
                stacklevel=2,
            )

    V = pd.DataFrame(V, index=X.columns, columns=X.columns)

    return b, V


def restricted_tsls(y, X, R=None, r=None, Z=None, cov="HC3"):
    """
    Estimate b in y = Xb + u subject to Rb = r.

    Parameters
    ----------
    y : Series or one-column DataFrame -- the dependent variable.
    X : DataMat/DataFrame of regressors (N x k).
    R : optional DataMat/DataFrame of restrictions (m x k).  Each row of
        R encodes one linear restriction Rb = r.  When R is None the
        function reduces to ordinary TSLS.
    r : right-hand side of the restrictions (length m).  Required when
        R is supplied.
    Z : instrument matrix (N x l); defaults to X.
    cov : 'OLS' or one of 'HC0'..'HC3' (default 'HC3').

    Returns
    -------
    Without R:  (b, Omega, V)
    With R:     (b, lm, Omega, V)

    where =lm= is the vector of Lagrange multipliers associated with
    the rows of R.

    Notes
    -----
    The restricted estimator solves the saddle-point system

        [ X' P_Z X    R' ] [ b  ]   [ X' P_Z y ]
        [ R           0  ] [ lm ] = [   r      ]

    in NumPy.  An earlier implementation tried to assemble this block
    via =datamat.concat= with custom MultiIndex levels and never
    reached the solver when R was supplied; the Lagrange-multiplier
    return path also referred to an undefined name.
    """
    if Z is None:
        Z = X

    N, k = X.shape
    _N, _l = Z.shape
    assert N == _N, "X & Z must have same number of rows."

    if R is not None:
        m, _k = R.shape
        assert _k == k, (
            "Matrix of restrictions must be conformable with the vector "
            "of parameters."
        )
        if r is None:
            raise ValueError("r must be supplied when R is supplied.")
        r_vec = np.asarray(r).reshape(-1)
        assert r_vec.shape[0] == m, "r and R must agree on the number of restrictions."
    else:
        m = 0
        r_vec = None

    assert _l + m >= k, (
        f"Need #instruments ({_l}) + #restrictions ({m}) >= " f"#parameters ({k})."
    )

    Qzz = Z.T @ Z / N
    Qxz = X.T @ Z / N
    Qzzinv = utils.inv(Qzz)

    # Q = X' P_Z X;  rhs_b = X' P_Z y.
    Q = N * Qxz @ Qzzinv @ Qxz.T
    rhs_b = Qxz @ Qzzinv @ Z.T @ y

    param_labels = X.columns

    if R is not None:
        # Build the (k+m) x (k+m) saddle-point block in NumPy to avoid the
        # label-juggling that broke the previous dm.concat-based path.
        Qv = np.asarray(Q)
        Rv = np.asarray(R)
        lhs_arr = np.zeros((k + m, k + m), dtype=float)
        lhs_arr[:k, :k] = Qv
        lhs_arr[:k, k:] = Rv.T
        lhs_arr[k:, :k] = Rv
        rhs_arr = np.concatenate([np.asarray(rhs_b).reshape(-1), r_vec])
        sol = np.linalg.solve(lhs_arr, rhs_arr)
        b = pd.Series(sol[:k], index=param_labels, name="Coefficients")
        lm = pd.Series(sol[k:], index=R.index, name="lm")
    else:
        b_arr = np.linalg.solve(np.asarray(Q), np.asarray(rhs_b).reshape(-1))
        b = pd.Series(b_arr, index=param_labels, name="Coefficients")
        lm = None

    # Residuals using X alone -- the restrictions add no observations.
    e = y - X @ b

    if cov in ("HC2", "HC3"):
        h = utils.leverage(X)
        if cov == "HC2":
            e = e / np.sqrt(1 - h)
        else:
            e = e / (1 - h)

    Ze = Z.multiply(e, axis=0)
    Omega = Ze.T @ Ze / N

    D = Qxz @ Qzzinv @ Qxz.T
    Dinv = utils.inv(D)
    V_unr = Dinv @ (Qxz @ Qzzinv @ Omega @ Qzzinv @ Qxz.T) @ Dinv / N

    if R is not None:
        # Project to the restricted subspace via
        #     P = I - V_unr R' (R V_unr R')^{-1} R,
        # then V_b = P V_unr P'.
        Rv = np.asarray(R)
        V_unr_v = np.asarray(V_unr)
        middle = np.linalg.inv(Rv @ V_unr_v @ Rv.T)
        P = np.eye(k) - V_unr_v @ Rv.T @ middle @ Rv
        V_b = pd.DataFrame(P @ V_unr_v @ P.T, index=param_labels, columns=param_labels)
    else:
        V_b = V_unr

    if cov == "HC1":
        V_b = V_b * (N / (N - k))

    if R is None:
        return b, Omega, V_b
    return b, lm, Omega, V_b


def tsls(X, y, Z, return_Omega=False, **kwargs):
    """
    Two-stage least squares estimator.
    """
    b, Omega, Vb = restricted_tsls(y, X, Z=Z, **kwargs)

    if return_Omega:
        return b, Omega
    else:
        return b, Vb


def factor_analysis(
    X,
    n_components=None,
    noise_variance_init=None,
    max_its=1000,
    tol=1e-2,
    svd_method="eig",
):
    """Fit the FactorAnalysis model to X using SVD based MLE approach.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features) Training data.

    n_components : Proposed rank (number of factors)
    """

    n_samples, n_features = X.shape

    assert n_samples >= n_features

    if n_components is None:
        n_components = n_features

    xbar = X.mean(axis=0)
    X = X - xbar

    # some constant terms
    llconst = n_features * np.log(2.0 * np.pi) + n_components
    var = X.var()

    if noise_variance_init is None:
        psi = np.ones(n_features)
    else:
        if len(noise_variance_init) != n_features:
            raise ValueError(
                "noise_variance_init dimension does not accord "
                "with number of features : %d != %d"
                % (len(noise_variance_init), n_features)
            )
        psi = np.array(noise_variance_init)

    loglike = []
    old_ll = -np.inf
    SMALL = 1e-12

    def squared_norm(x):
        return np.linalg.norm(x) ** 2

    def self_inner(X, min_obs=None):
        """Compute inner product X.T@X, allowing for possibility of missing data."""
        n, m = X.shape

        if n < m:
            axis = 1
            N = m
        else:
            axis = 0
            N = n

        mX = np.ma.masked_invalid(X)

        xbar = np.mean(mX, axis=axis)

        if axis:
            C = (N - 1) * np.ma.cov(mX)
        else:
            C = (N - 1) * np.ma.cov(mX.T)

        return (C + N * np.outer(xbar, xbar)).data

    # we'll modify svd outputs to return unexplained variance
    # to allow for unified computation of loglikelihood
    if svd_method == "lapack":
        # The previous implementation called `linalg.svd` without importing
        # `linalg`; gate the branch so it fails immediately and tells the
        # caller what to do instead of NameError'ing partway through.
        raise NotImplementedError(
            "factor_analysis(svd_method='lapack') is unfinished. "
            "Use svd_method='eig'."
        )

    elif svd_method == "randomized":
        # Likewise: the branch references self.random_state /
        # self.iterated_power, which don't exist in this free function.
        raise NotImplementedError(
            "factor_analysis(svd_method='randomized') is unfinished. "
            "Use svd_method='eig'."
        )

    elif svd_method == "eig":

        def my_svd(P):

            sigmas, v = np.linalg.eigh(P)
            vt = v.T

            order = np.argsort(-sigmas)
            sigmas = sigmas[order]

            # Truncate rank of representation using Kaiser criterion (positive eigenvalues)
            vt = vt[order, :]
            vt = vt[sigmas > 0, :]
            s = np.sqrt(sigmas[sigmas > 0])

            if n_components is not None and len(s) > n_components:
                vt = vt[:n_components, :]
                s = s[:n_components]

            return s, vt, squared_norm(P) - squared_norm(s)

    P = self_inner(X)
    for _ in range(max_its):
        # SMALL helps numerics
        sqrt_psi = np.sqrt(psi) + SMALL
        s, Vt, unexp_var = my_svd(P @ np.diag(1 / (psi * n_samples)))
        s **= 2
        # Use 'maximum' here to avoid sqrt problems.
        W = np.sqrt(np.maximum(s - 1.0, 0.0))[:, np.newaxis] * Vt
        del Vt
        W = W.squeeze() * sqrt_psi

        # loglikelihood
        ll = llconst + np.sum(np.log(s))
        ll += unexp_var + np.sum(np.log(psi))
        ll *= -n_samples / 2.0
        loglike.append(ll)
        if (ll - old_ll) < tol:
            break
        old_ll = ll

        psi = np.maximum(var - np.sum(W**2, axis=0), SMALL)
    else:
        # for/else: the loop completed without breaking, i.e. we exhausted
        # =max_its= iterations without satisfying the tolerance.  The
        # previous code referenced an unimported ConvergenceWarning and a
        # second never-called local =ll= function -- both removed.
        warnings.warn(
            f"factor_analysis did not converge in {max_its} iterations; "
            "increasing max_its may help.",
            RuntimeWarning,
            stacklevel=2,
        )

    return W, psi


def fwl_regression_step(D, X):
    """Regress each datamat in dictionary D on X.
    Return a dictionary of residuals, and a dictionary of least-squares coefficients.
    """
    b = {}
    u = {}
    if len(D) == 0:
        return D, {}

    for k, v in D.items():
        b[k] = X.lstsq(v)
        u[k] = dm.DataMat(v.resid(X))

    return u, b


def fwl_regression(D, B=None, U=None):
    """Regress each datamat in dictionary D on the last element X of D.
    Iterate.

    Return a dictionary of residuals, and a dictionary of least-squares coefficients.
    """
    if B is None:
        B = {}
    if U is None:
        U = {}

    if len(D) == 0:
        return {}, {}
    elif len(D) == 1:
        return U, B
    else:
        xk, x = D.popitem()
        D, B[xk] = fwl_regression_step(D, x)
        U[xk] = D.copy()
        return fwl_regression(D, B=B, U=U)


def reconstruct_coefficients_from_fwl(B: dict, as_dict=False):
    """
    Reconstructs OLS coefficient vectors from FWL inputs,
    generalized for matrix regressors.
    """
    # ## 1. Infer the dependent variable name ##
    top_level_keys = set(B.keys())
    if len(top_level_keys) == 0:
        return {}

    # Arbitrarily pick the first variable's sub-dictionary to inspect its keys
    first_var_key = next(iter(B))
    all_nested_keys = set(B[first_var_key].keys())

    # The dependent variable is the key in the nested dict that is NOT a top-level key
    dep_var_set = all_nested_keys - top_level_keys
    if len(dep_var_set) != 1:
        raise ValueError("Could not uniquely determine the dependent variable name.")
    dep_var_name = dep_var_set.pop()

    # ## 2. Proceed with the iterative reconstruction ##
    ordered_vars = list(B.keys())
    p = len(ordered_vars)
    coeffs: dict[str, pd.Series] = {}  # Stores the final coefficient vectors (b_i)

    # This loop proceeds backward from i = p-1 down to 0
    for i in range(p - 1, -1, -1):
        current_var = ordered_vars[i]

        # Use the inferred dependent variable name to get the G_iy vector
        G_iy = B[current_var][dep_var_name]

        summation_vector = np.zeros_like(G_iy)

        for j in range(i + 1, p):
            successor_var = ordered_vars[j]
            G_ij = B[current_var][successor_var]
            b_j = coeffs[successor_var]
            summation_vector += G_ij @ b_j

        coeffs[current_var] = G_iy - summation_vector
        coeffs[current_var].name = dep_var_name

    # Reverse order of dict
    coeffs = {k: coeffs[k] for k in reversed(list(coeffs.keys()))}

    if as_dict:
        return coeffs
    else:
        return dm.concat(coeffs, levelnames=True).squeeze()


def linear_gmm(X, y, Z, W=None, return_Omega=False):
    """
    Linear GMM estimator.
    """

    if W is None:  # Use 2sls to get initial estimate of W
        b1, Omega1 = tsls(X, y, Z, return_Omega=True)
        W = utils.inv(Omega1)
        # Forward return_Omega through the recursive call so callers asking
        # for Omega on a default-W invocation actually receive it.
        return linear_gmm(X, y, Z, W=W, return_Omega=return_Omega)
    else:
        n, k = X.shape

        Qxz = X.T @ Z / n

        b = lstsq(Qxz @ W @ Qxz.T, Qxz @ W @ Z.T @ y / n, rcond=None)[0]

        b = pd.Series(b.squeeze(), index=X.columns)

        # Cov matrix
        e = y.squeeze() - X @ b

        # Omega = Z.T@(e**2).dg()@Z/n
        # Rather than forming even a sparse nxn matrix, just use element-by-element multiplication
        ZTe = Z.T.multiply(e)
        Omega = ZTe @ ZTe.T / n

        if return_Omega:
            return b, Omega
        else:
            Vb = utils.inv(Qxz @ utils.inv(Omega) @ Qxz.T) / n
            return b, Vb


def restricted_linear_gmm(X, y, Z, R, r, W=None, return_Omega=False):
    """
    Linear GMM with linear restrictions Rb = r.

    Not yet implemented.  The previous placeholder body was an unreachable
    copy-paste of =linear_gmm= sitting underneath =raise NotImplementedError=;
    it has been removed to keep the function honest.
    """
    raise NotImplementedError("restricted_linear_gmm has not yet been implemented.")


def factor_regression(Y, X, F=None, rank=1, tol=1e-3):

    if rank > 1:
        raise NotImplementedError("Factor regression for rank>1 is not reliable.")

    N, k = Y.shape

    def ols(X, Y):
        N, k = Y.shape
        XX = utils.self_inner(X) / N
        XY = utils.matrix_product(X.T, Y) / N
        B = np.linalg.lstsq(XX, XY, rcond=None)[0]
        return pd.DataFrame(B, index=X.columns, columns=Y.columns)

    if F is None:
        B = ols(X, Y)
        F = 0
    else:
        parms = ols(pd.concat([X, F], axis=1), Y)
        L = parms.iloc[-rank:, :]
        B = parms.iloc[:-rank, :]

    lastF = F
    F, s, vt = utils.svd_missing(Y - utils.matrix_product(X, B), max_rank=rank)
    scale = F.std()
    F = F.multiply(1 / scale)

    if np.linalg.norm(F - lastF) > tol:
        B, L, F = factor_regression(Y, X, F=F, rank=rank, tol=tol)

    return B, L, F
