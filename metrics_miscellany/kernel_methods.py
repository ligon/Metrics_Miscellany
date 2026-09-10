import pandas as pd
import datamat as dm
import numpy as np

sqrt3 = np.sqrt(3)  # Avoid repeated evaluation of this for speed...
sqrt2pi = np.sqrt(2 * np.pi)


def rectangular(u):
    return (np.abs(u) < sqrt3) / (2 * sqrt3)  # Rectangular kernel


def gaussian(u):
    return np.exp(-(u**2) / 2) / sqrt2pi  # Gaussian kernel


def gram(X, kernel=gaussian, bw=1):
    """
    Construct Gram matrix from vector of data X.
    """
    try:
        idx = X.index
        df = True
        x = X.values
    except AttributeError:
        df = False
        x = X

    assert len(x.shape) == 1
    K = kernel((x.reshape((-1, 1)) - x.reshape((1, -1))) / bw)

    if df:
        if isinstance(X, (dm.DataVec, dm.DataMat)):
            K = dm.DataMat(K, index=X.index, columns=X.index)
        elif isinstance(X, (pd.Series, pd.DataFrame)):
            K = pd.DataFrame(K, index=X.index, columns=X.index)

    return K


def kernel_regression(X, y, bw, kernel=gaussian):
    """
    Use data (X,y) to estimate E(y|x), using bandwidth bw.
    """

    def mhat(x):
        S = kernel((X - x) / bw)  # "Smooths"

        return S.dot(y) / S.sum()

    return mhat


def kernel_regression_variance(X, y, bw, kernel=gaussian):
    """
    Use data (X, y) to estimate E((y - m(x))^2 | x), using bandwidth bw.

    Leave-one-out residuals are used to estimate the conditional
    variance: each y_i is compared against the prediction from a
    kernel regression that excludes observation i.

    The previous version constructed =Km = K - K.dg().dg()= (the
    Gram matrix with the diagonal zeroed out, i.e. the leave-one-out
    weights) but then never used it -- the residuals were formed
    from the inclusive K, so they systematically understated the
    variance.  It also divided the kernel-weighted average at the
    query point =x= by =K.sum().sum()= (the *training-time* total
    weight, fixed at function-definition time and independent of
    =x=) rather than by the local sum at =x=.  Both are fixed.
    """
    K = gram(X, bw=bw, kernel=kernel)

    # leave-one-out weights: zero on the diagonal.  Use NumPy diag
    # explicitly so the function works for any matrix-like K (a
    # DataMat, a pandas DataFrame, or even a NumPy array); the
    # previous =K.dg().dg()= only succeeded for DataMat.
    K_vals = np.asarray(K)
    Km_vals = K_vals - np.diag(np.diag(K_vals))
    if isinstance(K, pd.DataFrame):
        Km = pd.DataFrame(Km_vals, index=K.index, columns=K.columns)
    else:
        Km = Km_vals

    e2 = (y - (Km @ y) / Km.sum(axis=1)) ** 2

    def sigmahat(x):
        S = kernel((X - x) / bw)  # "Smooths"
        return (S**2).dot(e2) / (S**2).sum()

    return sigmahat
