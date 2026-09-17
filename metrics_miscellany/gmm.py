import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, approx_fprime
from scipy.optimize import minimize as scipy_min

from . import utils

matrix_product = utils.matrix_product
diag = utils.diag
inv = utils.inv

# IPython.core.debugger.Pdb used to be imported here for ad-hoc
# breakpoint() use; load it lazily inside whichever routine needs it
# rather than paying the IPython import cost at every gmm.py import.

######################################################
# Beginning of procedural version of gmm routines
#
# These routines deliberately reference a module-level `gj(b)` that is not
# defined here: the caller supplies the moment conditions for their own data
# by binding `gmm.gj`, which is what makes the procedural interface generic.
# Ruff cannot see that, so each use carries `# noqa: F821`.  The name is left
# genuinely undefined rather than stubbed to None so that forgetting to bind
# it raises NameError naming `gj`, instead of a TypeError about NoneType.


def gN(b):
    """Averages of g_j(b).

    This is generic for data, to be passed to gj.
    """
    e = gj(b)  # noqa: F821

    gN.N, gN.k = e.shape
    gN.N = e.count()  # Allows for possibility of missing data
    # Check to see more obs. than moments.
    assert np.all(gN.N > gN.k), "More moments than observations"

    try:
        return e.mean(axis=0).reshape((-1, 1))
    except AttributeError:
        return e.mean(axis=0)


def Omegahat(b):
    e = gj(b)  # noqa: F821

    # Recenter! We have Eu=0 under null.
    # Important to use this information.
    e = e - e.mean(axis=0)
    sqrtN = np.sqrt(e.count())

    e = e / sqrtN

    ete = matrix_product(e.T, e)

    return ete


def JN(b, W=None):

    if W is None:
        W = utils.inv(Omegahat(b))

    m = gN(b)  # Sample moments @ b

    # Pdb().set_trace()

    # Scaling by diag(N) allows us to deal with missing values
    WN = pd.DataFrame(matrix_product(diag(gN.N), W))

    crit = (m.T @ WN @ m).squeeze()
    assert crit >= 0

    return crit


def minimize(f, b_init=None):
    if b_init is None:
        return minimize_scalar(f).x
    else:
        return scipy_min(f, b_init).x


def one_step_gmm(W=None, b_init=None):

    if b_init is None:
        b_init = 0

    if W is None:
        e = gj(b_init)  # noqa: F821
        W = pd.DataFrame(np.eye(e.shape[1]), index=e.columns, columns=e.columns)

    assert np.linalg.matrix_rank(W) == W.shape[0]

    b = minimize(lambda b: JN(b, W), b_init=b_init)

    return b, JN(b, W)


def two_step_gmm(b_init=None):

    # First step uses identity weighting matrix
    b1 = one_step_gmm(b_init=b_init)[0]

    # Construct 2nd step weighting matrix using
    # first step estimate of beta
    W2 = utils.inv(Omegahat(b1))

    return one_step_gmm(W=W2, b_init=b1)


def continuously_updated_gmm(b_init=None):

    # First step uses identity weighting matrix
    def W(b):
        return utils.inv(Omegahat(b))

    bhat = minimize(lambda b: JN(b, utils.inv(Omegahat(b))), b_init=b_init)

    return bhat, JN(bhat, W(bhat))


def dgN(b):
    """
    Average gradient of gj(b).

    This function provides numerical derivatives.

    One may wish to override this with a function dgN which returns analytical derivatives.
    """
    gradient = pd.DataFrame(approx_fprime(b, gN), index=gN(b).index)

    return gradient


def Vb(b):
    """Covariance of estimator of b.

    Note that one must supply gmm.dgN, the average gradient of gmm.gj at b.
    """
    Q = dgN(b)
    W = pd.DataFrame(matrix_product(diag(gN.N), Omegahat(b)))

    return utils.inv(Q.T @ utils.inv(W) @ Q)


def print_version():
    """CLI entrypoint registered in pyproject.toml as
    =metrics-miscellany-version=.  Reads the installed package
    version via importlib.metadata so the string stays in sync with
    the wheel rather than the long-stale =__version__ = "0.3.1"= that
    used to live at module top."""
    from metrics_miscellany import __version__

    print(__version__)


# End of procedural version of gmm routines
######################################################
