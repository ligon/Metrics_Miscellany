import pandas as pd
import numpy as np
from metrics_miscellany.estimators import gmm
from numpy.linalg import inv
from scipy.stats import distributions as iid

def dgp(N,beta,gamma,sigma_u,VXZ):
    """Generate a tuple of (y,X,Z).

    Satisfies model:
        y = X@beta + u
        E Z'u = 0
        Var(u) = sigma^2
        Cov(X,u) = gamma*sigma_u^2
        Var([X,Z]|u) = VXZ
        u,X,Z mean zero, Gaussian

    Each element of the tuple is an array of N observations.

    Inputs include
    - beta :: the coefficient of interest
    - gamma :: linear effect of disturbance on X
    - sigma_u :: Variance of disturbance
    - VXZ :: Var([X,Z]|u)
    """

    u = pd.Series(iid.norm.rvs(size=(N,))*sigma_u)

    # "Square root" of VXZ via eigendecomposition
    lbda,v = np.linalg.eig(VXZ)
    SXZ = v@np.diag(np.sqrt(lbda))

    # Generate normal random variates [X*,Z]
    XZ = pd.DataFrame(iid.norm.rvs(size=(N,VXZ.shape[0]))@SXZ.T)

    # But X is endogenous...
    X = XZ.loc[:,0].add(gamma*u,axis=0)
    Z = XZ.loc[:,1:]

    # Calculate y
    y = X*beta + u

    return y,X,Z

def test_gmm(N=10000):

    ## Play with us!
    beta = 1     # "Coefficient of interest"
    gamma = 1    # Governs effect of u on X
    sigma_u = 1  # Note assumption of homoskedasticity
    ## Play with us!

    # Let Z have order ell, and X order 1, with Var([X,Z]|u)=VXZ

    ell = 4 # Play with me too!

    # Arbitrary (but deterministic) choice for VXZ = [VX Cov(X,Z);
    #                                                 Cov(Z,X) VZ]
    # Pinned down by choice of a matrix A...
    A = np.sqrt(1/np.arange(1,(ell+1)**2+1)).reshape((ell+1,ell+1))

    ## Below here we're less playful.

    # Now Var([X,Z]|u) is constructed so guaranteed pos. def.
    VXZ = A.T@A

    Q = -VXZ[1:,[0]]  # -EZX', or generally Edgj/db'

    # Gimme some truth:
    truth = (beta,gamma,sigma_u,VXZ)

    ## But play with Omega if you want to introduce heteroskedascity
    Omega = (sigma_u**2)*VXZ[1:,1:] # E(Zu)(u'Z')

    # Asymptotic variance of optimally weighted GMM estimator:
    AVar_b = inv(Q.T@inv(Omega)@Q)

    data = dgp(N,*truth)

    def gj(b):
        y,X,Z = data
        e = (y.squeeze()-b*X.squeeze())

        Ze = Z.multiply(e,axis=0)

        return Ze

    def dgN(b):
        y,X,Z = data
        return Z.T@X

    gmm.gj = gj
    gmm.dgN = dgN

    b,J = gmm.two_step_gmm()

    assert (b-beta)**2 < 0.01, f"Estimate {b} outside tolerance."


if __name__ == '__main__':
    test_gmm()
