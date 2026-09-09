from metrics_miscellany import utils
from scipy import stats
import pandas as pd
import numpy as np

def chi2_test(b, V, var_selection=None, R=None, TEST=False):
    """Construct chi2 test of R'b = 0.

    If R is None then test is b = 0.

    If one wishes to test a hypothesis regarding only a subset of elements
    of b, this subset can be chosen by specifying =var_selection= as either
    a query string or as a list of variable names.

    When =var_selection= and =R= are both supplied, =R= is auto-restricted
    via =R.loc[myb.index]= so that the algebra =R' myb= and =R' myV R= is
    well-defined.  Previously the code applied =R.T @ b= (the *full*
    coefficient vector) against an already-restricted =myV=, raising
    =ValueError: matrices are not aligned= before any test could run.
    """

    if var_selection is not None:
        if isinstance(var_selection, str):
            myb = b.query(var_selection)
        elif isinstance(var_selection, (list, tuple)):
            myb = b.loc[list(var_selection)]
        else:
            raise ValueError(
                "var_selection should be a query string or a list/tuple "
                "of variable names.")
    else:
        myb = b

    # Drop parts of matrix not involved in test
    myV = V.reindex(myb.index, axis=0).reindex(myb.index, axis=1)
    myV = utils.cov_nearest(myV, threshold=1e-10)

    if R is not None:
        # When the caller restricted to a subset, restrict R the same way
        # so the linear combinations remain conformable.
        myR = R.loc[myb.index] if var_selection is not None else R
        myV = myR.T @ myV @ myR
        myb = myR.T @ myb
        if np.isscalar(myV):
            myV = np.array([[myV]])
            myb = np.array([[myb]])

    if TEST: # Generate values of my that satisfy Var(myb)=Vb and Emyb=0
        myb = myb*0 + stats.multivariate_normal(cov=((1e0)*np.eye(myV.shape[0]) + myV)).rvs().reshape((-1,1))

    # "Invert"...

    L = np.linalg.cholesky(myV)
    y = np.linalg.solve(L.T,myb)

    chi2 = y.T@y

    y = pd.Series(y.squeeze(),index=myb.index)

    return chi2,1-stats.distributions.chi2.cdf(chi2,df=len(myb))

def skillings_mack(df,bootstrap=False):
    """
    Non-parametric test of correlation across columns of df.

    Algorithm from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2761045/
    """
    def construct_statistic(R, kay, X):
        """
        Once we have ranks R (and the original observation matrix X
        for the missingness pattern), construct the SM statistic.

        =X= is passed in explicitly rather than closed over: in the
        previous version =construct_statistic= reached back to a
        name =X= that was bound in the outer scope only *after* the
        function was defined.  That worked by coincidence -- the
        function is only called once =X= exists -- but stashed a
        latent closure bug that broke as soon as you rearranged the
        body.
        """
        # Fill missing ranks with (k_i+1)/2
        R = R.where(~np.isnan(R),(kay+1)/2,axis=1)

        # Construct adjusted observation matrix
        A = R.subtract((kay.values+1)/2,axis=1)@np.sqrt(12/(kay.values+1))

        # Count of observations in both columns k and l
        O = ~np.isnan(X)+0.

        Sigma = np.eye(O.shape[0]) - O@O.T

        # Delete diagonal
        Sigma = Sigma - np.diag(np.diag(Sigma))

        # Add minus column sums to diagonal
        Sigma = Sigma - np.diag(Sigma.sum())

        return A.T@np.linalg.pinv(Sigma)@A

    # Drop any rows with only one column
    X = df.loc[df.count(axis=1)>0]

    n,k = X.shape

    # Counts of obs per row ("treatments")
    kay = X.count(axis=0)

    # Counts of obs per column ("blocks")
    en = X.count(axis=1)

    R = X.rank(axis=0)

    SM = construct_statistic(R, kay, X)

    if not bootstrap:
        p = 1-stats.distributions.chi2.cdf(SM,df=n-1)
    else:
        if bootstrap == True:
            tol = 1e-03
        else:
            tol = bootstrap

        SE = 0
        lastSE = np.inf
        its = 0
        sms = []
        while (its < 30) or (np.abs(SE-lastSE) > tol):
            lastSE = SE
            scrambled = pd.DataFrame(np.apply_along_axis(np.random.permutation,axis=0,arr=R.values),
                                     index=R.index,columns=R.columns)
           
            sms.append(construct_statistic(scrambled, kay, X))
            SE = np.std(sms)
            its += 1
        p = np.mean(sms>SM)

    return SM,p

friedman = skillings_mack

import pandas as pd
import numpy as np
from metrics_miscellany.estimators import ols
from metrics_miscellany.random import permutation as _permutation

def randomization_inference(vars,X,y,permute_levels=None,R=None,tol=1e-3,VERBOSE=False,return_draws=False):
    """
    Return p-values associated with hypothesis that coefficients
    associated with vars are jointly equal to zero.

    The optional argument =permute_levels= names index levels along which
    the values of X[vars] are shuffled; the complementary "fixed" levels
    are held in place and the block of values across them is reassigned
    jointly.  Delegates to =metrics_miscellany.random.permutation= so the
    two entry points share one (correct) implementation.

    Ethan Ligon                                       June 2021
    """

    assert np.all([v in X.columns for v in vars]), "vars must correspond to columns of X."

    b,V = ols(X,y)

    beta = b.squeeze()[vars]
    chi2 = chi2_test(beta,V,R=R)[0]

    last = np.inf
    p = 0
    i = 0
    Chi2 = []
    while (np.linalg.norm(p-last)>tol) or (i < 30):
        last = p
        P = _permutation(X, columns=vars, permute_levels=permute_levels)

        myX = pd.concat([X.loc[:,X.columns.difference(vars)],P],axis=1)
        b,V = ols(myX,y)
        Chi2.append(chi2_test(b.squeeze()[vars],V,R=R)[0])
        p = (chi2<Chi2[-1])/(i+1) + last*i/(i+1)
        i += 1
        if VERBOSE: print("Latest chi2 (randomized,actual,p): (%6.2f,%6.2f,%6.4f)" % (Chi2[-1],chi2,p))

    if return_draws:
        return p,pd.Series(Chi2)
    else:
        return p

import numpy as np
from scipy.stats.distributions import chi2

def maunchy(C,N):
    """Given a sample covariance matrix C estimating using N observations,
       return p-value associated with test of whether the population
       covariance matrix is proportional to the identity matrix.
    """

    raise NotImplementedError

    m = C.shape[0]

    V = np.linalg.det(C)/((np.trace(C)/m)**m)

    rho = 1 - (2*m**2 + m + 2)/(6*m*(N-1))

    w2 = (m-1)*(m-2)*(m+2)*(2*m**3 + 6*m**2 + 3*m + 2)/(288*(m**2) * ((N-1)**2) * rho**2)

    gamma = (((N-1)*rho)**2)*w2

    x2 = -2*(N-1)*rho*np.log(V)  # Chi-squared statistic

    df = (m+2)*(m-1)/2

    px2 = chi2.cdf(x2,df)

    p = px2 + gamma/(((N-1)*rho)**2) * (chi2.cdf(x2,df+4) - px2)

    return x2,1 - px2

import numpy as np
from scipy.stats.distributions import chi2

def kr79(C,q,N):
    """Given a sample mxm covariance matrix C estimating using N observations,
       return p-value associated with test of whether the population
       covariance matrix has last q eigenvalues equal or not, where q+k=m.
    """

    m = C.shape[0]

    l = np.linalg.eigvalsh(C)  # eigenvalues in *ascending* order

    Q = (np.prod(l[:q])/(np.mean(l[:q])**q))**(N/2) # LR test statistic

    x2 = -2*np.log(Q)  # Chi-squared statistic

    df = (q-1)*(q+2)/2

    px2 = chi2.cdf(x2,df)

    #p = px2 + gamma/(((N-1)*rho)**2) * (chi2.cdf(x2,df+4) - px2)

    return x2,1 - px2

def cragg_donald(X, Q):
    r"""
    Cragg-Donald (1993) reduced-rank / weak-instruments test.

    In the reduced form

        X = Q\Pi + v,

    where X is N x m (endogenous regressors), Q is N x l (instruments)
    and \Pi is l x m, we test

        H_0:  rank(\Pi) = m - 1
        H_1:  rank(\Pi) = m  (full rank, i.e. the instruments identify
                              every column of X).

    The statistic is

        (N - l) x lambda_min( (X' M_Q X)^{-1} X' P_Q X ),

    which under H_0 is asymptotically chi-squared with

        df = l - m + 1.

    Previous versions used =df = N - m + 1= (a sample-size-based
    degree of freedom), which made the p-value collapse to 1.0 for
    any reasonable N -- the test never rejected.  The correct df
    depends only on the rank deficit, not on the sample size.

    Returns a (statistic, p-value) pair.
    """
    n, k = Q.shape   # k is the number of instruments l
    m = X.shape[1]   # number of endogenous regressors

    assert k >= m, (
        f"Cragg-Donald requires at least as many instruments as endogenous "
        f"regressors (got l={k}, m={m}).")

    teststat = utils.inv(X.T @ X.resid(Q)) @ (X.T @ X.proj(Q))
    teststat = (n - k) * teststat.eig()[0].min()

    df = k - m + 1
    pvalue = 1 - chi2(df).cdf(teststat)

    return teststat, pvalue
