import pandas as pd
from scipy import stats
from metrics_miscellany.estimators import factor_regression
from metrics_miscellany import utils
import numpy as np

def generate_multivariate_normal(N,k,V=None,colidx='a'):

    try:
        a = ord(colidx)
        labels = list(map(chr, range(a, a+k)))
    except TypeError:
        labels = range(colidx,colidx+k)

    if V is None:
        D = pd.DataFrame(np.random.randn(k,k),index=labels,columns=labels)
        V = D.T@D
    else:
        V = pd.DataFrame(V,index=labels,columns=labels)

    X = pd.DataFrame(stats.multivariate_normal(cov=V).rvs(N),columns=labels)

    return X

def main(N,k,l,r):

    U = generate_multivariate_normal(N,k,V=np.eye(k),colidx='A')/100

    X = generate_multivariate_normal(N,l)

    a = ord('a')
    A = ord('A')
    rlabels = list(map(chr, range(a, a+l)))
    clabels = list(map(chr, range(A, A+k)))
    B = pd.DataFrame(np.arange(1,l*k+1).reshape(l,k),index=rlabels,columns=clabels)

    F = generate_multivariate_normal(N,r,colidx=0)
    F = F - F.mean()
    scale = F.std()
    F = F.multiply(1/scale)

    L = pd.DataFrame(np.arange(1,k*r+1).reshape(r,k)/10,index=F.columns,columns=U.columns)
    L = L.multiply(scale,axis=0)

    Y = utils.matrix_product(X,B) + utils.matrix_product(F,L) + U

    return Y,X,F,B,L,U

def test_factor_regression(N=1000,k=10,l=2,r=1):
    Y,X,F0,B0,L0,U0 = main(N,k,l,r)
    X['Constant'] = 1

    B,L,F = factor_regression(Y,X,rank=r)

    assert np.linalg.norm(F0-F) < np.linalg.norm(F0)

    assert np.all(Y.var()>(Y-X@B).var())

    assert np.all((Y-X@B).var()>(Y-X@B-F@L).var())

    assert np.linalg.norm((B0-B).dropna())/np.linalg.norm(B0) < 0.02

if __name__ == '__main__':
    test_factor_regression(N=10000,r=1)
