import pandas as pd
from metrics_miscellany.estimators import linear_gmm
from datamat import DataMat, DataVec
import numpy as np

def test_linear_gmm(N=500000,tol=1e-2):

    z = DataMat(np.random.standard_normal((N,2)),columns=['z1','z2'])
    u = DataVec(np.random.standard_normal((N,)))
    x = DataMat({'x':z.sum(axis=1) + u})

    x['Constant'] = 1
    z['Constant'] = 1

    beta = DataMat({'Coefficients':[1,0]},index=['x','Constant'])

    y = (x@beta).squeeze() + u

    b,V = linear_gmm(x,y,z)

    assert np.allclose(b,beta.squeeze(),atol=tol)


if __name__=='__main__':
    test_linear_gmm()
