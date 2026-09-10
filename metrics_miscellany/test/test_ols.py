import pandas as pd
from metrics_miscellany.estimators import ols
import numpy as np

def test_ols(N=500000,tol=1e-2):

    x = pd.DataFrame({'x':np.random.standard_normal((N,))})
    x['Constant'] = 1

    beta = pd.DataFrame({'Coefficients':[1,0]},index=['x','Constant'])

    u = pd.DataFrame(np.random.standard_normal((N,))/10)

    y = (x@beta).values + u.values
    b,V = ols(x,y)

    assert np.allclose(b,beta,atol=tol)

if __name__=='__main__':
    test_ols()
