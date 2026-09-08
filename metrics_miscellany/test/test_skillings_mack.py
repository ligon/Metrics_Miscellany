import pandas as pd
import numpy as np
from metrics_miscellany import tests

def test_sm_against_R():
    """This is an example given in https://cran.r-project.org/web/packages/Skillings.Mack/Skillings.Mack.pdf
    """
    X = pd.DataFrame([[3.2, 3.1, 4.3, 3.5, 3.6, 4.5, np.nan, 4.3, 3.5],
                      [4.1, 3.9, 3.5, 3.6, 4.2, 4.7, 4.2, 4.6, np.nan],
                      [3.8, 3.4, 4.6, 3.9, 3.7, 3.7, 3.4, 4.4, 3.7],
                      [4.2, 4.,  4.8, 4., 3.9, np.nan, np.nan, 4.9, 3.9]])

    # This value of SM statistic matches that from R Skill.Mack routine
    assert np.allclose(tests.skillings_mack(X)[0],15.493)

def test_sm_type1():
    # Now try a random matrix
    ps = pd.Series([tests.skillings_mack(pd.DataFrame(np.random.rand(100,10)),bootstrap=0.02)[1] for i in range(100)])

    # p values should be distributed uniformly, with mean of 1/2
    tstat = (ps.mean()-1/2)/ps.std()

    assert np.abs(tstat)<2

if __name__=='__main__':
    test_sm_against_R()
    test_sm_type1()
