import pandas as pd
import numpy as np
import pytest
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

@pytest.mark.slow
def test_sm_type1(reps=100):
    """Under the null the p-values are uniform, so their mean is 1/2.

    Marked slow: each call bootstraps, and 100 of them take about 15
    minutes, which is more than `make quick-check` should carry.

    Seeded, because the assertion is a two-sided test at roughly the 5%
    level -- unseeded it would fail about one run in twenty by construction.
    """
    rng = np.random.default_rng(0)

    ps = pd.Series([tests.skillings_mack(pd.DataFrame(rng.random((100,10))),
                                         bootstrap=0.02)[1]
                    for _ in range(reps)])

    # Standard error of the mean, not the standard deviation.  Dividing by
    # ps.std() tolerates a deviation sqrt(reps) times too large: p-values
    # drawn from beta(0.5,2) -- mean 0.174, grossly non-uniform -- pass that
    # version, as do p-values uniform on [0,0.5].
    tstat = (ps.mean() - 1/2)/(ps.std()/np.sqrt(reps))

    assert np.abs(tstat) < 2, (
        f"mean p-value {ps.mean():.4f} is not 1/2 (t = {tstat:.3f})")

if __name__=='__main__':
    test_sm_against_R()
    test_sm_type1()
