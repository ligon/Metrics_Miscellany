import pandas as pd
from metrics_miscellany.utils import use_indices
import numpy as np

def test_use_indices():
    idx = pd.MultiIndex.from_tuples([(i,) for i in range(4)],names=['i'])
    foo = pd.DataFrame({'cat':['a','b','b','c']},index=idx)

    assert use_indices(foo,['i']).shape[0]==foo.shape[0]
    assert use_indices(foo,['cat']).size==0
    assert use_indices(foo,['cat','i']).shape[0]==foo.shape[0]

    assert np.all(use_indices(foo,['i']).index==idx)

if __name__=='__main__':
    test_use_indices()
