import pandas as pd
from pandas.errors import InvalidIndexError
from metrics_miscellany.utils import use_indices
import numpy as np

def test_use_indices():
    idx = pd.MultiIndex.from_tuples([(i,) for i in range(4)],names=['i'])
    foo = pd.DataFrame({'cat':['a','b','b','c']},index=idx)

    assert use_indices(foo,['i']).shape[0]==foo.shape[0]
    assert use_indices(foo,['cat']).size==0
    assert use_indices(foo,['cat','i']).shape[0]==foo.shape[0]

    assert np.all(use_indices(foo,['i']).index==idx)

def test_use_indices_invalid_index_falls_back():
    """The InvalidIndexError handler must be reachable.

    It named a class that was never imported, so *any* exception raised
    in the try body surfaced as `NameError: name 'InvalidIndexError' is
    not defined` instead of the intended fallback.
    """
    class Raising(pd.DataFrame):
        @property
        def _constructor(self):
            return Raising

        def reset_index(self,*args,**kwargs):
            raise InvalidIndexError("forced")

    idx = pd.MultiIndex.from_tuples([(i,) for i in range(3)],names=['i'])
    df = Raising({'cat':list('abc')},index=idx)

    assert use_indices(df,['i']) is df

if __name__=='__main__':
    test_use_indices()
    test_use_indices_invalid_index_falls_back()
