from metrics_miscellany.utils import dummies, use_indices
import pandas as pd

def test_dummies():
    idx = pd.MultiIndex.from_tuples([(i,) for i in range(4)],names=['i'])
    foo = pd.DataFrame({'cat':['a','b','b','c']},index=idx)

    assert dummies(foo,['i']).shape == (4,4)
    assert dummies(foo,['cat']).shape == (4,3)

def test_dummies_preserves_level_order():
    """The order of MultiIndex level names must match the caller-supplied
    order of =cols= and not depend on Python's hash randomization.
    """
    idx = pd.MultiIndex.from_product([['m','f'],['young','old']],
                                     names=['sex','age'])
    df = pd.DataFrame({'val':[1,2,3,4]}, index=idx)

    out = dummies(df, ['sex','age'])
    assert list(out.columns.names) == ['sex','age']
    assert list(out.columns) == [
        ('f','old'), ('f','young'), ('m','old'), ('m','young'),
    ]

    out2 = dummies(df, ['age','sex'])
    assert list(out2.columns.names) == ['age','sex']

def test_dummies_mixed_index_and_column():
    """When =cols= mixes index levels and regular columns, the user-supplied
    order must be honored.
    """
    idx = pd.MultiIndex.from_product([[0,1],[10,20]], names=['i','t'])
    df = pd.DataFrame({'cat':['a','b','b','c']}, index=idx)

    out_cat_first = dummies(df, ['cat','i'])
    assert list(out_cat_first.columns.names) == ['cat','i']

    out_i_first = dummies(df, ['i','cat'])
    assert list(out_i_first.columns.names) == ['i','cat']

if __name__=='__main__':
    test_dummies()
    test_dummies_preserves_level_order()
    test_dummies_mixed_index_and_column()
