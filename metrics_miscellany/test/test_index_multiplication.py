import inspect

import pytest
import datamat as dm
import pandas as pd
import numpy as np

@pytest.fixture
def setup_indices():
    idx = pd.MultiIndex.from_tuples([(0,0,0),(0,0,1),(1,0,0),(1,0,1)], names=['i','j','k'])
    return idx

@pytest.fixture
def setup_data_matrices(setup_indices):
    X = dm.DataMat([[1,2,3,4]], columns=setup_indices, idxnames=['l'])
    Y = dm.DataMat([[1,2,3,0]], columns=setup_indices.droplevel('j'), idxnames='m').T
    return X, Y

def test_index_multiplication(setup_data_matrices):
    X, Y = setup_data_matrices
    result = X @ Y
    assert result.index.names == ['l']

    # X.columns carries a vestigial level ('j') that Y.index lacks, so this
    # is the *reconciling* multiplication.  Its spelling moved between
    # DataMat releases: through 0.2.1 `strict=True` attempted reconciliation,
    # while 0.2.4 redefined `strict=True` as "labels must match exactly" and
    # introduced `align=True` for the reconciling behaviour.  The estimators
    # here only ever use `@`, which is unchanged, so pick whichever spelling
    # the installed DataMat offers rather than pinning a version.
    kwargs = ({'align': True}
              if 'align' in inspect.signature(X.matmul).parameters
              else {'strict': True})
    reconciled = X.matmul(Y, **kwargs)
    assert reconciled.index.names == ['l']

if __name__=='__main__':
    pytest.main()
