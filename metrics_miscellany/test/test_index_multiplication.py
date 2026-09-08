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
    X.matmul(Y, strict=True)

if __name__=='__main__':
    pytest.main()
