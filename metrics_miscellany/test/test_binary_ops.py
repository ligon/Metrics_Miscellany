import pytest
import datamat as dm
import pandas as pd
import numpy as np

@pytest.fixture
def numpy_matrices():
    A = np.array([[1,2],[3,4]])
    B = np.array([[1,1]]).T
    return A, B

@pytest.fixture
def pandas_matrices():
    A = pd.DataFrame([[1,2],[3,4]])
    B = pd.DataFrame([[1,1]]).T
    return A, B

@pytest.fixture
def datamat_matrices():
    A = dm.DataMat([[1,2],[3,4]])
    B = dm.DataMat([[1,1]]).T
    return A, B

def test_matmul(numpy_matrices, pandas_matrices, datamat_matrices):
    for A, B in [numpy_matrices, pandas_matrices, datamat_matrices]:
        C = A @ B
        if isinstance(A, dm.DataMat):
            expected = dm.DataVec if getattr(B, "shape", (None, None))[1] == 1 else dm.DataMat
            assert isinstance(C, expected)
        else:
            assert isinstance(C, type(A))

@pytest.fixture
def datamat_vector():
    A = dm.DataMat([[1,2],[3,4]])
    b = dm.DataVec([1,1])
    return A, b

@pytest.fixture
def pandas_vector():
    A = dm.DataMat([[1,2],[3,4]])
    b = pd.Series([1,1])
    return A, b

def test_matmul_matvec(datamat_vector, pandas_vector):
    for A, b in [datamat_vector, pandas_vector]:
        C = A @ b
        assert isinstance(C, type(b))

if __name__=='__main__':
    pytest.main()
