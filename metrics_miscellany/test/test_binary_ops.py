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


def test_kron_flat_index_string_labels():
    """Two flat-Index DataFrames with multi-character string labels must
    round-trip through kron without character-unpacking the labels."""
    from metrics_miscellany.utils import kron
    A = pd.DataFrame([[1, 2], [3, 4]],
                     index=pd.Index(['alpha', 'beta'], name='l'),
                     columns=pd.Index(['xx', 'yy'], name='c'))
    B = pd.DataFrame([[5, 6], [7, 8]],
                     index=pd.Index(['gamma', 'delta'], name='k'),
                     columns=pd.Index(['pp', 'qq'], name='d'))
    K = kron(A, B)
    assert K.shape == (4, 4)
    assert list(K.index.names) == ['l', 'k']
    assert list(K.columns.names) == ['c', 'd']
    np.testing.assert_array_equal(K.values, np.kron(A.values, B.values))
    assert ('alpha', 'gamma') in K.index
    assert ('xx', 'pp') in K.columns


def test_kron_mixed_multi_and_flat():
    """Mixing MultiIndex and flat Index should produce a properly named
    cartesian-product MultiIndex on the output."""
    from metrics_miscellany.utils import kron
    A = pd.DataFrame(np.arange(4).reshape(2, 2),
                     index=pd.MultiIndex.from_tuples([(0, 'a'), (0, 'b')],
                                                    names=['i', 'j']),
                     columns=pd.Index(['c0', 'c1'], name='c'))
    B = pd.DataFrame(np.arange(6).reshape(3, 2),
                     index=pd.Index(range(3), name='k'),
                     columns=pd.MultiIndex.from_tuples([('x', 1), ('x', 2)],
                                                      names=['s', 't']))
    K = kron(A, B)
    assert list(K.index.names) == ['i', 'j', 'k']
    assert list(K.columns.names) == ['c', 's', 't']
    assert K.shape == (6, 4)
    np.testing.assert_array_equal(K.values, np.kron(A.values, B.values))


def test_kron_with_ndarray_operand():
    """kron should accept a DataFrame paired with a NumPy ndarray."""
    from metrics_miscellany.utils import kron
    A = pd.DataFrame([[1, 2], [3, 4]],
                     index=['r0', 'r1'], columns=['c0', 'c1'])
    B = np.array([[5], [6]])
    K = kron(A, B)
    assert K.shape == (4, 2)
    np.testing.assert_array_equal(K.values, np.kron(A.values, B))


if __name__=='__main__':
    pytest.main()
