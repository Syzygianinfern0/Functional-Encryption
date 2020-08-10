from math import log2

from exceptions import WrongInputError
from schemes.models import Serializable
from utils import batch_exp
from utils import is_array
from utils import is_scalar
from vectors.enc_vector import EncryptedVector


class Projection(Serializable):
    ext_ = 'proj'

    def __init__(self, matrix=None, source=''):
        if source:
            self.fromFile(source)
            return
        if matrix is None:
            return
        assert is_array(matrix)
        n = len(matrix)
        assert n > 0
        assert (is_array(matrix[i]) for i in range(n))
        k = len(matrix[0])
        assert k > 0
        assert (len(matrix[i]) == k for i in range(n))
        assert (is_scalar(matrix[i][j]) for i in range(n) for j in range(k))
        self.n = n
        self.k = k
        self.columns = [[int(matrix[i][j]) for j in range(k)] for i in range(n)]
        self.nbits = []
        for j in range(n):
            m = 1
            for x in self.columns[j]:
                if abs(x) > m:
                    m = abs(x)
            self.nbits.append(int(log2(m)) + 1)

    def __mul__(self, X):
        if isinstance(X, EncryptedVector):
            left = [[1, 1] for i in range(self.k)]
            right = [[1, 1] for i in range(self.k)]
            for j in range(self.n):
                l1 = batch_exp(X.left[j][0], self.columns[j], self.nbits[j])
                l2 = batch_exp(X.left[j][1], self.columns[j], self.nbits[j])
                r1 = batch_exp(X.right[j][0], self.columns[j], self.nbits[j])
                r2 = batch_exp(X.right[j][1], self.columns[j], self.nbits[j])
                for i in range(self.k):
                    left[i][0] *= l1[i]
                    left[i][1] *= l2[i]
                    right[i][0] *= r1[i]
                    right[i][1] *= r2[i]
            return EncryptedVector(group=X.group, simplifier=X.simplifier, left=left, right=right,)
        elif is_array(X):
            import numpy as np

            assert self.n == len(X)
            return np.dot(np.transpose(self.columns), X)
        else:
            raise WrongInputError(X)


class DiagonalQuadraticForms(Serializable):
    ext_ = 'dqf'

    def __init__(self, matrix=None, source=''):
        if source:
            self.fromFile(source)
            return
        if matrix is None:
            return
        assert is_array(matrix)
        k = len(matrix)
        assert k
        assert (is_array(matrix[i]) for i in range(k))
        classes = len(matrix[0])
        assert classes
        assert (len(matrix[i]) == classes for i in range(k))
        assert (is_scalar(matrix[i][j]) for i in range(k) for j in range(classes))
        self.classes = classes
        self.k = k
        self.content = [[int(matrix[i][j]) for j in range(classes)] for i in range(k)]
        self.nbits = []
        for j in range(k):
            m = 1
            for x in self.content[j]:
                if abs(x) > m:
                    m = abs(x)
            self.nbits.append(int(log2(m)) + 1)
