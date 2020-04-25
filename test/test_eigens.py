import sys
import unittest
sys.path.append('/code/Hessian-analysis-with-tf2_0/')

import numpy as np
from numpy.testing import assert_array_equal

from src.eigens import calculate_topK_eigens

class TestCalculateEigens(unittest.TestCase):
    """
    check eigenvalues and eigenvectors of some real symmetric matrices
    """
    def test_calculate_topK_eigens(self):
        m = np.array([
            [5, 2],
            [2, 2]
        ])
        eig_val, eig_vec = calculate_topK_eigens(m)
        assert_array_equal(eig_val, np.array([1, 6]))
        # scipy.linalg.eigh gives the normalized selected eigenvector.
        assert_array_equal(eig_vec, np.array([
            [1,  -2] / np.linalg.norm([1,  -2]),
            [-2, -1] / np.linalg.norm([-2, -1])]))
    
if __name__ == '__main__':
    unittest.main()