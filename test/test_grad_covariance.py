import sys
import unittest
sys.path.append('/code/Hessian-analysis-with-tf2_0/')

import numpy as np
import tensorflow as tf

from src.grad_covariance import get_gradients, get_grad_covariance
from models.test_func import SimpleFunc

class TestGradCovarianceMatrix(unittest.TestCase):

    def setUp(self):
        self.model = SimpleFunc()
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.X = tf.convert_to_tensor([[1., 2.]])
        self.y = [[11.]]
        
    def test_get_hessian(self):
        # k = -1 because of no bias
        grads = get_gradients(self.X, self.y, self.model, self.loss_object)
        grad_cov_mat = get_grad_covariance(grads, k=-1)
        self.assertEqual(grad_cov_mat.shape, (2, 2))
        np.testing.assert_array_equal(grad_cov_mat, [[400., 320.],
                                                     [320., 256.]])

if __name__ == '__main__':
    unittest.main()