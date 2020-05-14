import sys
import unittest
sys.path.append('/code/Hessian-analysis-with-tf2_0/')

import numpy as np
import tensorflow as tf

from src.hessians import get_hessian, reshape_hessian
from models.test_func import SimpleFunc

class TestCalculateHessianMatrix(unittest.TestCase):

    def setUp(self):
        self.model = SimpleFunc()
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.X = tf.convert_to_tensor([[1., 2.]])
        self.y = [[13.]]
        
    def test_get_hessian(self):
        # k = -1 because of no bias
        hessian = get_hessian(self.X, self.y, self.model, self.loss_object, k=-1)
        hessian = reshape_hessian(hessian)
        self.assertEqual(hessian.shape, (2, 2))
        np.testing.assert_array_equal(hessian, [[50., 40.],
                                                [40., 32.]])

if __name__ == '__main__':
    unittest.main()