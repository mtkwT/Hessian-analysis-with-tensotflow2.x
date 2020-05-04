import sys
import unittest
sys.path.append('/code/Hessian-analysis-with-tf2_0/')

import tensorflow as tf

from src.preprocess import load_mnist
from src.hessians import get_hessian, reshape_hessian
from models.small_cnn import SmallCNN

class TestCalculateHessianMatrix(unittest.TestCase):

    def setUp(self):
        X_train, X_test, y_train, y_test = load_mnist()
        self.model = SmallCNN()
        self.loss_object = tf.losses.CategoricalCrossentropy()
        self.X = X_train[:10]
        self.y = y_train[:10]

    def test_get_hessian_dim(self):
        hessian = get_hessian(self.X, self.y, self.model, self.loss_object, k=-2)
        hessian = reshape_hessian(hessian)
        self.assertEqual(hessian.shape, (1280,1280))

    # TODO: Add test of check the value of the Hessian Matrix (for example Trace of Hessian)

if __name__ == '__main__':
    unittest.main()