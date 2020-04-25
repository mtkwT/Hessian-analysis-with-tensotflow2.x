import sys
import unittest
sys.path.append('/code/Hessian-analysis-with-tf2_0/')

from src.preprocess import load_mnist, load_fmnist, load_cifar10

class TestPreprocessData(unittest.TestCase):
    """
    check dimentions of dataset
    """
    def test_load_mnist(self):
        X_train, X_test, y_train, y_test = load_mnist()
        self.assertEqual(X_train.shape, (60000, 28, 28, 1))
        self.assertEqual(X_test.shape,  (10000, 28, 28, 1))
        self.assertEqual(y_train.shape, (60000, 10))
        self.assertEqual(y_test.shape,  (10000, 10))
    
    def test_load_fmnist(self):
        X_train, X_test, y_train, y_test = load_fmnist()
        self.assertEqual(X_train.shape, (60000, 28, 28, 1))
        self.assertEqual(X_test.shape,  (10000, 28, 28, 1))
        self.assertEqual(y_train.shape, (60000, 10))
        self.assertEqual(y_test.shape,  (10000, 10))
    
    def test_load_cifar10(self):
        X_train, X_test, y_train, y_test = load_cifar10()
        self.assertEqual(X_train.shape, (50000, 32, 32, 3))
        self.assertEqual(X_test.shape,  (10000, 32, 32, 3))
        self.assertEqual(y_train.shape, (50000, 10))
        self.assertEqual(y_test.shape,  (10000, 10))


if __name__ == '__main__':
    unittest.main()