import sys
import unittest
sys.path.append('/code/Hessian-analysis-with-tf2_0/')

from src.preprocess import load_mnist, load_fmnist, load_cifar10

class TestPreprocessData(unittest.TestCase):
    """
    check dimentions of dataset
    """
    def test_load_cifar10(self):
        train_ds, valid_ds, test_ds = load_cifar10()
        for n, train_image in enumerate(train_ds.take(1)):
            self.assertEqual(train_image[0].shape, (128, 32, 32, 3))
            self.assertEqual(train_image[1].shape, (128, 1))

if __name__ == '__main__':
    unittest.main()