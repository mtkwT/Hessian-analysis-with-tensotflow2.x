import sys
import unittest
sys.path.append('/code/Hessian-analysis-with-tf2_0/')

from src.preprocess import load_mnist
from models.small_cnn import SmallCNN

class TestModelForward(unittest.TestCase):
    """
    check dimention of model's output
    """
    def test_small_cnn(self):
        X_train, X_test, y_train, y_test = load_mnist()
        model = SmallCNN()
        pred_train = model(X_train)
        pred_test = model(X_test)
        self.assertEqual(pred_train.shape, y_train.shape)
        self.assertEqual(pred_test.shape, y_test.shape)

if __name__ == '__main__':
    unittest.main()