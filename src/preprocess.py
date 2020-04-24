import numpy as np
import tensorflow as tf

#TODO: use tf.data

def load_mnist():
    (train_images, train_labels), (test_images, test_labels) = \
        tf.keras.datasets.mnist.load_data()
    
    train_images = (train_images.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    test_images = (test_images.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    
    train_labels = np.eye(10)[train_labels].astype(np.float32).reshape(-1, 10)
    test_labels = np.eye(10)[test_labels].astype(np.float32).reshape(-1, 10)
    
    return train_images, test_images, train_labels, test_labels

def load_fmnist():
    (train_images, train_labels), (test_images, test_labels) = \
        tf.keras.datasets.fashion_mnist.load_data()
    train_images = (train_images.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    test_images = (test_images.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    
    train_labels = np.eye(10)[train_labels].astype(np.float32).reshape(-1, 10)
    test_labels = np.eye(10)[test_labels].astype(np.float32).reshape(-1, 10)
    
    return train_images, test_images, train_labels, test_labels

def load_cifar10():
    (train_images, train_labels), (test_images, test_labels) = \
        tf.keras.datasets.cifar10.load_data()
    train_images = (train_images.reshape(-1, 32, 32, 3) / 255).astype(np.float32)
    test_images = (test_images.reshape(-1, 32, 32, 3) / 255).astype(np.float32)
    
    train_labels = np.eye(10)[train_labels].astype(np.float32).reshape(-1, 10)
    test_labels = np.eye(10)[test_labels].astype(np.float32).reshape(-1, 10)
    
    return train_images, test_images, train_labels, test_labels
