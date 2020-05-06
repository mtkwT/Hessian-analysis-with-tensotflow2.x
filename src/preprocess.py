import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

#TODO: use tf.data

def data_augmentation(image):
    # image = tf.image.resize_image_with_crop_or_pad(image, image_size + 4, image_size + 4)
    image = tf.cast(image, tf.float32) / 255.0  # uint8 -> float32
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_brightness(image, max_delta=63. / 255.)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.pad(image, tf.constant([[2, 2], [2, 2], [0, 0]]), "REFLECT")
    image = tf.image.random_crop(image, [32, 32, 3])
    return image

def load_cifar10():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train, X_valid, y_train, y_valid   = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(
        lambda image, label: (data_augmentation(image), tf.cast(label, tf.float32))
    ).shuffle(len(X_train)).repeat().batch(128)

    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    valid_dataset = valid_dataset.map(
        lambda image, label: (tf.cast(image, tf.float32) / 255.0, tf.cast(label, tf.float32))
    ).batch(128)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.map(
        lambda image, label: (tf.cast(image, tf.float32) / 255.0, tf.cast(label, tf.float32))
    ).batch(128)

    return train_dataset, valid_dataset, test_dataset

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
