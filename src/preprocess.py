import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split

#TODO: use tf.data

def data_augmentation_cifar(image):
    image = tf.cast(image, tf.float32) / 255.0  # uint8 -> float32
    image = tf.image.random_flip_left_right(image)
    image = tf.pad(image, tf.constant([[2, 2], [2, 2], [0, 0]]), "REFLECT")
    image = tf.image.random_crop(image, [32, 32, 3])
    return image

def load_cifar10():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    X_train, X_valid, y_train, y_valid   = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(
        # lambda image, label: (data_augmentation_cifar(image), tf.cast(label, tf.float32))
        lambda image, label: (tf.cast(image, tf.float32) / 255.0, tf.cast(label, tf.float32))
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

def load_cifar100():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar100.load_data()
    X_train, X_valid, y_train, y_valid   = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(
        # lambda image, label: (data_augmentation_cifar(image), tf.cast(label, tf.float32))
        lambda image, label: (tf.cast(image, tf.float32) / 255.0, tf.cast(label, tf.float32))
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
    
def load_cifar10_for_hessian():
     (train_images, train_labels), (test_images, test_labels) = \
         tf.keras.datasets.cifar10.load_data()
     train_images = (train_images.reshape(-1, 32, 32, 3) / 255).astype(np.float32)
     test_images = (test_images.reshape(-1, 32, 32, 3) / 255).astype(np.float32)

    #  train_labels = np.eye(10)[train_labels].astype(np.float32).reshape(-1, 10)
    #  test_labels = np.eye(10)[test_labels].astype(np.float32).reshape(-1, 10)

    #  train_labels = np.eye(10)[train_labels].astype(np.float32)
    #  test_labels = np.eye(10)[test_labels].astype(np.float32)

     return train_images, test_images, train_labels, test_labels

def load_cifar100_for_hessian():
     (train_images, train_labels), (test_images, test_labels) = \
         tf.keras.datasets.cifar100.load_data()
     train_images = (train_images.reshape(-1, 32, 32, 3) / 255).astype(np.float32)
     test_images = (test_images.reshape(-1, 32, 32, 3) / 255).astype(np.float32)

     train_labels = np.eye(100)[train_labels].astype(np.float32).reshape(-1, 100)
     test_labels = np.eye(100)[test_labels].astype(np.float32).reshape(-1, 100)

     return train_images, test_images, train_labels, test_labels

def load_mnist():
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train, X_valid, y_train, y_valid   = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.map(
        lambda image, label: (tf.cast(image, tf.float32) / 255.0, tf.cast(label, tf.float32))
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
    
def load_mnist_for_hessian():
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

def load_reuters(batch_size=32):
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.reuters.load_data(
        path='reuters.npz', num_words=None, skip_top=0, maxlen=None, test_split=0.2,
        seed=113, start_char=1, oov_char=2, index_from=3
    )

    vocab_size = len(tf.keras.datasets.reuters.get_word_index(path="reuters_word_index.npz"))
    X_train, X_valid, y_train, y_valid   = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # for convert non-rectangular Python sequence to Tensor 
    X_train = tf.ragged.constant(X_train.tolist())
    X_valid = tf.ragged.constant(X_valid.tolist())
    X_test  = tf.ragged.constant(X_test.tolist())
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = (train_dataset.shuffle(1024).batch(batch_size))

    valid_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    valid_dataset = (valid_dataset.shuffle(1024).batch(batch_size))

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = (test_dataset.shuffle(1024).batch(batch_size))

    return vocab_size, train_dataset, valid_dataset, test_dataset

def load_imdb_reviews(batch_size=32):
    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)

    train_examples, test_examples = dataset['train'], dataset['test']
    encoder = info.features['text'].encoder
    print('Vocabulary size: {}'.format(encoder.vocab_size))

    train_dataset = (train_examples.shuffle(1024).padded_batch(batch_size))
    test_dataset = (test_examples.padded_batch(batch_size))

    return encoder, train_dataset, test_dataset

def labeler(example, index):
    return example, tf.cast(index, tf.int64)  

def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label

def encode_map_fn(text, label):
    # py_func は返り値の Tensor に shape を設定しません
    encoded_text, label = tf.py_function(encode, 
                                        inp=[text, label], 
                                        Tout=(tf.int64, tf.int64))
    # `tf.data.Datasets` はすべての要素に shape が設定されているときにうまく動きます
    #  なので、shape を手動で設定しましょう
    encoded_text.set_shape([None])
    label.set_shape([])
    
    return encoded_text, label

def load_illiad(batch_size=64):
    DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
    FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

    for name in FILE_NAMES:
        text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL+name)
    
    parent_dir = os.path.dirname(text_dir)

    labeled_data_sets = []
    for i, file_name in enumerate(FILE_NAMES):
        lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
        labeled_data_sets.append(labeled_dataset)
    
    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
    all_labeled_data = all_labeled_data.shuffle(50000, reshuffle_each_iteration=False)

    tokenizer = tfds.features.text.Tokenizer()
    vocabulary_set = set()
    for text_tensor, _ in all_labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)
    vocab_size = len(vocabulary_set)

    encoder = tfds.features.text.TokenTextEncoder(vocabulary_set)

    all_encoded_data = all_labeled_data.map(encode_map_fn)

    train_dataset = all_encoded_data.skip(5000).shuffle(50000)
    train_dataset = train_dataset.padded_batch(batch_size)

    test_dataset  = all_encoded_data.take(5000)
    test_dataset  = test_dataset.padded_batch(batch_size)

    vocab_size += 1

    return vocab_size, train_dataset, test_dataset


if __name__ == "__main__":
    vocab_size, train_dataset, valid_dataset, test_dataset = load_reuters()
    print(vocab_size)

    encoder, train_dataset, test_dataset = load_imdb_reviews()
    print(encoder.vocab_size)

    vocab_size, train_dataset, test_dataset = load_illiad()
    print(vocab_size)