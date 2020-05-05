import sys
sys.path.append('/code/Hessian-analysis-with-tf2_0/')

import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from src.preprocess import load_mnist, load_cifar10
from models.small_cnn import SmallCNN
from models.deepobs_3c3d import DeepOBS_3c3d
from models.resnet import ResNet50
from models.vgg16 import Vgg16

@tf.function
def train_step(X, y, model, loss_object, optimizer, train_loss, train_acc):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different behavior during training versus inference (e.g. Dropout).
        preds     = model(X, training=True)
        loss      = loss_object(y, preds)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_acc(y, preds)

@tf.function
def test_step(X, y, model, loss_object, test_loss, test_acc):
    preds = model(X, training=False)
    loss  = loss_object(y, preds)
    
    test_loss(loss)
    test_acc(y, preds)

def split_train_data(X, y):
    """
    create validation dataset for prevent information leak
    """
    X_train, X_valid, y_train, y_valid = \
        train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_valid, y_train, y_valid

def train(train_ds, valid_ds, model, batch_size, epochs, is_saved=True):
    """
    train step using validation dataset for epochs
    """
    train_loss     = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    valid_loss     = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

    # n_batches = X_train.shape[0] // batch_size    
    for epoch in range(epochs):
        for images, labels in train_ds:
            train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy)
 
        for valid_images, valid_labels in valid_ds:
            test_step(valid_images, valid_labels, model, loss_object, valid_loss, valid_accuracy)
        # _X_train, _y_train = shuffle(X_train, y_train, random_state=42)

        # for batch in range(n_batches):
        #     start = batch * batch_size
        #     end = start + batch_size
        #     train_step(
        #         _X_train[start:end], _y_train[start:end],
        #         model, loss_object, optimizer,
        #         train_loss, train_accuracy
        #     )
        
        # test_step(
        #     X_valid, y_valid, 
        #     model, loss_object, 
        #     valid_loss, valid_accuracy
        # )
        template = 'Epoch {}, Loss: {}, Accuracy: {}%, Valid Loss: {}, Valid Accuracy: {}%'
        print(template.format(
            epoch + 1,
            train_loss.result(),
            train_accuracy.result() * 100,
            valid_loss.result(),
            valid_accuracy.result() * 100))

        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()

    # if is_saved:
    #     checkpoint_dir = f'model_checkpoint/{args.name}/{args.arch}/{args.optimizer}/seed_{str(args.seed)}/epochs_{str(args.epochs)}/batch_size_{str(args.batch_size)}/'
    #     os.makedirs(checkpoint_dir, exist_ok=True)
    #     checkpoint_prefix = os.path.join(checkpoint_dir, f'ckpt-{epoch+1}')
    #     root = tf.train.Checkpoint(optimizer=optimizer, model=model)
    #     root.save(checkpoint_prefix)
    #     root.restore(tf.train.latest_checkpoint(checkpoint_dir))

    return train_loss, train_accuracy, valid_loss, valid_accuracy

def main():
    # set random seed to reproduce the work
    np.random.seed(1234)
    tf.random.set_seed(1234)

    # setting validation dataset
    # X_train, X_test,  y_train, y_test  = load_mnist()
    # X_train, X_test,  y_train, y_test  = load_cifar10()
    # X_train, X_valid, y_train, y_valid = split_train_data(X_train, y_train)
    train_ds, valid_ds, test_ds = load_cifar10()

    # build model
    # model = SmallCNN()
    model = DeepOBS_3c3d(output_dim=10, weight_decay=0.002)
    # model = ResNet50(output_dim=10)
    # model = Vgg16(output_dim=10)

    # train step
    train_loss, train_accuracy, valid_loss, valid_accuracy = \
        train(train_ds, valid_ds, model, batch_size=128, epochs=300)
            
if __name__ == '__main__':

    # gpu config setting
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
            print('memory growth:', tf.config.experimental.get_memory_growth(physical_devices[k]))
    else:
        print("Not enough GPU hardware devices available")
    
    if physical_devices:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.experimental.set_visible_devices(physical_devices[1], 'GPU')
            logical_physical_devices = tf.config.experimental.list_logical_devices('GPU')
            print(len(physical_devices), "Physical GPUs,", len(logical_physical_devices), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    main()