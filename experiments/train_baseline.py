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

def lr_scheduler(epoch):
    if epoch <= 60: return 0.01
    elif epoch <= 85: return 0.001
    else: return 0.00001

def setting_train_option():
    # build baseline model
    model = DeepOBS_3c3d(output_dim=10, weight_decay=0.0002)
    
    # setting optimization option
    optimizer_object = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    acc_object  = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    
    # setting learning scheduler
    scheduler = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    history = tf.keras.callbacks.History()

    return model, loss_object, acc_object, optimizer_object, scheduler, history

def train(
        train_ds, valid_ds, 
        model, optimizer_object, loss_object, acc_object, scheduler, history,
        batch_size, epochs, is_saved=True):
    
    model.compile(optimizer=optimizer_object, loss=loss_object, metrics=[acc_object])
    model.fit(
        train_ds, validation_data=valid_ds, 
        epochs=epochs, steps_per_epoch=40000 // batch_size, # Here, len(train_images) == 40000
        callbacks=[scheduler, history]
    )

    if is_saved:
        checkpoint_dir = 'model_checkpoint/cifar10/baseline/'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_prefix = os.path.join(checkpoint_dir, f'ckpt-{epoch+1}')
        root = tf.train.Checkpoint(optimizer=optimizer, model=model)
        root.save(checkpoint_prefix)
        root.restore(tf.train.latest_checkpoint(checkpoint_dir))

def main():
    # set random seed to reproduce the work
    np.random.seed(42)
    tf.random.set_seed(42)
    # download dataset
    train_ds, valid_ds, test_ds = load_cifar10()
    # setting train option 
    model, loss_object, acc_object, optimizer_object, scheduler, history = setting_train_option()
    # training step
    train(
        train_ds, valid_ds, 
        model, optimizer_object, loss_object, acc_object, scheduler, history,
        batch_size=128, epochs=150, is_saved=True
    )
            
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