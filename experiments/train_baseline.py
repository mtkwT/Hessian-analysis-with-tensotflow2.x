import os
import sys
sys.path.append('/code/Hessian-analysis-with-tf2_0/')

import numpy as np
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from models.deepobs_3c3d import DeepOBS_3c3d
from src.preprocess import load_mnist, load_cifar10

def setting_train_option():
    # build baseline model
    model = DeepOBS_3c3d(output_dim=10, weight_decay=0.0002)
    
    # setting optimization option
    optimizer_object = tf.keras.optimizers.SGD(learning_rate=6.31e-2)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    acc_object  = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
    
    return model, optimizer_object, loss_object, acc_object

def save_model(checkpoint_dir, model, optimizer_object, epochs):
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoint_dir, f'ckpt-{epochs}')
    
    root = tf.train.Checkpoint(optimizer=optimizer_object, model=model)
    root.save(checkpoint_prefix)
    root.restore(tf.train.latest_checkpoint(checkpoint_dir))

def train(train_ds, valid_ds, 
          model, optimizer_object, loss_object, acc_object,
          batch_size, epochs, checkpoint_dir, is_saved=True):
    
    model.compile(optimizer=optimizer_object, loss=loss_object, metrics=[acc_object])
    model.fit(
        train_ds, validation_data=valid_ds, 
        epochs=epochs, steps_per_epoch=40000 // batch_size, # Here, len(train_images) == 40000
        callbacks=[scheduler, history]
    )
    if is_saved:
        save_model(checkpoint_dir, model, optimizer_object, epochs)

def main():
    # set random seed to reproduce the work
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # download dataset
    train_ds, valid_ds, test_ds = load_cifar10()
    
    # setting train option 
    model, loss_object, acc_object, optimizer_object, scheduler, history = setting_train_option()
    
    # training step
    checkpoint_dir = 'model_checkpoint/cifar10/baseline/'
    train(
        train_ds, valid_ds, 
        model, optimizer_object, loss_object, acc_object, scheduler, history,
        batch_size=128, epochs=100, is_saved=True
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