import argparse
import os
import sys
sys.path.append('/code/Hessian-analysis-with-tf2_0/')

import numpy as np
import tensorflow as tf
import wandb

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from wandb.keras import WandbCallback
wandb.init(config={"hyper": "parameter"})

from models.build_model_keras_app import build_densenet121
from src.preprocess import load_cifar10, load_cifar100
from src.utils import setup_gpu_config

def setup_train_option():
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    acc_object  = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    if   args.optimizer == 'sgd':
        optimizer_object = tf.keras.optimizers.SGD( learning_rate=args.lr)
    elif args.optimizer == 'adam':
        optimizer_object = tf.keras.optimizers.Adam(learning_rate=args.lr)

    return loss_object, acc_object, optimizer_object

def setup_callbacks():
    """
    add callback
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        WandbCallback()
    ]

    return callbacks

def train(train_dataset, valid_dataset, 
          model, optimizer_object, loss_object, acc_object, 
          callbacks, batch_size, epochs, checkpoint_dir):

    model.compile(optimizer=optimizer_object, loss=loss_object, metrics=[acc_object])
    history = model.fit(
        train_dataset, validation_data=valid_dataset,
        epochs=epochs, steps_per_epoch=40000 // batch_size,
        callbacks=callbacks
    )
    model.save_weights(checkpoint_dir)

    return history
    
def main():
    # download dataset
    if   args.dataset == 'cifar10':
        train_ds, valid_ds, test_ds = load_cifar10()
    elif args.dataset == 'cifar100':
        train_ds, valid_ds, test_ds = load_cifar100()

    # setting train option
    model = build_densenet121(input_shape=(32, 32, 3), output_dim=args.output_dim)
    loss_object, acc_object, optimizer_object = setup_train_option()
    callbacks = setup_callbacks()

    # training step
    history = train(
        train_ds, valid_ds, 
        model, optimizer_object, loss_object, acc_object, # scheduler, history,
        callbacks=callbacks, batch_size=128, epochs=30, checkpoint_dir=args.save_dir
    )
            
if __name__ == '__main__':
    # set random seed to reproduce the work
    np.random.seed(42)
    tf.random.set_seed(42)

    # setting hyper-parameter
    parser = argparse.ArgumentParser(description='Training DenseNet121')
    parser.add_argument('--dataset'   , type=str  , default='cifar10')
    parser.add_argument('--output-dim', type=int  , default=10)
    parser.add_argument('--optimizer' , type=str  , default='sgd')
    parser.add_argument('--lr'        , type=float, default=1e-2)
    parser.add_argument('--device-num', type=int  , default=0)
    parser.add_argument('--save-dir'  , type=str)
    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    # setting gpu config
    setup_gpu_config(device_num=args.device_num)

    main()