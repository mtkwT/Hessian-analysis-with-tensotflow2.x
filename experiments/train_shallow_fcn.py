import argparse
import os
import sys
sys.path.append('/code/Hessian-analysis-with-tf2_0/')

import numpy as np
import tensorflow as tf
import wandb

from wandb.keras import WandbCallback
wandb.init(config={"hyper": "parameter"})

from models.shallow_fcn import *
from src.eigens import *
from src.grad_covariance import *
from src.hessians import *
from src.preprocess import *
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
    checkpoint_path = args.save_dir + "cp-{epoch:04d}.ckpt"
    callbacks = [
        # tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, 
            save_weights_only=True,
            period=5,
            verbose=1),
        WandbCallback()
    ]

    return callbacks

def train(train_dataset, valid_dataset, 
          model, optimizer_object, loss_object, acc_object, 
          callbacks, batch_size, epochs):

    model.compile(optimizer=optimizer_object, loss=loss_object, metrics=[acc_object])
    
    history = model.fit(
        train_dataset, validation_data=valid_dataset,
        epochs=epochs, steps_per_epoch=50000 // batch_size,
        callbacks=callbacks
    )
    
    return history

def main():
    # loading dataset
    train_ds, valid_ds, test_ds = load_mnist()

    # setting train option
    model = ShallowFCN(output_dim=args.output_dim)
    loss_object, acc_object, optimizer_object = setup_train_option()
    callbacks = setup_callbacks()

    # training step
    history = train(
        train_ds, valid_ds, 
        model, optimizer_object, loss_object, acc_object, # scheduler, history,
        callbacks=callbacks, batch_size=args.batch_size, epochs=args.epochs
    )

if __name__ == '__main__':
    # set random seed to reproduce the work
    np.random.seed(42)
    tf.random.set_seed(42)

    # setting hyper-parameter
    parser = argparse.ArgumentParser(description='Training Shallow FCN on MNIST')
    parser.add_argument('--dataset'   , type=str  , default='mnist')
    parser.add_argument('--output-dim', type=int  , default=10)
    parser.add_argument('--optimizer' , type=str  , default='sgd')
    parser.add_argument('--lr'        , type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int  , default=128)
    parser.add_argument('--epochs'    , type=int  , default=100)
    parser.add_argument('--device-num', type=int  , default=1)
    parser.add_argument('--save-dir'  , type=str)
    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    # setting gpu config
    setup_gpu_config(device_num=args.device_num)

    main()