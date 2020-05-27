import argparse
import os
import sys
sys.path.append('/code/Hessian-analysis-with-tf2_0/')

import tensorflow as tf

from models.deepobs_3c3d import DeepOBS_3c3d
from src.eigens import *
from src.hessians import *
from src.preprocess import load_cifar10_for_hessian, load_cifar100_for_hessian
from src.utils import setup_gpu_config
from experiments.train_densenet121 import build_densenet121

def setup_train_option():
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    acc_object  = tf.keras.metrics.CategoricalAccuracy(name='accuracy')
    optimizer_object = tf.keras.optimizers.SGD( learning_rate=1e-2)
    # optimizer_object = tf.keras.optimizers.Adam(learning_rate=1e-4)
    return loss_object, acc_object, optimizer_object

def calc_hessian_spectrum(
    X, y, model, loss_object, 
    optimizer_object, batch_size, k):

    # calculate mean Hessian matrix (train/test) & eigen-vector
    hessian_matrix = calculate_mean_hessian( X, y, model, loss_object, batch_size)
    
    eigen_values, _ = calculate_topK_eigens(hessian_matrix, k)

    return eigen_values

def main():
    # X_train, X_test, y_train, y_test = load_cifar10_for_hessian()
    X_train, X_test, y_train, y_test = load_cifar100_for_hessian()
    model = build_densenet121(output_dim=100)
    loss_object, acc_object, optimizer_object = setup_train_option()

    checkpoint_dir = 'model_checkpoint/cifar100/densenet121/Adam/'
    # checkpoint_dir = 'model_checkpoint/cifar100/densenet121/SGD/'
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # print(latest)
    # model.load_weights(latest)
    # latest = tf.train.latest_checkpoint(checkpoint_dir)
    # print(latest)
    # checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer_object)
    # checkpoint.restore(latest)

    model.compile(optimizer=optimizer_object, loss=loss_object, metrics=[acc_object])
    model.load_weights(checkpoint_dir)
    model.evaluate(X_test, y_test, verbose=2)

    hessian_spectrum = \
        calc_hessian_spectrum(
            X_train, y_train, model, loss_object, optimizer_object, 
            batch_size=1024, k=200
        )
    print(hessian_spectrum)

if __name__ == "__main__":

    setup_gpu_config(device_num=0)
    main()
