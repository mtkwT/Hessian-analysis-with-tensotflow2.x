import argparse
import os
import sys
import time
sys.path.append('/code/Hessian-analysis-with-tf2_0/')

import numpy as np
import tensorflow as tf
import wandb

from models.deepobs_3c3d import DeepOBS_3c3d
from src.eigens import *
from src.hessians import *
from src.preprocess import load_mnist_for_hessian
from src.utils import setup_gpu_config
from models.shallow_fcn import *

def setup_train_option():
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    acc_object  = tf.keras.metrics.CategoricalAccuracy(name='accuracy')

    if   args.optimizer == 'sgd':
        optimizer_object = tf.keras.optimizers.SGD( learning_rate=args.lr)
    elif args.optimizer == 'adam':
        optimizer_object = tf.keras.optimizers.Adam(learning_rate=args.lr)

    return loss_object, acc_object, optimizer_object

def create_block_diag_hessian(mat_A, mat_B):
    block_diag_hessian = np.block([
        [mat_A, np.zeros(shape=(mat_A.shape[0], mat_B.shape[0]))],
        [np.zeros(shape=(mat_B.shape[0], mat_A.shape[0])), mat_B] 
    ])
    return block_diag_hessian

def calc_hessian_spectrum(
    X, y, model, loss_object, 
    optimizer_object, batch_size, top_k, mode):

    # calculate mean Hessian matrix (train/test) & eigen-vector
    if mode == 'penultimate':
        hessian_matrix = calculate_mean_hessian(X, y, model, loss_object, batch_size, k=2)
    elif mode == 'full':
        hessian_block_A = calculate_mean_hessian(X, y, model, loss_object, batch_size, k=0)
        hessian_block_B = calculate_mean_hessian(X, y, model, loss_object, batch_size, k=2)
        hessian_matrix = create_block_diag_hessian(hessian_block_A, hessian_block_B)

    print(hessian_matrix.shape)
    eigen_values, _ = calculate_topK_eigens(hessian_matrix, top_k)

    return eigen_values

def main():
    # download dataset
    X_train, X_test, y_train, y_test = load_mnist_for_hessian()
    model = ShallowFCN(hidden_dim=args.hidden_dim, output_dim=args.output_dim)
    # elif args.dataset == 'mnist':
    #     X_train, X_test, y_train, y_test = load_mnist_for_hessian()
    #     model = 

    # setting train option
    loss_object, acc_object, optimizer_object = setup_train_option()

    # setting model for estimating hessian
    model.compile(optimizer=optimizer_object, loss=loss_object, metrics=[acc_object])
    checkpoint_dir = f'/code/Hessian-analysis-with-tf2_0/model_checkpoint/mnist/fcn/{args.optimizer}/hidden_dim={args.hidden_dim}/'
    model.load_weights(checkpoint_dir+'cp-0100.ckpt')
    model.evaluate(X_test, y_test, verbose=2)

    # estimating hessian spectrum on full train data
    hessian_spectrum_train = \
        calc_hessian_spectrum(
            X_train, y_train, model, loss_object, optimizer_object, 
            batch_size=args.batch_size, top_k=args.top_k, mode=args.mode
        )
    print(hessian_spectrum_train)

    # estimating hessian spectrum on full test data
    hessian_spectrum_test = \
        calc_hessian_spectrum(
            X_test, y_test, model, loss_object, optimizer_object, 
            batch_size=args.batch_size, top_k=args.top_k, mode=args.mode
        )
    print(hessian_spectrum_test)

    wandb.log({
        'max_eigenvalue_train': hessian_spectrum_train[0],
        'max_eigenvalue_test' : hessian_spectrum_test[0]
    })

if __name__ == "__main__":

    # setting hyper-parameter
    parser = argparse.ArgumentParser(description='Compare Hessian spectrum between full hessian and final-layer-wise hessian')
    parser.add_argument('--dataset'       , type=str  , default='mnist')
    parser.add_argument('--output-dim'    , type=int  , default=10)
    parser.add_argument('--hidden-dim'    , type=int  , default=10)
    parser.add_argument('--optimizer'     , type=str  , default='sgd')
    parser.add_argument('--lr'            , type=float, default=1e-2)
    parser.add_argument('--device-num'    , type=int  , default=0)
    parser.add_argument('--batch-size'    , type=int  , default=1024)
    parser.add_argument('--top-k'         , type=int  , default=1)
    parser.add_argument('--mode'          , type=str  , default='penultimate')
    # parser.add_argument('--checkpoint-dir', type=str)
    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))

    ###############################################################################
    # wandb setup
    ###############################################################################
    hyperparams = {
        'optimizer' : args.optimizer  ,
        'hidden_dim': args.hidden_dim ,
        'mode'      : args.mode
    }
    exp_name = time.strftime('%Y%m%d-%H%M%S')
    wandb.init(config=hyperparams, project='compare-hessian-spctral-between-full_H-and-penultimate_H', entity='mtkwt', name=exp_name)

    setup_gpu_config(device_num=0)

    main()
