import argparse
import copy
import json
import os
import sys
sys.path.append('/code/Hessian-analysis-with-tf2_0/')
import time

import numpy as np
import tensorflow as tf
import wandb

from tqdm import tqdm
from logging import getLogger, StreamHandler, DEBUG
logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(DEBUG)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

from models.deepobs_3c3d import DeepOBS_3c3d
from src.eigens import *
from src.hessians import *
from src.preprocess import load_cifar10_for_hessian, load_cifar100_for_hessian, load_mnist_for_hessian
from src.utils import setup_gpu_config
from models.build_model_keras_app import *
from models.shallow_fcn import *

def setup_train_option():
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    acc_object  = tf.keras.metrics.CategoricalAccuracy(name='accuracy')

    if   args.optimizer == 'sgd':
        optimizer_object = tf.keras.optimizers.SGD( learning_rate=args.lr)
    elif args.optimizer == 'adam':
        optimizer_object = tf.keras.optimizers.Adam(learning_rate=args.lr)

    return loss_object, acc_object, optimizer_object

def calc_top2_eigenvectors(X, y, model, loss_object, optimizer_object, batch_size, layer_num):

    # calculate mean Hessian matrix (train/test) & eigen-vector
    hessian_matrix = calculate_mean_hessian( X, y, model, loss_object, batch_size, k=layer_num)
    
    _, top2_eigen_vectors = calculate_topK_eigens(hessian_matrix, k=2)

    return top2_eigen_vectors

def calc_loss_surface(model, X, y, top2_eigen_vectors, init_penultimate_weights):
    
    v1 = top2_eigen_vectors.T[-1].reshape(model.trainable_variables[-2].shape)
    v2 = top2_eigen_vectors.T[-2].reshape(model.trainable_variables[-2].shape)

    loss_values_2D = []
    for a1 in tqdm([(_ / 100) for _ in range(-50, 51, 5)]):
        loss_values_1D = []

        for a2 in [(_ / 100) for _ in range(-50, 51, 5)]:
            model.trainable_variables[-2].assign_add(a1*v1 + a2*v2)                          # give weight perturbators
            loss, acc = model.evaluate(X[:args.batch_size], y[:args.batch_size], verbose=0)  # calc loss on grid point
            loss_values_1D.append(loss)
            model.trainable_variables[-2].assign(init_penultimate_weights)                   # weight initialization
        
        loss_values_2D.append(loss_values_1D)
    
    return np.array(loss_values_2D)

def main():
     # download dataset
    if   args.dataset.lower() == 'cifar10':
        X_train, X_test, y_train, y_test = load_cifar10_for_hessian()
    elif args.dataset.lower() == 'cifar100':
        X_train, X_test, y_train, y_test = load_cifar100_for_hessian()
    
    for seed in range(1,6):
        if args.model.lower() == 'densenet121':
            model = build_densenet121(
                input_shape=(32, 32, 3), 
                output_dim=args.output_dim)
        elif args.model.lower() == 'resnet50':
            model = build_resnet50(
                input_shape=(32, 32, 3), 
                output_dim=args.output_dim)
        elif args.model.lower() == 'mobilenet':
            model = build_mobilenet(
                input_shape=(32, 32, 3), 
                output_dim=args.output_dim)

        # setting train option
        loss_object, acc_object, optimizer_object = setup_train_option()

        # setting model
        model.compile(optimizer=optimizer_object, loss=loss_object, metrics=[acc_object])
        model.load_weights(args.checkpoint_dir+f'seed={seed}/')
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
        # Note: If you use a shallow copy, the value will be changed together with the perturbation when you change it later, so use deepcopy
        init_penultimate_weights = copy.deepcopy(model.trainable_variables[-2])

        # estimating train loss surface on full train data
        logger.debug('Calculate Hessian Top2-EigenVectors on Train Dataset...')
        train_top2_eigen_vectors = calc_top2_eigenvectors(
            X=X_train, 
            y=y_train, 
            model=model, 
            loss_object=loss_object, 
            optimizer_object=optimizer_object, 
            batch_size=args.batch_size, 
            layer_num=args.layer_num
        )
        logger.debug('Calculate Local Loss Surface on Train Dataset...')
        try:
            train_loss_values_2D += calc_loss_surface(
                model=model, 
                X=X_train, 
                y=y_train, 
                top2_eigen_vectors=train_top2_eigen_vectors,
                init_penultimate_weights=init_penultimate_weights
            )
        except:
            train_loss_values_2D  = calc_loss_surface(
                model=model, 
                X=X_train, 
                y=y_train, 
                top2_eigen_vectors=train_top2_eigen_vectors,
                init_penultimate_weights=init_penultimate_weights
            )            
        
        # estimating  test loss surface on full test data
        logger.debug('Calculate Hessian Top2-EigenVectors on Test Dataset...')
        test_top2_eigen_vectors = calc_top2_eigenvectors(
            X=X_test, 
            y=y_test, 
            model=model, 
            loss_object=loss_object, 
            optimizer_object=optimizer_object, 
            batch_size=args.batch_size, 
            layer_num=args.layer_num
        )
        logger.debug('Calculate Local Loss Surface on Test Dataset...')
        try:
            test_loss_values_2D += calc_loss_surface(
                model=model, 
                X=X_test, 
                y=y_test, 
                top2_eigen_vectors=test_top2_eigen_vectors,
                init_penultimate_weights=init_penultimate_weights
            )
        except:
            test_loss_values_2D  = calc_loss_surface(
                model=model, 
                X=X_test, 
                y=y_test, 
                top2_eigen_vectors=test_top2_eigen_vectors,
                init_penultimate_weights=init_penultimate_weights
            )

    train_loss_values_2D /= 5
    test_loss_values_2D  /= 5

    train_output_dict = {
        'train_loss_surface':train_loss_values_2D.tolist()
    }

    test_output_dict = {
        'test_loss_surface' :test_loss_values_2D.tolist()
    }

    with open(args.checkpoint_dir+'train_loss_surface.json', mode='w') as f:
        json.dump(train_output_dict, f, indent=4)
    
    with open(args.checkpoint_dir+'test_loss_surface.json',  mode='w') as f:
        json.dump( test_output_dict, f, indent=4)

if __name__ == "__main__":

    # setting hyper-parameter
    parser = argparse.ArgumentParser(description='Calculate local loss surface')
    parser.add_argument('--model'         , type=str  , default='densenet121')
    parser.add_argument('--dataset'       , type=str  , default='cifar10')
    parser.add_argument('--output-dim'    , type=int  , default=10)
    parser.add_argument('--optimizer'     , type=str  , default='sgd')
    parser.add_argument('--lr'            , type=float, default=1e-2)
    parser.add_argument('--device-num'    , type=int  , default=0)
    parser.add_argument('--batch-size'    , type=int  , default=1024)
    parser.add_argument('--layer-num'     , type=int  , default=-2)
    parser.add_argument('--top-k'         , type=int  , default=20)
    parser.add_argument('--checkpoint-dir', type=str)
    args = parser.parse_args()

    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    ###############################################################################
    # wandb setup
    ###############################################################################
    exp_name = time.strftime('%Y%m%d-%H%M%S')
    wandb.init(config=args, project='Calculate-local-loss-surface', entity='mtkwt', name=exp_name)

    setup_gpu_config(device_num=args.device_num)

    main()
