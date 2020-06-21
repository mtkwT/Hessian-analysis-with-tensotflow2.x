import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

@tf.function # when we use lstm, don't use tf.function
def get_hessian(X, y, model, loss_object, k=-2):
    """
    get hessian matrix w.r.t the last layer's parameters
    """
    with tf.GradientTape(persistent=True) as tape:
        preds = model(X)
        loss = loss_object(y, preds)
        gradients = tape.gradient(loss, model.trainable_variables) 
    hessian = tape.jacobian(gradients[k], model.trainable_variables[k])
    return hessian

def reshape_hessian(hessian):
    """
    ### Example ###

    class MyModel(Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = Conv2D(32, 3, activation='relu')
            self.flatten = Flatten()
            self.d1 = Dense(128, activation='relu')
            self.d2 = Dense(10)

        def call(self, x):
            x = self.conv1(x)
            x = self.flatten(x)
            x = self.d1(x)
            return self.d2(x)
    
    In this case, Hessian shape is (128, 10, 128, 10) before reshape.
    ==> After reshape, shape is (1280, 1280). It's Matrix shape!
    """
    return tf.reshape(hessian, (hessian.shape[0]*hessian.shape[1], hessian.shape[2]*hessian.shape[3]))

def calculate_mean_hessian(X, y, model, loss_object, batch_size, k):
    """
    calculate mean layer-wise hessian matrix for full dataset
    """
    n_batches = X.shape[0] // batch_size
    for batch in tqdm(range(n_batches)):
        start = batch * batch_size
        end = start + batch_size
        try:
            mean_hessian += get_hessian(X[start:end], y[start:end], model, loss_object, k)
        except:
            mean_hessian  = get_hessian(X[start:end], y[start:end], model, loss_object, k)

    mean_hessian /= n_batches

    return reshape_hessian(mean_hessian)

def get_hessian_for_lstm(X, y, model, loss_object, k=-2):
    """
    get hessian matrix w.r.t the last layer's parameters
    """
    with tf.GradientTape(persistent=True) as tape:
        preds = model(X)
        # print(y.shape, preds.shape)
        loss = loss_object(y, preds)
        gradients = tape.gradient(loss, model.trainable_variables) 
        # print(gradients)
    hessian = tape.jacobian(gradients[k], model.trainable_variables[k])
    return hessian
    
def calculate_mean_hessian_for_tfdata(tf_dataset, model, loss_object, batch_size, k):
    """
    calculate mean layer-wise hessian matrix for tf.data.dataset
    """
    n_batches = 0
    for (X, y) in tqdm(tfds.as_numpy(tf_dataset)):
        y = tf.reshape(y, (y.shape[0], 1)) # when we use SparseCategoricalCrossentropy, we must reshape
        try:
            mean_hessian += get_hessian_for_lstm(X, y, model, loss_object, k)
        except:
            mean_hessian  = get_hessian_for_lstm(X, y, model, loss_object, k)
       
        n_batches += 1

    mean_hessian /= n_batches

    return reshape_hessian(mean_hessian)