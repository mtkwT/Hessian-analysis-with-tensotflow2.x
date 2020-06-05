import tensorflow as tf
from tqdm import tqdm

@tf.function
def get_gradients(X, y, model, loss_object):
    """
    get gradients w.r.t the model parameters
    """
    with tf.GradientTape() as tape:
        preds = model(X)
        loss = loss_object(y, preds)
        gradients = tape.gradient(loss, model.trainable_variables)
    return gradients

def get_grad_covariance(gradients, k=-2):
    """
    get gradient covariance matrix 
    """
    grad = gradients[k]
    grad = tf.reshape(grad, (grad.shape[0]*grad.shape[1], 1))
    return tf.linalg.matmul(a=grad, b=grad, transpose_a=False, transpose_b=True)

def calc_mean_grad_covariance(X, y, model, loss_object, batch_size, k=-2):
    """
    calculate mean gradient covariance matrix for full dataset
    """
    n_batches = X.shape[0] // batch_size
    mean_grad_covariance = None
    for batch in tqdm(range(n_batches)):
        start = batch * batch_size
        end = start + batch_size
        grads = get_gradients(X[start:end], y[start:end], model, loss_object)
        try:
            mean_grad_covariance += get_grad_covariance(grads, k)
        except:
            mean_grad_covariance  = get_grad_covariance(grads, k)

    mean_grad_covariance /= n_batches
    return mean_grad_covariance