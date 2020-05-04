import tensorflow as tf
from tqdm import tqdm

@tf.function
def get_hessian(X, y, model, loss_object, k=-2):
    """
    get hessian matrix w.r.t the final layer's parameters
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

def calculate_mean_hessian(X, y, model, loss_object, batch_size):
    """
    calculate mean hessian matrix for full dataset
    """
    n_batches = X.shape[0] // batch_size
    for batch in tqdm(range(n_batches)):
        start = batch * batch_size
        end = start + batch_size
        try:
            mean_hessian += get_hessian(X[start:end], y[start:end], model, loss_object)
        except:
            mean_hessian = get_hessian(X[start:end], y[start:end], model, loss_object)

    mean_hessian /= n_batches

    return mean_hessian

def calculate_batch_hessian(X, y, model, loss_object, batch_size):
    """
    calculate hessian matrix for batch in dataset
    """
    batch_hessian = get_hessian(X[:batch_size], y[:batch_size], model, loss_object)
    return batch_hessian