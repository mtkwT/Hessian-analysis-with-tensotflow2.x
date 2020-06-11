import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

class DeepFCN(Model):
    def __init__(self, layer_num, hidden_dim, output_dim):
        super(DeepFCN, self).__init__()
        self.layer_num = layer_num
        self.flatten = Flatten()
        self.d_hidden_layers = []
        
        for k in range(self.layer_num):
            d_k = Dense(hidden_dim, activation='relu')
            self.d_hidden_layers.append(d_k)

        self.d_out = Dense(output_dim,  activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        for k in range(self.layer_num):
            x = self.d_hidden_layers[k](x)
        return self.d_out(x)