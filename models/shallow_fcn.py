import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

class ShallowFCN(Model):
    def __init__(self, hidden_dim, output_dim):
        super(ShallowFCN, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(hidden_dim, activation='relu')
        self.d2 = Dense(output_dim,  activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)