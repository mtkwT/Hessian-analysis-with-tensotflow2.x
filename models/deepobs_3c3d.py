import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, MaxPool2D, Flatten

class DeepOBS_3c3d(Model):
    def __init__(self, output_dim, weight_decay=0.001):
        super(DeepOBS_3c3d, self).__init__()
        self.conv1 = Conv2D(64 , kernel_size = (5, 5), padding = 'valid', activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.pool1 = MaxPool2D(  pool_size   = (3, 3), strides = (2, 2) , padding    = 'same')
        self.conv2 = Conv2D(96 , kernel_size = (3, 3), padding = 'valid', activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.pool2 = MaxPool2D(  pool_size   = (3, 3), strides = (2, 2) , padding    = 'same') 
        self.conv3 = Conv2D(128, kernel_size = (3, 3), padding = 'same' , activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.pool3 = MaxPool2D(  pool_size   = (3, 3), strides = (2, 2) , padding    = 'same')
        self.flatten = Flatten()
        self.fc1 = Dense(512       , activation='relu'   , kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.fc2 = Dense(256       , activation='relu'   , kernel_regularizer=tf.keras.regularizers.l2(weight_decay))
        self.out = Dense(output_dim, activation='softmax')
    
    def call(self, x):
        h = self.conv1(x)
        h = self.pool1(h)
        h = self.conv2(h)
        h = self.pool2(h)
        h = self.conv3(h)
        h = self.pool3(h)
        h = self.flatten(h)
        h = self.fc1(h)
        h = self.fc2(h)
        y = self.out(h)
        return y
