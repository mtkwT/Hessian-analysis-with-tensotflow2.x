import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

class SimpleFunc(Model):
    def __init__(self):
        super(SimpleFunc, self).__init__()
        self.d1 = Dense(2,  
                        activation='linear', 
                        use_bias=False, 
                        kernel_initializer=tf.constant_initializer([[1., 2.],
                                                                    [2., 1.]])
        )

        self.d2 = Dense(1,  
                        activation='linear', 
                        use_bias=False, 
                        kernel_initializer=tf.constant_initializer([1., 2.])
        )

    def call(self, x):
        x = self.d1(x)
        return self.d2(x)

if __name__ == "__main__":
    model = SimpleFunc()
    print(model(tf.convert_to_tensor([[1., 2.]]))) # answer 13