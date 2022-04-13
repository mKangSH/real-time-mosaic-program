import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

class MyDense(Layer):
    def __init__(self, units=32, input_shape=None):
        super(MyDense, self).__init__(input_shape=input_shape)
        self.units = units

    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name="weight",
                             initial_value=w_init(shape=(input_shape[-1], self.units), dtype='float32'),
                             trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias", 
                             initial_value=b_init(shape=(self.units,), dtype='float32'),
                             trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=float)

model = tf.keras.Sequential([
        MyDense(units=1, input_shape=[1])
])

model.summary()

model.compile(optimizer='sgd', loss='mse')

model.fit(x, y, epochs=1000, verbose=0)

print(model.predict([10.0]))