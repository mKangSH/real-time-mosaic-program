import tensorflow as tf
import os
from tensorflow.keras.utils import plot_model

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin'

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / x_train.max()
x_test = x_test / x_test.max()

class MyModel(tf.keras.Model):
    def __init__(self, units, num_classes):
        super(MyModel, self).__init__()
        # initial value
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units, activation='relu')
        self.dense2 = tf.keras.layers.Dense(units/4, activation='relu')
        self.dense3 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

myModel = MyModel(256, 10)

myModel._name = 'subclass_model'
myModel(tf.keras.layers.Input(shape=(28,28)))
myModel.summary()

myModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
myModel.fit(x_train, y_train, epochs=3)

loss, acc = myModel.evaluate(x_test, y_test, verbose=0)
print(f'loss: {loss:3f}, acc: {acc:3f}')

# method

"""
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # initial value
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

myModel = MyModel()
"""