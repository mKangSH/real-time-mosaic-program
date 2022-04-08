import tensorflow as tf
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin'

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / x_train.max()
x_test = x_test / x_test.max()

input_layer = tf.keras.Input(shape = (28, 28), name='InputLayer')

x1 = tf.keras.layers.Flatten(name='Flatten')(input_layer)
x2 = tf.keras.layers.Dense(256, activation='relu', name='Dense1')(x1)
x3 = tf.keras.layers.Dense(64, activation='relu', name='Dense2')(x2)
x4 = tf.keras.layers.Dense(10, activation='softmax', name='OutputLayer')(x3)

func_model = tf.keras.Model(inputs=input_layer, outputs=x4, name='FunctionalModel')

from tensorflow.keras.utils import plot_model

plot_model(func_model, show_shapes=True, show_layer_names=True, to_file='model.png')
