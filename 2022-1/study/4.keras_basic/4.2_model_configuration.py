import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('train set: ', x_train.shape, y_train.shape)
print('test set: ', x_test.shape, y_test.shape)

# data nomalization
x_train = x_train / x_train.max()
x_test = x_test / x_test.max()

# initializers
dense = tf.keras.layers.Dense(256, kernel_initializer='he_normal' ,activation='relu')

# Regularization: L1, L2 (preventing from overfitting)
dense = tf.keras.layers.Dense(256, kernel_regularizer='l1', activation='relu')

# dense configuration
print(dense.get_config())

# dropout
tf.keras.layers.Dropout(0.25) # delete 25% learning node

# batch normalization (scale adjustment)
model_a = tf.keras.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(64, activation='relu'),
          tf.keras.layers.Dense(32, activation='relu'),
          tf.keras.layers.Dense(10, activation='softmax'),
])

model_b = tf.keras.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(64),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Activation('relu'),

          tf.keras.layers.Dense(32),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Activation('relu'),

          tf.keras.layers.Dense(10, activation='softmax'),
])

model_c = tf.keras.Sequential([
          tf.keras.layers.Flatten(input_shape=(28, 28)),
          tf.keras.layers.Dense(64),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.LeakyReLU(alpha=0.2),

          tf.keras.layers.Dense(32),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.LeakyReLU(alpha=0.2),

          tf.keras.layers.Dense(10, activation='softmax'),
])

model_a.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_b.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_c.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_a = model_a.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
history_b = model_b.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)
history_c = model_c.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12,9))
plt.plot(np.arange(1,11), history_a.history['val_loss'], color='navy', linestyle='-')
plt.plot(np.arange(1,11), history_b.history['val_loss'], color='tomato', linestyle='-')
plt.plot(np.arange(1,11), history_c.history['val_loss'], color='green', linestyle='-')

plt.title('Losses', fontsize=20)
plt.xlabel('epochs')
plt.ylabel('Losses')
plt.legend(['ReLU', 'BatchNorm + ReLU', 'BatchNorm + LeakyReLU'], fontsize=12)
plt.show()
# method
"""
dense = tf.keras.layers.Dense(256, kernel_initializer='he_normal', activation='relu')

he_normal = tf.keras.layers.initializers.HeNormal()
dense = tf.keras.layers.Dense(256, kernel_initializer=he_normal, activation='relu')
"""

# method
"""
dense = tf.keras.layers.Dense(256, kernel_regularizer='l1', activation='relu')

regularizer = tf.keras.regularizers.l1(l1=0.1) # alpha = 0.1 
dense = tf.keras.layers.Dense(256, kernel_regularizer=regularizer, activation='relu')
"""