import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import (Dense, BatchNormalization, Dropout, Flatten, Convolution2D, MaxPooling2D)
from keras import backend as K
from keras.models import Sequential, load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.core import Activation
from tensorflow.keras import layers

groups_folder_path = './cnn_sample/'
categories = ["kang", "people"]
num_classes = len(categories)

train_images, test_images, train_labels, test_labels = np.load('./img_data.npy', allow_pickle=True)

np.random.seed(1234)
index_list = np.arange(0, len(train_labels))
valid_index = np.random.choice(index_list, size = 800, replace = False)

valid_images = train_images[valid_index]
valid_labels = train_labels[valid_index]

train_index = set(index_list) - set(valid_index)
train_images = train_images[list(train_index)]
train_labels = train_labels[list(train_index)]

min_key = np.min(train_images)
max_key = np.max(train_images)

train_images = (train_images - min_key) / (max_key - min_key)
valid_images = (valid_images - min_key) / (max_key - min_key)
test_images = (test_images - min_key) / (max_key - min_key)

model = Sequential()
model = Sequential()

model.add(Convolution2D(32, (3, 3), input_shape=train_images.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10, restore_best_weights=True)
history = model.fit(train_images, train_labels, batch_size=32, epochs=100, validation_data=(valid_images, valid_labels))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(100)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model.save('kang.h5')