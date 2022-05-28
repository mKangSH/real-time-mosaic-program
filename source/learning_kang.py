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

groups_folder_path = 'C:/'
categories = ["jmk", "ksh"]
num_classes = len(categories)

train_images, test_images, train_labels, test_labels = np.load('./img_data.npy', allow_pickle=True)

np.random.seed(1234)
index_list = np.arange(0, len(train_labels))
valid_index = np.random.choice(index_list, size = 10, replace = False)

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

"""
from tensorflow.keras.applications import ResNet50V2

pre_trained_base = ResNet50V2(include_top=False,
                              weights='imagenet',
                              input_shape=train_images.shape[1:])

pre_trained_base.trainable = False

def build_transfer_classifier():
    model = tf.keras.Sequential([
            pre_trained_base,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model

tc_model = build_transfer_classifier()
tc_model.summary()

def build_model():
    model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32,(3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model
"""

opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(train_images, train_labels, epochs=20, validation_data=(valid_images, valid_labels))

def plot_loss_acc(history, epoch):
    loss, val_loss = history.history['loss'], history.history['val_loss']
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(range(1, epoch + 1), loss, label='.Training')
    axes[0].plot(range(1, epoch + 1), val_loss, label='.Validation')
    axes[0].legend(loc='best')
    axes[0].set_title('Loss')

    axes[1].plot(range(1, epoch + 1), acc, label='.Training')
    axes[1].plot(range(1, epoch + 1), val_acc, label='.Validation')
    axes[1].legend(loc='best')
    axes[1].set_title('Accuracy')

    plt.show()

plot_loss_acc(history, 20)

model.save('kang.h5')