import tensorflow as tf
import numpy as np
import json
import matplotlib.pylab as plt
import pandas as pd

import tensorflow_datasets as tfds

DATA_DIR = "2022-1/data/eurosat/"

(train_ds, valid_ds), info = tfds.load("eurosat/rgb", split=['train[:80%]', 'train[80%:]'],
                                        shuffle_files=True,
                                        as_supervised=True,
                                        with_info=True,
                                        data_dir=DATA_DIR)

#print(train_ds)
#print(valid_ds)
#print(info)

#tfds.show_examples(train_ds, info)
#df = tfds.as_dataframe(valid_ds.take(10), info)

NUM_CLASSES = info.features["label"].num_classes
#print(NUM_CLASSES)

#print(info.features["label"].int2str(6))

# Data preprocess pipeline
BATCH_SIZE = 64
BUFFER_SIZE = 1000

def preprocess_data(image, label):
    image = tf.cast(image, tf.float32) / 255.
    return image, label

train_data = train_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
valid_data = valid_ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)

train_data = train_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
valid_data = valid_data.batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

# Create sample model

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

model = build_model()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, validation_data=valid_data, epochs=50)

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

plot_loss_acc(history, 50)