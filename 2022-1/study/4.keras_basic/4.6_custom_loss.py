import tensorflow as tf
import numpy as np

x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
y = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=float)

model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1, input_shape=[1])
])

def custom_huber_loss(y_true, y_pred):
    threshold = 1

    error = y_true - y_pred

    small = tf.abs(error) <= threshold

    # l2 loss
    small_error = tf.square(error) / 2

    # l1 loss
    big_error = threshold * (tf.abs(error) - (threshold / 2))

    return tf.where(small, small_error, big_error)

model.compile(optimizer='sgd', loss=custom_huber_loss)
model.fit(x, y, epochs=1000, verbose=0)

print(model.predict([10.0]))