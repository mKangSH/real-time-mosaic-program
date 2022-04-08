import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 6)

y = 3 * x + 2

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
    ])
model.summary()

model.compile(optimizer='sgd', loss='mse', metrics=['mae'])
"""
model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.005),
    loss=tf.keras.losses.MeanAbsoluteError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)
"""

history = model.fit(x, y, epochs=1200, verbose = 0)

print(model.evaluate(x, y))

tensor1 = model.predict([30])
print(tensor1)



"""

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape = [4]), 
    tf.keras.layers.Dense(5),
    tf.keras.layers.Dense(1),
    ])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(10))
model.add(tf.keras.layers.Dense(5))
model.add(tf.keras.layers.Dense(1))

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['mae'], label='mae')
plt.xlim(-1, 100)
plt.title('Loss')
plt.legend()
plt.show()

"""

