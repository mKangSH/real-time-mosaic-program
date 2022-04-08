import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train / x_train.max()

x_test = x_test / x_test.max()

# The reason of normalizing value between 0 and 1 is convergence speed by gradient descent algorithm
# Effect of preventing from a local optimum
# Dense layer input value is 1 rank tensor

model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        # node = 10 counts(class counts).
        tf.keras.layers.Dense(10, activation='softmax')
])

# y is one-hot vector
# [0., 0., 0., 0., 1., 0., 0.,]
# model.compile(loss='categorical_crossentropy')

# y is not one-hot vector
# [5]
# model.compile(loss='sparse_categorical_crossentropy')

# adam = tf.keras.optimizers.Adam(lr=0.001)
# acc = tf.keras.metrics.SparseCategoricalAccuracy()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('validation set accuracy: ', test_acc)

# image load
# write your path_dir
path_dir = '/real-time-mosaic-program/2022-1/data/Mnist'
tensor = [0, 1, 2, 3, 4]

for i in range(5):
    test_image = Image.open(path_dir + '/test%d.png' % i)
    test_array = np.array(test_image)

    #gray_image = ((test_array[:,:,0]) + (test_array[:,:,1]) + (test_array[:,:,2])) / 3
    gray_image = (0.299 * test_array[:,:,0]) + (0.587 * test_array[:,:,1]) + (0.114 * test_array[:,:,2])

    gray_image = (-(gray_image - 255)) / 255
    
    tensor[i] = (gray_image)

tensor = tf.constant(tensor)
print(tensor.shape)

# prediction
predictions = model.predict(tensor)
print(predictions[:5])
print(np.argmax(predictions[:5], axis=1))

# print(np.argmax(predictions[:10], axis=1))

tensor_true = [7, 2, 4, 6, 8]
# Graph
def get_one_result(idx):
    img, y_true, y_pred, confidence = tensor[idx], tensor_true[idx], np.argmax(predictions[idx]), 100 * np.max(predictions[idx])

    return img, y_true, y_pred, confidence

fig, axes = plt.subplots(1, 5)
fig.set_size_inches(12, 10)

for i in range(5):
    ax = axes[i%5]
    img, y_true, y_pred, confidence = get_one_result(i)

    ax.imshow(img, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'True: {y_true}')
    ax.set_xlabel(f'Prediction: {y_pred}\nConfidence: ({confidence:.2f} %)')
plt.tight_layout()
plt.show()

# mehod 
"""
x_train.reshape(60000, -1)

x = tf.keras.layers.Flatten()(x_train)
"""
# method
"""
model = tf.keras.Sequential([
        tf.keras.layers.Dense(128),
        tf.keras.layers.Activation('relu')
])

tf.keras.layer.Dense(128, activation='relu')
"""
# method
"""
adam = tf.keras.optimizers.Adam(lr=0.001)
acc = tf.keras.metrics.SparseCategoricalAccuracy()
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=[acc])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
"""