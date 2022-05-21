import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)

y_train_odd = []

for y in y_train:
    if y % 2 == 0:
        y_train_odd.append(0)

    else:
        y_train_odd.append(1)

y_train_odd = np.array(y_train_odd)

# print(y_train_odd.shape)

# print(y_train[:10])
# print(y_train_odd[:10])

y_valid_odd = []

for y in y_valid:
    if y % 2 == 0:
        y_valid_odd.append(0)

    else:
        y_valid_odd.append(1)

# print(type(y_valid_odd))

y_valid_odd = np.array(y_valid_odd)
# print(y_valid_odd.shape)

x_train = x_train / 255.0
x_valid = x_valid / 255.0

x_train_in = tf.expand_dims(x_train, -1)
x_valid_in = tf.expand_dims(x_valid, -1)

# print(x_train_in.shape, x_valid_in.shape)

inputs = tf.keras.layers.Input(shape=(28,28, 1), name='inputs')

conv = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d_layer')(inputs)
pool = tf.keras.layers.MaxPooling2D((2, 2), name='maxpool_layer')(conv)
flat = tf.keras.layers.Flatten(name='flatten_layer')(pool)

flat_inputs = tf.keras.layers.Flatten()(inputs)
odd_outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='odd_dense')(flat_inputs)

concat = tf.keras.layers.Concatenate()([flat, flat_inputs])
digit_outputs = tf.keras.layers.Dense(10, activation='softmax', name='digit_dense')(concat)

model = tf.keras.models.Model(inputs=inputs, outputs=[digit_outputs, odd_outputs])

model.summary()
print(model.input)
print(model.output)

from tensorflow.python.keras.utils.vis_utils import plot_model
plot_model(model, show_shapes=True, show_layer_names=True, to_file='functional_cnn.png')
'''
model.compile(optimizer='adam', 
              loss={'digit_dense':'sparse_categorical_crossentropy', 'odd_dense':'binary_crossentropy'},
              loss_weights={'digit_dense':1, 'odd_dense': 0.5},
              # loss = 1.0 * sparse_categorical_crossentropy + 0.5 * binary_crossentropy
              metrics=['accuracy'])

history = model.fit({'inputs': x_train_in},{'digit_dense':y_train, 'odd_dense':y_train_odd},
                     validation_data=({'inputs':x_valid_in}, {'digit_dense':y_valid, 'odd_dense':y_valid_odd}), epochs=10)

model.evaluate({'inputs':x_valid_in}, {'digit_dense':y_valid, 'odd_dense':y_valid_odd})
'''
import matplotlib.pyplot as plt

def plot_image(data, idx):
    plt.figure(figsize=(5,5))
    plt.imshow(data[idx])
    plt.axis("off")
    plt.show()

# plot_image(x_valid, 0)

# digit_preds, odd_preds = model.predict(x_valid_in)

# print(digit_preds[0])
# print(odd_preds[0])

# digit_labels = np.argmax(digit_preds, axis=-1)
# print(digit_labels[0:10])

# odd_labels = (odd_preds > 0.5).astype(np.int).reshape(1, -1)[0]
# print(odd_labels[0:10])

base_model_output = model.get_layer('flatten_layer').output

base_model = tf.keras.models.Model(inputs=model.input, outputs=base_model_output, name='base')
# base_model.summary()

plot_model(base_model, show_shapes=True, show_layer_names=True, to_file='base_model.png')

transfer_model = tf.keras.Sequential([
                 base_model,
                 tf.keras.layers.Dense(10, activation='softmax')
])

transfer_model.summary()
# plot_model(transfer_model, show_shapes=True, show_layer_names=True, to_file='transfer_model.png')

transfer_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = transfer_model.fit(x_train_in, y_train, validation_data=(x_valid_in, y_valid), epochs=10)

base_model_frozen = tf.keras.models.Model(inputs=model.input, outputs=base_model_output, name='base_frozen')
base_model_frozen.trainable = False
# base_model_frozen.summary()
dense_output = tf.keras.layers.Dense(10, activation='softmax')(base_model_frozen.output)

digit_model_frozen = tf.keras.models.Model(inputs=base_model_frozen.input, outputs=dense_output)
digit_model_frozen.summary()

digit_model_frozen.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = digit_model_frozen.fit(x_train_in, y_train, validation_data=(x_valid_in, y_valid), epochs=10)

base_model_frozen2 = tf.keras.models.Model(inputs=model.input, outputs=base_model_output, name='base_frozen2')
base_model_frozen2.get_layer('conv2d_layer').trainable = False
base_model_frozen2.summary()

dense_output2 = tf.keras.layers.Dense(10, activation='softmax')(base_model_frozen2.output)

digit_model_frozen2 = tf.keras.models.Model(inputs=base_model_frozen2.input, outputs=dense_output2)
digit_model_frozen2.summary()

digit_model_frozen2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history=digit_model_frozen2.fit(x_train_in, y_train, validation_data=(x_valid_in, y_valid), epochs=10)
