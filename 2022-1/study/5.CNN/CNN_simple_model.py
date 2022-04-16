from re import X
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_valid, y_valid) = mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_valid.shape, y_valid.shape)

import matplotlib.pylab as plt

def plot_image(data, idx):
    plt.figure(figsize=(5, 5))
    plt.imshow(data[idx], cmap="gray")
    plt.axis("off")
    plt.show()

# plot_image(x_train, 0)

x_train = x_train / 255.0
x_valid = x_valid / 255.0

# adding new axis 
# x_train_in = x_train[..., tf.newaxis] 
# x_valid_in = x_valid[:, tf.newaxis,:,:]

x_train_in = x_train[..., tf.newaxis]
x_valid_in = x_valid[..., tf.newaxis]

# 3x3 kernel max counts: 128
print(x_train_in.shape, x_valid_in.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1), name='conv'),
    tf.keras.layers.MaxPooling2D((2,2), name='pool'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax'),
])

# model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train_in, y_train, validation_data=(x_valid_in, y_valid), epochs=10)

model.evaluate(x_valid_in, y_valid)

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

plot_loss_acc(history, 10)

# print(madel.layers[0].input)
# print(madel.layers[0].output)
# print(madel.layers[0].weights) # print(madel.layers[0].kernel) + print(madel.layers[0].bias)

# the number of parameter: (3x3) x 1 x 32 + 32(bias) = 288(kernel) + 32(bias)

# print(model.get_layer('conv')

activator = tf.keras.Model(inputs=model.input, outputs=[layer.output for layer in model.layers[:2]])
activations = activator.predict(x_train_in[0][tf.newaxis, ...])

len(activations)

conv_activation = activations[0]
print(conv_activation.shape)

def show_graph(counts, layer):
    fig, axes = plt.subplots(counts // 8, 8)
    fig.set_size_inches(10, 5)

    for i in range(counts):
        axes[i//8, i%8].matshow(layer[0,:,:,i], cmap='viridis')
        axes[i//8, i%8].set_title('kernel %s'%str(i), fontsize=10)
        plt.setp(axes[i//8, i%8].get_xticklabels(), visible=False)
        plt.setp(axes[i//8, i%8].get_yticklabels(), visible=False)

    plt.tight_layout()
    plt.show()

show_graph(32, conv_activation)

pooling_activation = activations[1]
print(pooling_activation.shape)

show_graph(32, pooling_activation)

