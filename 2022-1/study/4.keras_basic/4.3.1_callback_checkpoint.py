import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / x_train.max()
x_test = x_test / x_test.max()

model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ckpt: weights checkpoint file format
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='tmp_checkpoint.ckpt',
                                                save_weights_only=True,
                                                save_best_only=True,
                                                verbose=1)

model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, callbacks=[checkpoint])

# before model checkpoint load
loss, acc = model.evaluate(x_test, y_test)
print(f'befor checkpoint load= loss: {loss:3f}, acc: {acc:3f}')

model.load_weights('tmp_checkpoint.ckpt')
loss, acc = model.evaluate(x_test, y_test)
print(f'after checkpoint load= loss: {loss:3f}, acc: {acc:3f}')