import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / x_train.max()
x_test = x_test / x_test.max()

model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def get_batches(x, y, batch_size=32):
    for i in range(int(x.shape[0] // batch_size)):
        x_batch = x[i * batch_size:(i+1) * batch_size]
        y_batch = y[i * batch_size:(i+1) * batch_size]
        yield(np.asarray(x_batch), np.asarray(y_batch))

x, y = next(get_batches(x_train, y_train))

print(x.shape, y.shape)

MONITOR_STEP = 50

for epoch in range(1, 4):
    batch = 1
    total_loss = 0
    losses = []

    for x,y in get_batches(x_train, y_train, batch_size=128):
        loss, acc = model.train_on_batch(x, y)
        total_loss += loss

        if batch % MONITOR_STEP == 0:
            losses.append(total_loss / batch)
            print(f'epoch:{epoch}, batch:{batch}, batch_loss: {loss:.4f},\
                    batch_accuracy: {acc:.4f}, avg_loss: {total_loss / batch:.4f}')
        
        batch += 1

    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(1, batch // MONITOR_STEP + 1), losses)
    plt.title(f'epoch: {epoch}, losses over batches')
    plt.show()

    loss, acc = model.evaluate(x_test, y_test)

    print('----' * 10)
    print(f'epoch:{epoch}, val_loss: {loss:.4f}, val_accuracy: {acc:.4f}')
    print()

    

