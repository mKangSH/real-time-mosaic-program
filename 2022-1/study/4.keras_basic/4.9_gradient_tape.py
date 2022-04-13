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

loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

optimizer = tf.keras.optimizers.Adam()

def get_batches(x, y, batch_size=32):
    for i in range(int(x.shape[0] // batch_size)):
        x_batch = x[i * batch_size : (i + 1) * batch_size]
        y_batch = y[i * batch_size : (i + 1) * batch_size]
        yield(np.asarray(x_batch), np.asarray(y_batch))

# lazy execution mode define (only tensorflow function)
# preventing from unexpected constant conversion
@tf.function 
def train_step(images, labels):
    with tf.GradientTape() as tape:
        prediction = model(images, training=True)
        loss = loss_function(labels, prediction)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, prediction)

@tf.function
def valid_step(images, labels):
    prediction = model(images, training=False)
    loss = loss_function(labels, prediction)

    valid_loss(loss)
    valid_accuracy(labels, prediction)

train_loss.reset_states()
train_accuracy.reset_states()
valid_loss.reset_states()
valid_accuracy.reset_states()

for epoch in range(5):
    for images, labels in get_batches(x_train, y_train):
        train_step(images, labels)

    for images, labels in get_batches(x_test, y_test):
        valid_step(images, labels)

    metric_template = 'epoch: {}, loss: {:.4f}, acc: {:.2f}%, val_loss: {:.4f}, val_acc: {:.2f}%'

    print(metric_template.format(epoch+1, train_loss.result(), train_accuracy.result() * 100,
                                 valid_loss.result(), valid_accuracy.result() * 100))




# simple gradient tape model 
"""
a = tf.Variable([1, 2, 3, 4, 5], dtype=tf.float32)
b = tf.Variable([10, 20, 30, 40, 50], dtype=tf.float32)

# Check Differentiable for a tensor and b tensor
print(f'a.trainable: {a.trainable}\nb.trainable: {b.trainable}')

with tf.GradientTape() as tape:
    f = 3 * a + 2 * a * b

grads = tape.gradient(f, [a, b])

tf.print(f'df/da: {grads[0]}')
tf.print(f'df/db: {grads[1]}')

x = tf.Variable(np.random.normal(size=(100,)), dtype=tf.float32)
y = (x ** 2) + (4 * x) + 2

print(f'x[:5]: {x[:5].numpy()}\ny[:5]: {y[:5].numpy()}')

#plt.scatter(x, y)
#plt.show()

learning_rate = 0.25
EPOCHS = 1000

w = tf.Variable(0.0)
b = tf.Variable(0.0)

for epoch in range(EPOCHS):
    with tf.GradientTape() as tape:
        y_hat = x ** 2 + w * x + b
        loss = tf.reduce_mean((y_hat - y) ** 2)

    dw, db = tape.gradient(loss, [w, b])

    w = tf.Variable(w - learning_rate * dw)
    b = tf.Variable(b - learning_rate * db)

    print(f'epoch: {epoch}, loss: {loss.numpy():.4f}, w: {w.numpy():.4f}, b: {b.numpy():.4f}')

    if loss.numpy() < 0.0005:
        break
"""