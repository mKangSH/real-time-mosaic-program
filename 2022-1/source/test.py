import tensorflow as tf

def cal_mse(x, y, a, b):
    y_pred = a * x + b
    squared_error = (y_pred - y) ** 2
    mean_squared_error = tf.reduce_mean(squared_error)

    return mean_squared_error

g = tf.random.Generator.from_seed(2022)
x = g.normal(shape=(10, ))
y = 3 * x - 2

a = tf.Variable(0.0)
b = tf.Variable(0.0)

EPOCHS = 120

for epoch in range(1, EPOCHS + 1):

    with tf.GradientTape() as tape:
        mse = cal_mse(x, y, a, b)

    grad = tape.gradient(mse, {'c':a, 'd': b})
    d_a, d_b = grad['c'], grad['d']

    a.assign_sub(d_a * 0.05)
    b.assign_sub(d_b * 0.05)

    if epoch % 10 == 0:
        print("EPOCH %d - MSE: %.4f - a: %.2f - b: %.2f"%(epoch, mse, a, b))