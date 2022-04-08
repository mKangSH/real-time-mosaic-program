import numpy as np
import matplotlib.pyplot as plt

def make_linear(w = 0.5, b = 0.8, size = 50, noise = 1.0):
    x = np.random.rand(size)
    y = w * x + b
    noise = np.random.uniform(-abs(noise), abs(noise), size = y.shape)
    yy = y + noise

    """
    plt.figure(figsize=(10, 7))
    plt.plot(x, y, color = 'r', label = f'y = {w} * x + {b}')
    plt.scatter(x, yy, label = 'data')
    plt.legend(fontsize = 20)
    plt.show()
    """

    print(f'w:{w}, b: {b}')
    return x, yy

x, y = make_linear(w = 0.3, b = 0.5, size = 100, noise = 0.01)

num_epoch = 1000

learning_rate = 0.005

errors = []

w = np.random.uniform(low=0.0,high=1.0)
b = np.random.uniform(low=0.0,high=1.0)

for epoch in range(num_epoch):
    y_hat = w * x + b

    error = 0.5 * ((y_hat - y) ** 2).sum()
    if error < 0.005:
        break

    w = w - learning_rate * ((y_hat - y) * x).sum()
    b = b - learning_rate * (y_hat - y).sum()

    errors.append(error)

    if epoch % 5 == 0:
        print("{0:2} w = {1:.5f}, b = {2:.5f}, error = {3:.5f}".format(epoch, w, b, error))


print("------" * 15)
print("{0:2} w = {1:.1f}, b = {2:.1f}, error = {3:.5f}".format(epoch, w, b, error))

plt.figure(figsize=(10, 7))
plt.plot(errors)
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()