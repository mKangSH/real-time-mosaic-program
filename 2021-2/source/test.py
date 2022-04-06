import os
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import (Dense, BatchNormalization, Dropout, Flatten)
from tensorflow.keras.datasets.mnist import load_data
from keras import backend as K

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = load_data()

#print(train_images.shape)
#print(train_labels.shape)
#print(test_images.shape)
#print(test_labels.shape)

#plt.hist(train_labels)
#plt.show()

np.random.seed(1234)
index_list = np.arange(0, len(train_labels))
valid_index = np.random.choice(index_list, size = 5000, replace = False)

valid_images = train_images[valid_index]
valid_labels = train_labels[valid_index]

train_index = set(index_list) - set(valid_index)
train_images = train_images[list(train_index)]
train_labels = train_labels[list(train_index)]

#print(pd.Series(valid_labels).value_counts())
min_key = np.min(train_images)
max_key = np.max(train_images)

train_images = (train_images - min_key) / (max_key - min_key)
valid_images = (valid_images - min_key) / (max_key - min_key)
test_images = (test_images - min_key) / (max_key - min_key)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28], name="Flatten"))
model.add(Dense(300, activation="relu", name="Hidden1"))
model.add(Dense(200, activation="relu", name="Hidden2"))
model.add(Dense(100, activation="relu", name="Hidden3"))
model.add(Dense(10, activation="softmax", name="Output"))
#print(model.summary())

opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer = opt, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10, restore_best_weights=True)
#checkpoint_path = "training_1/cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)
#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

history = model.fit(train_images, train_labels, epochs = 100, batch_size=5000, validation_data=(valid_images, valid_labels), callbacks=[early_stop])
model.save("MNIST_211128.h5")


#model.save("./MNIST_211128.h5")

#model.evaluate(test_images, test_labels)
#model.save_weights("MNIST_211128.h5")
#with open("model.json", "w") as json_file :
#    json_file.write(model.to_json())


'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense, BatchNormalization, Dropout, Flatten)
from tensorflow.keras.datasets.mnist import load_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = load_data()

#print(train_images.shape)
#print(train_labels.shape)
#print(test_images.shape)
#print(test_labels.shape)

#plt.hist(train_labels)
#plt.show()

np.random.seed(1234)
index_list = np.arange(0, len(train_labels))
valid_index = np.random.choice(index_list, size = 5000, replace = False)

valid_images = train_images[valid_index]
valid_labels = train_labels[valid_index]

train_index = set(index_list) - set(valid_index)
train_images = train_images[list(train_index)]
train_labels = train_labels[list(train_index)]

#print(pd.Series(valid_labels).value_counts())
min_key = np.min(train_images)
max_key = np.max(train_images)

train_images = (train_images - min_key) / (max_key - min_key)
valid_images = (valid_images - min_key) / (max_key - min_key)
test_images = (test_images - min_key) / (max_key - min_key)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28], name="Flatten"))
model.add(Dense(300, activation="relu", name="Hidden1"))
model.add(Dense(200, activation="relu", name="Hidden2"))
model.add(Dense(100, activation="relu", name="Hidden3"))
model.add(Dense(10, activation="softmax", name="Output"))

#print(model.summary())

opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer = opt, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
#from time import time
#start = time()
history = model.fit(train_images, train_labels, epochs = 30, batch_size = 5000,  validation_data = (valid_images, valid_labels))
#print("코드 동작 시간: {} minutes".format(round((time()- start)/60, 2)))
#print(history.history)

history_DF = pd.DataFrame(history.history)
#print(history_DF)

# 꺾은 선 그래프
# 그래프의 크기와 선의 굵기
history_DF.plot(figsize=(12,8), linewidth=3)

# 교차선
plt.grid(True)

# 그래프 요소
plt.legend(loc="upper right", fontsize=15)
plt.title("Learning Curve", fontsize=30, pad =30)
plt.xlabel('Epoch', fontsize=20, loc = 'center', labelpad=20)
plt.ylabel('Variable', fontsize=20, rotation=0, loc='center', labelpad=40)


ax=plt.gca()
ax.spines["right"].set_visible(False) # 오른쪽 테두리 제거
ax.spines["top"].set_visible(False) # 위 테두리 제거

plt.show()
'''

'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense, BatchNormalization, Dropout, Flatten)
from tensorflow.keras.datasets.mnist import load_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = load_data()

def show_images(dataset, label, nrow, ncol):

    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(2*ncol, 2*nrow))
    ax = axes.ravel()

    xlabels = label[0:nrow*ncol]

    for i in range(nrow*ncol):

        image = dataset[i]
        ax[i].imshow(image, cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_xlabel(xlabels[i])

    plt.tight_layout()
    plt.show()

show_images(train_images, train_labels, 4, 5)
print(train_images[0])
'''

"""
np.random.seed(1234)
# Data setting
def function(x1, x2):
    return 0.5*(x1**2) - (3 * x2) + 5

X1 = np.random.randint(0, 100, (30000))
X2 = np.random.randint(0, 100, (30000))
X = np.c_[X1, X2]
Y = function(X1, X2)

Xy = np.c_[X, Y]
Xy = np.unique(Xy, axis=0)
np.random.shuffle(Xy)
test_len = int(np.ceil(len(Xy) * 0.2))
X = Xy[:, [0,1]]
Y = Xy[:, 2]

X_test = X[:test_len]
Y_test = Y[:test_len]

X_train = X[test_len:]
Y_train = Y[test_len:]

model = keras.Sequential()
model.add(Dense(32, activation='elu'))
model.add(Dense(32, activation='elu'))
model.add(Dense(1, activation='linear'))

opt = keras.optimizers.Adam(learning_rate = 0.005)
model.compile(optimizer=opt, loss='mse')

min_key = np.min(X_train)
max_key = np.max(X_train)

X_train_std = (X_train - min_key) / (max_key - min_key)
Y_train_std = (Y_train - min_key) / (max_key - min_key)

X_test_std = (X_test - min_key) / (max_key - min_key)

model.fit(X_train_std, Y_train_std, epochs = 100)

pred = (model.predict(X_test_std) * (max_key - min_key)) + min_key
pred = pred.reshape(pred.shape[0])
print("Accuracy:", np.sqrt(np.sum((Y_test - pred) ** 2)) / len(Y_test))

result_DF = pd.DataFrame({"predict":pred, "label":Y_test})
result_DF["gap"] = result_DF["label"] - result_DF["predict"]
print(result_DF)
"""
'''
def function1(x1, x2, x3, x4):
    return 0.3*x1 + 0.2*x2 - 0.4*x3 + 0.1*x4 + 2

def function2(x1, x2, x3, x4):
    return 0.5*x1 - 0.1-x2 + 0.3*x3 + 2

def make_dataset(start_N, end_N):

    x1 = np.arange(start_N, end_N)
    x2 = x1 + 1
    x3 = x1 + 2
    x4 = x1 + 3

    y1 = function1(x1, x2, x3 , x4)
    y2 = function2(x1, x2, x3 , x4)

    append_for_shuffle = np.c_[x1, x2, x3, x4, y1, y2]
    np.random.shuffle(append_for_shuffle)

    X = append_for_shuffle[:, [0,1,2,3]]
    y = append_for_shuffle[:, [4,5]]

    return X, y

X, y = make_dataset(0, 1000)
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

model = keras.Sequential()
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="linear"))

opt = keras.optimizers.Adam(learning_rate = 0.005)
model.compile(optimizer = opt, loss = "mse")

min_key = np.min(X_train)
max_key = np.max(X_train)

X_std_train = (X_train - min_key) / (max_key - min_key)
y_std_train = (y_train - min_key) / (max_key - min_key)
X_std_test = (X_test - min_key) / (max_key - min_key)

model.fit(X_std_train, y_std_train, epochs = 100)

def MAE(x, y):
    return np.mean(np.abs(x-y))

pred = model.predict(X_std_test) * (max_key - min_key) + min_key
print("Accuracy", MAE(pred, y_test))

DF = pd.DataFrame(pred, columns=["y1_pred", "y2_pred"])
DF[["y1_label", "y2_label"]] = y_test
DF["y1_gap"] = DF["y1_label"] - DF["y1_pred"]
DF["y2_gap"] = DF["y2_label"] - DF["y2_pred"]
print(DF[["y1_pred", "y1_label", "y1_gap", "y2_pred", "y2_label", "y2_gap"]])
'''