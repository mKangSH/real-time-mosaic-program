import os, re, glob
import cv2, pickle
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import (Dense, BatchNormalization, Dropout, Flatten)
from tensorflow.keras.datasets.mnist import load_data
from keras import backend as K
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

groups_folder_path = 'D:/workingFolder/Tensor/cnn_sample/'
categories = ["kang", "people"]
num_classes = len(categories)

image_w = 180
image_h = 180

X = []
Y = []

for idex, category in enumerate(categories):
    label = [0 for i in range(num_classes)]
    label[idex] = 1
    image_dir = groups_folder_path + category + '/'

    for top, dir, func in os.walk(image_dir):
        for filename in func:
            print(image_dir+filename)
            img = cv2.imread(image_dir+filename)
            img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
            X.append(img/255)
            Y.append(label)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
xy = (X_train, X_test, Y_train, Y_test)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

np.save("./img_data.npy", xy, allow_pickle=True)