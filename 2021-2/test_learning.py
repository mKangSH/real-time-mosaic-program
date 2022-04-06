import os, re, glob
import cv2, pickle
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import (Dense, BatchNormalization, Dropout, Flatten)
from tensorflow.keras.datasets.mnist import load_data
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.models import load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

groups_folder_path = './cnn_sample/'
categories = ["kang", "people"]
num_classes = len(categories)

def Dataization(img_path):
    image_w = 180
    image_h = 180
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=image_w/img.shape[1], fy=image_h/img.shape[0])
    return (img/256)


src = []
name = []
test = []
#image_dir = "C:/"
image_dir = "D:/workingFolder/Tensor/cnn_sample/people/"

for file in os.listdir(image_dir):
    if (file.find(".jfif") is not -1):      
        src.append(image_dir + file)
        name.append(file)
        test.append(Dataization(image_dir + file))
 
test = np.array(test)
model = load_model('kang.h5')
temp = model.predict(test)
predict = np.argmax(temp, axis=1)
 
for i in range(len(test)):
    print(name[i] + ":, Predict: "+ str(categories[predict[i]]))

