import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#write your path_dir
path_dir = '/real-time-mosaic-program/2022-1/data'

tensor = [0, 1, 2, 3, 4]

for i in range(5):
    test_image = Image.open(path_dir + '/test%d.png' % i)
    test_array = np.array(test_image)

    #gray_image = ((test_array[:,:,0]) + (test_array[:,:,1]) + (test_array[:,:,2])) / 3
    gray_image = (0.299 * test_array[:,:,0]) + (0.587 * test_array[:,:,1]) + (0.114 * test_array[:,:,2])

    gray_image = (-(gray_image - 255)) / 255
    
    tensor[i] = (gray_image)

tensor = tf.constant(tensor)

print(tensor.shape)