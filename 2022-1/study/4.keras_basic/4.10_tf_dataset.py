# https://www.tensorflow.org/datasets
# https://www.tensorflow.org/datasets/catalog/overview

import tensorflow_datasets as tfds

dataset_name = 'cifar100'

# shuffle_files=True : shuffle the data set
# with_info=True : return information of the data set
ds = tfds.load(dataset_name, split='train')

print(ds)

for data in ds.take(5):
    image = data['image']
    label = data['label']

    print(image.shape)
    print(label)

# tfds load method 
"""
ds = tfds.load(dataset_name, split='train', as_supervised=True) 

for image, label in ds.take(5):
    print(image.shape, label)
"""