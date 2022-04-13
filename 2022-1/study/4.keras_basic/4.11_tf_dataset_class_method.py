# https://www.tensorflow.org/api_docs/python/tf/data/Dataset

import numpy as np
import tensorflow as tf

# as_numpy_iterator
dataset = tf.data.Dataset.range(10)
print( list(dataset.as_numpy_iterator()) )

# apply
dataset = tf.data.Dataset.range(10)

def filter_five(x):
    return x.filter(lambda x: x < 5)

print( list(dataset.apply(filter_five).as_numpy_iterator()) )

# from_tensor_slices
ds = tf.data.Dataset.from_tensor_slices([1,2,3,4,5])
print(type(ds))
print( list(ds.as_numpy_iterator()) )

# iteration
ds = tf.data.Dataset.from_tensor_slices([2,3,4,5,6])

for d in ds:
    print(d)

# range
ds = tf.data.Dataset.range(1, 10 ,2) # parameter: start, stop, step

# batch & drop_remainder
for d in ds.batch(3, drop_remainder=True):
    print(d)

# window
ds = tf.data.Dataset.range(10)
ds = ds.window(5, shift=1, drop_remainder=False)

for d in ds:
    print(list(d.as_numpy_iterator()))

# flat_map
ds = tf.data.Dataset.range(10)
ds = ds.window(5, shift=1, drop_remainder=True)
ds = ds.flat_map(lambda w: w.batch(5))

for d in ds:
    print(d)

# shuffle
ds = tf.data.Dataset.from_tensor_slices(np.arange(10)).shuffle(buffer_size=5)

for d in ds:
    print(d)

# map (Utilize separating input data and label data)
window_size = 5
ds = tf.data.Dataset.range(10)
ds = ds.window(window_size, shift=1, drop_remainder=True)
ds = ds.flat_map(lambda w: w.batch(window_size))
ds = ds.shuffle(10)

ds = ds.map(lambda x:(x[ :-2], x[-2: ]))
for x, y in ds:
    print('train set: {}'.format(x))
    print('label set: {}'.format(y))
    print('===' * 10)