# CNN(Convolution Neural Network)
Algorithm learning pattern of data characteristics.
 - Extracting feature for input image using squared kernel(filter).
 - Extracted image called by feature map.
 - Update weights of kernel to back propagation.
 - Fewer calculation than that of full connected layer(dense layer).

ex) When extracting feature map to 5x5 shape input image using 3x3 filter.

![convolution example img](https://github.com/mKangSH/real-time-mosaic-program/blob/main/2022-1/data/Convolution%20example.JPG)

Channel
 - Counts of 2-dimensional array.

multi-channel convolution process
 1. Create as many kernels as the number of channels.
 2. Convolution operation for each channel.
 3. Element-wise add operation for result of convolution. (feature map)

The number of gradient of weights to update on one layer
 - result of multiplying the kernel size, the number of input channel and the number of output filter.
 - ex) (filter size).(3 x 3) x (the number of RGB channel).(3x3) x (the number of output filter).(20) = 540
