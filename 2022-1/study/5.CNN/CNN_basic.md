# CNN(Convolution Neural Network)
Algorithm learning pattern of data characteristics.
 - Extracting feature for input image using squared kernel(filter).
 - Extracted image called by feature map.
 - Update weights of kernel to back propagation.
 - Fewer calculation than that of full connected layer(dense layer).

ex) When extracting feature map to 5x5 shape input image using 3x3 filter.

![convolution example img](https://github.com/mKangSH/real-time-mosaic-program/blob/main/2022-1/data/Convolution%20example.JPG)

Multi-channel convolution process
 1. Create as many kernels as the number of channels.
 2. Convolution operation for each channel.
 3. Element-wise add operation for result of convolution. (feature map)

The number of gradient of weights to update on one layer
 - Result of multiplying the kernel size, the number of input channel and the number of output filter + biases.
 - Counts of bias are same to the number of output filter.
 - ex) (filter size).(3 x 3) x (the number of RGB channel).(3) x (the number of output filter).(9) + (biases).(9) = 252

Size of feature map
 - $\frac{ImageHeight+(2*Padding)-KernelHeight+1}{Strides}$
 - <img src="https://latex.codecogs.com/svg.latex?\Large&space;HeightofFeatureMap=\frac{ImageHeight+(2*Padding)-KernelHeight}{Strides} + 1" title="\Large x=\frac{ImageHeight+(2*Padding)-KernelHeight}{Strides} + 1" />
 - ![\Large Height=\frac{ImageHeight+(2*Padding)-KernelHeight}{Strides} + 1](https://latex.codecogs.com/svg.latex?x%3D%5Cfrac%7B-b%5Cpm%5Csqrt%7Bb%5E2-4ac%7D%7D%7B2a%7D)

_Channel_
 - Counts of 2-dimensional array.

_Stride_
 - Moving interval of the kernel (in pixels).
 - Usually set to 1 or 2 value.

_Padding_
 - Filling the edges with padding values.
 - Support to maintain size of feature map relative to input image.

