import cnn
import numpy as np
image = np.zeros((1 ,28, 28))
filter = np.zeros((8, 1, 5, 5))
bias = np.zeros((8, 1))
net = cnn.CNN()

output = net.convolution(filter, image, bias)

print(output.shape)

output = net.maxPooling(output)

print(output.shape)