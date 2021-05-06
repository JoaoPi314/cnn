import cnn
import numpy as np
image = np.zeros((1 ,28, 28))
filter = np.zeros((8, 1, 5, 5))
bias = np.zeros((8, 1))
net = cnn.CNN((1, 28,28))

# output = net.convolution(filter, image, bias)

# print(output.shape)

# output = net.maxPooling(output)

# print(output.shape)

# output = net.flatten(output)

# print(output.shape)


# data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# print(data.shape)
# output = np.array([0, 1, 2, 3, 4, 5])
# w = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# 			  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# 			  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# 			  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# 			  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
# 			  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])

# b = np.array([0, 1, 2, 3, 4, 5])

# result = net.dense(data, w, b)

# print(result)


print(net.get_inputShape())
