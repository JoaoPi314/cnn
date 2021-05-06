import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

#The ####... means that in future these methods will be private


class CNN:

	def __init__(self, inputShape=(0, 0, 0)):
		self.__input_channels = inputShape[0]
		self.__input_size = inputShape[1]


	'''
		Method to concolve the image.
		Params: @filter: filter in format (n_filters, n_channels, size, size)
				@image: image in format: (n_channels, size, size)
				@bias: Bias in format (n_filters, 1)
				@stride: Step that will control convolution. Default = 1

	'''

	def convolution(self, filter, image, bias, stride=1): ################################################################################################################
		
		(n_filters, n_channels, size_f, _) = filter.shape

		(n_channels_img, size_img, _) = image.shape

		#Calculates the output size of the image (After convolution, the image is reducted a little bit)
		size_out = int((size_img - size_f)/stride) + 1


		#Certifies that the number of channels of image and filters are the same
		assert n_channels == n_channels_img, 'N_channels of filter differs from n_channels of image'

		#Creates the output image
		output = np.zeros((n_filters, size_out, size_out))


		#Sweeps towards the image
		for resource_map in range(n_filters):
			#Sweeps vertically
			out_row = 0
			for i in range(0, size_img - size_f + 1, stride):
				#sweeps horizontally
				out_col = 0
				for j in range(0, size_img - size_f + 1, stride):
					#Convolve the actual sizexsize window
					# ouput[out_row, out_col] = [f00, f01, ..., f0n] 		[ij,         i(j+1), ...,     i(j+n)]
					#							[f10, f11, ..., f1n]  (*)	[(i+1)j, (i+1)(j+1), ..., (i+1)(j+n)]
					#							[..., ..., ..., ...]		[...,           ..., ...,        ...]
					#							[fn0, fn1, ..., fnn]		[(i+n)j, (i+n)(j+1), ..., (i+n)(j+n)]
					#							
					output[resource_map, out_row, out_col] = np.sum(filter[resource_map] * image[:, i:i + size_f, j:j + size_f]) + bias[resource_map]
					out_col += 1
				
				out_row += 1

		return output



	'''
		Method to downsample image using MaxPooling
		Params: @image: Image in format (n_channels, size, size)
				@kernel: size of MaxPooling kernel. Default: 2
				@stride: Step that will control Pooling . Default: 2
	'''

	def maxPooling(self, image, kernel=2, stride=2): ################################################################################################################

		(n_channels, size_img, _) = image.shape

		#calculates size of output image
		output_size = int((size_img - kernel)/stride) + 1 

		#Creates output image
		output = np.zeros((n_channels, output_size, output_size))
		
		#Starts sweep
		for resource_map in range(n_channels):
			out_row = 0
			#Sweeps vertically
			for i in range(0, size_img - kernel + 1, stride):
				out_col = 0
				#Sweeps horizontally
				for j in range(0, size_img - kernel + 1, stride):
					#MaxPooling. the output[row, col] will be max value of the kernelxkernel window
					output[resource_map, out_row, out_col] = np.max(image[resource_map, i:i+kernel, j:j+kernel])
					out_col += 1
				out_row += 1


		return output

	'''
		Private method to flatten image after convolutional layers
		Paramas: @image: Image in format (n_channels, size, size)
	'''

	def flatten(self, image):	################################################################################################################
		(n_channels, size, _) = image.shape
		flat = image.reshape((n_channels * size * size, 1))
		return flat


	'''
		Private method to feed foward fully connected layer
		Params: @prev_layer: Outputs from previous layer (n, 1)
				@w: Weights that connect both layers (m, n)
				@b: Bias of each neuron (m,  1)
	'''
	def dense(self, prev_layer, w, b): ###########################################################################################################
		#Certifies that shapes are correct to multiply
		assert prev_layer.shape[0] == w.shape[1], 'Invalid shapes of prev_layer and weights'
		assert w.shape[0] == b.shape[0], 'Invalid shapes of weights and bias'

		#Multiplies the values
		y = w.dot(prev_layer) + b

		output = self.__ReLU(y)

		return output

	'''
		Private method to calculate categorical cross entropy loss function
		Params: @measure: Output calculated by the fully connected layers
				@real: Real output
	'''
	def __categoricalCrossEntropy(self, measure, real):

		output = -np.sum(real * np.log(measure))
		return output


	'''
		Private method to calcule ReLU function
		Params: @data: Data that will be mapped 
	'''

	def __ReLU(self, data):

		data [data <= 0] = 0

		return data

	'''
		Private method to map outputs to softmax function
		Params: @data: Data that will be mapped
	'''

	def __softmax(self, data):
		output = np.exp(data)
		output /= np.sum(output)
		return output

	'''
		Method to set input shape to CNN. Starts to define the weights
		params: @input: input shape (n_channels, size, size) format
	'''

	def set_inputShape(self, input):
		(self.__input_channels, self.__input_size, _) = input.shape

	'''
		Method to get the input size of the CNN
	'''

	def get_inputShape(self):
		return (self.__input_channels, self.__input_size, self.__input_size)


	



