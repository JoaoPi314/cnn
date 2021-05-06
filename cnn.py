import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

class CNN:

	'''
		Method to concolve the image.
		Params: @filter: filter in format (n_filters, n_channels, size, size)
				@image: image in format: (n_channels, size, size)
				@bias: Bias in format (n_filters, 1)
				@stride: Step that will control convolution. Default = 1

	'''

	def convolution(self, filter, image, bias, stride=1):
		
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