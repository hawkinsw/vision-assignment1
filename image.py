#!/usr/bin/python

from __future__ import print_function

import numpy
import skimage
import skimage.io
import skimage.transform

class Util:
	@classmethod
	def discretize_angle(cls, angle):
		input_angle = angle
		if angle < 0:
			angle += 180
		closest = 180.0
		for c in [0.0, 45.0, 90.0, 135.0]:
			if numpy.abs(angle-c) < numpy.abs(angle-closest):
				closest = c
		Debug.Print("discretize_angle(%f): %f" % (input_angle, closest))
		return closest

	@classmethod
	def values_at(cls, array, y, x, z = 0):
		(array_height, array_width, array_depth) = array.shape
		return array[y % array_height, x % array_width, z]

class IntensityWeight:
	Red = 0.3
	Green = 0.6
	Blue = 0.1

class ChannelIndex:
	Red = 0
	Green = 1
	Blue = 2

class Debug:
	@classmethod
	def Print(cls, string):
		#print(string)
		pass

class Gauss:
	#
	# Calculate 1d Gaussian first derivative.
	#
	@classmethod
	def Gaussian1d1d(cls, i, sigma):
		return -1.0 * i * (1.0/(sigma*sigma)) * Gauss.Gaussian1d(i, sigma)
	
	#
	# Calculate 1d Gaussian
	#
	@classmethod
	def Gaussian1d(cls, i, sigma):
		exponent = -1.0 * ( (i*i) / (2.0*sigma*sigma))
		e_to_the = numpy.exp(exponent)
		inverse_term = 1.0 / (sigma * numpy.sqrt(numpy.pi * 2.0))
		result = inverse_term*e_to_the

		Debug.Print("i: " + str(i))
		Debug.Print("exponent: " + str(exponent))
		Debug.Print("e_to_the: " + str(e_to_the))
		Debug.Print("inverse_term: " + str(inverse_term))
		Debug.Print("result: " + str(result))

		return result

	#
	# Calculate 2d Gaussian
	#
	@classmethod
	def Gaussian2d(cls, i, j, sigma):
		exponent = -1.0 * ( (i*i + j*j) / (2.0*sigma*sigma))
		e_to_the = numpy.exp(exponent)
		inverse_term = 1.0 / (sigma * sigma * (numpy.pi * 2.0))
		result = inverse_term*e_to_the

		Debug.Print("i: " + str(i))
		Debug.Print("j: " + str(j))
		Debug.Print("exponent: " + str(exponent))
		Debug.Print("e_to_the: " + str(e_to_the))
		Debug.Print("inverse_term: " + str(inverse_term))
		Debug.Print("result: " + str(result))

		return result
		

class Image:
	def __init__(self, path):
		self.image = skimage.img_as_float(skimage.io.imread(path))
		self.gradiant_image = None
		self.gradiant_direction = None

	def store(self, path):
		skimage.io.imsave(path, self.image)

	def native_gaussian(self, sigma):
		self.image = skimage.filter.gaussian_filter(self.image, sigma, mode='wrap', multichannel=True)

	def intensify(self):
		(image_height, image_width, image_channels) = self.image.shape
		intensity = numpy.zeros(image_height*image_width*1)
		intensity = intensity.reshape(image_height, image_width, 1)

		# Apply the convolution to the image and its channels.
		for x in range(image_width):
			for y in range(image_height):
				intense = 0.0
				intensity[y,x,0]=self.image[y,x,ChannelIndex.Red]*IntensityWeight.Red+\
					self.image[y,x,ChannelIndex.Green]*IntensityWeight.Green + \
					self.image[y,x,ChannelIndex.Blue]*IntensityWeight.Blue
		self.image = intensity
	
	def store_gradiant(self, path):
		if self.gradiant_image == None:
			return
		skimage.io.imsave(path, self.gradiant_image)
	
	def thin_gradiant(self):
		assert self.gradiant_image != None and self.gradiant_direction != None, \
			"Must compute_gradiant first."
		(grad_height, grad_width, grad_channels) = self.gradiant_image.shape
		thin_gradiant = numpy.zeros(grad_height*grad_width*grad_channels)
		thin_gradiant = thin_gradiant.reshape(grad_height, grad_width, grad_channels)
		for x in range(grad_width):
			for y in range(grad_height):
				a = Util.discretize_angle(self.gradiant_direction[y,x])
				if a == 0.0 or a == 180.0:
					# 
					# Compare to right and left.
					#
					if Util.values_at(self.gradiant_image,y,x,0) > \
						Util.values_at(self.gradiant_image,y,x+1,0) and \
						Util.values_at(self.gradiant_image,y,x,0) > \
						Util.values_at(self.gradiant_image,y,x-1,0):
						thin_gradiant[y,x,0] = self.gradiant_image[y,x,0]
				elif a == 45.0:
					#
					# Compare to right, up and left, down
					#
					if Util.values_at(self.gradiant_image,y,x,0) > \
						Util.values_at(self.gradiant_image,y+1,x+1,0) and \
						Util.values_at(self.gradiant_image,y,x,0) > \
						Util.values_at(self.gradiant_image,y-1,x-1,0):
						thin_gradiant[y,x,0] = self.gradiant_image[y,x,0]
				elif a == 90.0:
					#
					# Compare to up and down
					#
					if Util.values_at(self.gradiant_image,y,x,0) > \
						Util.values_at(self.gradiant_image,y+1,x,0) and \
						Util.values_at(self.gradiant_image,y,x,0) > \
						Util.values_at(self.gradiant_image,y-1,x,0):
						thin_gradiant[y,x,0] = self.gradiant_image[y,x,0]
				elif a == 135.0:
					#
					# Compare to right, down and left, up
					#
					if Util.values_at(self.gradiant_image,y,x,0) > \
						Util.values_at(self.gradiant_image,y-1,x+1,0) and \
						Util.values_at(self.gradiant_image,y,x,0) > \
						Util.values_at(self.gradiant_image,y+1,x-1,0):
						thin_gradiant[y,x,0] = self.gradiant_image[y,x,0]
				else:
					assert False, "discretize_angle failed."
		self.gradiant_image = thin_gradiant

	def compute_gradiant(self, sigma):
		if self.gradiant_image != None:
			return self.gradiant_image

		(image_height, image_width, image_channels) = self.image.shape

		#
		# We only allow this on images that are already intensified.
		#
		assert image_channels == 1, "One channel images only."

		self.gradiant_image = numpy.zeros(image_height*image_width*1)
		self.gradiant_image = self.gradiant_image.reshape(image_height, image_width, 1)

		(image_height, image_width, image_channels) = self.image.shape
		self.gradiant_direction = numpy.zeros(image_height*image_width)
		self.gradiant_direction = self.gradiant_direction.reshape(image_height, image_width)

		convolution_range = 6

		# calculate factor as the gaussian kernel of the 
		# convolution
		factor = numpy.zeros(convolution_range*2)
		d_factor = numpy.zeros(convolution_range*2)
		for c in range(-1*convolution_range, convolution_range):
			# use the 2d gaussian to calculate the
			# amount this pixel should contribute overall.
			factor[c + convolution_range] = Gauss.Gaussian1d(c, sigma)
			d_factor[c + convolution_range] = Gauss.Gaussian1d1d(c, sigma)

		for x in range(image_width):
			for y in range(image_height):
				x_grad = 0.0
				y_grad = 0.0
				# we are calculating the out(x,y)
				# at this point.
				Debug.Print("(y,x,0): (" + str(y) + "," + str(x) + ",0): "
					+ str(self.image[y,x,0]))

				for i in range(-1*convolution_range, convolution_range):
					for j in range(-1*convolution_range, convolution_range):
						# use the 2d gaussian to calculate the
						# amount this pixel should contribute overall.
						x_grad += Util.values_at(self.image, y+j, x+i, 0)* \
							d_factor[i+convolution_range]* \
							factor[j + convolution_range]

						y_grad += Util.values_at(self.image, y+j, x+i, 0)* \
							factor[i + convolution_range]* \
							d_factor[j + convolution_range]

				self.gradiant_image[y,x,0] = numpy.sqrt(x_grad*x_grad + y_grad*y_grad)
				if x_grad == 0.0:
					self.gradiant_direction[y,x] = numpy.rad2deg(numpy.pi/2.0)
				else:
					self.gradiant_direction[y,x] = numpy.rad2deg(numpy.arctan(y_grad/x_grad))
				Debug.Print("gradiant image (y,x,0): (" 
					+ str(y) 
					+ "," 
					+ str(x) 
					+ ",0): " 
					+ str(self.gradiant_image[y,x,0]))
				Debug.Print("gradiant direction (y,x,0): (" 
					+ str(y) 
					+ "," 
					+ str(x) 
					+ "): " 
					+ str(self.gradiant_direction[y,x])
					+ " -> "
					+ str(Util.discretize_angle(self.gradiant_direction[y,x])))

if __name__ == "__main__":
#	sigma = 2
#	i_range = 6
#	j_range = 6
#	s = 0
#	for i in range(-1*i_range, i_range):
#		for j in range(-1*j_range, j_range):
#			factor = Gauss.Gaussian2d(i,j,sigma)
#			s += factor
#			print("(%d,%d): %f" % (i,j,factor))
#	print("sum: %f" % s)
	print("Loading image.")
#	image = Image("./line.jpg")
#	image = Image("./circle.jpg")
	image = Image("./building-crop.jpg")
#	image_native = Image("./building-crop.jpg")

	print("Blurring image.")
	image.intensify()
#	image.edges(2.0, "building-edges.jpg")
	image.compute_gradiant(1.0)
	image.store_gradiant("me-thick.jpg")
	image.thin_gradiant()
	image.store_gradiant("me-thin.jpg")
#	image_native.native_gaussian(2.0)

#	print("Saving image.")
#	image.store("./me.jpg")
#	image_native.store("./native.jpg")
