#!/usr/bin/python

from __future__ import print_function
from operator import itemgetter

import math

import os

import numpy
import skimage
import skimage.io
import skimage.transform
import skimage.feature
import skimage.filter

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

class Derivative:
	WithRespectToY = 0
	WithRespectToX = 1

	#
	# Parameter function must look like this:
	#
	# function[s] -> indexes a numpy array using sigma.
	# function[s][y,x,c] -> indexes that numpy array.
	#

	#
	# References:
	#
	# siftppt-1.ppt
	# http://en.wikipedia.org/wiki/Finite_difference
	#

	@classmethod
	def Sigma1d(cls, (sigma, y, x, c), function):
		max_sigma = len(function)
		max_y, max_x, max_c = function[0].shape
		return (function[(sigma+1)%max_sigma][y,x,c] - function[sigma-1][y,x,c])/2.0

	@classmethod
	def X1d(cls, (sigma, y, x, c), function):
		max_sigma = len(function)
		max_y, max_x, max_c = function[0].shape
		return (function[sigma][y,(x+1)%max_x,c] - function[sigma][y,x-1,c])/2.0

	@classmethod
	def Y1d(cls, (sigma, y, x, c), function):
		max_sigma = len(function)
		max_y, max_x, max_c = function[0].shape
		return (function[sigma][(y+1)%max_y,x,c] - function[sigma][y-1,x,c])/2.0

	@classmethod
	def SigmaSigma2d(cls, (sigma, y, x, c), function):
		max_sigma = len(function)
		max_y, max_x, max_c = function[0].shape
		return (function[sigma-1][y,x,c]  -
			(2*function[sigma][y,x,c]) +
			function[(sigma+1)%max_sigma][y,x,c])

	@classmethod
	def YY2d(cls, (sigma, y, x, c), function):
		max_sigma = len(function)
		max_y, max_x, max_c = function[0].shape
		return (function[sigma][y-1,x,c]  -
			(2*function[sigma][y,x,c]) +
			function[sigma][(y+1)%max_y,x,c])

	@classmethod
	def XX2d(cls, (sigma, y, x, c), function):
		max_sigma = len(function)
		max_y, max_x, max_c = function[0].shape
		return (function[sigma][y,x-1,c]  -
			(2*function[sigma][y,x,c]) +
			function[sigma][y,(x+1)%max_x,c])

	@classmethod
	def SigmaY2d(cls, (sigma, y, x, c), function):
		max_sigma = len(function)
		max_y, max_x, max_c = function[0].shape
		return (
			((function[(sigma+1)%max_sigma][(y+1)%max_y,x,c] - function[sigma-1][(y+1)%max_y,x,c]) -
			(function[(sigma+1)%max_sigma][y-1,x,c] - function[sigma-1][y-1,x,c])) /
			4.0
		)

	@classmethod
	def SigmaX2d(cls, (sigma, y, x, c), function):
		max_sigma = len(function)
		max_y, max_x, max_c = function[0].shape
		return (
			((function[(sigma+1)%max_sigma][y,(x+1)%max_x,c] - function[sigma-1][y,(x+1)%max_x,c]) -
			(function[(sigma+1)%max_sigma][y,x-1,c] - function[sigma-1][y,x-1,c])) /
			4.0
		)

	@classmethod
	def YX2d(cls, (sigma, y, x, c), function):
		max_sigma = len(function)
		max_y, max_x, max_c = function[0].shape
		return (
			((function[sigma][(y+1)%max_y,(x+1)%max_x,c] - function[sigma][y-1,(x+1)%max_x,c]) -
			(function[sigma][(y+1)%max_y,x-1,c] - function[sigma][y-1,x-1,c])) /
			4.0
		)

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

	@classmethod
	def ImageFromFile(cls, path):
		img = Image()
		img.image = skimage.img_as_float(skimage.io.imread(path))
		return img

	@classmethod
	def ImageFromArray(cls, array):
		img = Image()
		img.image = array
		return img

	def __init__(self):
		self.image = None
		self.gradient_image = None
		self.gradient_direction = None

	def store_gradient(self, path):
		if self.gradient_image != None:
			skimage.io.imsave(path, self.gradient_image)

	def store_image(self, path):
		if self.image != None:
			skimage.io.imsave(path, self.image)

	def height(self):
		return self.image.shape[0]

	def width(self):
		return self.image.shape[1]

	def subsample(self, rate):
		""" Returns a new image subsampled at rate. """
		return Image.ImageFromArray(self._subsample(self.image, rate))

	def _subsample(self, image, rate):
		""" Returns a copy of image subsampled at rate. """
		return image[0::rate, 0::rate, 0::]

	def sift(self, k=math.pow(2,1.0/3.0), sigma=1.6, s=3):
		""" Perform SIFT.

		"""
		keypoints = []
		filtered_keypoints = []
		base_image = self.image

		biggest_sigma = sigma
		biggest_circle_radius = 10

		filtered_keypoint_image = numpy.zeros(self.image.shape)
		all_keypoint_image = numpy.zeros(self.image.shape)

		for o in range(4):
			# we are doing a new octave!
			octave = self._octave(base_image, k, sigma, s)

			Debug.Print("octave size: %d" % len(octave))

			# calculate DoGs
			dogs = self._dog(octave)
			dog_height, dog_width, dog_depth = dogs[0].shape

			Debug.Print("dogs size: %d" % len(dogs))

			# calculate scale space extrema
			extrema = self._scale_space_extrema(dogs)

			for (y,x,c,s) in extrema:
				x_orig = (s,y,x,c)
				x_hat = (s,y,x,c)

				Debug.Print("extrema at (s,y,x,c):(%d,%d,%d,%d)" % x_orig)
				Debug.Print("keypoint at (y,x,c,sigma):(%d,%d,%d,%f)" %
					(y*(2**o),x*(2**o),c,math.pow(k,o)*math.pow(k,s)*sigma))

				#
				# Limiting the number of
				# iterations is a trick from
				# Sift++. See writeup for more
				# information.
				#
				flapping = False
				for attempts in range(5):
					adjusted = False

					Debug.Print("x_hat (sigma,y,x): (%f,%f,%f)" % (x_hat[0],x_hat[1],x_hat[2]))
					a = numpy.array([
						[ Derivative.SigmaSigma2d(x_hat, dogs),
						  Derivative.SigmaY2d(x_hat, dogs),
						  Derivative.SigmaX2d(x_hat, dogs)
						],
						[ Derivative.SigmaY2d(x_hat, dogs),
						  Derivative.YY2d(x_hat, dogs),
						  Derivative.YX2d(x_hat, dogs)
						],
						[ Derivative.SigmaX2d(x_hat, dogs),
						  Derivative.YX2d(x_hat, dogs),
						  Derivative.XX2d(x_hat, dogs)
						]
					])
					b = numpy.array([-1.0*Derivative.Sigma1d(x_hat, dogs),
						-1.0*Derivative.Y1d(x_hat, dogs),
						-1.0*Derivative.X1d(x_hat, dogs)
					])

					Debug.Print("Hessian/Second derivative: " + repr(a))
					Debug.Print("First derivative: " + repr(b))

					#
					# Solve for x in ax = b
					#
					delta = numpy.linalg.solve(a,b)

					Debug.Print("Delta (sigma,y,x): (%f,%f,%f)" % (delta[0],delta[1],delta[2]))
					new_sigma = x_hat[0]
					new_y = x_hat[1]
					new_x = x_hat[2]
					new_c = x_hat[3]

					if delta[1] > 0.5:
						if new_y + 1 < dog_height:
							new_y += 1
							adjusted = True
						else:
							Debug.Print("Wanted to adjust y (up), but would have been OOB.")
							adjusted = False
					elif delta[1] < -0.5:
						if new_y - 1 >= 0:
							new_y -= 1
							adjusted = True
						else:
							Debug.Print("Wanted to adjust y (down), but would have been OOB.")
							adjusted = False
					if delta[2] > 0.5:
						if new_x + 1 < dog_width:
							new_x += 1
							adjusted = True
						else:
							Debug.Print("Wanted to adjust x (up), but would have been OOB.")
							adjusted = False
					elif delta[2] < -0.5:
						if new_x - 1 >= 0:
							new_x -= 1
							adjusted = True
						else:
							Debug.Print("Wanted to adjust x (down), but would have been OOB.")
							adjusted = False

					x_hat = (new_sigma, new_y, new_x, new_c)

					if not adjusted:
						break
					else:
						Debug.Print("Re-interpolating with a new x_hat (%d)" % attempts)
				else:
					flapping = True

				Debug.Print("Finished sub-pixel localization (Flapping? %s): (s,y,x,c)o vs (s,y,x,c)n: (%d,%d,%d,%d) vs (%d,%d,%d,%d)" % ((flapping,) + x_orig + x_hat))

				d_x_hat = dogs[s][y,x,c] + 0.5*(
					(Derivative.X1d(x_orig,dogs)*x_hat[2]) +
					(Derivative.Y1d(x_orig,dogs)*x_hat[1]) +
					(Derivative.Sigma1d(x_orig,dogs)*x_hat[0])
				)
				Debug.Print("d_x_hat: %f" % d_x_hat)

				keypoint_sigma = (
					math.pow(k,o)* # prefix the octave
					math.pow(k,s)* # now adjust for the scale
					sigma          # finally, give us sigma
				)

				#
				# Only keep the points with a good threshold value.
				#
				if d_x_hat >= 0.03:
					filtered_keypoints.append((
					y*(2**o), # each x,y coordinate may have been smushed when subsampling
					x*(2**o), # use this trick to restore them back to the original value
					c,
					keypoint_sigma
					))

				#
				# Keep all the candidate points!
				#
				keypoints.append((
					y*(2**o), # each x,y coordinate may have been smushed when subsampling
					x*(2**o), # use this trick to restore them back to the original value
					c,
					keypoint_sigma
				))
				if keypoint_sigma > biggest_sigma:
					biggest_sigma = keypoint_sigma

			# Grab the 2*sigma image
			base_image = octave[len(octave)-2-1]

			# subsample the 2*sigma image at 2x rate.
			base_image = self._subsample(base_image, 2)

			# and, repeat.

		#
		# Draw (filtered and all) keypoints with circles relative to their
		# scale. Bigger circles mean a higher scale.
		#
		for (y,x,c,sig) in filtered_keypoints:
			rr, cc= skimage.draw.circle(y,
				x,
				(sig/biggest_sigma)*biggest_circle_radius,
				shape=filtered_keypoint_image.shape
				)
			#
			# Note: This will not work unless you apply
			# https://github.com/sciunto/scikit-image/commit/b3f0c1eb963f791634fb3be982721bfcd1d536c2
			# to your draw.py.
			filtered_keypoint_image[rr,cc,0] = 1.0

		for (y,x,c,sig) in keypoints:
			rr, cc= skimage.draw.circle(y,
				x,
				(sig/biggest_sigma)*biggest_circle_radius,
				shape=all_keypoint_image.shape
				)
			#
			# Note: See above.
			all_keypoint_image[rr,cc,0] = 1.0

		Debug.Print("# all keypoints: %d" % len(keypoints))
		Debug.Print("# filtered keypoints: %d" % len(filtered_keypoints))

		return (Image.ImageFromArray(filtered_keypoint_image),
			Image.ImageFromArray(all_keypoint_image))

	def _is_extreme(self, y, x, c, upper, middle, lower):
		maximum = True
		minimum = True

		height, width, depth = upper.shape
		# iterate the neighborhood
		comparison = middle[y,x,c]
		for xx in range(-3, 4):
			if not maximum and not minimum:
				return False, False
			for yy in range(-3, 4):
				if not maximum and not minimum:
					return False, False
				c_yy = y + yy
				c_xx = x + xx
				if c_yy >= 0 and c_xx >= 0 and\
				   c_yy < height and c_xx < width:
					# Valid comparison point.

					Debug.Print("c_xx: %d" % c_xx)
					Debug.Print("c_yy: %d" % c_yy)
					# maximum
					if upper[c_yy, c_xx, c] >= comparison:
						maximum = False
					if lower[c_yy, c_xx, c] >= comparison:
						maximum = False
					# take special care not to compare
					# to ourselves in the middle
					if xx!=0 and yy!=0 and middle[c_yy, c_xx, c] >= comparison:
						maximum = False

					# minimum
					if upper[c_yy, c_xx, c] <= comparison:
						minimum = False
					if lower[c_yy, c_xx, c] <= comparison:
						minimum = False
					# take special care not to compare
					# to ourselves in the middle
					if xx!=0 and yy!=0 and middle[c_yy, c_xx, c] <= comparison:
						minimum = False
		return (maximum, minimum)

	def _scale_space_extrema(self, dogs):
		""" Calculate a list of the scale space extrema """
		assert len(dogs) != 0, "No dogs!"

		height, width, depth = dogs[0].shape

		extrema = []

		for x in range(width):
			for y in range(height):
				for c in range(depth):
					for s in range(len(dogs)):
						if s-2 >= 0:
							Debug.Print("(y,x,c,s): %d, %d, %d, %d" % (y, x, c, s))
							maxi, mini =\
								self._is_extreme(y, x, c, dogs[s], dogs[s-1], dogs[s-2])
							if maxi:
								Debug.Print("Maxi: (y,x,c,s): %d, %d, %d, %d" % (y, x, c, s))
								extrema.append((y,x,c,s-1))
							elif mini:
								Debug.Print("Mini: (y,x,c,s): %d, %d, %d, %d" % (y, x, c, s))
								extrema.append((y,x,c,s-1))
		return extrema

	def _dog(self, octave):
		""" Create the Differences of Gaussians from a set an octave. """
		dogs = []
		for s in range(len(octave)):
			if s-1 >= 0:
				# subtract
				dogs.append(numpy.subtract(octave[s], octave[s-1]))
		return dogs

	def octave(self, k=math.pow(2,1.0/3.0), sigma=1.6, s=3):
		height, width, depth = self.image.shape
		assert depth == 1, "Depth is not singular."

		return Image.ImageFromArray(self._octave(self.image, k, sigma, s))

	def _octave(self, image, k, sigma, s):
		""" Calculate an entire octave. """
		octave = (s+3)*[None]
		for level in range(s+3):
			level_sigma = math.pow(k,level)*sigma
			Debug.Print("level: %d" % level)
			Debug.Print("level_sigma: %f" % level_sigma)
			#
			# Use the built-in Gaussian filter here for simplicity.
			#
			octave[level] = skimage.filter.gaussian_filter(image, level_sigma, mode='wrap')
		return octave

	def canny(self, sigma, start_thresh, continue_thresh, save=None):
		#
		# First, change to grayscale
		#
		tmp = self._intensify(self.image)

		#
		# Compute the gradient.
		#
		g, gd = self._compute_gradient(tmp, sigma, save=save)

		#
		# Thin the gradient.
		#
		tmp = self._thin_gradient(g, gd)

		#
		# Relative the edges to the max
		#
		up = self._relative_up(tmp)

		#mean = numpy.mean(self._down_rank(up))
		mean = numpy.mean(up)
		Debug.Print("mean: %s" % mean)

		#
		# Calculate threshold values based on
		# the mean of the intensity of the edges.
		#
		start_thresh = start_thresh * mean
		continue_thresh = continue_thresh * mean

		#
		# Keep just the chained edges.
		#
		connected_edges = self._connected_edges(up, start_thresh, continue_thresh)

		return Image.ImageFromArray(connected_edges)

	def _up_rank(self, array):
		""" Add a z dimension to array. """
		height, width = array.shape

		aa = numpy.zeros(height*width*1)
		aa = aa.reshape(height, width, 1)
		for x in range(width):
			for y in range(height):
				aa[y,x,0] = array[y,x]
		return aa

	def _down_rank(self, array):
		""" Remove the z dimension from array. """
		height, width, depth = array.shape

		aa = numpy.zeros(height*width)
		aa = aa.reshape(height, width)
		for x in range(width):
			for y in range(height):
				aa[y,x] = array[y,x]
		return aa


	def _relative_up(self, image):
		""" Make an array where all values are relative to the max of image."""
		maxi = 0.0
		height, width, parts = image.shape
		up = numpy.zeros(height*width*parts)
		up = up.reshape(height, width, parts)
		for x in range(width):
			for y in range(height):
				for z in range(parts):
					if (image[y,x,z]>maxi): maxi = image[y,x,z]

		for x in range(width):
			for y in range(height):
				for z in range(parts):
					up[y,x,z] = image[y,x,z]/maxi
		return up

	def _save_separate_gradients(self, gradiants, path_base, extension="jpg"):
		"""Save the x and y gradients into separate -x and -y files. """
		height, width, depth = gradiants.shape

		assert depth == 2, "Depth must be 2 (ie, x and y derivatives)"

		x_deriv_base = numpy.zeros(height*width*1)
		x_deriv_base = x_deriv_base.reshape(height, width, 1)
		y_deriv_base = numpy.zeros(height*width*1)
		y_deriv_base = y_deriv_base.reshape(height, width, 1)
		for x in range(width):
			for y in range(height):
				x_deriv_base[y,x,0] = abs(gradiants[y,x,Derivative.WithRespectToX])
				y_deriv_base[y,x,0] = abs(gradiants[y,x,Derivative.WithRespectToY])
		x_deriv_base = self._relative_up(x_deriv_base)
		x_deriv_image = Image.ImageFromArray(x_deriv_base)
		x_deriv_image.store_image(path_base + "-x." + extension)

		y_deriv_base = self._relative_up(y_deriv_base)
		y_deriv_image = Image.ImageFromArray(y_deriv_base)
		y_deriv_image.store_image(path_base + "-y." + extension)

	def _connected_edges(self, edges, start_thresh, continue_thresh):
		"""Calculate the connected edges."""
		height, width, depth = edges.shape

		changes = []

		connected_edges = numpy.zeros(height*width*depth)
		connected_edges = connected_edges.reshape(height, width, depth)
		for x in range(width):
			for y in range(height):
				if edges[y,x] >= start_thresh:
					connected_edges[y,x,0] = edges[y,x]
					changes.append((y,x))

		while changes:
			updated_changes = []
			for (y,x) in changes:
				for i in (-1,0,1):
					for j in (-1,0,1):
						if i==0 and j==0: continue
						xx = x+i
						yy = y+j
						if xx >= 0 and xx < width and yy >= 0 and yy < height:
							Debug.Print("(%d,%d): %f vs %f (%f)" %
								(yy, xx, edges[yy, xx], continue_thresh, connected_edges[yy,xx,0]))
							if connected_edges[yy,xx,0] == 0 and \
							   edges[yy,xx] > continue_thresh:
								Debug.Print("(%d,%d): added" % (yy,xx))
								connected_edges[yy,xx,0] = edges[yy,xx]
								updated_changes.append((yy,xx))
			changes = updated_changes
		return connected_edges

	def corners(self, sigma, threshold, neighborhood_size = 4):
		sortable = []
		intensity = self._intensify(self.image)
		gs, gd = self._compute_separate_gradient(intensity, sigma)
		gradient_height, gradient_width, gradient_parts = gs.shape

		corners_image = numpy.zeros(gradient_height*gradient_width*1)
		corners_image = corners_image.reshape(gradient_height, gradient_width, 1)

		corners_covariance = numpy.zeros(gradient_height*gradient_width)
		corners_covariance = corners_covariance.reshape(
			gradient_height,
			gradient_width)

		for x in range(gradient_width):
			for y in range(gradient_height):
				#
				# Compute the covarient matrix
				# in the neighborhood of (y,x)
				#
				covar = numpy.zeros(4)
				covar = covar.reshape(2,2)
				for i in range(-1*neighborhood_size, neighborhood_size+1):
					for j in range(-1*neighborhood_size, neighborhood_size+1):
						covar[0,0] += \
							Util.values_at(gs,y+j,x+i,Derivative.WithRespectToX)*\
							Util.values_at(gs,y+j,x+i,Derivative.WithRespectToX)
						covar[0,1] += \
							Util.values_at(gs,y+j,x+i,Derivative.WithRespectToX)*\
							Util.values_at(gs,y+j,x+i,Derivative.WithRespectToY)
						covar[1,0] += \
							Util.values_at(gs,y+j,x+i,Derivative.WithRespectToX)*\
							Util.values_at(gs,y+j,x+i,Derivative.WithRespectToY)
						covar[1,1] += \
							Util.values_at(gs,y+j,x+i,Derivative.WithRespectToY)*\
							Util.values_at(gs,y+j,x+i,Derivative.WithRespectToY)
				#
				# Compute the eigenvalues
				#
				w, v = numpy.linalg.eig(covar)
				Debug.Print("Eigenvalues at (%d, %d): %s" % (x,y,str(w)))

				#
				# Find the smaller of the two eigenvalues.
				#
				e = 0.0
				if w[0] <= w[1]:
					e = w[0]
				else:
					e = w[1]

				#
				# Compare to a threshold.
				#
				if e > threshold:
					# Add this point to a sortable list and
					# update the neighborhood where this threshold
					# value is bigger than existing values.
					sortable.append((e, y, x))
					corners_covariance[y,x] = e
					for i in range(-1*neighborhood_size, neighborhood_size+1):
						for j in range(-1*neighborhood_size, neighborhood_size+1):
							# Using a slice index/assignment from numpy would be
							# awesome, but I don't know if we can.
							if (x+i) > 0 and (x+i) < gradient_width and \
							   (y+j) > 0 and (y+j) < gradient_height and\
								 corners_covariance[y+j,x+i] <= e:
								corners_covariance[y+j,x+i] = e

		#
		# Accentuate local maximums.
		#
		sortable = sorted(sortable, key=itemgetter(0), reverse=True)
		max_e = sortable[0][0]
		for e,y,x in sortable:
			Debug.Print("In neighborhood of (%d,%d):" % (y, x))
			corners_covariance[y,x] = e
			if e>max_e: max_e = e
			for i in range(-1*neighborhood_size, neighborhood_size+1):
				for j in range(-1*neighborhood_size, neighborhood_size+1):
					if (x+i) > 0 and (x+i) < gradient_width and \
					   (y+j) > 0 and (y+j) < gradient_height and\
						 corners_covariance[y+j,x+i] < e:
						Debug.Print("Clearing %f at (%d,%d) <= %f" %
							(corners_covariance[y+j,x+i], y+j, x+i, e))
						corners_covariance[y+j,x+i] = 0.0
		#
		# Relativize corners
		#
		for x in range(gradient_width):
			for y in range(gradient_height):
				corners_image[y,x,0] = corners_covariance[y,x]/max_e

		return Image.ImageFromArray(corners_image)

	def native_canny(self, sigma):
		i = self._intensify(self.image)
		i = self._down_rank(i)
		e = skimage.filter.canny(i, sigma=sigma)
		e = self._up_rank(e)
		return Image.ImageFromArray(e)

	def native_corners(self, sigma, threshold):
		i = self._intensify(self.image)
		i = self._down_rank(i)
		e = skimage.feature.corner_harris(i, k=threshold, sigma=sigma)
		e = self._up_rank(e)
		return Image.ImageFromArray(e)

	def native_gaussian(self, sigma):
		return Image.ImageFromArray(skimage.filter.gaussian_filter(
			self.image,
			sigma,
			mode='wrap',
			multichannel=True))

	def intensify(self):
		"""Convert to a grayscale image. """
		return Image.ImageFromArray(self._intensify(self.image))

	def _intensify(self, image):
		"""Convert to a grayscale array. """
		(image_height, image_width, image_channels) = image.shape
		intensity = numpy.zeros(image_height*image_width*1)
		intensity = intensity.reshape(image_height, image_width, 1)

		# Apply the convolution to the image and its channels.
		for x in range(image_width):
			for y in range(image_height):
				intense = 0.0
				intensity[y,x,0]=image[y,x,ChannelIndex.Red]*IntensityWeight.Red+\
					image[y,x,ChannelIndex.Green]*IntensityWeight.Green + \
					image[y,x,ChannelIndex.Blue]*IntensityWeight.Blue
		#self.image = intensity
		return intensity

	def thin_gradient(self):
		img = Image.ImageFromArray(None)
		if self.gradient_image == None or self.gradient_direction == None:
			return img
		img.gradient_direction = self.gradient_direction
		img.gradient_image = self._thin_gradient(
			self.gradient_image,
			self.gradient_direction)
		return img

	def _thin_gradient(self, gradient_image, gradient_direction):
		(grad_height, grad_width, grad_channels) = gradient_image.shape
		thin_gradient = numpy.zeros(grad_height*grad_width*grad_channels)
		thin_gradient = thin_gradient.reshape(grad_height,
			grad_width,
			grad_channels)

		for x in range(grad_width):
			for y in range(grad_height):
				a = Util.discretize_angle(gradient_direction[y,x])
				if a == 0.0 or a == 180.0:
					#
					# Compare to right and left.
					#
					if Util.values_at(gradient_image,y,x,0) > \
						Util.values_at(gradient_image,y,x+1,0) and \
						Util.values_at(gradient_image,y,x,0) > \
						Util.values_at(gradient_image,y,x-1,0):
						thin_gradient[y,x,0] = gradient_image[y,x,0]
				elif a == 45.0:
					#
					# Compare to right, up and left, down
					#
					if Util.values_at(gradient_image,y,x,0) > \
						Util.values_at(gradient_image,y+1,x+1,0) and \
						Util.values_at(gradient_image,y,x,0) > \
						Util.values_at(gradient_image,y-1,x-1,0):
						thin_gradient[y,x,0] = gradient_image[y,x,0]
				elif a == 90.0:
					#
					# Compare to up and down
					#
					if Util.values_at(gradient_image,y,x,0) > \
						Util.values_at(gradient_image,y+1,x,0) and \
						Util.values_at(gradient_image,y,x,0) > \
						Util.values_at(gradient_image,y-1,x,0):
						thin_gradient[y,x,0] = gradient_image[y,x,0]
				elif a == 135.0:
					#
					# Compare to right, down and left, up
					#
					if Util.values_at(gradient_image,y,x,0) > \
						Util.values_at(gradient_image,y-1,x+1,0) and \
						Util.values_at(gradient_image,y,x,0) > \
						Util.values_at(gradient_image,y+1,x-1,0):
						thin_gradient[y,x,0] = gradient_image[y,x,0]
				else:
					assert False, "discretize_angle failed."
		#self.gradient_image = thin_gradient
		return thin_gradient

	def compute_gradient(self, sigma):
		img = Image.ImageFromArray(None)
		if self.gradient_image != None:
			img.gradient_image = self.gradient_image
			img.gradient_direction = self.gradient_direction
		else:
			img.gradient_image, img.gradient_direction = \
				self._compute_gradient(self.image,sigma)
		return img

	def compute_gaussian(self, sigma):
		return Image.ImageFromArray(self._compute_gaussian(self.image, sigma))

	def _compute_gaussian(self, image, sigma):
		height, width, depth = image.shape

		support = int(sigma*2 + 0.5)
		kernel = numpy.zeros((support*2+1)*(support*2+1))
		kernel = kernel.reshape((support*2+1), (support*2+1))
		kernel_sum = 0.0
		for i in range(-1*support, support+1):
			for j in range(-1*support, support+1):
				kernel[j+support, i+support] = Gauss.Gaussian2d(i, j, sigma)
				kernel_sum += kernel[j+support, i+support]

		gimage = numpy.zeros(height*width*depth)
		gimage = gimage.reshape(height, width, depth)
		for x in range(width):
			for y in range(height):
				for d in range(depth):
					convolve = 0.0
					for i in range(-1*support, support+1):
						for j in range(-1*support, support+1):
							convolve += (kernel[j+support, i+support] *
								Util.values_at(image, i+y, j+x, d))
					Debug.Print("(%d, %d, %d): %f to %f" % (y,x,d,image[y,x,d], convolve))
					gimage[y,x,d] = convolve
		return gimage

	def _compute_gradient(self, image, sigma, save=None):
		separate_gradient, gradient_direction = \
			self._compute_separate_gradient(image, sigma)
		gradient_height, gradient_width, gradient_parts = separate_gradient.shape

		if save != None:
			self._save_separate_gradients(separate_gradient, save)

		gradient_image = numpy.zeros(gradient_height * gradient_width * 1)
		gradient_image = gradient_image.reshape(gradient_height, gradient_width, 1)
		for x in range(gradient_width):
			for y in range(gradient_height):
				y_grad = separate_gradient[y,x,Derivative.WithRespectToY]
				x_grad = separate_gradient[y,x,Derivative.WithRespectToX]
				gradient_image[y,x,0] = numpy.sqrt(x_grad*x_grad + y_grad*y_grad)
		return (gradient_image, gradient_direction)

	#@profile
	def _compute_separate_gradient(self, image, sigma):

		(image_height, image_width, image_channels) = image.shape

		#
		# We only allow this on images that are already intensified.
		#
		assert image_channels == 1, "One channel images only."

		separate_gradient = numpy.zeros(image_height*image_width * 2)
		separate_gradient = separate_gradient.reshape(image_height, image_width, 2)

		(image_height, image_width, image_channels) = image.shape
		gradient_direction = numpy.zeros(image_height*image_width)
		gradient_direction = gradient_direction.reshape(image_height, image_width)

		convolution_range = int(sigma*2 + 0.5)

		# calculate factor as the gaussian kernel of the
		# convolution
		factor = numpy.zeros(convolution_range*2+1)
		d_factor = numpy.zeros(convolution_range*2+1)

		dy_factor = numpy.zeros((convolution_range*2+1)*(convolution_range*2+1))
		dy_factor = dy_factor.reshape((convolution_range*2+1),(convolution_range*2+1))
		dx_factor = numpy.zeros((convolution_range*2+1)*(convolution_range*2+1))
		dx_factor = dx_factor.reshape((convolution_range*2+1),(convolution_range*2+1))

		for c in range(-1*convolution_range, convolution_range+1):
			# use the 2d gaussian to calculate the
			# amount this pixel should contribute overall.
			factor[c + convolution_range] = Gauss.Gaussian1d(c, sigma)
			d_factor[c + convolution_range] = Gauss.Gaussian1d1d(c, sigma)

		for i in range(-1*convolution_range, convolution_range+1):
			for j in range(-1*convolution_range, convolution_range+1):
				dx_factor[j+convolution_range,i+convolution_range] = \
					d_factor[i+convolution_range]* \
					factor[j+convolution_range]
				dy_factor[j+convolution_range,i+convolution_range] = \
					factor[i + convolution_range]* \
					d_factor[j + convolution_range]

		for x in range(image_width):
			for y in range(image_height):
				x_grad = 0.0
				y_grad = 0.0
				# we are calculating the out(x,y)
				# at this point.
				Debug.Print("(y,x,0): (" + str(y) + "," + str(x) + ",0): "
					+ str(image[y,x,0]))

				#
				# This is a very hot part of the code. It is optimized
				# for speed and not for readability.
				#
				for i in range(-1*convolution_range, convolution_range+1):
					for j in range(-1*convolution_range, convolution_range+1):
						x_grad += image[(y+j)%image_height, (x+i)%image_width,0] * \
							dx_factor[j+convolution_range,i+convolution_range]
						y_grad += image[(y+j)%image_height, (x+i)%image_width,0] * \
							dy_factor[j+convolution_range,i+convolution_range]

				separate_gradient[y,x,Derivative.WithRespectToY] = y_grad
				separate_gradient[y,x,Derivative.WithRespectToX] = x_grad

				if x_grad == 0.0:
					gradient_direction[y,x] = numpy.rad2deg(numpy.pi/2.0)
				else:
					gradient_direction[y,x] = numpy.rad2deg(numpy.arctan(y_grad/x_grad))

				Debug.Print("gradient image (y,x,0): ("
					+ str(y)
					+ ","
					+ str(x)
					+ ",0): "
					+ str(separate_gradient[y,x,0]))
				Debug.Print("gradient direction (y,x,0): ("
					+ str(y)
					+ ","
					+ str(x)
					+ "): "
					+ str(gradient_direction[y,x])
					+ " -> "
					+ str(Util.discretize_angle(gradient_direction[y,x])))
		return (separate_gradient, gradient_direction)

def frange(low, high, step):
	current = low
	# Rounding errors!
	while current<=(high+0.0000001):
		yield current
		current+=step

def test_harris(images):
	sigma_low = 1.0
	sigma_high = 1.0
	sigma_step = 0.1

	thresh_low = 0.3
	thresh_high = 0.3
	thresh_step = 0.1

	neighborhood_low = 5
	neighborhood_high = 6
	neighborhood_step = 1

	for i in images:
		print("<table>")
		for sigma in frange(sigma_low,
			sigma_high,
			sigma_step):
			for thresh in frange(
				thresh_low,
				thresh_high,
				thresh_step):
				for neighborhood in range(
					neighborhood_low,
					neighborhood_high,
					neighborhood_step):
					Debug.Print("sigma, thresh, neighborhood: %f %f %d" % (sigma, thresh, neighborhood))
					image = Image.ImageFromFile(i + ".jpg")
					try:
						os.mkdir("./results-corners/" + i + "/")
					except:
						pass
					save = "./results-corners/" + i + "/" +\
						str(sigma) + "-" +\
						str(thresh) + "-" +\
						str(neighborhood)

					corners = image.corners(sigma,
						thresh,
						neighborhood_size=neighborhood)
					corners.store_image(save+".jpg")
					print("<tr>")
					print("<td colspan=1>")
					print("Sigma: " + str(sigma) + "&nbsp;")
					print("Threshold: " + str(thresh) + "&nbsp;")
					print("Neighborhood: " + str(neighborhood) + "&nbsp;")
					print("</td>")
					print("</tr>")

					print("<tr>")
					print("<td>Final Result</td>")
					print("</tr>")
					print("<tr>")
					print("<td><img src=" + save + ".jpg height=" + str(image.height()/2) +  " width=" + str(image.width()/2) + "></td>")
					print("</tr>")
		print("</table>")

def test_canny(images):
	sigma_low = 1.0
	sigma_high = 2.0
	sigma_step = 0.1

	start_thresh_low = 0.4
	start_thresh_high = 0.6
	start_thresh_step = 0.1

	continue_thresh_low = 0.1
	continue_thresh_high = 0.3
	continue_thresh_step = 0.1

	for i in images:
		print("<table>")
		for sigma in frange(sigma_low,
			sigma_high,
			sigma_step):
			for start_thresh in frange(
				start_thresh_low,
				start_thresh_high,
				start_thresh_step):
				for continue_thresh in frange(
					continue_thresh_low,
					continue_thresh_high,
					continue_thresh_step):
					Debug.Print("sigma, start, continue: %f %f %f" % (sigma, start_thresh, continue_thresh))
					image = Image.ImageFromFile(i + ".jpg")
					try:
						os.mkdir("./results/" + i + "/")
					except:
						pass
					save = "./results/" + i + "/" +\
						str(sigma) + "-" +\
						str(start_thresh) + "-" +\
						str(continue_thresh)

					edges = image.canny(sigma,
						start_thresh,
						continue_thresh,
						save=save+"-grad")
					edges.store_image(save+".jpg")
					print("<tr>")
					print("<td colspan=3>")
					print("Sigma: " + str(sigma) + "&nbsp;")
					print("Start: " + str(start_thresh) + "&nbsp;")
					print("Continue: " + str(continue_thresh) + "&nbsp;")
					print("</td>")
					print("</tr>")

					print("<tr>")
					print("<td>X Derivative</td><td>Y Derivative</td><td>Final Result</td>")
					print("</tr>")
					print("<tr>")
					print("<td><img src=" + save + "-grad-x.jpg height=" + str(image.height()/2) +  " width=" + str(image.width()/2) + "></td>")
					print("<td><img src=" + save + "-grad-y.jpg height=" + str(image.height()/2) +  " width=" + str(image.width()/2) + "></td>")
					print("<td><img src=" + save + ".jpg height=" + str(image.height()/2) +  " width=" + str(image.width()/2) + "></td>")
					print("</tr>")
		print("</table>")

def test_sift(images):
	for i in images:
		print("<table>")
		image = Image.ImageFromFile(i + ".jpg")
		try:
			os.mkdir("./results-sift/" + i + "/")
		except:
			pass
		save = "./results-sift/" + i + "/"

		image = image.intensify()
		(filtered_keypoints, all_keypoints) = image.sift()
		filtered_keypoints.store_image(save+"filtered.jpg")
		all_keypoints.store_image(save+"all.jpg")

		print("<tr>")
		print("<td>Original</td><td>All</td><td>Filtered</td>")
		print("</tr>")
		print("<tr>")
		print("<td><img src=" + i + ".jpg height=" + str(image.height()/2) +  " width=" + str(image.width()/2) + "></td>")
		print("<td><img src=" + save + "all.jpg height=" + str(image.height()/2) +  " width=" + str(image.width()/2) + "></td>")
		print("<td><img src=" + save + "filtered.jpg height=" + str(image.height()/2) +  " width=" + str(image.width()/2) + "></td>")
		print("</tr>")
		print("</table>")

if __name__ == "__main__":
	try:
		os.mkdir("./results-sift/")
		os.mkdir("./results-corners/")
		os.mkdir("./results/")
	except:
		pass
	#test_canny(["circle", "beams", "building", "flower", "mandrill"])
	test_canny(["circle"])
	#test_harris(["beams"])
	#test_sift(["beams"])
