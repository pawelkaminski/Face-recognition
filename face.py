#! /usr/bin/env python3

import argparse
from PIL import Image, ImageFilter, ImageDraw, ImageColor
from matplotlib import colors
from collections import deque
import numpy as np
import math
import json

#max values
HUE_DEPTH = 360
SATURATION_DEPTH = 100

BFS_VISITED = 0
BFS_UNVISITED = 255

EPS = 100
SCALE = 15

def neighbourhood(x, y, width, height):
	if x + 1 < width:
		yield (x + 1, y)
	if x - 1 > 0:
		yield (x - 1, y)
	if y + 1 < height:
		yield (x, y + 1)
	if y - 1 > 0:
		yield (x, y - 1)


def rgb_to_hsv(input_img):
	rgb_img = input_img.convert(mode='RGB')
	width, height = rgb_img.size
	rgb_array = np.array(rgb_img.getdata()).reshape(height, width, 3) / 255
	return colors.rgb_to_hsv(rgb_array)


def bin_coordinates(h, s):
	h = math.floor(h * (HUE_DEPTH - 1))
	s = math.floor(s * (SATURATION_DEPTH - 1))
	return h, s


def sobel(img):
	"""
	Sobel operator implementation
	"""
	img = img.convert(mode='L')
	width, height = img.size
	filter_x = ImageFilter.Kernel(
		size=(3, 3),
		kernel=(1, 0, -1, 2, 0, -2, 1, 0, -1),
		scale=1,
		offset=128
	)
	filter_y = ImageFilter.Kernel(
		size=(3, 3),
		kernel=(1, 2, 1, 0, 0, 0, -1, -2, -1),
		scale=1,
		offset=128
	)
	image_x = img.filter(filter_x)
	image_y = img.filter(filter_y)
	for x in range(0, width):
		for y in range(0, height):
			img.putpixel(
				(x, y),
				math.floor((
					(128 - image_x.getpixel((x, y))) ** 2 +
					(128 - image_y.getpixel((x, y))) ** 2
				) ** 0.5) + 128
			)
	return img


class SkinHistogram:
	def __init__(self):
		self.histogram = np.zeros((HUE_DEPTH, SATURATION_DEPTH))

	def process(self, input_img):
		hsv_array = rgb_to_hsv(input_img)
		width, height, _ = hsv_array.shape
		for x in range(0, width):
			for y in range(0, height):
				h, s, _ = hsv_array[x, y]
				h, s = bin_coordinates(h, s)
				if h != 0:
					self.histogram[h, s] += 1
		del hsv_array

	def normalized(self):
		return self.histogram / np.amax(self.histogram)

	def open(self, input_path):
		f = open(input_path, 'r')
		self.histogram = np.array(json.loads(f.read()))

	def save(self, output_path):
		f = open(output_path, 'w')
		f.write(json.dumps(self.normalized().tolist()))


class SkinFinder:
	def __find_skin(self, histogram, input_img):
		input_img = input_img.filter(ImageFilter.GaussianBlur(5))
		mask_image = sobel(input_img)
		hsv_array = rgb_to_hsv(input_img)
		width, height = input_img.size
		for x in range(0, width):
			for y in range(0, height):
				h, s, v = hsv_array[y, x]
				h, s = bin_coordinates(h, s)
				#Magic values from paper
				if (
					histogram[h, s] < 0.05 or
					not (0.3 < v < 0.97) or
					not (128 - EPS < mask_image.getpixel((x, y)) < 128 + EPS)
				):
					color = BFS_VISITED
				else:
					color = BFS_UNVISITED
				mask_image.putpixel((x, y), color)
		return mask_image

	def find_face(self, histogram, input_img):
		skin = self.__find_skin(histogram, input_img)
		width, height = input_img.size
		rects = []

		# BFS implementation
		def spread(sx, sy):
			fmin_x, fmin_y, fmax_x, fmax_y = width, height, -1, -1
			q = deque()
			skin.putpixel((sx, sy), BFS_VISITED)
			q.append((sx, sy))
			while q:
				(qx, qy) = q.popleft()
				fmax_x = max(fmax_x, qx)
				fmin_x = min(fmin_x, qx)
				fmin_y = min(fmin_y, qy)
				fmax_y = max(fmax_y, qy)
				for (nx, ny) in neighbourhood(qx, qy, width, height):
					# If pixel is white then pixel := BFS_UNVISITED.
					if skin.getpixel((nx, ny)) == BFS_UNVISITED:
						skin.putpixel((nx, ny), BFS_VISITED)
						q.append((nx, ny))
			return fmin_x, fmin_y, fmax_x, fmax_y

		# BFS
		for x in range(0, width):
			for y in range(0, height):
				if skin.getpixel((x, y)) == 255:
					min_x, min_y, max_x, max_y = spread(x, y)
					#checks if has face shape
					if (
						(max_y - min_y) / (max_x - min_x + 1) < 2 and
						(max_x - min_x) / (max_y - min_y + 1) < 2 and
						(max_x - min_x) > width / SCALE and
						(max_y - min_y) > height / SCALE
					):
						rects.append((min_x, min_y, max_x, max_y))
		return rects

	def draw_rects(self, input_img, rects):
		rgb_img = input_img.convert(mode='RGB')
		draw = ImageDraw.Draw(rgb_img)
		for (min_x, min_y, max_x, max_y) in rects:
			draw.rectangle([(min_x, min_y), (max_x, max_y)], None, (255,0, 0))
		return rgb_img


def generate_histogram(input_img_path, output_hist_path):
	try:
		input_img = Image.open(input_img_path)
	except IOError:
		print("Error:", input_img_path)
	histogram = SkinHistogram()
	histogram.process(input_img)
	try:
		histogram.save(output_hist_path)
	except IOError:
		print("Error:", output_hist_path)



def save_face(input_hist_path, input_img_path, output_img_path):
	histogram = SkinHistogram()
	try:
		histogram.open(input_hist_path)
	except IOError:
		print("Error:", input_hist_path)
	try:
		input_img = Image.open(input_img_path)
	except IOError:
		print("Error:", input_img_path)
	finder = SkinFinder()
	rects = finder.find_face(histogram.normalized(), input_img)
	output_img = finder.draw_rects(input_img, rects)
	try:
		output_img.save(output_img_path)
	except IOError:
		print("Error:", output_img_path)



parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-g', '--generate', action='store', nargs=2, dest='generate',
				   metavar=("Image", "Histogram"))
group.add_argument('-s', '--save', action='store', nargs=3, dest='save',
				   metavar=("Histogram", "Input", "Output"))
args = parser.parse_args()

if args.generate is not None:
	generate_histogram(args.generate[0], args.generate[1])
elif args.save is not None:
	save_face(args.save[0], args.save[1], args.save[2])

