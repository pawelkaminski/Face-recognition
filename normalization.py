#! /usr/bin/env python3
# (c) 2015 Paweł Kamiński

from PIL import Image
import math
import numpy as np
import sys


DEPTH = 256


def create_histogram(img):
    """
    Gets PIL Image, returns tuple of three lists: (red, green, blue)
    """
    rgb = (np.zeros(DEPTH), np.zeros(DEPTH), np.zeros(DEPTH))
    width, height = img.size
    # Create a histogram
    for x in range(width):
        for y in range(height):
            pixel = img.getpixel((x, y))
            for i in range(3):
                rgb[i][pixel[i]] += 1
    return rgb


def auto_levels(input_img):
    rgb = create_histogram(input_img)
    # Create new image
    output_img = input_img.copy()
    width, height = output_img.size
    # Find peak values in histograms
    peaks = [0, 0, 0]
    for i in range(3):
        for j in range(DEPTH - 1):
            if rgb[i][j] > peaks[i]:
                peaks[i] = rgb[i][j]
    # Find where each channel starts on a histogram
    mins = [0, 0, 0]
    for i in range(3):
        for j in range(DEPTH - 1):
            if rgb[i][j] > peaks[i] * 0.02:
                mins[i] = j
                break
    # Find where each channel ends on a histogram
    maxs = [DEPTH - 1, DEPTH - 1, DEPTH - 1]
    for i in range(3):
        for j in range(DEPTH - 1, 0, -1):
            if rgb[i][j] > peaks[i] * 0.02:
                maxs[i] = j
                break
    # Expand histograms
    for x in range(width):
        for y in range(height):
            pixel = list(output_img.getpixel((x, y)))
            for i in range(3):
                if mins[i] <= maxs[i]:
                    pixel[i] = math.floor((pixel[i] - mins[i]) / (maxs[i] - mins[i]) * (DEPTH - 1) )
            output_img.putpixel((x, y), tuple(pixel))
    return output_img



def auto_levels_io(input_img_path, output_img_path):
    # Load input image
    try:
        input_img = Image.open(input_img_path)
    except IOError:
        print("Cannot load", input_img_path)
    input_img = input_img.convert(mode='RGB')
    # Auto histogram
    output_img = auto_levels(input_img)
    # Save output image
    try:
        output_img.save(output_img_path)
    except IOError:
        print("Cannot save", output_img_path)



if len(sys.argv) != 3:
	print('./normalization.py file.in file.out')
else:
	auto_levels_io(sys.argv[1], sys.argv[2])


