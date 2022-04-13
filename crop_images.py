import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import scipy.stats as st
from PIL import Image
import random
import sys


def get_random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop


def random_crop_images(reflection, transmission, contains_waterstamp):

    frame = 80
    if contains_waterstamp:
        rand_y_crop_top = frame + random.randint(0, 10) * 4
        rand_y_crop_bottom = frame + random.randint(0, 10) * 4
        frame = frame*2
    else:
        rand_y_crop_top = random.randint(1, 50) * 4
        rand_y_crop_bottom = random.randint(1, 50) * 4
    rand_x_crop = frame + random.randint(0, 50) * 4
    
    reflection_crop = reflection[rand_y_crop_bottom:1080 - rand_y_crop_top,
                      rand_x_crop:1920 - rand_x_crop]
    h, w, c = reflection_crop.shape
    transmission_crop = get_random_crop(transmission, h, w)

    return reflection_crop, transmission_crop
