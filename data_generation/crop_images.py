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


def random_crop_images(reflection, transmission):
    # Opens a image in RGB mode![]()
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("output2", cv2.WINDOW_NORMAL)
    # im_copy_ref = reflection
    # im_copy_trans = transmission

    frame = 80
    rand_x_crop = random.randint(0, 50) * 4
    rand_y_crop_top = random.randint(0, 50) * 4
    rand_y_crop_bottom = random.randint(0, 50) * 4

    reflection_crop = reflection[frame + rand_y_crop_bottom:1080 - frame - rand_y_crop_top,
                      frame + rand_x_crop:1920 - frame - rand_x_crop]
    h, w, c = reflection_crop.shape
    transmission_crop = get_random_crop(transmission, h, w)

    # cv2.imshow("output", transmission_crop)
    # cv2.imshow("output2", reflection_crop)
    # print(transmission_crop.shape)
    # print(reflection_crop.shape)
    return reflection_crop, transmission_crop
