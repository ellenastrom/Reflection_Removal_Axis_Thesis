# import argparse
import random

import cv2
import numpy as np
from PIL import Image
import os

def augment_image(image, is_reflection):
    if is_reflection:
        rand_flip = random.randint(0, 3)
        if rand_flip == 0:
            image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
        elif rand_flip == 1:
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
        else:
            h_flip_img = image.transpose(method=Image.FLIP_TOP_BOTTOM)
            image = h_flip_img.transpose(method=Image.FLIP_LEFT_RIGHT)
    else:
        rand_flip = random.randint(0, 2)
        if rand_flip == 0:
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
    return np.asarray(image)
