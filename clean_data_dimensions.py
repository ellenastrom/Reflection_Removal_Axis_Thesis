import os
import sys
from skimage import io

# Script to delete images of dimension other than 1080x1920

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


# function to check if is image file
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def clean(path):
    # loop through all images in folder
    for filename in os.listdir(path):
        # delete if is image file and wrong dimension
        if is_image_file(filename):
            if io.imread(os.path.join(path, filename)).shape != (1080, 1920, 3):
                os.remove(os.path.join(path, filename))


if __name__ == "__main__":
    clean(sys.argv[1])
