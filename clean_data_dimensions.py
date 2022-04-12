import os
import sys

from skimage import io

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def clean(path):
    for filename in os.listdir(path):
        if is_image_file(filename):
            if io.imread(os.path.join(path,filename)).shape != (1080, 1920, 3):
                os.remove(os.path.join(path, filename))

if __name__ == "__main__":
    clean(sys.argv[1])