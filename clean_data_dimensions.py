import os
import sys

from skimage import io


def clean(path):
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            if io.imread(os.path.join(path,filename)).shape != (1080, 1920, 3):
                os.remove(os.path.join(path, filename))

if __name__ == "__main__":
    clean(sys.argv[1])