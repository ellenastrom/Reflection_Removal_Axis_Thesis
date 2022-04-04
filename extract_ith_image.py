import os
import sys
from PIL import Image


def is_image_file(filename):
    # check if the image ends with png
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def extract_trans_IBCLN(folder_path, save_path, search):
    for filename in os.listdir(folder_path):
        if is_image_file(filename):
            if filename.find(search) != -1:
                img = Image.open(os.path.join(folder_path, filename))
                img.save(os.path.join(save_path, filename))


if __name__ == "__main__":
    extract_trans_IBCLN(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]))
