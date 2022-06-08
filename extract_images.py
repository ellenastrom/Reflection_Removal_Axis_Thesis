import os
import sys
from PIL import Image

# script to extract image from folder based on image file name (can be used to extract filter results from networks). Inputs are image folder path, path to folder where to save, and search word to determine which file names to save  


#check if file is image
def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


#extract image based on if it contains word "search"
def extract_images(folder_path, save_path, search):
    for filename in os.listdir(folder_path):
        if is_image_file(filename):
            if filename.find(search) != -1:
                img = Image.open(os.path.join(folder_path, filename))
                img.save(os.path.join(save_path, filename))


if __name__ == "__main__":
    extract_images(str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]))
