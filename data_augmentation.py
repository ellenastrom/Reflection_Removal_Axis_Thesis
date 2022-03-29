#import argparse
from PIL import Image
import os

#parser = argparse.ArgumentParser()

#-db DATABSE
#parser.add_argument("--input_path", type=str, default='images', help="Path to images to augment")
#parser.add_argument("--output_path", type=str, default=input_path, help="Path where to save images that has been augmented")

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

input_path='images/synthetic_dataset/reflection_layer/first_session/'
output_path='images/synthetic_dataset/reflection_layer/'

# iterate through the names of contents of the folder
for filename in os.listdir(input_path):
	image_path = os.path.join(input_path, filename)
	print(image_path)
	if is_image_file(filename):
		can_load = True
		try: 
			original_img = Image.open(image_path)
		except: 
			print('Was not able to load ', image_path)
			can_load = False
			continue
		if can_load and original_img.size == 1920, 1080:
			vertical_img = original_img.transpose(method=Image.FLIP_TOP_BOTTOM)
			horz_img = original_img.transpose(method=Image.FLIP_LEFT_RIGHT)
			both_img = vertical_img.transpose(method=Image.FLIP_LEFT_RIGHT)

			vertical_img.save(output_path + os.path.splitext(filename)[0] + "_vertical.jpg")
			horz_img.save(output_path + os.path.splitext(filename)[0] + "_horizontal.jpg")
			both_img.save(output_path + os.path.splitext(filename)[0] + "_both.jpg")

			# close all our files object
			original_img.close()
			vertical_img.close()
			horz_img.close()
			both_img.close()