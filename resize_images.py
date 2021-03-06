from PIL import Image
import os, sys

<<<<<<< HEAD

inputpath = "/home/ellena/Documents/test_real_final/test_real_reflections"
outputpath = "/home/ellena/Documents/test_real_final/test_real_reflections_256x256"
=======
inputpath = "/data/generated_final/train/blended/"
outputpath = "/data/generated_final/train/downsampled"
>>>>>>> Tove

# script to resize images in input path and save in output path

inputpath = "generated_1/transmission/"
outputpath = inputpath + 'resized/'


if not os.path.exists(outputpath):
    os.makedirs(outputpath)

def resize():
    for item in os.listdir(inputpath):
    
        if os.path.isfile(os.path.join(inputpath,item)) and (item.endswith('.jpg') or item.endswith('.jpeg')):
            im = Image.open(os.path.join(inputpath,item))
            w, h = im.size
            imResize = im.resize((w*3/4,h*3/4), Image.ANTIALIAS)
            file=os.path.splitext(os.path.basename(item))[0]
            imResize.save(os.path.join(outputpath, file), 'JPEG', quality=90)
            

resize()
