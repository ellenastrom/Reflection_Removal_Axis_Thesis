from PIL import Image
import os, sys

inputpath = "generated_1/transmission/"
outputpath = inputpath + 'resized/'
dirs = os.listdir(inputpath)

def resize():
    for item in dirs:
        if os.path.isfile(inputpath+item) and (item.endswith('.jpg') or item.endswith('.jpeg')):
            im = Image.open(inputpath+item)
            imResize = im.resize((960,540), Image.ANTIALIAS)
            file=os.path.splitext(os.path.basename(item))[0]
            imResize.save(outputpath + item + '_resized.jpg', 'JPEG', quality=90)

resize()
