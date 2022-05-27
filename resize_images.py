from PIL import Image
import os, sys

inputpath = "/data/generated_final/train/blended/"
outputpath = "/data/generated_final/train/downsampled"

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
