from PIL import Image
import os, sys

inputpath = "/home/ellena/Documents/test_real_final/test_real_reflections"
outputpath = "/home/ellena/Documents/test_real_final/test_real_reflections_256x256"

if not os.path.exists(outputpath):
    os.makedirs(outputpath)

def resize():
    for item in os.listdir(inputpath):
    
        if os.path.isfile(os.path.join(inputpath,item)) and (item.endswith('.jpg') or item.endswith('.jpeg')):
            im = Image.open(os.path.join(inputpath,item))
            imResize = im.resize((256,256), Image.ANTIALIAS)
            file=os.path.splitext(os.path.basename(item))[0]
            imResize.save(os.path.join(outputpath, file), 'JPEG', quality=90)
            

resize()
