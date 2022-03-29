from PIL import Image
import os, sys

path = "images/chose_model_test"
#path = ('drive-download-20220309T100926Z-001')
dirs = os.listdir( path )

def resize():
    for item in dirs:
        if os.path.isfile(path+'/'+item) and (item.endswith('.jpg') or item.endswith('.jpeg')):
            im = Image.open(path+'/'+item)
            imResize = im.resize((960,540), Image.ANTIALIAS)
            imResize.save(path +'/'+ item + 'resized.jpg', 'JPEG', quality=90)

resize()
