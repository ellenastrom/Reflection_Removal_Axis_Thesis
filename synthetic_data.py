import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import scipy.stats as st
from PIL import Image

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def syn_data(t,r,sigma, threshold):
    t=np.power(t,2.2)
    r=np.power(r,2.2)
    
    sz=int(2*np.ceil(2*sigma)+1) # controls the size of kernel, the larger the blurrier. Must be an uneven int
    # sigma controls variance of of the Gaussian filter, the higher value, the blurrier
    r_blur=cv2.GaussianBlur(r,(sz,sz),sigma,sigma,0)
    blend=r_blur+t
    
    # higher value gives higher threshold for what is kept in the reflection image
    att= threshold + np.random.random()/10 #I like this value but original was #1.08+np.random.random()/10.0
    
    for i in range(3):
        maski=blend[:,:,i]>1
        mean_i=max(1.,np.sum(blend[:,:,i]*maski)/(maski.sum()+1e-6))
        r_blur[:,:,i]=r_blur[:,:,i]-(mean_i-1)*att
    r_blur[r_blur>=1]=1
    r_blur[r_blur<=0]=0

    h,w=r_blur.shape[0:2]
    alpha1=g_mask
    alpha2 = 1-np.random.random()/5.0; #the lower value the darker blended image. Not sure we should make any changes
    r_blur_mask=np.multiply(r_blur,alpha1)
    blend=r_blur_mask+t*alpha2
    
    t=np.power(t,1/2.2)
    r_blur_mask=np.power(r_blur_mask,1/2.2)
    blend=np.power(blend,1/2.2)
    blend[blend>=1]=1
    blend[blend<=0]=0

    return t,r_blur_mask,blend

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def prepare_data(train_path):
    input_names=[]
    transmission_images=[]
    reflection_images=[]
    train_t_gt = train_path + "transmission_layer/"
    train_r_gt = train_path + "reflection_layer/"
    for root, _, fnames in sorted(os.walk(train_t_gt)):
        for fname in fnames:
            if is_image_file(fname):
                path_output = os.path.join(train_t_gt, fname)
                transmission_images.append(path_output)
    for root, _, fnames in sorted(os.walk(train_r_gt)):
        for fname in fnames:
            if is_image_file(fname):
                path_output = os.path.join(train_r_gt, fname)
                reflection_images.append(path_output)
    return transmission_images, reflection_images

def gkern(width=100, height=100, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(width)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., width+1)
    kern1d_width = np.diff(st.norm.cdf(x))
    interval = (2*nsig+1.)/(height)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., height+1)
    kern1d_height = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d_height, kern1d_width))
    kernel = kernel_raw/kernel_raw.sum()
    kernel = kernel/kernel.max()
    return kernel

train_syn_in = 'images/synthetic_dataset/'
transmission_list,reflection_list=prepare_data(train_syn_in) # image pairs for generating synthetic training images

train_syn_out = train_syn_in + 'generated/train/'
test_syn_out = train_syn_in + 'generated/test/'

dir_train = [train_syn_out + "reflection_org/", train_syn_out + "blended/", train_syn_out + "reflection/", train_syn_out + "transmission/"]
dir_test = [test_syn_out + "reflection_org/", test_syn_out + "blended/", test_syn_out + "reflection/", test_syn_out + "transmission/"]
    
for directory in dir_train:
    if not os.path.exists(directory):
        os.makedirs(directory)
for directory in dir_test:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(directory)

k_sz=np.linspace(0.2, 4, 80) #1,5,80) # for synthetic images
w=1920
h=1080

for id, transmission_name in enumerate(transmission_list):

    # create a vignetting mask
    g_mask=gkern(w,h,np.random.randint(1, 3)) # 3))
    g_mask=np.dstack((g_mask,g_mask,g_mask))

    r_id = np.random.randint(0, len(reflection_list))
    r_name=os.path.splitext(os.path.basename(reflection_list[r_id]))[0]

    if r_name.startswith('first'):
        threshold = 0.55
    elif r_name.startswith('P5655'):
        threshold = 0.05
    elif r_name.startswith('Q6135'):
        threshold = 0.25
    elif r_name.startswith('Q6315'):
        threshold = 0.35
    elif r_name.startswith('victor'):
        threshold = 0.3

    t_image=cv2.imread(transmission_name, -1)
    if t_image.shape[0] != h and t_image.shape[1] != w:
        continue
    r_image = cv2.imread(reflection_list[r_id],-1)
    t_image = cv2.cvtColor(t_image, cv2.COLOR_BGR2RGB)
    r_image = cv2.cvtColor(r_image, cv2.COLOR_BGR2RGB)

    t_image_out=cv2.resize(np.float32(t_image),(w,h),cv2.INTER_CUBIC)/255.0
    r_image_out=cv2.resize(np.float32(r_image),(w,h),cv2.INTER_CUBIC)/255.0
    sigma=k_sz[np.random.randint(0, len(k_sz))]
    t_image_out,r_image_out,b_image=syn_data(t_image_out,r_image_out,sigma, threshold)

    r_image = Image.fromarray((r_image).astype(np.uint8))
    b_image = Image.fromarray((b_image * 255).astype(np.uint8))
    r_image_out = Image.fromarray((r_image_out * 255).astype(np.uint8))
    t_image_out = Image.fromarray((t_image_out * 255).astype(np.uint8))

    file=os.path.splitext(os.path.basename(transmission_name))[0]
    file = r_name

    if id%5 == 0:
        r_image.save(dir_test[0] + file + ".jpg")
        b_image.save(dir_test[1] + file + ".jpg")
        r_image_out.save(dir_test[2] + file + ".jpg")
        t_image_out.save(dir_test[3] + file + ".jpg")        
        print(dir_test[2] + file + ".jpg")
    else:
        r_image.save(dir_train[0] + file + ".jpg")
        b_image.save(dir_train[1] + file + ".jpg")
        r_image_out.save(dir_train[2] + file + ".jpg")
        t_image_out.save(dir_train[3] + file + ".jpg")
