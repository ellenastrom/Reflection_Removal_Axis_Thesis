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

def syn_data(t,r,sigma):
    t=np.power(t,2.2)
    r=np.power(r,2.2)
    
    sz=int(2*np.ceil(2*sigma)+1)
    r_blur=cv2.GaussianBlur(r,(sz,sz),sigma,sigma,0)
    blend=r_blur+t
    
    att=1.08+np.random.random()/10.0
    
    for i in range(3):
        maski=blend[:,:,i]>1
        mean_i=max(1.,np.sum(blend[:,:,i]*maski)/(maski.sum()+1e-6))
        r_blur[:,:,i]=r_blur[:,:,i]-(mean_i-1)*att
    r_blur[r_blur>=1]=1
    r_blur[r_blur<=0]=0

    h,w=r_blur.shape[0:2]
    alpha1=g_mask
    alpha2 = 1-np.random.random()/5.0;
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
    image1=[]
    image2=[]
    for dirname in train_path:
        train_t_gt = dirname + "transmission_layer/"
        train_r_gt = dirname + "reflection_layer/"
        train_b = dirname + "blended/"
        for root, _, fnames in sorted(os.walk(train_t_gt)):
            for fname in fnames:
                if is_image_file(fname):
                    path_input = os.path.join(train_b, fname)
                    path_output = os.path.join(train_t_gt, fname)
                    input_names.append(path_input)
                    image1.append(path_output)
        for root, _, fnames in sorted(os.walk(train_r_gt)):
            for fname in fnames:
                if is_image_file(fname):
                    path_output = os.path.join(train_r_gt, fname)
                    image2.append(path_output)
    return input_names,image1,image2

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

train_syn_root = ['images/synthetic_dataset/']
names,syn_image1_list,syn_image2_list=prepare_data(train_syn_root) # image pairs for generating synthetic training images

k_sz=np.linspace(1,5,80) # for synthetic images

for id, t_layer in enumerate(syn_image1_list):
    r_id = np.random.randint(0, len(syn_image2_list))

    syn_image1=cv2.imread(syn_image1_list[id],-1)
    #w, h = syn_image1.shape[:2]
    if 1080 != syn_image1.shape[0] and 1920 != syn_image1.shape[1]:
        continue
    syn_image2=cv2.imread(syn_image2_list[r_id],-1)
    syn_image1 = cv2.cvtColor(syn_image1, cv2.COLOR_BGR2RGB)
    syn_image2 = cv2.cvtColor(syn_image2, cv2.COLOR_BGR2RGB)
    w=1920
    h=1080

    # create a vignetting mask
    g_mask=gkern(w,h,np.random.randint(1, 4))#3)
    g_mask=np.dstack((g_mask,g_mask,g_mask))

    output_image_t=cv2.resize(np.float32(syn_image1),(w,h),cv2.INTER_CUBIC)/255.0
    output_image_r=cv2.resize(np.float32(syn_image2),(w,h),cv2.INTER_CUBIC)/255.0
    sigma=k_sz[np.random.randint(0, len(k_sz))]
    output_image_t,output_image_r,input_image=syn_data(output_image_t,output_image_r,sigma)

    directories = ["images/synthetic_dataset/generated/reflection_org/", "images/synthetic_dataset/generated/blended/",
    "images/synthetic_dataset/generated/reflection/", "images/synthetic_dataset/generated/transmission/"]

    directories_test = ["images/synthetic_dataset/generated/reflection_org/test/", "images/synthetic_dataset/generated/blended/test/",
    "images/synthetic_dataset/generated/reflection/test/", "images/synthetic_dataset/generated/transmission/test/"]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    for directory in directories_test:
        if not os.path.exists(directory):
            os.makedirs(directory)

    im_0 = Image.fromarray((syn_image2).astype(np.uint8))
    im_1  = Image.fromarray((input_image * 255).astype(np.uint8))
    im_2 = Image.fromarray((output_image_r * 255).astype(np.uint8))
    im_3 = Image.fromarray((output_image_t * 255).astype(np.uint8))

    file=os.path.splitext(os.path.basename(syn_image1_list[id]))[0]

    if id%100 == 0:
        im_0.save(directories_test[0] + file + ".jpeg")
        im_1.save(directories_test[1] + file + ".jpeg")
        im_2.save(directories_test[2] + file + ".jpeg")
        im_3.save(directories_test[3] + file + ".jpeg")
    else:
        im_0.save(directories[0] + file + ".jpeg")
        im_1.save(directories[1] + file + ".jpeg")
        im_2.save(directories[2] + file + ".jpeg")
        im_3.save(directories[3] + file + ".jpeg")
