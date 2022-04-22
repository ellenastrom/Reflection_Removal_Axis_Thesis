import argparse
import numpy as np
import cv2
import os
import scipy.stats as st
from PIL import Image
import random

parser = argparse.ArgumentParser('Synthetic Data')
parser.add_argument('--path', default=None, help="path to folder with background and reflection layer folders")
parser.add_argument('--nbr_train_images', type=int, default=4000, help="nbr of train images")
parser.add_argument('--is_test', default=False, help="is True if test set is supposed to be created otherwise False")
parser.add_argument('--test_ratio', type=float, default=0.1, help="ratio of total images being test images")


def get_random_crop(image, crop_height, crop_width):
    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop = image[y: y + crop_height, x: x + crop_width]

    return crop


def random_crop_images(reflection, transmission, contains_waterstamp):
    frame = 80
    if contains_waterstamp:
        rand_y_crop_top = frame + random.randint(0, 10) * 4
        rand_y_crop_bottom = frame + random.randint(0, 10) * 4
        frame = frame * 2
    else:
        rand_y_crop_top = random.randint(1, 50) * 4
        rand_y_crop_bottom = random.randint(1, 50) * 4
    rand_x_crop = frame + random.randint(0, 50) * 4

    reflection_crop = reflection[rand_y_crop_bottom:1080 - rand_y_crop_top,
                      rand_x_crop:1920 - rand_x_crop]
    h, w, c = reflection_crop.shape
    transmission_crop = get_random_crop(transmission, h, w)

    return reflection_crop, transmission_crop


def augment_image(image, is_reflection):
    if is_reflection:
        rand_flip = random.randint(0, 3)
        if rand_flip == 0:
            image = image.transpose(method=Image.FLIP_TOP_BOTTOM)
        elif rand_flip == 1:
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
        else:
            h_flip_img = image.transpose(method=Image.FLIP_TOP_BOTTOM)
            image = h_flip_img.transpose(method=Image.FLIP_LEFT_RIGHT)
    else:
        rand_flip = random.randint(0, 2)
        if rand_flip == 0:
            image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
    return np.asarray(image)


def gkern(width=100, height=100, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    interval = (2 * nsig + 1.) / (width)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., width + 1)
    kern1d_width = np.diff(st.norm.cdf(x))
    interval = (2 * nsig + 1.) / (height)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., height + 1)
    kern1d_height = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d_height, kern1d_width))
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel / kernel.max()
    return kernel


def syn_data(t, r, sigma, threshold, g_mask):
    t = np.power(t, 2.2)
    r = np.power(r, 2.2)

    sz = int(2 * np.ceil(2 * sigma) + 1)  # controls the size of kernel, the larger the blurrier. Must be an uneven int
    # sigma controls variance of of the Gaussian filter, the higher value, the blurrier
    r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
    blend = r_blur + t

    # higher value gives higher threshold for what is kept in the reflection image
    att = threshold + np.random.random() / 10  # I like this value but original was #1.08+np.random.random()/10.0

    for i in range(3):
        maski = blend[:, :, i] > 1
        mean_i = max(1., np.sum(blend[:, :, i] * maski) / (maski.sum() + 1e-6))
        r_blur[:, :, i] = r_blur[:, :, i] - (mean_i - 1) * att
    r_blur[r_blur >= 1] = 1
    r_blur[r_blur <= 0] = 0

    h, w = r_blur.shape[0:2]
    alpha1 = g_mask
    alpha2 = 1 - np.random.random() / 5.0;  # the lower value the darker blended image. Not sure we should make any changes
    r_blur_mask = np.multiply(r_blur, alpha1)
    blend = r_blur_mask + t * alpha2

    t = np.power(t, 1 / 2.2)
    r_blur_mask = np.power(r_blur_mask, 1 / 2.2)
    blend = np.power(blend, 1 / 2.2)
    blend[blend >= 1] = 1
    blend[blend <= 0] = 0

    return t, r_blur_mask, blend


class SyntheticDataGenerator:
    def __init__(self, args):
        # ARGUMENTS
        self.args = args
        self.path = args.path
        self.nbr_train_images = args.nbr_train_images
        self.is_test = args.is_test
        self.test_ratio = args.test_ratio

        self.IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        ]

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def prepare_data(self, train_path):
        input_names = []
        transmission_images = []
        reflection_images = []
        train_t_gt = os.path.join(train_path, 'transmission_layer')
        train_r_gt = os.path.join(train_path, 'reflection_layer')
        for root, _, fnames in sorted(os.walk(train_t_gt)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path_output = os.path.join(train_t_gt, fname)
                    transmission_images.append(path_output)
        for root, _, fnames in sorted(os.walk(train_r_gt)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path_output = os.path.join(train_r_gt, fname)
                    reflection_images.append(path_output)
        return transmission_images, reflection_images

    def create_data(self):
        train_syn_in = self.path

        transmission_list, reflection_list = self.prepare_data(
            train_syn_in)  # image pairs for generating synthetic training images

        train_syn_out = os.path.join(train_syn_in, 'generated/train')
        test_syn_out = os.path.join(train_syn_in, 'generated/test')

        dir_train = [os.path.join(train_syn_out, 'reflection_org'), os.path.join(train_syn_out, 'blended'),
                     os.path.join(train_syn_out, 'reflection'), os.path.join(train_syn_out, 'transmission')]
        dir_test = [os.path.join(test_syn_out, 'reflection_org'), os.path.join(test_syn_out, 'blended'),
                    os.path.join(test_syn_out, 'reflection'), os.path.join(test_syn_out, 'transmission')]

        for directory in dir_train:
            if not os.path.exists(directory):
                os.makedirs(directory)
        if self.is_test:
            for directory in dir_test:
                if not os.path.exists(directory):
                    os.makedirs(directory)
                    print(directory)

        if self.is_test:
            nbr_images = int(
                np.ceil(self.test_ratio / (1 - self.test_ratio) * self.nbr_train_images + self.nbr_train_images))
        else:
            nbr_images = self.nbr_train_images

        for i in range(0, nbr_images):

            r_id = np.random.randint(0, len(reflection_list))
            t_id = np.random.randint(0, len(transmission_list))

            t_image = Image.open(transmission_list[t_id])
            r_image = Image.open(reflection_list[r_id])

            r_name = os.path.splitext(os.path.basename(reflection_list[r_id]))[0]
            t_name = os.path.splitext(os.path.basename(transmission_list[t_id]))[0]

            t_image = augment_image(t_image, False)
            r_image = augment_image(r_image, True)

            r_image, t_image = random_crop_images(r_image, t_image, 'waterstamp' in r_name)

            w = r_image.shape[1]
            h = r_image.shape[0]

            t_image_out = cv2.resize(np.float32(t_image), (w, h), cv2.INTER_CUBIC) / 255.0
            r_image_out = cv2.resize(np.float32(r_image), (w, h), cv2.INTER_CUBIC) / 255.0

            if r_name.startswith('Q6315_1st'):
                threshold = 0.55
            elif r_name.startswith('Q6315_2nd'):
                threshold = 0.37
            elif r_name.startswith('Q6315_3rd'):
                threshold = 0.30
            elif r_name.startswith('P5655'):
                threshold = 0.25
            elif r_name.startswith('Q6135_1st'):
                threshold = 0.25
            elif r_name.startswith('Q6135_2nd'):
                threshold = 0.22
            elif r_name.startswith('Q6075_d'):
                threshold = 0.30
            elif r_name.startswith('Q6075_l'):
                threshold = 0.40

            k_sz = np.linspace(0.2, 4, 80)  # 1,5,80) # for synthetic images
            sigma = k_sz[np.random.randint(0, len(k_sz))]

            # create a vignetting mask
            g_mask = gkern(w, h, np.random.randint(1, 3))  # 3))
            g_mask = np.dstack((g_mask, g_mask, g_mask))

            t_image_out, r_image_out, b_image = syn_data(t_image_out, r_image_out, sigma, threshold, g_mask)

            r_image = Image.fromarray(r_image.astype(np.uint8))
            b_image = Image.fromarray((b_image * 255).astype(np.uint8))
            r_image_out = Image.fromarray((r_image_out * 255).astype(np.uint8))
            t_image_out = Image.fromarray((t_image_out * 255).astype(np.uint8))

            file = r_name + '_' + str(i) + '.jpg'

            if self.is_test:
                if i % 10 == 0:
                    r_image.save(os.path.join(dir_test[0], file))
                    b_image.save(os.path.join(dir_test[1], file))
                    r_image_out.save(os.path.join(dir_test[2], file))
                    t_image_out.save(os.path.join(dir_test[3], file))
                else:
                    r_image.save(os.path.join(dir_train[0], file))
                    b_image.save(os.path.join(dir_train[1], file))
                    r_image_out.save(os.path.join(dir_train[2], file))
                    t_image_out.save(os.path.join(dir_train[3], file))
            else:
                r_image.save(os.path.join(dir_train[0], file))
                b_image.save(os.path.join(dir_train[1], file))
                r_image_out.save(os.path.join(dir_train[2], file))
                t_image_out.save(os.path.join(dir_train[3], file))


def main():
    args = parser.parse_args()
    generator = SyntheticDataGenerator(args)
    generator.create_data()


if __name__ == "__main__":
    main()
