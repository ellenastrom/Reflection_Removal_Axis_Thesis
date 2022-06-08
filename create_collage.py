import argparse
import os.path
import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm

# Script to create collage of images and their filtered result for the different networks.
# The input folder should contain the original data folder (folder name must contain 'real' or 'original'), and
# one folder for each network (DADnet, ERRNet, IBCLN and RAGNet) with filter results

parser = argparse.ArgumentParser('Collage')
parser.add_argument('--test_result_dir', default="./results_real",
                    help="path to result folder containing real data results")


# function to check if file is image
def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# Add title to image, either original or filtered result
def add_title(image_path, text):
    original = PIL.Image.open(image_path)
    rows, cols = original.size
    font_size = int(rows / 20)

    font = ImageFont.truetype('Ubuntu-R.ttf', font_size)
    draw = PIL.ImageDraw.Draw(original)
    if 'real' or 'original' in text.lower():
        draw.text((0, 0), 'Unfiltered image', font=font, fill='yellow')
    else:
        draw.text((0, 0), 'filtered by: ' + text, font=font, fill='yellow')
    return original


# Create collage of original image and filter results
def create_collage(width, height, listofimages, collage_nbr, save_path):
    cols = 2
    rows = 4
    thumbnail_width = width // cols
    thumbnail_height = height // rows
    spacing = 20
    size = thumbnail_width - spacing, thumbnail_height - spacing
    new_im = Image.new('RGB', (width, height))
    ims = []
    for im in listofimages:
        im.thumbnail(size)
        ims.append(im)
    i = 0
    x = 0
    y = 0
    for col in range(cols):
        for row in range(rows):
            new_im.paste(ims[i], (x + int(spacing / 2), y))
            i += 1
            y += thumbnail_height
        x += thumbnail_width
        y = 0
    nbr = '{0:03}'.format(collage_nbr)
    if not os.path.isdir(os.path.join(save_path, 'Collages')):
        os.makedirs(os.path.join(save_path, 'Collages'))
    new_im.save(os.path.join(save_path, "Collages/{}_Collage.jpg".format(nbr)))


class Collage:
    def __init__(self, args):
        # ARGUMENTS
        self.args = args
        self.dir = args.test_result_dir

    # function to create one collage for each image and filtered results
    def create_collages(self):
        path = self.dir
        sub_dir_list = []
        # loop through all folders in input folder
        for sub_dir in sorted(os.listdir(path)):
            # if folder name contains any network name, add to list of sub dirs
            if 'dad' or 'err' or 'ibcln' or 'rag' in sub_dir.lower():
                sub_dir_list.append(sub_dir)
        collage_nbr = 0  # counting collages
        # loop through images in all 5 folders at the same time
        for im_path_dad, im_path_err, im_path_ibcln, im_path_rag, im_path_org in tqdm(
                zip(sorted(os.listdir(os.path.join(path, sub_dir_list[0])), key=str.casefold),
                    sorted(os.listdir(os.path.join(path, sub_dir_list[1])), key=str.casefold),
                    sorted(os.listdir(os.path.join(path, sub_dir_list[2])), key=str.casefold),
                    sorted(os.listdir(os.path.join(path, sub_dir_list[3])), key=str.casefold),
                    sorted(os.listdir(os.path.join(path, sub_dir_list[4])), key=str.casefold))):
            list_of_images = []
            # check if all files are images
            if is_image_file(im_path_dad) and is_image_file(im_path_err) and is_image_file(
                    im_path_ibcln) and is_image_file(im_path_rag) and is_image_file(im_path_rag):
                can_load = True
                try:
                    # if able to load images, add title to each image
                    im_dad_text = add_title(os.path.join(path, sub_dir_list[0], im_path_dad), sub_dir_list[0])
                    im_err_text = add_title(os.path.join(path, sub_dir_list[1], im_path_err), sub_dir_list[1])
                    im_ibcln_text = add_title(os.path.join(path, sub_dir_list[2], im_path_ibcln), sub_dir_list[2])
                    im_rag_text = add_title(os.path.join(path, sub_dir_list[3], im_path_rag), sub_dir_list[3])
                    im_org_text = add_title(os.path.join(path, sub_dir_list[4], im_path_org), sub_dir_list[4])
                except:
                    print(
                        'Was not able to load ' + im_path_rag + 'or ' + im_path_ibcln + 'or ' + im_path_dad + 'or ' + im_path_err)
                    can_load = False
                    continue
                if can_load:
                    # append all images to one list and pass on to create_collage function
                    collage_nbr += 1
                    list_of_images.append(im_dad_text)
                    list_of_images.append(im_err_text)
                    list_of_images.append(im_ibcln_text)
                    list_of_images.append(im_rag_text)
                    list_of_images.append(im_org_text)
                    list_of_images.append(im_org_text)
                    list_of_images.append(im_org_text)
                    list_of_images.append(im_org_text)
                create_collage(550, 1100, list_of_images, collage_nbr, path)


def main():
    args = parser.parse_args()
    collage = Collage(args)
    collage.create_collages()


if __name__ == "__main__":
    main()
