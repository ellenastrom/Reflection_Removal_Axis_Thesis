import argparse
import os
from datetime import datetime
import openpyxl
from PIL import Image
from openpyxl.styles import PatternFill
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
import numpy as np
from tqdm import tqdm

# Script to evaluate and create excel file of results. The evaluation is either on synthetic or real data.
# For synthetic data, evaluation is based on SSIM and PSNR. For real data, a file with possible grading is created

date = str(datetime.now())
parser = argparse.ArgumentParser('Evaluation')
parser.add_argument('--background_dir', default="./background",
                    help="path to background image folder, not used for real data")
parser.add_argument('--filtered_dir', default="./filtered", help="path to filtered image folder")
parser.add_argument('--original_data_dir', default="./original", help="path to original image folder")
parser.add_argument('--is_real_data', type=bool, default=False, help="is True if evaluating real data otherwise False")
parser.add_argument('--collage_dir', default='./Collages',
                    help="Directory for pre-saved collages, only used for real data evaluation")
parser.add_argument('--nbr_marked', type=int, default=5,
                    help="nbr of best and worst metrics to mark, not used for real data")
parser.add_argument('--save_dir', default='./metrics_results_DATA_SET_NAME_{}.xlsx'.format(date),
                    help="path where to save metric results")


# check if file is image
def is_image_file(filename):
    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class Evaluator:
    def __init__(self, args):
        # ARGUMENTS
        self.args = args
        self.dir_bg = args.background_dir
        self.dir_res = args.filtered_dir
        self.dir_test = args.original_data_dir
        self.dir_save = args.save_dir
        self.collage_dir = args.collage_dir
        if not self.dir_save.endswith('xlsx'):
            self.dir_save = os.path.normpath(self.dir_save + '/metrics_results_DATA_SET_NAME_{}.xlsx'.format(date))
        self.nbr_marked = args.nbr_marked
        self.is_real_data = args.is_real_data

        # COLORS
        self.color_mean = PatternFill(patternType='solid',
                                      fgColor='7777C9')
        self.color_median = PatternFill(patternType='solid',
                                        fgColor='4B4B9F')
        self.color_std = PatternFill(patternType='solid',
                                     fgColor='70709A')
        self.color_best = PatternFill(patternType='solid',
                                      fgColor='31A91F')
        self.color_worst = PatternFill(patternType='solid',
                                       fgColor='D81F1F')
        self.color_five_best = PatternFill(patternType='solid',
                                           fgColor='8FBC89')
        self.color_five_worst = PatternFill(patternType='solid',
                                            fgColor='DE8E7E')

    # function to evaluate synthetic data
    def evaluate_synthetic_data(self):
        # allocate lists of psnr and ssim results, ground truth images, filtered result images and original images (
        # test set)
        psnr_list = []
        ssim_list = []
        image_list_gt = []
        image_list_result = []
        image_list_test = []

        # loop through all image folders at once and store image path
        for img_bg, img_res, img_test in tqdm(zip(sorted(os.listdir(self.dir_bg), key=str.casefold),
                                                  sorted(os.listdir(self.dir_res), key=str.casefold),
                                                  sorted(os.listdir(self.dir_test), key=str.casefold))):
            ground_truth_image_path = os.path.join(self.dir_bg, img_bg)
            result_image_path = os.path.join(self.dir_res, img_res)
            test_image_path = os.path.join(self.dir_test, img_test)

            # if is image, open image as PIL.Image
            if is_image_file(ground_truth_image_path) and is_image_file(result_image_path) and is_image_file(
                    test_image_path):
                can_load = True
                try:
                    ground_truth_img = Image.open(ground_truth_image_path)
                    result_img = Image.open(result_image_path)
                    test_img = Image.open(test_image_path)
                except:
                    print('Was not able to load ' + ground_truth_image_path + 'or' + result_image_path)
                    can_load = False
                    continue
                if can_load:
                    # if loaded, append image to corresponding list
                    image_list_gt.append(ground_truth_image_path)
                    image_list_result.append(result_image_path)
                    image_list_test.append(test_image_path)

                    # convert to RGB if needed
                    if len(result_img.split()) == 4:
                        result_img = result_img.convert('RGB')

                    # convert to float to calculate psnr and ssim
                    ground_truth_img = img_as_float(ground_truth_img)
                    result_img = img_as_float(result_img)

                    ssim_val = ssim(ground_truth_img, result_img, multichannel=True)
                    psnr_val = psnr(ground_truth_img, result_img)
                    ssim_list.append(ssim_val)
                    psnr_list.append(psnr_val)
        # create pandas data frame of results and convert to excel sheet
        d = {'GT': image_list_gt, 'Filtered': image_list_result, 'SSIM': ssim_list, 'PSNR': psnr_list,
             'Original': image_list_test}
        df = pd.DataFrame(data=d)
        df.to_excel(self.dir_save, index=False, sheet_name='sheet1')

        # mean and std values of psnr and ssim
        ssim_mean = np.mean(ssim_list)
        psnr_mean = np.mean(psnr_list)
        ssim_median = np.median(ssim_list)
        psnr_median = np.median(psnr_list)
        ssim_std = np.std(ssim_list)
        psnr_std = np.std(psnr_list)

        # open sheet as work sheet
        wb = openpyxl.load_workbook(self.dir_save)
        ws = wb['sheet1']  # Name of the working sheet

        # find best ssim and psnr
        max_ssim_idx = np.argmax(ssim_list) + 2
        max_psnr_idx = np.argmax(psnr_list) + 2

        # find worst ssim and psnr
        min_ssim_idx = np.argmin(ssim_list) + 2
        min_psnr_idx = np.argmin(psnr_list) + 2

        # find the second to sixth best ssim and psnr
        five_max_ssim_idx = np.argsort(ssim_list)[-self.nbr_marked - 1:-1]
        five_max_psnr_idx = np.argsort(psnr_list)[-self.nbr_marked - 1:-1]

        # find the second to sixth worst ssim and psnr
        five_min_ssim_idx = np.argsort(ssim_list)[1:self.nbr_marked + 1]
        five_min_psnr_idx = np.argsort(psnr_list)[1:self.nbr_marked + 1]

        # mark the found indices with different colors. Green is best results (dark green the best),
        # red is worst (dark red most worst)
        for i in range(0, self.nbr_marked):
            ws[int(five_max_ssim_idx[i] + 2)][2].fill = self.color_five_best
            ws[int(five_max_psnr_idx[i] + 2)][3].fill = self.color_five_best

            ws[int(five_min_ssim_idx[i] + 2)][2].fill = self.color_five_worst
            ws[int(five_min_psnr_idx[i] + 2)][3].fill = self.color_five_worst

        ws[int(max_ssim_idx)][2].fill = self.color_best
        ws[int(max_psnr_idx)][3].fill = self.color_best

        ws[int(min_ssim_idx)][2].fill = self.color_worst
        ws[int(min_psnr_idx)][3].fill = self.color_worst

        # go from image path to hyperlink to be able to click on and open the image in Excel
        for i in range(2, len(ssim_list) + 2):
            ws[i][0].value = '=HYPERLINK("{}", "im{}")'.format(ws[i][0].value, i - 1)
            ws[i][0].style = 'Hyperlink'
            ws[i][1].value = '=HYPERLINK("{}", "im{}")'.format(ws[i][1].value, i - 1)
            ws[i][1].style = 'Hyperlink'
            ws[i][4].value = '=HYPERLINK("{}", "im{}")'.format(ws[i][4].value, i - 1)
            ws[i][4].style = 'Hyperlink'

        # add mean, median, and std in Excel sheet
        ws[int(len(ssim_list) + 3)][1].value = 'mean:'
        ws[int(len(ssim_list) + 4)][1].value = 'median:'
        ws[int(len(ssim_list) + 5)][1].value = 'std:'

        ws[len(ssim_list) + 3][2].value = ssim_mean
        ws[int(len(ssim_list) + 3)][3].value = psnr_mean

        ws[len(ssim_list) + 4][2].value = ssim_median
        ws[int(len(ssim_list) + 4)][3].value = psnr_median

        ws[len(ssim_list) + 5][2].value = ssim_std
        ws[int(len(ssim_list) + 5)][3].value = psnr_std

        wb.save(self.dir_save)
        print("results saved in: " + self.dir_save)

    # evaluate real data (evaluation is manually done after sheet is created)
    def evaluate_real_data(self):
        grade_list = []
        image_list_collage = []
        rank_list = []

        for img_collage in tqdm(sorted(os.listdir(self.collage_dir), key=str.casefold)):
            collage_image_path = os.path.join(self.collage_dir, img_collage)

            if is_image_file(collage_image_path):
                can_load = True
                try:
                    collage_img = Image.open(collage_image_path)
                except:
                    print('Was not able to load ' + collage_image_path)
                    can_load = False
                    continue
                if can_load:
                    image_list_collage.append(collage_image_path)
                    grade_list.append(0)
                    rank_list.append(' ')

        d = {'Results': image_list_collage, 'DADNet': grade_list, 'ERRNet': grade_list, 'IBCLN': grade_list,
             'RAGNet': grade_list, 'Rank': rank_list}
        df = pd.DataFrame(data=d)
        df.to_excel(self.dir_save, index=False, sheet_name='sheet1')

        wb = openpyxl.load_workbook(self.dir_save)
        ws = wb['sheet1']  # Name of the working sheet

        for i in range(2, len(rank_list) + 2):
            ws[i][0].value = '=HYPERLINK("{}", "im{}")'.format(ws[i][0].value, i - 1)
            ws[i][0].style = 'Hyperlink'

        ws[int(len(rank_list) + 3)][0].value = 'mean:'
        ws[int(len(rank_list) + 4)][0].value = 'median:'
        ws[int(len(rank_list) + 5)][0].value = 'std:'
        for i in range(1, 5):
            col = '{}'.format(chr(ord('A') + i))
            ws[len(rank_list) + 3][i].value = '=AVERAGE({}{}:{}{})'.format(col, 2, col, len(rank_list) + 1)
            ws[len(rank_list) + 4][i].value = '=MEDIAN({}{}:{}{})'.format(col, 2, col, len(rank_list) + 1)
            ws[len(rank_list) + 5][i].value = '=STDEV({}{}:{}{})'.format(col, 2, col, len(rank_list) + 1)

        ws[len(rank_list) + 3][5].value = '=INDEX(F2:F{0}, MODE(MATCH(F2:F{0}, F2:F{0}, 0)))'.format(len(rank_list) + 1)

        wb.save(self.dir_save)
        print("results saved in: " + self.dir_save)


def main():
    args = parser.parse_args()
    evaluator = Evaluator(args)
    if not args.is_real_data:
        evaluator.evaluate_synthetic_data()
    else:
        evaluator.evaluate_real_data()


if __name__ == "__main__":
    main()
