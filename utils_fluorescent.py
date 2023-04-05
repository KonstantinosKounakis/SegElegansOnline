import os
import cv2
import random
import tifffile
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.morphology import skeletonize
from utils import *
import warnings
warnings.filterwarnings('ignore')


def mesuare_gray_image(gray_image, BW):
    gray_image_BW = gray_image * BW
    mass_y, mass_x = np.where(gray_image_BW > 0)
    gray_value = []
    for i in range(len(mass_y)):
        gray_value.append(gray_image_BW[mass_y[i], mass_x[i]])

    V_area = len(gray_value)
    V_min = min(gray_value)
    V_max = max(gray_value)
    V_mean = sum(gray_value) / len(gray_value)
    return V_area, V_min, V_max, V_mean


def mesuare_dots(worm, img_fluorecent, labels_F):
    labels = ((worm > 0) * 1) * labels_F
    area_total = np.count_nonzero(labels > 0)
    labels_list_worm = labels_list(labels)
    n_dot = len(labels_list_worm)

    dots_area = []
    dots_min = []
    dots_max = []
    dots_avg = []
    dots_id = []
    i_dot = 1
    for i in labels_list_worm:
        dot_i = (labels == i) * 1
        V_area, V_min, V_max, V_mean = mesuare_gray_image(img_fluorecent, dot_i)
        dots_id.append(i_dot)
        dots_area.append(V_area)
        dots_min.append(V_min)
        dots_max.append(V_max)
        dots_avg.append(V_mean)
        i_dot = i_dot + 1

    dict_dot = {'dot_id': dots_id,
                'area': dots_area,
                'min': dots_min,
                'max': dots_max,
                'mean': dots_avg
                }
    return dict_dot, area_total, n_dot


def mask2dict(masks, fluorescent_gray, fluorescent_segmentation, labels_F):
    if len(masks.shape) == 2:
        h, w = masks.shape
        d = 1
        f = 1
    else:
        d, h, w = masks.shape
        f = 0

    worm_id = []
    worm_area = []
    worm_lenght = []
    worm_width = []
    list_dict = []

    dots_area = []
    dots_min = []
    dots_max = []
    dots_avg = []

    dots_total = []
    dots_count = []
    sum_mask = np.zeros((h, w))
    for wi in range(d):
        if f != 1:
            bw = masks[wi, :, :]
        else:
            bw = masks
        if bw.max() > 0:
            sum_mask = sum_mask + bw
            worm_id.append(wi + 1)
            worm_area.append(np.count_nonzero(bw > 0))

            dt_worm = ndimage.distance_transform_edt((bw > 0) * 1)
            worm_width.append(dt_worm.max()*2)

            skeleton = skeletonize((bw > 0) * 1) * 1
            worm_lenght.append(np.count_nonzero(skeleton > 0))

            if fluorescent_segmentation.max() > 0:
                list_info, area_total, n_dot = mesuare_dots(bw, fluorescent_gray, labels_F)
                list_dict.append(list_info)
                dots_count.append(n_dot)

                # V_area, V_min, V_max, V_mean = mesuare_gray_image(img_fluorecent, bw)
                dots_area.append(area_total)
                if len(list_info['min']) != 0:
                    dots_min.append(min(list_info['min']))
                else:
                    dots_min.append(0)

                if len(list_info['max']) != 0:
                    dots_max.append(max(list_info['max']))
                else:
                    dots_max.append(0)

                if len(list_info['mean']) != 0:
                    dots_avg.append(sum(list_info['mean']) / len(list_info['mean']))
                else:
                    dots_avg.append(0)
            else:
                dots_area.append(0)
                dots_total.append(0)
                dots_count.append(0)
                dots_min.append(0)
                dots_max.append(0)
                dots_avg.append(0)

    sum_mask = (sum_mask > 0) * 1
    dict_worm = {'worm_id': worm_id,
                 'area': worm_area,
                 'lenght': worm_lenght,
                 'width': worm_width,
                 'n_dots': dots_count,
                 'dots_area': dots_area,
                 'min': dots_min,
                 'max': dots_max,
                 'mean': dots_avg
                 }
    return dict_worm, list_dict, sum_mask


def dict2xlxs(name_xlxs, dict_worm, list_dict):
    writer = pd.ExcelWriter(name_xlxs, engine='xlsxwriter')  # Create a Pandas Excel writer
    for wii in range(- 1, len(list_dict)):
        if wii == -1:
            sheet_name = 'all_worms'
            df = pd.DataFrame(dict_worm)  # Create a Pandas dataframe from the data.
            df.to_excel(writer, sheet_name=sheet_name)  # Convert the dataframe to an XlsxWriter Excel object.
        else:
            worm_p = list_dict[wii]
            sheet_name = 'worm_' + str(wii)
            df = pd.DataFrame(worm_p)  # Create a Pandas dataframe from the data.
            df.to_excel(writer, sheet_name=sheet_name)  # Convert the dataframe to an XlsxWriter Excel object.
    writer.close()  # Close the Pandas Excel writer and output the Excel file.


def label2contour(image_fluorescent, labels_F):
    labels_F_b = (labels_F.copy() > 0) * 255
    img2 = image_fluorescent.copy()
    markers1 = labels_F_b.astype(np.uint8)
    ret, m2 = cv2.threshold(markers1, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow('m2', m2)
    # contours, hierarchy = cv2.findContours(m2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(image=m2, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    for c in contours:
        R = random.randint(10, 255)
        G = random.randint(10, 255)
        B = random.randint(10, 255)
        cv2.drawContours(img2, c, -1, (R, G, B), 2)
    return img2


def save_fluorescent_img_results(name_image_final, image_fluorescent, image_complete_mask, im2, true_worms):
    worms_good, new_map = Ndims2image(image_complete_mask, 1)

    im3 = im2.copy()
    for i in range(3):
        im3[:, :, i] = im3[:, :, i] * ((true_worms > 0) * 1)

    plt.ioff()  # Turn interactive plotting off
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(131)
    ax1.title.set_text('Fluorescent image')
    ax1.imshow(image_fluorescent)

    ax2 = fig.add_subplot(132)
    ax2.title.set_text('Complete masks')
    ax2.imshow(worms_good, cmap=new_map, interpolation='None')

    ax3 = fig.add_subplot(133)
    ax3.title.set_text('Dots in complete masks')
    ax3.imshow(im3)

    # plt.show()
    plt.savefig(name_image_final)
    plt.close()
