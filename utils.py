import gc
import torch
import random
from tqdm import tqdm
from glob import glob
import tifffile
import numpy as np
import pandas as pd
import os
import cv2
import math
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage import measure, morphology
from scipy import ndimage
import matplotlib
from PIL import Image, ImageDraw
import math
import numpy.matlib as npm

from xlsxwriter.workbook import Workbook


from skimage.measure import find_contours, label, regionprops
from roifile import ImagejRoi, roiwrite

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# from models.UMF import UMF
from models.UMF_ConvLSTM import UMF_ConvLSTM


def imread_image(path_image):
    head, tail = os.path.split(path_image)
    ext = tail.split('.')[1]

    if ext == 'tif' or ext == 'tiff' or ext == 'TIF' or ext == 'TIFF':
        image = tifffile.imread(path_image)
    else:
        # image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(path_image)
    return image.astype('uint8')


def update_mask(true_mask, mask):
    if len(true_mask.shape) == 2:
        true_mask = mask
    else:
        true_mask = np.concatenate((true_mask, mask), axis=0)
    return true_mask


def rotate_rectangle(HH, WW, cx, cy, w, h, angle):
    value = 1
    img = Image.new('L', (WW, HH), 0)
    theta = math.radians(angle)
    bbox = npm.repmat([[cx], [cy]], 1, 5) + \
           np.matmul([[math.cos(theta), math.sin(theta)],
                      [-math.sin(theta), math.cos(theta)]],
                     [[-w / 2, w / 2, w / 2, -w / 2, w / 2 + 8],
                      [-h / 2, -h / 2, h / 2, h / 2, 0]])

    x1, y1 = bbox[0][0], bbox[1][0]  # add first point
    x2, y2 = bbox[0][1], bbox[1][1]  # add second point
    x3, y3 = bbox[0][2], bbox[1][2]  # add third point
    x4, y4 = bbox[0][3], bbox[1][3]  # add forth point
    polygon = [x1, y1, x2, y2, x3, y3, x4, y4]
    ImageDraw.Draw(img).polygon(polygon, outline=value, fill=value)
    return np.asarray(img)


def setps_crop(h, in_size, max_crops):
    diff_h = h - in_size

    steps = []
    for i in range(2, max_crops):
        if diff_h % i == 0:
            steps.append(i)

    step_i = int(diff_h / steps[-1])
    v_step = []
    for i in range(steps[-1] + 1):
        v_step.append(i * step_i)
        # print('crop', i * step_i, '-', in_size + (i * step_i))
    return v_step


def build_skel(H, W, XY_skels):
    mask = np.zeros((H, W))
    for XY_skel in XY_skels:
        mask[int(XY_skel[0]), int(XY_skel[1])] = 1
    return mask


def list_files(path, extension):
    res = []
    # Iterate directory
    for file in os.listdir(path):
        # check only text files
        if file.endswith(extension):
            res.append(file)
    return res


def labels_list(label_segmentation):
    list_parts = list(np.unique(label_segmentation))
    labels_list_noneZero = []
    for i in list_parts:
        if i != 0:
            labels_list_noneZero.append(i)
    return labels_list_noneZero


def get_image_network(device, dir_checkpoint, n_classes, in_size, image_gray):
    # model = UMF_ConvLSTM(n_channels=1, n_classes=n_classes, bilinear=False)
    # model = UMF(n_channels=1, n_classes=n_classes, bilinear=True, type_net=1)
    model = UMF_ConvLSTM(n_channels=1, n_classes=n_classes, bilinear=True, type_net=1)
    model.load_state_dict(torch.load(dir_checkpoint))
    model.eval()
    model.to(device=device)

    h, w = image_gray.shape
    h_steps = setps_crop(h, in_size, 4)
    w_steps = setps_crop(w, in_size, 4)
    list_box = []
    for i in h_steps:
        for j in w_steps:
            crop = [i, i + in_size, j, j + in_size]
            list_box.append(crop)

    if n_classes == 1:
        masK_edge = np.zeros((h, w), dtype="uint8")

    if n_classes == 4:
        masK_edge = np.zeros((h, w, 3), dtype="uint8")

    for i in range(len(list_box)):
        with torch.no_grad():
            image_i = image_gray[list_box[i][0]:list_box[i][1], list_box[i][2]:list_box[i][3]]
            image_i = torch.from_numpy(image_i).to(device=device, dtype=torch.float32).unsqueeze(0)
            image_i = model(image_i.unsqueeze(0))
            image_i = (torch.sigmoid(image_i) > 0.5) * 255

            if n_classes == 1:
                image_i = image_i.squeeze(0).squeeze(0).cpu().numpy().astype('uint8')
                masK_edge[list_box[i][0]:list_box[i][1], list_box[i][2]:list_box[i][3]] = image_i

            if n_classes == 4:
                image_i = image_i.squeeze(0).cpu().numpy().astype('uint8')
                masK_edge[list_box[i][0]:list_box[i][1], list_box[i][2]:list_box[i][3], 0] = image_i[1, :, :]
                masK_edge[list_box[i][0]:list_box[i][1], list_box[i][2]:list_box[i][3], 1] = image_i[2, :, :]
                masK_edge[list_box[i][0]:list_box[i][1], list_box[i][2]:list_box[i][3], 2] = image_i[3, :, :]

    del model, image_i
    gc.collect()
    torch.cuda.empty_cache()
    return masK_edge


def build_edge(image_bw):
    image_bw = ((image_bw > 0) * 255).astype('uint8')
    contours, hierarchy = cv2.findContours(image_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for j in range(len(contours)):
        XY = contours[j][:, 0, :]
        X = XY[:, 0]
        Y = XY[:, 1]
        XY = np.stack((Y, X), axis=-1)
        if j == 0:
            XY_points = XY
        else:
            XY_points = np.vstack((XY_points, XY))

    H, W = image_bw.shape
    SKL = build_skel(H, W, XY_points)
    return ((SKL > 0) * 255).astype('uint8')


def edge_img(img):
    # create edge
    masK_edge = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")
    masK_edge[0:img.shape[0], 0:1] = 255
    masK_edge[0:img.shape[0], img.shape[1] - 1:img.shape[1]] = 255

    masK_edge[0:1, 0:img.shape[1]] = 255
    masK_edge[img.shape[0] - 1:img.shape[0], 0:img.shape[1]] = 255
    return masK_edge


def skeleton_check_edge(img):
    masK_edge = edge_img(img)
    # skeleton = morphology.medial_axis(img) * 255
    skeleton_fix = cv2.bitwise_and(img.astype('uint8'), cv2.bitwise_not(masK_edge))
    return skeleton_fix > 0


def check_edge_worms(image_seg, image_edge):
    h, w = image_edge.shape
    edge_seg = image_seg[:, :, 2]
    edge_net = image_edge
    edge_worm = build_edge(image_seg[:, :, 0])
    try:
        edge_overlap = build_edge(image_seg[:, :, 1])
    except:
        edge_overlap = np.zeros((h, w), dtype="uint8")
    edge_final = (((edge_net + edge_seg + edge_worm + edge_overlap) > 0) * 255).astype('uint8')
    return edge_final


def obtain_overlappings(image_seg, edge_final):
    # Obtain none overlapings and overlappings
    worms_seg = (image_seg[:, :, 0] > 0) * 1
    edge_overlap_seg = ((edge_final + image_seg[:, :, 1]) == 0) * 1
    none_overlappings = worms_seg * edge_overlap_seg

    overlapping_parts = (image_seg[:, :, 1] > 0) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    overlapping_parts_dilated = cv2.dilate(overlapping_parts.astype('uint8'), kernel, iterations=1)
    overlapping = (overlapping_parts_dilated > 0) * 1
    return none_overlappings, overlapping


def check_overlapping(labels_overlapping, labels_none_overlapping, image_skl, kernel_size):
    h, w = labels_overlapping.shape
    masK_delete_overlaps = np.zeros((h, w))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    for i in range(1, labels_overlapping.max() + 1):
        overlap_i = (labels_overlapping == i) * 1
        worms_connected_overlap = overlap_i * labels_none_overlapping
        worms_connected_overlap = labels_list(worms_connected_overlap)

        if len(worms_connected_overlap) < 2:  # overlapping in one worm
            masK_delete_overlaps = masK_delete_overlaps + overlap_i
        else:
            # overlapp touch don't touch skeleton => overlapworm = edge
            skeleton_touch = ((overlap_i * image_skl) > 0) * 1
            count_px = np.count_nonzero(skeleton_touch > 0)  # overlapp touch worm edge = 0
            if count_px == 0:
                masK_delete_overlaps = masK_delete_overlaps + overlap_i
            else:  # delete if overlap touch = points
                for j in worms_connected_overlap:
                    if j != 0:
                        part_none_overlap = (labels_none_overlapping == j) * 1
                        overlapping_touch = ((part_none_overlap * overlap_i) > 0) * 1
                        image_skl_dilated = cv2.dilate(image_skl.astype('uint8'), kernel, iterations=1)
                        overlapping_touch_dilated = cv2.dilate((overlapping_touch*255).astype('uint8'), kernel, iterations=1)
                        skeleton_touch_dilated = ((overlapping_touch_dilated * image_skl_dilated) > 0) * 1
                        count_px = np.count_nonzero(skeleton_touch_dilated > 0)  # overlapp touch worm edge = 0
                        if count_px == 0:
                            masK_delete_overlaps = masK_delete_overlaps + overlapping_touch
                        else:  # touching endpoint?
                            # part_none_overlap_overlap = ((part_none_overlap + overlapping_touch) > 0) * 1
                            part_none_overlap_overlap = ((part_none_overlap + overlap_i) > 0) * 1
                            dt_part_none_overlap_overlap = ndimage.distance_transform_edt(part_none_overlap_overlap)
                            max_overlap_i = (dt_part_none_overlap_overlap * ((skeleton_touch_dilated > 0) * 1)).max()
                            # max_overlap_i = (dt_part_none_overlap_overlap * ((image_skl > 0) * 1)).max()

                            if max_overlap_i < 4:  # is touching with endpoint
                                masK_delete_overlaps = masK_delete_overlaps + overlapping_touch

    # check again overlapping connections
    true_overlaps = (masK_delete_overlaps > 0) * 1
    true_overlaps1 = ((true_overlaps == 0) * 1) * labels_overlapping

    blobs_labels_overlaps = measure.label((true_overlaps1 > 0) * 1, background=0)
    for i in range(len(blobs_labels_overlaps) + 1):
        if i != 0:
            overlap_i = (blobs_labels_overlaps == i) * 1
            worms_connected_overlap = overlap_i * labels_none_overlapping
            worms_connected_overlap = labels_list(worms_connected_overlap)

            if len(worms_connected_overlap) < 2:
                masK_delete_overlaps = masK_delete_overlaps + overlap_i
    true_overlaps = (masK_delete_overlaps > 0) * 1
    true_overlaps1 = ((true_overlaps == 0) * 1) * labels_overlapping
    return (true_overlaps1 > 0) * 1


def get_none_overlapping(labels_none_overlapping, true_overlaps, area_min, kernel_size):

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    true_overlaps_ic = np.copy(true_overlaps)
    true_overlaps_ic = cv2.dilate(((true_overlaps_ic > 0) * 255).astype('uint8'), kernel, iterations=1)

    worms_overlaps = ((true_overlaps_ic > 0) * 1) * labels_none_overlapping
    overlap_parts = labels_list(worms_overlaps)
    all_elements = []
    for i in range(labels_none_overlapping.max() + 1):
        if i != 0:
            all_elements.append(i)

    all_elements1 = all_elements + overlap_parts
    None_overlap_parts = [i for i in all_elements if all_elements1.count(i) == 1]

    # select worms > 600 pixeles (area)
    h, w = labels_none_overlapping.shape
    mask_true_worms = np.zeros((h, w))
    for i in None_overlap_parts:
        part_i = (labels_none_overlapping == i) * 1
        count_px = np.count_nonzero(part_i > 0)
        if count_px > area_min:
            mask_true_worms = mask_true_worms + part_i
    return (mask_true_worms > 0) * 1


def get_angle(XY00, XYFF):
    angle = math.atan2(XYFF[1] - XY00[1], XYFF[0] - XY00[0]) * (180 / math.pi)
    return angle


def TwoPoints(none_overlappings_mpart, image_skl, dt_overlappings_ov_part):
    skl_none_overlappings_mpart = ((none_overlappings_mpart * image_skl) > 0) * 1
    value_skl = skl_none_overlappings_mpart * dt_overlappings_ov_part
    (rows, cols) = np.nonzero(value_skl)
    skl_val = []
    skl_px = []
    for px in range(len(rows)):
        Y = rows[px]
        X = cols[px]
        skl_val.append(value_skl[Y, X])
        skl_px.append([X, Y])

    XY_points = []
    angle = -1
    if len(skl_val) > 0:
        Value_sorted, XY_sorted = zip(*sorted(zip(skl_val, skl_px)))

        XY0 = XY_sorted[0]
        if len(XY_sorted) > 10:
            XYF = XY_sorted[10]
        else:
            XYF = XY_sorted[-1]

        XY_points = [XY0, XYF]
        angle = get_angle(XYF, XY0)

    return XY_points, angle


def connected_parts(none_overlappings_part, overlappings_ov_part, none_overlappings_i, image_skl):
    dt_overlappings_ov_part = ndimage.distance_transform_edt((overlappings_ov_part == 0) * 1)
    XY_points0, angle0 = TwoPoints(none_overlappings_part, image_skl, dt_overlappings_ov_part)

    none_overlappings_ov_parts = overlappings_ov_part * none_overlappings_i * ((none_overlappings_part == 0) * 1)
    none_overlappings_ov_list = labels_list(none_overlappings_ov_parts)

    XY_none_overlap = []
    angle_none_overlap = []
    id_part = []
    for mpart in none_overlappings_ov_list:
        none_overlappings_mpart = (none_overlappings_i == mpart) * 1  # labels_none_overlapping
        XY_points, angle = TwoPoints(none_overlappings_mpart, image_skl, dt_overlappings_ov_part)
        if len(XY_points) > 0:
            XY_none_overlap.append(XY_points)
            angle_none_overlap.append(angle)
            id_part.append(mpart)

    # select best part
    list_angles = []
    if len(XY_points0) > 0:
        for i in range(len(XY_none_overlap)):
            XY_pointsF = XY_none_overlap[i]
            angleF = get_angle(XY_points0[0], XY_pointsF[0])
            diff_angle = abs(angle0 - angleF)
            list_angles.append(diff_angle)

    best_id_part = 0
    best_points = []
    if len(XY_points0) > 0:
        best_points = [XY_points0[0]]
    if len(list_angles) > 0:
        min_angle = min(list_angles)
        if min_angle < 22:
            best_idex = list_angles.index(min_angle)
            best_id_part = id_part[best_idex]
            best_points.append(XY_none_overlap[best_idex][0])
    return best_id_part, best_points


def connection_overlap(conecction_points, overlappings_ov_part, none_overlappings_part):
    ci = conecction_points[0]
    cf = conecction_points[1]
    dt_overlap_i = ndimage.distance_transform_edt((overlappings_ov_part > 0) * 1)

    cx = ci[0] - (ci[0] - cf[0]) / 2
    cy = ci[1] - (ci[1] - cf[1]) / 2
    angle = get_angle([cx, cy], ci)

    wr = math.sqrt(((abs(ci[0] - cx)) ** 2) + ((abs(ci[1] - cy)) ** 2))
    hR = dt_overlap_i[int(cy), int(cx)]

    h, w = overlappings_ov_part.shape
    img = rotate_rectangle(h, w, cx, cy, wr*2, hR*2, -angle)
    img = img * (((overlappings_ov_part + none_overlappings_part) > 0) * 1)
    return (img > 0) * 1


def recursive_connect_parts(list_part, masK_worm, none_overlappings_part, none_ovp, overlappings_i, none_overlappings_i, image_skl):
    list_part.append(none_ovp)
    masK_worm = masK_worm + none_overlappings_part

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    none_overlappings_i = none_overlappings_i * ((none_overlappings_part == 0) * 1)

    none_overlappings_part_ic = np.copy(none_overlappings_part)
    none_overlappings_part_ic = cv2.dilate(((none_overlappings_part_ic > 0) * 255).astype('uint8'), kernel, iterations=1)
    none_overlappings_part_ic = (none_overlappings_part_ic > 0) * 1
    overlap_touch = none_overlappings_part_ic * overlappings_i
    overlappings_i_list = labels_list(overlap_touch)  # list overlap touching none-overlap
    for ovp in overlappings_i_list:
        overlappings_ov_part = (overlappings_i == ovp) * 1  # select first

        overlappings_ov_part_ic = np.copy(overlappings_ov_part)
        overlappings_ov_part_ic = cv2.dilate(((overlappings_ov_part_ic > 0) * 255).astype('uint8'), kernel, iterations=1)
        overlappings_ov_part_ic = (overlappings_ov_part_ic > 0) * 1
        best_id_part, best_points = connected_parts(none_overlappings_part_ic, overlappings_ov_part_ic,
                                                    none_overlappings_i, image_skl)
        if best_id_part != 0:  # select next part
            # conecction_points.append(best_points)
            img_connection = connection_overlap(best_points, overlappings_ov_part, none_overlappings_part)
            masK_worm = masK_worm + img_connection

            none_overlappings_part_ovp = (none_overlappings_i == best_id_part) * 1
            overlappings_i = overlappings_i * ((overlappings_ov_part == 0) * 1)
            list_part, masK_worm = recursive_connect_parts(list_part, masK_worm, none_overlappings_part_ovp,
                                                           best_id_part, overlappings_i, none_overlappings_i, image_skl)
    return list_part, (masK_worm > 0) * 1


def overlapping_worms(true_overlaps, mask_worms, labels_overlapping, labels_none_overlapping, image_skl, area_min, kernel_size):
    overlapping = (labels_overlapping > 0) * 1
    none_overlappings = (labels_none_overlapping > 0) * 1

    deleted_overlaps = ((true_overlaps == 0) * 1) * overlapping
    deleted_segmentation = ((mask_worms + deleted_overlaps) > 0) * 1

    worms_overlapping = ((mask_worms == 0) * 1) * none_overlappings
    all_segmentation = ((deleted_segmentation == 0) * 1) * ((overlapping + worms_overlapping) > 0) * 1
    blobs_all_segmentation = measure.label(all_segmentation, background=0)

    #  Delete none worms
    h, w = blobs_all_segmentation.shape
    delete_none_worms = np.zeros((h, w))
    connected_worms = []
    for i in range(blobs_all_segmentation.max() + 1):
        if i != 0:
            part_i = (blobs_all_segmentation == i) * 1
            count_px = np.count_nonzero((part_i * overlapping) > 0)
            if count_px == 0:
                delete_none_worms = delete_none_worms + part_i
            else:
                connected_worms.append(i)
    delete_none_worms = (delete_none_worms > 0) * 1

    # optimizer: select best parts per worm
    masK_worms = np.zeros((h, w), dtype="uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    for i in connected_worms:  # select blob
        blob_i = (blobs_all_segmentation == i) * 1
        overlappings_i = blob_i * labels_overlapping
        none_overlappings_i = blob_i * labels_none_overlapping
        none_overlappings_i_list = labels_list(none_overlappings_i)  # obtain none-overlappings parts
        for none_ovp in none_overlappings_i_list:  # part_worm not taken
            if none_ovp != 0:
                list_part = []
                # conecction_points = []
                masK_worm = np.zeros((h, w))
                none_overlappings_part = (none_overlappings_i == none_ovp) * 1  # select binary none-overlap
                overlappings_ic = np.copy(overlappings_i)
                none_overlappings_ic = np.copy(none_overlappings_i)
                list_part, masK_worm = recursive_connect_parts(list_part, masK_worm, none_overlappings_part, none_ovp,
                                                               overlappings_ic, none_overlappings_ic, image_skl)

                count_px = np.count_nonzero(masK_worm > 0)
                if count_px > area_min:
                    # update list none-overlappings and none_overlappings_i
                    for wm in list_part:
                        dx_list = none_overlappings_i_list.index(wm)
                        none_overlappings_i_list[dx_list] = 0
                    none_overlappings_i = none_overlappings_i * ((masK_worm == 0) * 1)

                    # save result
                    masK_worm = (masK_worm * 255).astype('uint8')
                    masK_worm = cv2.dilate(masK_worm, kernel, iterations=1)  # dilate to recover edge
                    masK_worm = np.expand_dims(masK_worm, axis=0)
                    masK_worms = update_mask(masK_worms, masK_worm)
                    # plt.imshow(masK_worm)
                    # plt.show()

    if len(masK_worms.shape) == 2:
        masK_worms = np.zeros((0, h, w), dtype="uint8")

    return masK_worms


def worms2NDims(mask_worms):
    h, w = mask_worms.shape
    mask_worms = (mask_worms > 0) * 1
    labels_mask_worms = measure.label(mask_worms, background=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if labels_mask_worms.max() > 0:
        masK_worms_NDims = np.zeros((h, w), dtype="uint8")
        for i in range(1, labels_mask_worms.max()+1):
            mask_i = ((labels_mask_worms == i) * 255).astype('uint8')
            mask_i = cv2.dilate(mask_i, kernel, iterations=1)  # dilate to recover edge
            mask_i = np.expand_dims(mask_i, axis=0)
            masK_worms_NDims = update_mask(masK_worms_NDims, mask_i)
    else:
        masK_worms_NDims = np.zeros((0, h, w), dtype="uint8")

    return masK_worms_NDims


def Ndims2image(all_worms, init):
    # change NDims to single image
    if len(all_worms.shape) == 2:
        h, w = all_worms.shape
        d = 1
    else:
        d, h, w = all_worms.shape
    image_color = np.zeros((h, w), dtype="uint16")
    for z in range(d):
        image_color = np.where(all_worms[z] > 0, image_color + z + init, image_color)

    colors = [(0, 0, 0)]
    for i in range(255):
        R = random.randint(10, 100) / 100
        G = random.randint(10, 100) / 100
        B = random.randint(10, 100) / 100
        colors.append((R, G, B))

    new_map = matplotlib.colors.LinearSegmentedColormap.from_list('new_map', colors, N=256)
    return image_color, new_map


def chk_endPoints_edge(worm):
    chkL = 0
    dt_worm = ndimage.distance_transform_edt(worm)
    mask_edge = (edge_img(worm) > 0) * 1

    edge_dtimg = dt_worm * mask_edge
    if edge_dtimg.max() > 4:
        chkL = 1
    # plt.imshow(BW_SKL_j)
    # plt.show()
    return chkL


def check_noneOverlapping(mask_worms_Dims, th):
    d, h, w = mask_worms_Dims.shape
    worms_an = []
    worms_size = []
    for i in range(d):
        f_worm = 0
        worm = mask_worms_Dims[i, :, :]
        chk_edge = chk_endPoints_edge(worm)  # touching edge?
        count_px = np.count_nonzero(worm > 0)  # mesuare worm size
        if chk_edge == 0:  # is not on the edge
            f_worm = 1

        worms_an.append(f_worm)
        worms_size.append(count_px)

    max_size = max(worms_size)
    worms_good = np.zeros((h, w), dtype="uint8")
    worms_bads = np.zeros((h, w), dtype="uint8")
    # worms_good_0 = np.zeros((h, w), dtype="uint8")
    # worms_bads_0 = np.zeros((h, w), dtype="uint8")

    index_good = []
    index_bad = []
    for i in range(len(worms_an)):
        worm = mask_worms_Dims[i, :, :].astype('uint8')
        worm = np.expand_dims(worm, axis=0)
        if worms_an[i] == 1:
            pSize = worms_size[i]/max_size
            if pSize >= (th/100):
                # worms_good_0 = worms_good_0 + worm[0, :, :]
                worms_good = update_mask(worms_good, worm)
                index_good.append(i)
            else:
                # worms_bads_0 = worms_bads_0 + worm[0, :, :]
                worms_bads = update_mask(worms_bads, worm)
                index_bad.append(i)
        else:
            # worms_bads_0 = worms_bads_0 + worm[0, :, :]
            worms_bads = update_mask(worms_bads, worm)
            index_bad.append(i)

    # rgbArray = np.zeros((h, w, 3), 'uint8')
    # rgbArray[..., 2] = worms_good_0
    # rgbArray[..., 0] = worms_bads_0
    #
    # plt.imshow(rgbArray)
    # plt.show()

    if len(worms_good.shape) == 2:
        worms_good = np.zeros((0, h, w), dtype="uint8")

    if len(worms_bads.shape) == 2:
        worms_bads = np.zeros((0, h, w), dtype="uint8")

    results = {
        # *****************
        # [masks]
        # *****************
        'worms_good': worms_good,
        'worms_bads': worms_bads,
        # *****************
        # [labels]
        # *****************
        'index_good': index_good,
        'index_bad': index_bad
    }
    return results


def get_centroid(mask, init):
    centroid = []
    label_worm = []
    if mask.shape[0] > 0:
        for i in range(mask.shape[0]):
            img00 = mask[i, :, :]
            mass_x, mass_y = np.where(img00 > 0)
            cent_x = np.average(mass_x)
            cent_y = np.average(mass_y)
            centroid.append([cent_x, cent_y])
            label_worm.append(i + init)
    return centroid, label_worm


def save_results_mask(name_image_final, image_gray, results_masks_NO, mask_overlaps_Dims, gray_color):
    init = 1
    worms_good, new_map = Ndims2image(results_masks_NO['worms_good'], init)
    centroid_good, label_good = get_centroid(results_masks_NO['worms_good'], init)

    init1 = init + results_masks_NO['worms_good'].shape[0]
    worms_bads, _ = Ndims2image(results_masks_NO['worms_bads'], init1)
    centroid_bads, label_bads = get_centroid(results_masks_NO['worms_bads'], init1)

    init2 = init1 + results_masks_NO['worms_bads'].shape[0]
    worms_overlap, _ = Ndims2image(mask_overlaps_Dims, init2)
    centroid_overlap, label_overlap = get_centroid(mask_overlaps_Dims, init2)

    font = {'family': 'serif',
            'color': 'white',
            'weight': 'bold',
            'size': 8,
            }

    plt.ioff()  # Turn interactive plotting off
    fig = plt.figure(figsize=(15, 15))
    ax1 = fig.add_subplot(221)
    ax1.title.set_text('Gray image')
    if gray_color == 1:
        ax1.imshow(image_gray, cmap='gray', vmin=0, vmax=255)
    else:
        ax1.imshow(image_gray, cmap='magma')

    ax2 = fig.add_subplot(222)
    ax2.title.set_text('Good masks')
    ax2.imshow(worms_good, cmap=new_map, interpolation='None')
    for i in range(len(centroid_good)):
        ax2.text(centroid_good[i][1], centroid_good[i][0], label_good[i], fontdict=font,
                 bbox=dict(facecolor='black', edgecolor='red', linewidth=2))

    ax3 = fig.add_subplot(223)
    ax3.title.set_text('Bad masks')
    ax3.imshow(worms_bads, cmap=new_map, interpolation='None')
    for i in range(len(centroid_bads)):
        ax3.text(centroid_bads[i][1], centroid_bads[i][0], label_bads[i], fontdict=font,
                 bbox=dict(facecolor='black', edgecolor='red', linewidth=2))

    ax4 = fig.add_subplot(224)
    ax4.title.set_text('Overlap masks')
    ax4.imshow(worms_overlap, cmap=new_map, interpolation='None')
    for i in range(len(centroid_overlap)):
        ax4.text(centroid_overlap[i][1], centroid_overlap[i][0], label_overlap[i], fontdict=font,
                 bbox=dict(facecolor='black', edgecolor='red', linewidth=2))
    # plt.show()
    plt.savefig(name_image_final)
    plt.close()


def save_mask_tif(path_save, img):
    if img.shape[0] == 0:
        d, h, w = img.shape
        img = np.zeros((h, w))
    tifffile.imsave(path_save, img.astype(np.uint8))


def save_mask_rois(path_save_rois, mask_worms_Dims):
    list_rois = []

    for i in range(mask_worms_Dims.shape[0]):
        image_bw = ((mask_worms_Dims[i, :, :] > 0) * 255).astype('uint8')
        contours = measure.find_contours(image_bw, 0.8)

        # select bigger contour
        contours_size = []
        for j in range(len(contours)):
            contours_size.append(len(contours[j]))

        max_value = max(contours_size)
        max_index = contours_size.index(max_value)

        contour = contours[max_index]
        X = contour[:, 0]
        Y = contour[:, 1]
        XY = np.stack((Y, X), axis=-1)
        roi = ImagejRoi.frompoints(XY)
        list_rois.append(roi)

    if len(list_rois) != 0:
        roiwrite(path_save_rois, list_rois)


def img2edge(image):
    h, w = image.shape
    bw = np.copy(image)
    bw = ((bw > 0) * 255).astype('uint8')

    contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for j in range(len(contours)):
        XY = contours[j][:, 0, :]
        X = XY[:, 0]
        Y = XY[:, 1]
        XY = np.stack((Y, X), axis=-1)
        XY = np.stack((Y, X), axis=-1)
        if j == 0:
            XY_points = XY
        else:
            XY_points = np.vstack((XY_points, XY))

    return build_skel(h, w, XY_points)


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


def mesuare_dots(worm, init_BW, img_fluorecent, bw_fluorecent):

    fluorecent_worm = ((worm > 0) * 1) * bw_fluorecent
    distance = ndimage.distance_transform_edt(fluorecent_worm)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=fluorecent_worm)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = watershed(-distance, markers, mask=fluorecent_worm)
    # ***********************************
    area_total = np.count_nonzero(labels > 0)
    n_dot = labels.max()

    dots_area = []
    dots_min = []
    dots_max = []
    dots_avg = []
    dots_id = []
    h, w = worm.shape
    dot_edge = np.zeros((h, w))
    for i in range(1, labels.max()+1):
        dot_i = (labels == i) * 1
        dot_edge = dot_edge + (img2edge(dot_i)*(i + init_BW))
        V_area, V_min, V_max, V_mean = mesuare_gray_image(img_fluorecent, dot_i)
        dots_id.append(i)
        dots_area.append(V_area)
        dots_min.append(V_min)
        dots_max.append(V_max)
        dots_avg.append(V_mean)

    dict_dot = {'dot_id': dots_id,
                'area': dots_area,
                'min': dots_min,
                'max': dots_max,
                'mean': dots_avg
                }
    # list_info.append([dots_area, dots_min, dots_max, dots_avg])
    return dict_dot, dot_edge, area_total, n_dot


def mask2dict(masks, img_fluorecent, bw_fluorecent, init):

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
    BW_dot_edge = np.zeros((h, w))
    init_BW = 0
    for wi in range(d):
        if f != 1:
            bw = masks[wi, :, :]
        else:
            bw = masks
        if bw.max() > 0:
            sum_mask = sum_mask + bw
            worm_id.append(wi + init)
            worm_area.append(np.count_nonzero(bw > 0))

            dt_worm = ndimage.distance_transform_edt((bw > 0) * 1)
            worm_width.append(dt_worm.max()*2)

            skeleton = skeletonize((bw > 0) * 1) * 1
            worm_lenght.append(np.count_nonzero(skeleton > 0))

            if bw_fluorecent.max() > 0:
                list_info, dot_edge, area_total, n_dot = mesuare_dots(bw, init_BW, img_fluorecent, bw_fluorecent)

                list_dict.append(list_info)
                init_BW = dot_edge.max()
                BW_dot_edge = BW_dot_edge + dot_edge
                dots_total.append(area_total)
                dots_count.append(n_dot)

                # V_area, V_min, V_max, V_mean = mesuare_gray_image(img_fluorecent, bw)
                # dots_area.append(V_area)
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
                dots_total.append(0)
                dots_count.append(0)
                dots_min.append(0)
                dots_max.append(0)
                dots_avg.append(0)

    BW_dot_edge_RGB = np.zeros((h, w, 3), dtype="uint8")
    for icw in range(int(BW_dot_edge.max())):
        if icw != 0:
            BW = (BW_dot_edge == icw) * 1
            BW_dot_edge_RGB[:, :, 0] = BW_dot_edge_RGB[:, :, 0] + (BW * random.randint(60, 200)).astype('uint8')
            BW_dot_edge_RGB[:, :, 1] = BW_dot_edge_RGB[:, :, 1] + (BW * random.randint(60, 200)).astype('uint8')
            BW_dot_edge_RGB[:, :, 2] = BW_dot_edge_RGB[:, :, 2] + (BW * random.randint(60, 200)).astype('uint8')

    # BW_dot_edge_RGB = np.zeros((h, w, 3), dtype="uint8")
    # BW_dot_edge_RGB[:, :, 2] = (BW_dot_edge > 0) * 255
    sum_mask = (sum_mask > 0) * 1
    img_fluorecent_RGB = img_fluorecent * ((BW_dot_edge == 0) * 1)
    img_fluorecent_RGB = img_fluorecent_RGB * sum_mask
    img_fluorecent_RGB = cv2.cvtColor(img_fluorecent_RGB.astype('uint8'), cv2.COLOR_GRAY2RGB)
    dst = cv2.addWeighted(img_fluorecent_RGB, 0.4, BW_dot_edge_RGB, 0.99, 0)
    # plt.imshow(dst)
    # plt.show()

    dict_worm = {'worm_id': worm_id,
                 'area': worm_area,
                 'lenght': worm_lenght,
                 'width': worm_width,
                 'total_area': dots_total,
                 'n_dots': dots_count,
                 'min': dots_min,
                 'max': dots_max,
                 'mean': dots_avg
                 }
    return dict_worm, list_dict, dst


def dict2xlxs(name_xlxs, list_dict):
    writer = pd.ExcelWriter(name_xlxs, engine='xlsxwriter')  # Create a Pandas Excel writer
    for wii in range(len(list_dict)):
        worm_p = list_dict[wii]
        sheet_name = 'worm_' + str(wii)
        df = pd.DataFrame(worm_p)  # Create a Pandas dataframe from the data.
        df.to_excel(writer, sheet_name=sheet_name)  # Convert the dataframe to an XlsxWriter Excel object.
    writer.close()  # Close the Pandas Excel writer and output the Excel file.


def save_csv_info(dict_paths, dict_xlxs, fluorescent_threshold_min, channel_fluorescent):
    img_good_mask = imread_image(dict_paths['path_good_mask'])
    img_bad_mask = imread_image(dict_paths['path_bad_mask'])
    img_overlap_mask = imread_image(dict_paths['path_overlap_mask'])
    path_fluorecent = dict_paths['path_fluorecent']

    if len(img_good_mask.shape) == 2:
        h, w = img_good_mask.shape
        d = 1
    else:
        d, h, w = img_good_mask.shape

    if len(path_fluorecent) > 0:
        img_fluorecent_org = imread_image(path_fluorecent)
        img_fluorecent = np.copy(img_fluorecent_org)
        img_fluorecent = img_fluorecent[:, :, channel_fluorescent]
        bw_fluorecent = (img_fluorecent > fluorescent_threshold_min) * 1
    else:
        img_fluorecent = np.zeros((h, w), dtype="uint8")
        bw_fluorecent = np.zeros((h, w), dtype="uint8")

    init1 = 1
    dic_good_mask, list_dict_good, BW_dots_good = mask2dict(img_good_mask, img_fluorecent, bw_fluorecent, init1)
    init2 = init1 + len(dic_good_mask['worm_id'])
    dic_bad_mask, list_dict_bad, BW_dots_bad = mask2dict(img_bad_mask, img_fluorecent, bw_fluorecent, init2)
    init3 = init2 + len(dic_bad_mask['worm_id'])
    dic_overlap_mask, list_dict_overlap, BW_dots_overlap = mask2dict(img_overlap_mask, img_fluorecent, bw_fluorecent, init3)

    if len(path_fluorecent) > 0:
        path_fluorescent_folder_result = dict_paths['path_fluorescent_folder_result']
        head, tail = os.path.split(path_fluorescent_folder_result)
        path_fluorescent_good = head + '/' + tail.split('.')[0] + '_good.jpg'
        path_fluorescent_bad = head + '/' + tail.split('.')[0] + '_bad.jpg'
        path_fluorescent_overlap = head + '/' + tail.split('.')[0] + '_overlap.jpg'
        cv2.imwrite(path_fluorescent_good, BW_dots_good)
        cv2.imwrite(path_fluorescent_bad, BW_dots_bad)
        cv2.imwrite(path_fluorescent_overlap, BW_dots_overlap)

        name_xlxs_good_mask = dict_xlxs['path_good_mask']
        name_xlxs_bad_mask = dict_xlxs['path_bad_mask']
        name_xlxs_overlap_mask = dict_xlxs['path_overlap_mask']

        dict2xlxs(name_xlxs_good_mask, list_dict_good)
        dict2xlxs(name_xlxs_bad_mask, list_dict_bad)
        dict2xlxs(name_xlxs_overlap_mask, list_dict_overlap)

    # Save global info
    name_xlxs_all_mask = dict_xlxs['path_all']
    with pd.ExcelWriter(name_xlxs_all_mask) as writer:
        # use to_excel function and specify the sheet_name and index
        # to store the dataframe in specified sheet
        pd.DataFrame(dic_good_mask).to_excel(writer, sheet_name="Good_mask", index=False)
        pd.DataFrame(dic_bad_mask).to_excel(writer, sheet_name="Bad_mask", index=False)
        pd.DataFrame(dic_overlap_mask).to_excel(writer, sheet_name="Overlap_mask", index=False)
