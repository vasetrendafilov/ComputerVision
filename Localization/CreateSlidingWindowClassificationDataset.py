"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2022
Date: 15.03.2022

Description: select and save training samples from photographs with bounding box annotations
Python version: 3.6
"""

# python imports
import os
import cv2
import numpy as np
from tqdm import tqdm

from copy import deepcopy

import xml.etree.ElementTree as ET

# custom package imports
from Helpers_Localization import helper_postprocessing


# --- paths ---
# NOTE: specify destination paths
srcPath = r'D:\Science\Elena\MachineVision\Data\M-30\images_split'
annotationsPath = r'D:\Science\Elena\MachineVision\Data\M-30\GRAM-RTMv4\Annotations\M-30\xml'

dstPath = r'D:\Science\Elena\MachineVision\Data\M-30\windows12'
annotVisPath = r'D:\Science\Elena\MachineVision\Data\M-30\annot_vis'

# --- variables ---
subset = 'train'  # options; 'train', 'val', 'test'   NOTE: specify subset

imgSize = (800, 640)  # cols, rows
depth = 3  # color images

min_height = 25  # reject annotations with smaller height

# IOU overlap thresholds for selecting positive and negative samples
thr_pos = 0.7
thr_neg = 0.6

# sliding window variables
stride_pos = 5  # stride for translation augmentation of positive samples
stride_neg = 5  # sliding window stride

stride_frame = 5  # get data from every Nth frame

window_sizes = [(32, 32), (48, 48), (64, 64), (96, 96)]  # sizes of the selected negative samples

# counters
ctr_processed_files = 0
ctrPos = 0  # number of selected positive samples (used for naming samples)
ctrNeg = 0  # number of selected negative samples (used for naming samples)

# --- load images ---
fNames = [x for x in os.listdir(srcPath) if x[-4:] == '.jpg']

# how many images to select negative samples from
if subset == 'train':
    num_neg_images = 15
else:
    num_neg_images = 3

# --- create dataset ---
file_names = [x for x in os.listdir(os.path.join(srcPath, subset))]

for num_file in tqdm(range(0, len(file_names), stride_frame)):  # iterate through image names

    # load image
    imgName = file_names[num_file]
    img = cv2.imread(os.path.join(srcPath, subset, imgName))    # color image
    imgVis = deepcopy(img)

    # --- load annotations ---
    annotationName = str(int(imgName.split('.')[0][5:].lstrip('0')) - 1) + '.xml'
    annotationPath = os.path.join(annotationsPath, annotationName)

    root = ET.parse(annotationPath).getroot()

    objects = []    # coordinates of objects of all classes
    cars = []       # coordinates of cars

    for obj in root.findall('object'):  # find all object tags

        cl = obj.find('class').text  # get content of class tag

        bb_xml = obj.find('bndbox')  # get content of bndbox tag
        bb = [np.int(bb_xml.find('xmin').text),
              np.int(bb_xml.find('xmax').text),
              np.int(bb_xml.find('ymin').text),
              np.int(bb_xml.find('ymax').text),
              ]
        # bb = [min_col, max_col, min_row, max_col]

        objects.append(bb)
        cv2.rectangle(imgVis, (bb[0], bb[2]), (bb[1], bb[3]), color=(0, 255, 0),
                      thickness=1)  # visualize all annotated objects

        # save annotated positive car samples
        if cl == 'car' and bb[3] - bb[2] > min_height:
            # save original sample (manually annotated bounding box)
            cars.append(bb)
            cv2.imwrite(os.path.join(dstPath, subset, 'cars', str(ctrPos).zfill(8) + '.bmp'),
                        img[bb[2]:bb[3], bb[0]:bb[1]])
            ctrPos += 1

    # --- select and save negative samples ---
    for win_size in window_sizes:

        # sliding window
        for row in range(0, img.shape[0] - win_size[0], stride_neg):
            for col in range(0, img.shape[1] - win_size[1], stride_neg):

                neg_flag = 1    # flag marking a negative sample

                for obj in objects:
                    # check overlap of window with all objects in the image

                    # --- calculate overlap (IOU) ---
                    bbox_1 = [obj[2], obj[0], obj[3], obj[1]]
                    bbox_2 = [row, col, row + win_size[0], col + win_size[1]]
                    # coordinates: row1, row2, col1, col2

                    iou = helper_postprocessing.calc_iou(bbox_1, bbox_2)

                    if obj in cars and iou > thr_pos:
                        # positive sample, overlaps with ground truth car
                        cv2.imwrite(os.path.join(dstPath, subset, 'cars', str(ctrPos).zfill(8) + '.bmp'),
                                    img[row:row + win_size[0], col:col + win_size[1]])
                        ctrPos += 1

                    if iou > thr_neg:
                        # the sample is between  the IoU thresholds (not positive, not negative)
                        neg_flag = 0

                if ctr_processed_files < num_neg_images:    # select negative samples only from a subset of images
                    if neg_flag == 1:
                        cv2.imwrite(os.path.join(dstPath, subset, 'bgr', str(ctrNeg).zfill(8) + '.bmp'),
                                    img[row:row + win_size[0], col:col + win_size[1]])
                        ctrNeg += 1

    ctr_processed_files += 1
