"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2022
Date: 16.03.2022

Description: non-maximum suppression of bounding boxes
             merges bounding boxes of all sizes overlapping more than a given threshold
             averaging of coordinates of top left and bottom right point
Python version: 3.6
"""

# python imports
import cv2
import numpy as np
import os
from tqdm import tqdm

# custom package imports
from Helpers_Localization import helper_postprocessing


if __name__ == '__main__':

    # paths
    srcPath = r'D:\Science\Elena\MachineVision\Data\M-30\images_split\test'
    srcCoordsPath = r'D:\Science\Elena\MachineVision\Results_tmp\GRAM_sliding_window_v1\coords'
    dstPath = r'D:\Science\Elena\MachineVision\Results_tmp\GRAM_sliding_window_v1\nms'

    # --- variables ---
    iou_thr = 0.1   # minimum overlap needed to combine

    # --- paths ---
    # NOTE: make sure the lists contain the corresponding file names of the images and coordinate files (naming in alphabetical order)
    imgNames = [x for x in os.listdir(srcPath) if x[-4:] == '.jpg']
    coordsNames = [x for x in os.listdir(srcCoordsPath) if x[-4:] == '.txt']

    for ind, imgName in tqdm(enumerate(imgNames)):      # iterate through images

        img = cv2.imread(os.path.join(srcPath, imgName))    # load image
        coords = np.loadtxt(os.path.join(srcCoordsPath, coordsNames[ind]), ndmin=2)  # load bounding box coordinates

        new_windows = []   # list of coordinates of output windows
        deletedIndices = []     # indices of bounding boxes which have been merged

        for ind1 in range(len(coords)):     # get initial bbox (ind1)

            # --- find all bboxes which overlap with the initial bbox---

            to_combine = [coords[ind1]]     # initialize list of bounding boxes overlapping with the initial bbox

            for ind2 in range(ind1 + 1, len(coords)):   # iterate through all remaining bounding boxes
                if ind2 not in deletedIndices:

                    iou = helper_postprocessing.calc_iou(coords[ind1], coords[ind2])
                    # print(iou)

                    if iou > iou_thr:
                        to_combine.append(coords[ind2])
                        deletedIndices.append(ind2)

            deletedIndices.append(ind1)     # mark initial bounding box as merged
            to_combine = np.array(to_combine)

            if len(to_combine) > 1:  # if at least one window overlaps with the initial (delete isolated windows)
                # form new window with mean coordinates for the two opposite points
                new = [np.mean(to_combine[:, 0]), np.mean(to_combine[:, 1]), np.mean(to_combine[:, 2]), np.mean(to_combine[:, 3])]
                new_windows.append(new)

        new_windows = np.array(new_windows).astype(np.int)  # coordinates in integer format is needed for the drawing function

        for bbox in new_windows:
            cv2.rectangle(img, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color=(0, 255, 0), thickness=1)
            # expected input bbox format: [row_start, col_start, row_end, col_end]

        cv2.imwrite(os.path.join(dstPath, imgName), img)

