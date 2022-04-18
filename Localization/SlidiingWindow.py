"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2021
Date: 26.03.2021

Description: sliding window approach for object localization (applying a multi-class classification model)
Python version: 3.6
"""

# python imports
import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# custom package imports
from Helpers_Localization import helper_model


if __name__ == '__main__':

    # paths
    srcPath = r'D:\Science\Elena\MachineVision\Data\M-30\images_split\test'
    dstPath = r'D:\Science\Elena\MachineVision\Results_tmp\GRAM_sliding_window_v1'
    dstCoordsPath = r'D:\Science\Elena\MachineVision\Results_tmp\GRAM_sliding_window_v1\coords'

    modelPath = r'D:\Science\Elena\MachineVision\Models_tmp\LV03_v1\model.json'
    weightsPath = r'D:\Science\Elena\MachineVision\Models_tmp\LV03_v1\model.h5'

    # variables
    prob_thr = 0.5      # probability threshold for detected objects (range 0 - 1)
    img_dims = (32, 32)     # cols, rows

    windowSizes = [(32, 32), (64, 64), (96, 96)]  # rows, cols
    stride = [5, 5]     # rows, cols
    depth = 1  # test image depth

    num_frames = 50000   # 200

    # --- load model ---
    model = helper_model.load_model(modelPath, weightsPath)

    # --- load images ---
    fNames = [x for x in os.listdir(srcPath) if x[-4:] == '.jpg']

    # --- sliding window ---
    for ind, fName in tqdm(enumerate(fNames)):      # iterate through test images

        # load image
        img = cv2.imread(os.path.join(srcPath, fName))

        windows = []
        coords = []
        cars_coords = []

        # classify all windows
        for windowSize in windowSizes:

            # sliding window
            for row in range(0, img.shape[0] - windowSize[0], stride[0]):   # if windowSize is not subtracted, the border windows will be smaller thm the specified size
                for col in range(0, img.shape[1] - windowSize[1], stride[1]):

                    window = img[row:row+windowSize[0], col:col+windowSize[1], 0]
                    window = cv2.resize(window, img_dims, cv2.INTER_AREA)   # resize to the input dimensions of the model

                    windows.append(window)
                    coords.append([row, col, windowSize[0], windowSize[1]])

        windows = np.array(windows)
        windows = windows.reshape((len(windows), windows[0].shape[0], windows[0].shape[1], depth))
        coords = np.array(coords)

        preds = model.predict(windows)  # prediction as a batch is significantly faster than calling predict for one sample

        # plot results
        for i, pred in enumerate(preds):  # set of probabilities of each window

            prob_car = pred[1]

            if prob_car > prob_thr:
                cv2.rectangle(img, (coords[i, 1], coords[i, 0]), (coords[i, 1] + coords[i, 3], coords[i, 0] + coords[i, 2]),
                              color=(0, 255, 0), thickness=1)
                cars_coords.append([coords[i, 0], coords[i, 1], coords[i, 0] + coords[i, 2], coords[i, 1] + coords[i, 3]])

        cv2.imwrite(os.path.join(dstPath, fName), img)

        cars_coords = np.array(cars_coords).astype(np.int)
        np.savetxt(os.path.join(dstCoordsPath, 'bboxes_' + str(ind).zfill(4) + '.txt'), cars_coords, fmt="%d")

        if ind >= num_frames:
            break

        '''
        plt.hist(preds[:, 1], density=False, bins=100)  # density=False would make counts
        plt.axvline(prob_thr, color='k', linestyle='dashed', linewidth=1)
        plt.ylabel('Count')
        plt.xlabel('Probability values')
        plt.show()
        '''
