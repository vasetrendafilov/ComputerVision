"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2021
Date: 09.03.2021

Description: download MNIST data and save to destination folders organized by class
Python version: 3.6
"""

# python modules
import os
import cv2
import numpy as np
from tqdm import tqdm

from keras.datasets import mnist


def create_folders(path):
    """
    create folders with class names 0 - 9
    :param path: global path of the parent folder [str]
    :return: None
    """

    for i in range(10):     # 0 - 9
        if not os.path.exists(os.path.join(path, str(i))):
            os.mkdir(os.path.join(path, str(i)))


def save_data_to_folders(path, samples, labels):
    """
    save loaded MNIST data as PNG images to folders by class
    :param path: global path of the parent folder containing the class folders [str]
    :param samples: training samples [numpy.ndarray]
    :param labels: labels, corresponding to each sample [numpy.ndarray]
    :return: None
    """

    for i, sample in tqdm(enumerate(samples)):     # tqdm shows progress bar in console
        dst_path = os.path.join(path, str(labels[i]))
        cv2.imwrite(os.path.join(dst_path, str(i).zfill(6) + '.png'), sample.astype(np.uint8))


if __name__ == '__main__':

    # --- create destination folders ---
    dstPath = r'C:\Users\vase_\Downloads\ComputerVision\Data\Minst'     # NOTE: supply destination path
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)

    if not os.path.exists(os.path.join(dstPath, 'train')):
        os.mkdir(os.path.join(dstPath, 'train'))
    if not os.path.exists(os.path.join(dstPath, 'test')):
        os.mkdir(os.path.join(dstPath, 'test'))

    create_folders(os.path.join(dstPath, 'train'))
    create_folders(os.path.join(dstPath, 'test'))

    # --- download MNIST data ---
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  # the data, split between train and test sets

    if len(x_train) > 0 and len(x_test) > 0:
        # 60000 training samples, 100000 test samples with size 28x28px
        # 22.8 MB
        print(f"Loaded {len(x_train)} training samples with shape {x_train[0].shape}.")
        print(f"Loaded {len(x_test)} training samples with shape {x_test[0].shape}.")
    else:
        print("No data is loaded.")

    # --- save data to folders ---
    save_data_to_folders(os.path.join(dstPath, 'train'), x_train, y_train)
    save_data_to_folders(os.path.join(dstPath, 'test'), x_test, y_test)
