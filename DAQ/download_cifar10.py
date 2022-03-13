"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2021
Date: 21.03.2021

Description: download CIFAR10 data and save to destination folders organized by class
Python version: 3.6
"""

# python modules
import os
import cv2
import numpy as np
from tqdm import tqdm

from keras.datasets import cifar10


def create_folders(path, class_names):
    """
    create folders with class names
    :param path: global path of the parent folder [str]
    :param class_names: tuple of class names (the item position will be associated with the numeric value of the class) [tuple]
    :return: None
    """

    for class_name in class_names:
        if not os.path.exists(os.path.join(path, class_name)):
            os.mkdir(os.path.join(path, class_name))


def save_data_to_folders(path, samples, labels, class_names):
    """
    save loaded CIFAR10 data as PNG images to folders by class
    :param path: global path of the parent folder containing the class folders [str]
    :param samples: training samples [numpy.ndarray]
    :param labels: labels, corresponding to each sample [numpy.ndarray]
    :param class_names: tuple of class names [tuple]
    :return: None
    """

    for i, sample in tqdm(enumerate(samples)):     # tqdm shows progress bar in console
        dst_path = os.path.join(path, class_names[np.int(labels[i])])
        cv2.imwrite(os.path.join(dst_path, str(i).zfill(6) + '.png'), sample.astype(np.uint8))


if __name__ == '__main__':

    # --- variables ---
    class_names = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # immutable data type (tuple) to prevent modification of the position of the class names, as the position will be associated with the class code (0 - 9)

    # --- create destination folders ---
    dstPath = r'C:\Users\vase_\Downloads\ComputerVision\Data\Cifar10'     # NOTE: supply destination path
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)

    if not os.path.exists(os.path.join(dstPath, 'train')):
        os.mkdir(os.path.join(dstPath, 'train'))
    if not os.path.exists(os.path.join(dstPath, 'test')):
        os.mkdir(os.path.join(dstPath, 'test'))

    create_folders(os.path.join(dstPath, 'train'), class_names)
    create_folders(os.path.join(dstPath, 'test'), class_names)

    # --- download MNIST data ---
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()  # the data, split between train and test sets

    if len(x_train) > 0 and len(x_test) > 0:
        # 60000 training samples, 100000 test samples with size 28x28px
        # 22.8 MB
        print(f"Loaded {len(x_train)} training samples with shape {x_train[0].shape}.")
        print(f"Loaded {len(x_test)} training samples with shape {x_test[0].shape}.")
    else:
        print("No data is loaded.")

    # --- save data to folders ---
    save_data_to_folders(os.path.join(dstPath, 'train'), x_train, y_train, class_names)
    save_data_to_folders(os.path.join(dstPath, 'test'), x_test, y_test, class_names)
