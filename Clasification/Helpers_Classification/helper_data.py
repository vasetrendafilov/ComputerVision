"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2022
Date: 01.03.2022

Description: function library
             data operations: load, save, process
Python version: 3.6
"""

# python imports
import os
import numpy as np
import cv2
from tqdm import tqdm

def trim_train_data(x_train, y_train, num_samples):

    index = 0
    images_list = []
    labels_list = []
    for i,num in enumerate(y_train):
        if y_train[i-1] != num:
            index = i
        if num_samples is not None:
            if i-index >= num_samples:
                continue        
        labels_list.append(num)
        images_list.append(x_train[i])
        
    return np.array(images_list).astype(np.float32), np.array(labels_list).astype(np.int)

def read_images(path, num_samples, depth):
    """
    load, crop, normalize and binarize image data
    loads images from all class subfolders
    :param path: global path of folder containing class folders [string]
    :param num_samples: number of samples to load from each class [int]
    :param depth: required depth of the loaded images (value: 1 or 3) [int]
    :return: array of normalized depth maps [ndarray]
    """

    images_list = []       # array of normalized images
    labels_list = []

    # list class folders
    for folder_name in os.listdir(path):    # iterate through class folders

        src_path = os.path.join(path, folder_name)

        for i, file_name in tqdm(enumerate(os.listdir(src_path))):

            if num_samples is not None:
                if i >= num_samples:
                    break

            image = cv2.imread(os.path.join(src_path, file_name), 0)
            image = image.reshape(image.shape[0], image.shape[1], depth)

            if image is None:   # if file is read incorrectly
                continue

            images_list.append(image)
            labels_list.append(np.int(folder_name))

    if len(images_list) == 0:
        print("No images were read.")
        exit(100)

    images_list = np.array(images_list).astype(np.float32)
    images_list = images_list / 255.0   # normalize value range to 0 - 1
    images_list = images_list.reshape(images_list.shape[0], images_list[0].shape[0], images_list[0].shape[1], depth)

    labels_list = np.array(labels_list).astype(np.int)

    return images_list, labels_list


def save_images(images, names, dst_path):
    """
    save images as they are, no changes applied
    :param images: images to save [numpy array]
    :param names: list of full names of images [list of strings]
    :param dst_path: destination folder path [string]
    :return: None
    """

    for ind, im in enumerate(images):
        cv2.imwrite(os.path.join(dst_path, names[ind]), im)


def read_images_cifar(path, depth):
    """
    load, crop, normalize and binarize image data
    loads images from all class subfolders
    :param path: global path of folder containing class folders [string]
    :param depth: required depth of the loaded images (value: 1 or 3) [int]
    :return: array of normalized depth maps [ndarray]
    """

    images_list = []       # array of normalized images
    labels_list = []

    # list class folders
    for class_num, folder_name in enumerate(os.listdir(path)):    # iterate through class folders

        src_path = os.path.join(path, folder_name)

        for file_name in tqdm(os.listdir(src_path)):

            if depth == 3:
                image = cv2.imread(os.path.join(src_path, file_name))
            else:
                image = cv2.imread(os.path.join(src_path, file_name), 0)

            image = image.reshape(image.shape[0], image.shape[1], depth)

            if image is None:   # if file is read incorrectly
                continue

            images_list.append(image)
            labels_list.append(np.int(class_num))

    if len(images_list) == 0:
        print("No images were read.")
        exit(100)

    images_list = np.array(images_list).astype(np.float32)
    images_list = images_list / 255.0   # normalize value range to 0 - 1
    images_list = images_list.reshape(images_list.shape[0], images_list[0].shape[0], images_list[0].shape[1], depth)

    labels_list = np.array(labels_list).astype(np.int)

    return images_list, labels_list
