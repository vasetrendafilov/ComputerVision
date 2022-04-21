"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2021
Date: 09.03.2021

Description: function library
             data operations: load, save, process
Python version: 3.6
"""

# python imports
import os
import numpy as np
import cv2
import random


def read_data_unet(path_images, path_masks, img_dims, norm, stretch, shuffle):
    """
    load, crop and normalize image data
    option to select images form a folder
    :param image_path: path of the folder containing the images [string]
    :param mask_path: path of the folder containing the masks [string]
    :param scans_to_use: list of 4-zero-padded sequence numbers of the scans to be used in the current training [list of strings]
    :param img_dims: [dict]
    :param norm: is normalization to the range of [0, 1] required [bool]
    :param stretch: is contrast stretch to the range of [0, 255] required [bool]
    :return: list - array of normalized depth maps [array of numpy arrays]
    """

    images = []       # array of normalized multidiemnsional distance maps (frequencies: 20 - 120MHz)
    masks = []

    im_names = [fname for fname in os.listdir(path_images) if fname[-4:] == '.jpg']   # image and corresponding mask have the same filename
    mask_names = [fname for fname in os.listdir(path_masks) if fname[-4:] == '.jpg']   # image and corresponding mask have the same filename

    for ind, im_name in enumerate(im_names):

        # load image
        im = cv2.imread(os.path.join(path_images, im_name), 0)
        if stretch:
            im = cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX)
        images.append(im)

        # load mask
        mask = cv2.imread(os.path.join(path_masks, mask_names[ind]), 0)
        masks.append(mask)

    if len(images) == 0:
        print("No images were read.")
        exit(101)

    if shuffle:
        data = list(zip(images, masks))
        random.shuffle(data)
        images, masks = zip(*data)

    # convert data to ndarrays
    images = np.array(images)
    images = np.reshape(images, (len(images), img_dims['rows'], img_dims['cols'], img_dims['depth']))

    masks = np.array(masks)
    masks = np.reshape(masks, (len(masks), img_dims['rows'], img_dims['cols'], 1))

    # binarize masks
    masks[masks > 0] = 1
    masks[masks <= 0] = 0

    if norm:
        images = images / 255.0

    return images, masks


def save_probability_maps(maps, dst_path, file_names):
    """
    saves probability maps raw values as txt files and normalized (no contrast stretch) as grayscale images
    :param maps: maps to save [numpy array]
    :param dst_path: destination folder path [string]
    :param file_names: list of sequence numbers of the files to save [list of integers]
    :return: None
    """

    i = 0  # image counter
    for map in maps:
        outputImageName = file_names[i]
        map = map * 255
        map = np.round(map).astype(np.uint8)
        # map = cv2.normalize(map, map, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(os.path.join(dst_path, outputImageName), map)

        i += 1


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


def save_overlay(images, bin_masks, overlay_coef, names, dst_path):
    """
    save images as they are, no changes applied
    :param images: input images [numpy array]
    :param bin_masks: binary output masks (values: 0, 1) [numpy array]
    :param overlay_coef: transparency coefficient of overlayed mask [float]
    :param names: list of full names of images [list of strings]
    :param dst_path: destination folder path [string]
    :return: None
    """

    for ind, im in enumerate(images):

        mask = bin_masks[ind]   # get binary mask which corresponds to im
        mask = 255 - mask

        im = cv2.merge((im, im, im))    # convert images to 3-channel to overlay color
        im = (im * 255).astype(np.uint8)

        overlay = im.copy()     # will contain areas of mask with full opacity
        output = im.copy()      # overlay of the full opacity mask

        # create overlay
        overlay[:, :, 0][mask[:, :, 0] > 0] = 0     # blue
        overlay[:, :, 1][mask[:, :, 0] > 0] = 255     # green
        overlay[:, :, 2][mask[:, :, 0] > 0] = 0     # red

        cv2.addWeighted(overlay, overlay_coef, output, 1 - overlay_coef, 0, output)

        cv2.imwrite(os.path.join(dst_path, names[ind]), output)
