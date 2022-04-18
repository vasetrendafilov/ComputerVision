"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2021
Date: 26.03.2021

Description: function to load annotations for the GRAM Road-Traffic Monitoring dataset provided in XML format
Python version: 3.6
"""

# python imports
import os
from collections import Counter

import xml.etree.ElementTree as ET


def load_annotations(path):
    """
    load bounding box annotation data provided in XML format
    :param path: path of folder containing XML annotation files
    :return: list of dictionaries
    """

    fNames = [x for x in os.listdir(srcPath) if x[-4:] == '.xml']

    annotations = []
    classes = []

    for fName in fNames:    # iterate through annotation files

        image_annotations = []  # all objects in an image

        path = os.path.join(srcPath, fName)

        root = ET.parse(path).getroot()

        for object in root.findall('object'):   # find all object tags

            cl = object.find('class').text  # get content of class tag
            classes.append(cl)

            bb_xml = object.find('bndbox')  # get content of bndbox tag
            bb = [bb_xml.find('xmin').text,
                  bb_xml.find('xmax').text,
                  bb_xml.find('ymin').text,
                  bb_xml.find('ymax').text,
                  ]
            # xmin, ymin, xmax, ymax

            image_annotations.append({cl: bb})

        annotations.append(image_annotations)

        break

    print(set(classes))
    print(Counter(classes).keys())  # all classes
    print(Counter(classes).values())    # number of samples from each class

    return annotations


if __name__ == '__main__':

    srcPath = r'C:\Users\User\MashinskiVid\GRAM\xml'

    annotations = load_annotations(srcPath)

    # output format:
    # [[{'motorbike': ['380', '390', '74', '89']}, {'car': ['404', '423', '74', '92']}, {'car': ['376', '413', '127', '155']}, {'car': ['402', '473', '271', '337']}]]

