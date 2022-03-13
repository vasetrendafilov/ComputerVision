"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2021
Date: 09.03.2021

Description: function library
             model operations: construction, loading, saving
Python version: 3.6
"""

# python imports
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Dropout, Flatten, Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, BatchNormalization, Input, ZeroPadding2D, Concatenate


def load_model(model_path, weights_path):
    """
    loads a pre-trained model configuration and calculated weights
    :param model_path: path of the serialized model configuration file (.json) [string]
    :param weights_path: path of the serialized model weights file (.h5) [string]
    :return: model - keras model object
    """

    # --- load model configuration ---
    json_file = open(model_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)     # load model architecture

    model.load_weights(weights_path)     # load weights

    return model


def construct_model(num_classes):
    """
    construct model architecture
    :param num_classes: number of output classes of the model [int]
    :return: model - Keras model object
    """

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))     # softmax for multi-class classification

    return model


def construct_model_cnn(num_classes):
    """
    construct model architecture
    :param num_classes: number of output classes of the model [int]
    :return: model - Keras model object
    """

    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    # model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))     # softmax for multi-class classification

    return model
