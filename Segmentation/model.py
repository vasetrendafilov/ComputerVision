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
from keras.layers import Conv2D, Conv2DTranspose, MaxPool2D, UpSampling2D, Input, Concatenate
from keras.models import Model, model_from_json


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


def construct_model_unet_orig(input_shape):
    """
    construct semantic segmentation model architecture (encoder-decoder)
    :param input_shape: list of input dimensions (height, width, depth) [tuple]
    :return:  model - Keras model object
    """

    input = Input(shape=input_shape)

    # --- encoder ---

    conv1 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv11 = Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv11)

    conv2 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv22 = Conv2D(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv22)

    conv3 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv33 = Conv2D(filters=256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv33)

    conv4 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv44 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(conv44)


    # --- decoder ---
    conv5 = Conv2D(filters=1024, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv55 = Conv2D(filters=512, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up1 = UpSampling2D(size=(2, 2))(conv55)
    merge1 = Concatenate(axis=3)([conv44, up1])
    deconv1 = Conv2DTranspose(filters=512, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    deconv11 = Conv2DTranspose(filters=256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(deconv1)

    up2 = UpSampling2D(size=(2, 2))(deconv11)
    merge2 = Concatenate(axis=3)([conv33, up2])
    deconv2 = Conv2DTranspose(filters=256, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge2)
    deconv22 = Conv2DTranspose(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(deconv2)

    up3 = UpSampling2D(size=(2, 2))(deconv22)
    merge3 = Concatenate(axis=3)([conv22, up3])
    deconv3 = Conv2DTranspose(filters=128, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge3)
    deconv33 = Conv2DTranspose(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(deconv3)

    up4 = UpSampling2D(size=(2, 2))(deconv33)
    merge4 = Concatenate(axis=3)([conv11, up4])
    deconv4 = Conv2DTranspose(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(merge4)
    deconv44 = Conv2DTranspose(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(deconv4)

    output = Conv2DTranspose(filters=input_shape[2], kernel_size=1, padding='same', activation='sigmoid')(deconv44)

    model = Model(input=input, output=output)

    return model
