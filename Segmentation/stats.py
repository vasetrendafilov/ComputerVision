"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2021
Date: 09.03.2021

Description: function library
             training process monitoring and result statistics
Python version: 3.6
"""

# python imports
import matplotlib.pyplot as plt
import os
import numpy as np


def save_training_logs(history, dst_path):
    """
    saves graphs for the loss and accuracy of both the training and validation dataset throughout the epochs for comparison
    :param history: Keras callback object which stores accuracy information in each epoch [Keras history object]
    :param dst_path: destination for the graph images
    :return: None
    """

    # --- save accuracy graphs of training and validation sets ---
    plt.plot(history.history['accuracy'], 'r')
    plt.plot(history.history['val_accuracy'], 'g')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='lower right')
    plt.grid()
    # plt.show()    # blocks execution until figure is closed
    plt.savefig(os.path.join(dst_path, 'acc.png'))      # acc.png - name of accuracy graph
    plt.close()

    # --- save loss graphs of training and validation sets ---
    plt.plot(history.history['loss'], 'r')
    plt.plot(history.history['val_loss'], 'g')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.grid()
    # plt.show()    # blocks execution until figure is closed
    plt.savefig(os.path.join(dst_path, 'loss.png'))     # loss.png - name of loss graph
    plt.close()

    # --- save loss and accuracy of training and validation sets as a txt file ---
    losses = np.column_stack((history.history['loss'], history.history['val_loss']))
    np.savetxt(os.path.join(dst_path, 'loss.txt'), losses, fmt='%.4f', delimiter='\t', header="TRAIN_LOSS\tVAL_LOSS")

    accuracies = np.column_stack((history.history['accuracy'], history.history['val_accuracy']))
    np.savetxt(os.path.join(dst_path, 'acc.txt'), accuracies, fmt='%.4f', delimiter='\t', header="TRAIN_ACC\tVAL_ACC")
