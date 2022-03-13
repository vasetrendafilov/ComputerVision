"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2022
Date: 01.03.2022

Description: design, train, evaluate and apply a fully connected neural network for multi-class image classification
Python version: 3.6
"""

# python imports
import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

# custom package imports
from Helpers_Classification import helper_model
from Helpers_Classification import helper_data
from Helpers_Classification import helper_stats


# --- paths ---
version = 'LV1_v3'

# NOTE: specify destination paths
srcPath = r'C:\Users\vase_\Downloads\ComputerVision\Data\Minst'
dstResultsPath = r'C:\Users\vase_\Downloads\ComputerVision\Data\Results'
dstModelsPath = r'C:\Users\vase_\Downloads\ComputerVision\Data\Models'

# create folders to save data from the current execution
if not os.path.exists(os.path.join(dstResultsPath, version)):
    os.mkdir(os.path.join(dstResultsPath, version))
else:
    # to avoid overwriting training results
    print(f"Folder name {version} exists.")
    exit(1)

resultsPath = os.path.join(dstResultsPath, version)

if not os.path.exists(os.path.join(dstModelsPath, version)):
    os.mkdir(os.path.join(dstModelsPath, version))
modelsPath = os.path.join(dstModelsPath, version)


# --- variables ---
imgDims = {'rows': 28, 'cols': 28}
num_classes = 10
image_depth = 1

num_samples_to_load = 100   # how many samples to load from each class, value None loads all available samples

# optimization hyperprameters
batch_size = 128
epochs = 10
lr = 0.0001


# --- load and format data ---
# load full dataset into memory - image data and labels
x_train, y_train = helper_data.read_images(os.path.join(srcPath, 'train'), num_samples_to_load, image_depth)
x_test, y_test = helper_data.read_images(os.path.join(srcPath, 'test'), None, image_depth)

print(f'Training dataset shape: {x_train.shape}')
print(f'Number of training samples: {x_train.shape[0]}')
print(f'Number of test samples: {x_test.shape[0]}')

# one-hot encoding of labels
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# create validation dataset (image and label data is shuffled in both datasets)
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train,
                                                  test_size=0.2,    # assign random 20% of the samples to the validation set
                                                  random_state=42)     # fixed random seed enables repeatability of sample choice across executions


# --- construct model ---
model = helper_model.construct_model_cnn(num_classes)   # build model architecture

# compile model
model.compile(loss=categorical_crossentropy,    # categorical crossentropy for multi-class classification
              optimizer=Adam(lr=lr),
              metrics=['accuracy'])
# SGD(lr=lr, momentum=0.0, decay=0.0)

# --- fit model ---
model_checkpoint = ModelCheckpoint(filepath=os.path.join(modelsPath, 'checkpoint-{epoch:03d}-{val_accuracy:.4f}.hdf5'),   # epoch number and val accuracy will be part of the weight file name
                                   monitor='val_accuracy',      # metric to monitor when selecting weight checkpoints to save
                                   verbose=1,
                                   save_best_only=True)     # True saves only the weights after epochs where the monitored value (val accuracy) is improved

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,  # number of samples to process before updating the weights
                    epochs=epochs,
                    callbacks=[model_checkpoint],
                    verbose=1,
                    validation_data=(X_val, Y_val))


# --- save model ---
# save model architecture
print(model.summary())      # parameter info for each layer
with open(os.path.join(modelsPath, 'modelSummary.txt'), 'w') as fh:     # save model summary
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
plot_model(model, to_file=os.path.join(modelsPath, 'modelDiagram.png'), show_shapes=True)   # save diagram of model architecture

# save model configuration and weights
model_json = model.to_json()  # serialize model architecture to JSON
with open(os.path.join(os.path.join(modelsPath, 'model.json')), "w") as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(modelsPath, 'model.h5'))  # serialize weights to HDF5
print("Saved model to disk.")


# --- save training curves and logs ---
helper_stats.save_training_logs(history=history, dst_path=modelsPath)


# --- apply model to test data ---
Y_test_pred = model.predict(x_test, verbose=1)


# --- evaluate model ---
# accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# confusion matrix
labels = [x for x in range(10)]
print(labels)

# convert one-hot encoded vectors to 1D list of classes
y_test_list = np.argmax(y_test, axis=1)
Y_test_pred_list = np.argmax(Y_test_pred, axis=1)

cm = confusion_matrix(y_test_list, Y_test_pred_list, labels)    # takes 1D list of classes as input
print(cm)

# plot confusion matrix
target_names = [str(x) for x in labels]
fig = helper_stats.plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=False)
fig.savefig(os.path.join(modelsPath, 'confusionMatrix.png'), dpi=fig.dpi)    # save confusion matrix as figure


# --- save misclassified test samples ---

# find indices of misclassified samples
missed = [ind for ind, elem in enumerate(Y_test_pred_list) if elem != y_test_list[ind]]

for i in missed:
    cv2.imwrite(os.path.join(resultsPath, str(i).zfill(6) + '_' + str(Y_test_pred_list[i]) + '_' + str(y_test_list[i]) + '.png'),
                (x_test[i] * 255).astype(np.uint8))     # transform value range inback to [0, 255]
    # file name: OrdinalNumberOfSample_PredictedClass_TrueClass.png
