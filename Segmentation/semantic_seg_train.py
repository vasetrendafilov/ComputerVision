"""
Authors: Elena Vasileva, Zoran Ivanovski
E-mail: elenavasileva95@gmail.com, mars@feit.ukim.edu.mk
Course: Mashinski vid, FEEIT, Spring 2021
Date: 15.05.2021

Description: design, train and evaluate a fully convolutional encoder-decoder architecture for semantic segmentation
Python version: 3.6
"""

# python imports
import os
import matplotlib.pyplot as plt

from keras.callbacks import ModelCheckpoint
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.utils import plot_model

from sklearn.model_selection import train_test_split

# custom package imports
from Lab6 import data
from Lab6 import model
from Lab6 import stats

# --- paths ---
version = 'v1'

# NOTE: specify destination paths
path_images = r'D:\MachineVision\Data\ISBI2012\images'
path_masks = r'D:\MachineVision\Data\ISBI2012\labels'
path_results = r'D:\MachineVision\Results'
path_models = r'D:\MachineVision\Models'

# create folders to save data from the current execution
if not os.path.exists(os.path.join(path_results, version)):
    os.mkdir(os.path.join(path_results, version))
else:
    # to avoid overwriting training results
    print(f"Folder name {version} exists.")
    exit(1)

path_results_folder = os.path.join(path_results, version)
path_results_whole = os.path.join(path_results_folder, 'maps_whole')
if not os.path.exists(path_results_whole):
    os.mkdir(path_results_whole)
path_results_bin = os.path.join(path_results_folder, 'bin')
if not os.path.exists(path_results_bin):
    os.mkdir(path_results_bin)
path_results_overlay = os.path.join(path_results_folder, 'overlay')
if not os.path.exists(path_results_overlay):
    os.mkdir(path_results_overlay)

if not os.path.exists(os.path.join(path_models, version)):
    os.mkdir(os.path.join(path_models, version))
modelsPath = os.path.join(path_models, version)


# --- variables ---
img_dims = {'rows': 512, 'cols': 512, 'depth': 1}
num_classes = 2

# optimization hyperprameters
batch_size = 2
epochs = 100
lr = 0.0001


# --- load and format data ---
# load full dataset into memory - images and ground truth masks
images, masks = data.read_data_unet(path_images, path_masks, img_dims, norm=True, stretch=False, shuffle=False)

# select test set
x_test = images[25:]
y_test = masks[25:]

x_train = images[:25]
y_train = masks[:25]

# select validation set
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.2,    # assign random 20% of the samples to the validation set
                                                  random_state=42)     # fixed random seed enables repeatability of sample choice across executions

print(f'Training dataset shape: {x_train.shape}')
print(f'Number of training samples: {x_train.shape[0]}')
print(f'Number of validation samples: {x_val.shape[0]}')
print(f'Number of test samples: {x_test.shape[0]}')


# --- construct model ---
input_shape = (img_dims['rows'], img_dims['cols'], img_dims['depth'])
model = model.construct_model_unet_orig(input_shape=input_shape)   # build model architecture

# compile model
model.compile(loss=binary_crossentropy,    # categorical crossentropy for multi-class classification
              optimizer=Adam(lr=lr),
              metrics=['accuracy'])
# SGD(lr=lr, momentum=0.0, decay=0.0)


# --- fit model ---
model_checkpoint = ModelCheckpoint(filepath=os.path.join(modelsPath, 'checkpoint-{epoch:03d}-{val_accuracy:.4f}.hdf5'),   # epoch number and val accuracy will be part of the weight file name
                                   monitor='val_accuracy',      # metric to monitor when selecting weight checkpoints to save
                                   verbose=1,
                                   save_best_only=False)     # True saves only the weights after epochs where the monitored value (val accuracy) is improved

history = model.fit(x_train, y_train,
                    batch_size=batch_size,  # number of samples to process before updating the weights
                    epochs=epochs,
                    callbacks=[model_checkpoint],
                    verbose=1,
                    validation_data=(x_val, y_val))


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
stats.save_training_logs(history=history, dst_path=modelsPath)


# --- apply model to test data ---
prob_maps_test = model.predict(x_test, verbose=1)


# --- evaluate model ---
# accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# --- plot histogram of probabilities ---
probs_flat = prob_maps_test.flatten()
plt.hist(probs_flat, density=False, bins=100)  # density=False shows counts, True shows density
plt.axvline(0.5, color='k', linestyle='dashed', linewidth=1)
plt.ylabel('Count')
plt.xlabel('Probability values')
plt.show()


# --- save test resutls ---
test_names = [str(x).zfill(2) + '.bmp' for x in range(25, 29 + 1)]
data.save_probability_maps(maps=prob_maps_test, dst_path=path_results_whole, file_names=test_names)


# --- binarize output and save overlay ---
prob_thr = 0.5
prob_maps_test[prob_maps_test > prob_thr] = 255
prob_maps_test[prob_maps_test <= prob_thr] = 0
data.save_images(images=prob_maps_test, names=test_names, dst_path=path_results_bin)


# --- save overlay of binary masks of test images ---
overlay_coef = 0.5
data.save_overlay(images=x_test, bin_masks=prob_maps_test, overlay_coef=overlay_coef, names=test_names, dst_path=path_results_overlay)
