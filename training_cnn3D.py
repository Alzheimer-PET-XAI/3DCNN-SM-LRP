#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Feb  4 21:12:13 2022

@author:

Training, Validation and Test of 3D CNN

ADNI Dataset
Data type: TFRecordDataset created from TFRecord files
Training set augmented (augmentation factor=13)
Augmentation type: 
    - Rotation
    - Translation
    - Rototranslation
    - Zoom
    
TFRecord files were created launching 'create_tfrecord.py'
file

"""

import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoard,
    EarlyStopping,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import auc, roc_auc_score, roc_curve
from itertools import cycle
import pandas as pd
from utilities_function import get_model_CNN, test_performance


classes = {0:"CN", 1:"MCI", 2:"AD"};
num_classes = len(classes); # 3

# Image size
img_depth = 96; # depth
img_width = 160; # width
img_height = 160; # heigth

# Training, Validation and Test Set sizes 
img_num_train = 21216; # training set
img_num_val = 409; # validation set
img_num_test = 511; # test set
# NOTE: The last two kfold (k=4, k=5) produces a different n° images (we 
# created training, validation and test set using 80%-20% and 80%-20% 
# percentage split starting from 2552 images):
#   img_num_train = 21229; # training set
#   img_num_val = 409; # validation set
#   img_num_test = 510; # test set

# Training, Validation and Test set directories
# All sets are stored in TFRecord format
main_dir = 'INSERT_YOUR_MAIN_DIRECTORY'; # directory of folder containing TFRecord files
train_filename = main_dir + '/train.tfrecords';  
validation_filename = main_dir + '/validation.tfrecords';
test_filename = main_dir + '/test.tfrecords';


""" Utilies functions used to manage dataset """

# PARSING TFRecord files

# We can create a Dataset of TFRecord files using tf.data.TFRecordDataset 
# class.
# TFRecordDataset loads TFRecords from the TFRecord files as they were written. 
# We need to define a parsing and decoding function which will be applied 
# using Dataset.map transformations after the TFRecordDataset.
# ----------------------------------------------------------------------

# Functions used to parse TFRecord files to extract images during training 
# process and to test the trained CNN.

def _float_feature(value):
    # Data to Feature
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def decode(serialized_example):
    # Decode examples stored in TFRecord
    # NOTE: make sure to specify the correct dimensions for the images
    features = tf.io.parse_single_example(
        serialized_example,
        features = {'train/image': tf.io.FixedLenFeature([img_width*img_height*img_depth,], tf.float32),
                    'train/label': tf.io.FixedLenFeature([3,], tf.float32)})
    # NOTE: No need to cast these features, as they are already `tf.float32` values.
    return features['train/image'], features['train/label']

def decode_val(serialized_example):
    # Decode examples stored in TFRecord
    # NOTE: make sure to specify the correct dimensions for the images
    features = tf.io.parse_single_example(
        serialized_example,
        features = {'validation/image': tf.io.FixedLenFeature([img_width*img_height*img_depth,], tf.float32),
                    'validation/label': tf.io.FixedLenFeature([3,], tf.float32)})
    # NOTE: No need to cast these features, as they are already `tf.float32` values.
    return features['validation/image'], features['validation/label']

def decode_test(serialized_example):
    # Decode examples stored in TFRecord
    # NOTE: make sure to specify the correct dimensions for the images
    features = tf.io.parse_single_example(
        serialized_example,
        features = {'test/image': tf.io.FixedLenFeature([img_width*img_height*img_depth,], tf.float32),
                    'test/label': tf.io.FixedLenFeature([3,], tf.float32)})
    # NOTE: No need to cast these features, as they are already `tf.float32` values.
    return features['test/image'], features['test/label']

# Function used to preprocess each element of TFRecordDataset

def dataset_preprocessing(volume, label, w = img_width, h = img_height, d = img_depth):
    # Process training data by:
    #     - Reshaping vectorized images
    #     - Adding a channel to perform 3D convolutions on the data
    volume = tf.reshape(volume,[w,h,d]);
    volume = tf.expand_dims(volume, axis = 3);
    return volume, label



""" Build Training, Validation and Test Dataset """
# We create the TFRecordDataset.
# We iterate over Training TFRecordDataset to determine the n° images in each 
# class (used to weight Loss Function during training process).

batch_size = 32; # mini-batch size

# Training set
train_dataset = tf.data.TFRecordDataset(train_filename);
train_dataset = train_dataset.map(decode);
train_dataset = train_dataset.map(dataset_preprocessing);
train_label = np.zeros((img_num_train, 3)); # true label (one-hot ecnding)
iter_train = iter(train_dataset);
for i in range(img_num_train):
    try:
        tensor_volume, tensor_label = next(iter_train);
        label = tensor_label.numpy();
        train_label[i,:] = label; 
    except:
        break;
freq_class_train = train_label.sum(axis=0).astype(int); # n° images in each class (training set)
train_dataset = train_dataset.batch(batch_size);
train_dataset = train_dataset.prefetch(2);
num_nc_train, num_mci_train, num_ad_train = freq_class_train;
# We weight loss function for the inverse of frequency of images in each class
# (used to manage class imbalance)
weight_cn = (1/num_nc_train)*(img_num_train/3); # weight CN
weight_mci = (1/num_mci_train)*(img_num_train/3); # weight MCI
weight_ad = (1/num_ad_train)*(img_num_train/3); # weight AD 
class_weight = {0:weight_cn, 1:weight_mci, 2:weight_ad}; 

# Validation set
validation_dataset = tf.data.TFRecordDataset(validation_filename);
validation_dataset = validation_dataset.map(decode_val);
validation_dataset = validation_dataset.map(dataset_preprocessing); 
validation_dataset = validation_dataset.batch(batch_size);
validation_dataset = validation_dataset.prefetch(2);

# Test set
test_dataset = tf.data.TFRecordDataset(test_filename);
test_dataset = test_dataset.map(decode_test);
test_dataset = test_dataset.map(dataset_preprocessing);
test_dataset = test_dataset.batch(1);
test_dataset = test_dataset.prefetch(2);

       
                                       
""" Build 3DCNN Model """

# Build model
model = get_model_CNN(width = img_width, height = img_height, depth = img_depth);
model.summary();

# Learning Rate
initial_learning_rate = 0.5*0.00001;
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
);

# Compile model
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"]
);

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "cnn3D_weights.h5", save_best_only=True
);  # save models weight

early_stopping_cb = keras.callbacks.EarlyStopping(monitor = "val_acc", 
                                                  patience = 15);


""" Neural Network Training """

t = time.time();

epochs = 100;

history = model.fit(
    train_dataset,
    epochs = epochs,
    batch_size = batch_size,
    shuffle = True,
    validation_data = validation_dataset,
    class_weight = class_weight,
    callbacks = [checkpoint_cb, early_stopping_cb]
)

fig, ax = plt.subplots(1, 2, figsize=(12,6))
ax = ax.ravel()
for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
    ax[i].grid(True)
plt.savefig('training.png');
elapsed = time.time() - t;



""" Test Neural Network Performance """

# Load best weights
model.load_weights("cnn3D_weights.h5");

# Test Set
# Accuracy Test set
test_loss, test_accuracy = model.evaluate((test_dataset));
print("Test accuracy: {:.2f}%".format(test_accuracy*100));

# CNN Prediction 
y_pred_test_prob = model.predict(test_dataset); # label predicted
y_pred_test = np.argmax(y_pred_test_prob, axis = 1); # label predicted: probabilities -> label

# True class Test set (one-hot encoding)
y_test = np.zeros((img_num_test, 3)); 
iter_test = iter(test_dataset);
for i in range(img_num_test):
    print('Immagine: '+str(i))
    tensor_volume, tensor_label = next(iter_test);
    label = tensor_label.numpy();
    y_test[i,:] = label;
    
# AUC
roc_test = roc_auc_score(y_test, y_pred_test_prob, multi_class='ovr', average=None);

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_test_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
lw = 2
plt.figure()
colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(num_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(classes[i], roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC multiclass curves")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curves.png')
plt.show()

# True class Label
y_test = np.argmax(y_test, axis = 1); # true label: one-hot encoding -> label

# Confusion Matrix
ConfusionMatrix = confusion_matrix(y_test, y_pred_test);
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test);
performance_cn, performance_mci, performance_ad = test_performance(ConfusionMatrix);
np.save('ConfusionMatrix', ConfusionMatrix)


print(performance_cn)
print(performance_mci)
print(performance_ad)


