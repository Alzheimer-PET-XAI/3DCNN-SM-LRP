#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:31:20 2022


@author: ##################

Create Training, Validation and Test TFRecord files performing Iterated Kfold
cross-validation and applying a radom data augmentation on the training set.

For every trial we need to change n_fold from 1 to K in order to perform the
CNN's training and its performances' evaluation with all the K fold.

Changing n_trial from 1 to tot_trial we change the initial shuffling of the 
dataset, so we can perform the Kfold cross-validation tot_trial times with a 
different shuffled dataset.

If we apply the Iterarated Kfold cross validation we need to evaluate the 
average over the trial of the performances (in the test set) averaged over
the K fold.

"""

tot_trial = 5;
k_fold = 5;

n_trial = 1; # change from 1 to tot_trial;
random_seed = n_trial*42; # seed for reproducible shuffling
n_fold = 1; # change from 1 to k_fold


import os
import numpy as np
import random
from scipy import ndimage
from skimage import transform
from scipy.ndimage.interpolation import zoom
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold
import SimpleITK as sitk

from utilities_function import plot_slices, ADNI_complete_directory


"""
DATA AUGMENTATION 
Functions:
    - Random rotation [-10°; +10°];
    - Random translation x/y/z;
    - Random rototranslation;
    - Random zoom.

Data Augmentation MUST be applied only on the Training set!

"""

# Dizionario che identifica il tipo di trasformazione applicata all'immagine
augmentation = {0:"Original",
                1:"Rotate",
                2:"Translate",
                3:"Rototrasl",
                4:"Zoom"};

def rotate(volume):
    """Rotate the volume by a few degrees"""
    # define some rotation angles
    angles = np.arange(-10,10,1);
    # pick angles at random
    angle = random.choice(angles);
    # rotate volume
    volume = ndimage.interpolation.rotate(volume, angle, mode='nearest', 
                                          axes=(0, 1), reshape=False,
                                          order=1);
    return volume
    
def translate(volume):
    # define some translation factor
    shift = np.arange(-5,5,1);
    # pick shift at random
    x_shift = random.choice(shift);
    y_shift = random.choice(shift);
    z_shift = random.choice(shift);
    volume = ndimage.shift(volume, [y_shift,x_shift,z_shift], order=1);
    return volume
    
def rototrasl(volume):
    # define some translation factor
    shift = np.arange(-5,5,1);
    # define some rotation angles
    angles = np.arange(-10,10,1);
    # pick translationa and angles at random
    x_shift = random.choice(shift);
    y_shift = random.choice(shift);
    angle = random.choice(angles);
    # shift volume
    volume = ndimage.shift(volume, [y_shift,x_shift,0], order=1);
    # rotate volume
    volume = ndimage.interpolation.rotate(volume, angle, reshape=False, 
                                          mode='nearest', axes=(0, 1),
                                          order=1);
    return volume

def zoomin(volume):
    Ny = volume.shape[0];
    Nx = volume.shape[1];
    Nz = volume.shape[2];
    zoom_factor = random.choice([1.1,1.2,1.3]);
    zoomed_img = zoom(volume, zoom_factor, order=1);
    zoomed_img = transform.resize(zoomed_img, (Ny,Nx,Nz), 
                                  preserve_range = True, anti_aliasing=(True),
                                  order=1);
    return zoomed_img



"""
--- DATASET CREATION ---
Array containing all the data path
ad/mci/nc_data -> dim: (n_imm, Nx, Ny, Nz)

"""

classes = {0:"NC",
           1:"MCI",
           2:"AD"};

root_dir = '/ADNI/';

# TFRecords filename (augmented)
train_aug_filename = "training.tfrecord";
val_filename = 'validation.tfrecords';
test_filename = 'test.tfrecords';

# Name used to save Exams ID
train_examID = 'exam_ID_train_kfold_' + str(n_fold);
val_examID = 'exam_ID_val_kfold_' + str(n_fold);
test_examID = 'exam_ID_test_kfold_' + str(n_fold);


d = 96;
test_split = 0.2;
valid_split = 0.2;


dicom_directories, label = ADNI_complete_directory(root_dir, d);
img_num = len(dicom_directories);

directories = np.array(dicom_directories);
label = np.array(label);

rng = np.random.default_rng(random_seed); # shuffling dataset of directories 
                                          # using random seed for reproducibility
shuffled_index = np.arange(img_num);
rng.shuffle(shuffled_index);
# Shuffled dataset
directories = directories[shuffled_index]; 
label = label[shuffled_index];


# Slitting of dataset into Training, Validation and Test set

# Partition of dataset to create the Kfold of Training+Validation and Test set
skf = StratifiedKFold(n_splits=k_fold, random_state=random_seed, shuffle=True);
kfold_generator = skf.split(directories, label);

for i in range(n_fold):
     train_index, test_index = next(kfold_generator);
     print("TRAIN:", train_index, "TEST:", test_index);
     X_train_val, X_test = directories[train_index], directories[test_index];
     y_train_val, y_test = label[train_index], label[test_index];

# Random splitting (using a random seed) into Training and Validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=valid_split, shuffle=True, 
    random_state=random_seed, stratify=y_train_val)

# Label -> Categorical (one-hot representation)
y_train = to_categorical(y_train, num_classes=3);
y_val = to_categorical(y_val, num_classes=3);
y_test = to_categorical(y_test, num_classes=3);


# Training test
freq_class_train = y_train.sum(axis=0).astype(int); # n° di immagini per classe 
num_nc_train, num_mci_train, num_ad_train = freq_class_train;

# Validation dataset
freq_class_valid = y_val.sum(axis=0).astype(int); 
num_nc_valid, num_mci_valid, num_ad_valid = freq_class_valid;

# Test dataset
freq_class_test = y_test.sum(axis=0).astype(int);
num_nc_test, num_mci_test, num_ad_test = freq_class_test;

y_test = np.argmax(y_test, axis = 1); # categorical -> label
y_val = np.argmax(y_val, axis = 1); # categorical -> label
y_train = np.argmax(y_train, axis = 1); # categorical -> label

img_num_train = X_train.shape[0]; # 
img_num_valid = X_val.shape[0]; # 
img_num_test = X_test.shape[0]; # 


""" Subj ID """

np.save(train_examID, X_train);
np.save(val_examID, X_val);
np.save(test_examID, X_test);


"""
--- BUILD TFRECORDS FILES ---

We can create memorize data into a TFRecord file and pass them to the CNN
during the training process using the TFRecordDataset API to process large
datasets that does not fit in memory.

The tf.data.TFRecordDataset class enables you to stream over the contents of 
ne of more TFRecord files as part of an input pipeline. 
We create three different TFRecord files:
    - train.tfrecord: Training set augmented
    - validation.tfrecord: Validation set
    - test.tfrecord: Test set

Images of training set augmented need to be shuffled in order to not provide 
blocks of the same image slightly modified. Shuffling cannot be performed in 
a multidimensional array containing the augmented dataset cause does not fit 
in memory. To do it so we create a matrix called augmentation_matrix where each 
row identy an image of the training set augmented and the column identify:
    
    - 1st col: Image's ID in X_train array
    - 2nd col: int which identify the type of transformation to apply to the
               image according to the following dictionary
               - 0: No transformation;
               - 1: Random rotation;
               - 2: Random translation;
               - 3: Random rototranslation;
               - 4: Random zoom.
    - 3rd col: Image's label
    
We shuffle matrix's rows and we write tfrecord files according to the order
specify in the matrix. 

"""

""" --- TRAINING SET --- """

aug_factor = 13;
augmentation_matrix = np.zeros((img_num_train*aug_factor, 3))
for num in range(img_num_train): 
    print('Immagine n°: ', str(num));
    # Images identifiers
    label = y_train[num];
    augmentation_matrix[aug_factor*num,:] = np.array([num, 0, label]);
    augmentation_matrix[aug_factor*num+1,:] = np.array([num, 1, label]);
    augmentation_matrix[aug_factor*num+2,:] = np.array([num, 1, label]);
    augmentation_matrix[aug_factor*num+3,:] = np.array([num, 1, label]);
    augmentation_matrix[aug_factor*num+4,:] = np.array([num, 2, label]);
    augmentation_matrix[aug_factor*num+5,:] = np.array([num, 2, label]);
    augmentation_matrix[aug_factor*num+6,:] = np.array([num, 2, label]);
    augmentation_matrix[aug_factor*num+7,:] = np.array([num, 3, label]);
    augmentation_matrix[aug_factor*num+8,:] = np.array([num, 3, label]);
    augmentation_matrix[aug_factor*num+9,:] = np.array([num, 3, label]);
    augmentation_matrix[aug_factor*num+10,:] = np.array([num, 4, label]);
    augmentation_matrix[aug_factor*num+11,:] = np.array([num, 4, label]);
    augmentation_matrix[aug_factor*num+12,:] = np.array([num, 4, label]);
    # Label 
augmentation_matrix = augmentation_matrix.astype(int);

# Random shuffle
rng = np.random.default_rng(random_seed);
shuffled_index = np.arange(img_num_train*aug_factor);
rng.shuffle(shuffled_index);
# Shuffled array 
augmentation_matrix = augmentation_matrix[shuffled_index,:]; 



# Data to features
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value)) 

writer = tf.io.TFRecordWriter(train_aug_filename)
i = 1;

# We cycle the augmentation_matrix to write the augmented training dataset 
# containing images and corresponding labels.

for num, aug_type, label in augmentation_matrix:
    
    volume_directory = X_train[num]; # 3D ndarray (image's directory)
    # PET volume, DICOM
    reader = sitk.ImageSeriesReader();
    dicom_names = reader.GetGDCMSeriesFileNames(volume_directory);
    reader.SetFileNames(dicom_names);
    volume = reader.Execute();
    volume = np.transpose(sitk.GetArrayFromImage(volume));
    print('Immagine ' + str(i) + '/' + str(img_num_train*aug_factor));

    # Augmentation transformation
    if aug_type == 1:
        volume = rotate(volume);
    elif aug_type == 2:
        volume = translate(volume);
    elif aug_type == 3:
        volume = rototrasl(volume);
    elif aug_type == 4:
        volume = zoomin(volume);
    else:
        volume = volume;

    label = to_categorical(label, num_classes=3);
    # Create a feature
    feature = {'train/image': _float_feature(volume.ravel()),
               'train/label': _float_feature(label)};
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature));
    # Serialize to string and write on the file
    writer.write(example.SerializeToString());
    i+=1;
    
writer.close()

print('Training saved!')


""" --- VALIDATION SET --- """

writer = tf.io.TFRecordWriter(val_filename)
i = 1;

for num in range(img_num_valid): 
    
    label = y_val[num];
    volume_directory = X_val[num]; 
    # PET volume, DICOM
    reader = sitk.ImageSeriesReader();
    dicom_names = reader.GetGDCMSeriesFileNames(volume_directory);
    reader.SetFileNames(dicom_names);
    volume = reader.Execute();
    volume = np.transpose(sitk.GetArrayFromImage(volume));

    label = to_categorical(label, num_classes=3);
    # Create a feature
    feature = {'validation/image': _float_feature(volume.ravel()),
               'validation/label': _float_feature(label)};
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature));
    # Serialize to string and write on the file
    writer.write(example.SerializeToString());
    i+=1;
    
writer.close()


""" --- TEST SET --- """

writer = tf.io.TFRecordWriter(test_filename)
i=1;

for num in range(img_num_test): 
    
    label = y_test[num];
    volume_directory = X_test[num]; 
    # PET volume, DICOM
    reader = sitk.ImageSeriesReader();
    dicom_names = reader.GetGDCMSeriesFileNames(volume_directory);
    reader.SetFileNames(dicom_names);
    volume = reader.Execute();
    volume = np.transpose(sitk.GetArrayFromImage(volume));
    label = to_categorical(label, num_classes=3);
    # Create a feature

    feature = {'test/image': _float_feature(volume.ravel()),
               'test/label': _float_feature(label)};
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature));
    # Serialize to string and write on the file
    writer.write(example.SerializeToString());
    i+=1;
    
writer.close()
































































