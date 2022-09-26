#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:38:25 2022

@author: 

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

import numpy as np
import matplotlib.pyplot as plt

import pandas
import tensorflow as tf
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split, StratifiedKFold
import SimpleITK as sitk

import random
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from skimage import transform
from utilities_function import list_ADNI_directories


"""
Global Variables

"""

tot_trial = 5;
k_fold = 5;

n_trial = 1; # change from 1 to tot_trial;
random_seed = n_trial*42; # seed for reproducible shuffling
n_fold = 5; # change from 1 to k_fold

classes = {0:"CN", 1:"MCI", 2:"AD"};
dic_classes = {"CN":0, "MCI":1, "AD":2};
augmentation = {0:"Original", 1:"Rotate", 2:"Translate", 3:"Rototrasl", 4:"Zoom"};

root_dir = ""; # INSERT YOUR DIRECTORY
adni_file = ""; # INSERT YOUR DIRECTORY

# paths to save exams ID
train_aug_filename = root_dir + 'train.tfrecords';
val_filename = root_dir + 'validation.tfrecords';
test_filename = root_dir + 'test.tfrecords';

results = "/RISULTATI_ADNI_completo_review1";

# paths to save the TFRecords file (augmented)
train_examID = root_dir + results + '/ExamsID/exam_ID_train_kfold_' + str(n_fold);
val_examID = root_dir + results + '/ExamsID/exam_ID_val_kfold_' + str(n_fold);
test_examID = root_dir + results + '/ExamsID/exam_ID_test_kfold_' + str(n_fold);

img_depth = 96; # depth
img_width = 160; # width
img_height = 160; # heigth
test_split = 0.2;
val_split = 0.2;


"""
DATA AUGMENTATION 
Functions:
    - Random rotation [-10°; +10°];
    - Random translation x/y/z;
    - Random rototranslation;
    - Random zoom.

Data Augmentation MUST be applied only on the Training set!

"""

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

Load ADNI directories 

"""

dicom_directories_dic = list_ADNI_directories(adni_file, img_depth);
img_num = len(dicom_directories_dic);

# Dataset of ADNI directories
directory_dataframe = pandas.DataFrame(dicom_directories_dic);
directory_dataframdic_classese = directory_dataframe.sort_values("SUBJ");
all_subj = list(set(directory_dataframe["SUBJ"]));
label = directory_dataframe["LABEL"];


# Split Dataset into Training(+ Valid) and Test set (using kfold)
skf = StratifiedKFold(n_splits=k_fold, random_state=None, shuffle=False);
kfold_generator = skf.split(directory_dataframe, label);

for i in range(n_fold):
    
     train_val_index, test_index = next(kfold_generator);
     X_train_val_df, X_test_df = directory_dataframe.loc[train_val_index], directory_dataframe.loc[test_index];
     y_train_val, y_test = label[train_val_index], label[test_index];
     
     # Check to not have data (exams) from the same subjects both in the 
     # training and test sets
     subj_train_val = np.array(X_train_val_df["SUBJ"]);
     subj_test = np.array(X_test_df["SUBJ"]);
     dup_subjects = np.intersect1d(subj_train_val, subj_test);
     
     # If a subjects has data in both sets move data to the training set
     # (this is an arbitrary choice)
     for dup_subj in dup_subjects:
         
         dup_subj_test = X_test_df.loc[X_test_df["SUBJ"]==dup_subj];
         id_dup_subj_test = np.array(dup_subj_test.index);
         to_train_val = X_test_df.loc[id_dup_subj_test];
         
         # Test set (without duplicated subjects)
         X_test_df = X_test_df.drop(id_dup_subj_test);
         X_test_df = X_test_df.sort_values("SUBJ");
         y_test = X_test_df["LABEL"];
         # Training+Validation set (without duplicated subjects)
         X_train_val_df = X_train_val_df.append(to_train_val);
         X_train_val_df = X_train_val_df.sort_values("SUBJ");
         y_train_val = X_train_val_df["LABEL"];


# Split into Training and Validation set
# select last 20% of the dataset 

X_train_df, X_val_df, y_train, y_val = train_test_split(
    X_train_val_df, y_train_val, test_size=val_split, shuffle=False, 
    random_state=None, stratify=None); # with shuffle False stratify is not support

# Check to not have data (exams) from the same subjects both in the 
# training and validation sets
subj_train = np.array(X_train_df["SUBJ"]);
subj_val = np.array(X_val_df["SUBJ"]);
dup_subjects = np.intersect1d(subj_train, subj_val);

# If a subjects has data in both sets move data to the training set
# (this is an arbitrary choice)
for dup_subj in dup_subjects:
    
    dup_subj_val = X_val_df.loc[X_val_df["SUBJ"]==dup_subj];
    id_dup_subj_val = np.array(dup_subj_val.index);
    to_train = X_val_df.loc[id_dup_subj_val];
    
    # val set (without duplicated subjects)
    X_val_df = X_val_df.drop(id_dup_subj_val);
    X_val_df = X_val_df.sort_values("SUBJ");
    y_val = X_val_df["LABEL"];
    # Training+Validation set (without duplicated subjects)
    X_train_df = X_train_df.append(to_train);
    X_train_df = X_train_df.sort_values("SUBJ");
    y_train = X_train_df["LABEL"];

n_train = len(X_train_df);
n_val = len(X_val_df);
n_test = len(X_test_df);
print('Training: ' + str(n_train))
print('Validation: ' + str(n_val))
print('Test: ' + str(n_test))

X_train = np.array(X_train_df['ROOT']) + np.array(X_train_df['LABEL']) + np.array(['/']*n_train) + np.array(X_train_df['SUBJ']) + np.array(['/']*n_train) + np.array(X_train_df['PREPROC']) + np.array(['/']*n_train)+ np.array(X_train_df['DATE']) + np.array(['/']*n_train)+ np.array(X_train_df['EXAM_ID']);
y_train = np.array(y_train);
y_train = np.array([dic_classes[yi] for yi in y_train]);
X_val = np.array(X_val_df['ROOT']) + np.array(X_val_df['LABEL']) + np.array(['/']*n_val) + np.array(X_val_df['SUBJ']) + np.array(['/']*n_val) + np.array(X_val_df['PREPROC']) + np.array(['/']*n_val)+ np.array(X_val_df['DATE']) + np.array(['/']*n_val)+ np.array(X_val_df['EXAM_ID']);
y_val = np.array(y_val);
y_val = np.array([dic_classes[yi] for yi in y_val]);
X_test = np.array(X_test_df['ROOT']) + np.array(X_test_df['LABEL']) + np.array(['/']*n_test) + np.array(X_test_df['SUBJ']) + np.array(['/']*n_test) + np.array(X_test_df['PREPROC']) + np.array(['/']*n_test)+ np.array(X_test_df['DATE']) + np.array(['/']*n_test)+ np.array(X_test_df['EXAM_ID']) ;
y_test = np.array(y_test);
y_test = np.array([dic_classes[yi] for yi in y_test]);


# Check how many different subjects there are for each class in Training set
subj_train = (X_train_df[["SUBJ","LABEL"]]);

subj_train_cn = np.array(subj_train.loc[subj_train["LABEL"]=="CN"]);
subj_train_cn = np.array(subj_train_cn[:,0]).tolist();
no_mult_subj_train_cn = set(subj_train_cn);
print(len(subj_train_cn))
print(len(no_mult_subj_train_cn))
subj_train_occurence_cn = {i:subj_train_cn.count(i) for i in subj_train_cn};
fig, ax = plt.subplots(figsize=(15,6));
plt.bar(range(len(subj_train_occurence_cn)), list(subj_train_occurence_cn.values()), align='center');
num_rep_cn = set(subj_train_occurence_cn.values());
occurrance_cn = list(subj_train_occurence_cn.values());
num_occurence_cn = {i:occurrance_cn.count(i) for i in occurrance_cn};
fig, ax = plt.subplots(figsize=(15,6));
plt.bar(range(max(num_occurence_cn.keys())), [num_occurence_cn[i+1] for i in range(max(num_occurence_cn.keys()))], align='center');

subj_train_mci = np.array(subj_train.loc[subj_train["LABEL"]=="MCI"]);
subj_train_mci = np.array(subj_train_mci[:,0]).tolist();
no_mult_subj_train_mci = set(subj_train_mci);
print(len(subj_train_mci))
print(len(no_mult_subj_train_mci))
subj_train_occurence_mci = {i:subj_train_mci.count(i) for i in subj_train_mci};
fig, ax = plt.subplots(figsize=(15,6));
plt.bar(range(len(subj_train_occurence_mci)), list(subj_train_occurence_mci.values()), align='center');
num_rep_mci = set(subj_train_occurence_mci.values());
occurrance_mci = list(subj_train_occurence_mci.values());
num_occurence_mci = {i:occurrance_mci.count(i) for i in occurrance_mci};
fig, ax = plt.subplots(figsize=(15,6));
plt.bar(range(max(num_occurence_mci.keys())), [num_occurence_mci[i+1] for i in range(max(num_occurence_mci.keys()))], align='center');

subj_train_ad = np.array(subj_train.loc[subj_train["LABEL"]=="AD"]);
subj_train_ad = np.array(subj_train_ad[:,0]).tolist();
no_mult_subj_train_ad = set(subj_train_ad);
print(len(subj_train_ad))
print(len(no_mult_subj_train_ad))
subj_train_occurence_ad = {i:subj_train_ad.count(i) for i in subj_train_ad};
fig, ax = plt.subplots(figsize=(15,6));
plt.bar(range(len(subj_train_occurence_ad)), list(subj_train_occurence_ad.values()), align='center');
num_rep_ad = set(subj_train_occurence_ad.values());
occurrance_ad = list(subj_train_occurence_ad.values());
num_occurence_ad = {i:occurrance_ad.count(i) for i in occurrance_ad};
fig, ax = plt.subplots(figsize=(15,6));
plt.bar(range(max(num_occurence_ad.keys())), [num_occurence_ad[i+1] for i in range(max(num_occurence_ad.keys()))], align='center');


# Augmentation factor used pre-review
aug_factor = 13;
Naug = n_train*aug_factor;

# Apply a different augmentation factor according to the the class
Nex_cn = len(subj_train_cn);
Nsubj_cn = len(no_mult_subj_train_cn);
Nex_mci = len(subj_train_mci);
Nsubj_mci = len(no_mult_subj_train_mci);
Nex_ad = len(subj_train_ad);
Nsubj_ad = len(no_mult_subj_train_ad);
Nsubj = Nsubj_cn + Nsubj_mci + Nsubj_ad;
# Determine the augmentation factor
aug_cn = round(Naug*Nsubj_cn/(Nex_cn*Nsubj));
aug_mci = round(Naug*Nsubj_mci/(Nex_mci*Nsubj));
aug_ad = round(Naug*Nsubj_ad/(Nex_ad*Nsubj));
Naug = Nex_cn*aug_cn + Nex_mci*aug_mci + Nex_ad*aug_ad; # update (due to rounding could be slightly different)
aug_dic = {0:aug_cn, 1:aug_mci, 2:aug_ad};

# Data shuffling 

rng = np.random.default_rng(random_seed);
shuffled_index = np.arange(n_train);
rng.shuffle(shuffled_index);
X_train = X_train[shuffled_index]; # Shuffled dataset
y_train = y_train[shuffled_index];
y_train = to_categorical(y_train, num_classes=3); # Label -> Categorical 

rng = np.random.default_rng(random_seed);
shuffled_index = np.arange(n_val);
rng.shuffle(shuffled_index);
X_val = X_val[shuffled_index]; # Shuffled dataset
y_val = y_val[shuffled_index];
y_val = to_categorical(y_val, num_classes=3); # Label -> Categorical 

rng = np.random.default_rng(random_seed);
shuffled_index = np.arange(n_test);
rng.shuffle(shuffled_index);
X_test = X_test[shuffled_index]; # Shuffled dataset
y_test = y_test[shuffled_index];
y_test = to_categorical(y_test, num_classes=3); # Label -> Categorical 

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


""" Save subject ID """

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

# """ --- TRAINING SET --- """

augmentation_matrix = np.zeros((Naug, 3));
count = 0;
for num in range(n_train): 
    print('Immagine n°: ', str(num));
    # Immagini modificate
    label = y_train[num];
    # check class and according to the class apply a different augmentation factor
    augmentation = aug_dic[label];
    augmentation_matrix[count,:] = np.array([num, 0, label]); # original image
    count += 1;
    for i in range(augmentation-1):
        # choose a random augmentation type
        aug_type = random.choice([1,2,3,4]);
        augmentation_matrix[count,:] = np.array([num, aug_type, label]);
        count += 1;
augmentation_matrix = augmentation_matrix.astype(int);

# Mescolo le righe dell'array2D
# Random shuffle
rng = np.random.default_rng(random_seed);
shuffled_index = np.arange(Naug);
rng.shuffle(shuffled_index);
# Shuffled array 
augmentation_matrix = augmentation_matrix[shuffled_index,:]; 



# # Converto i dati in features
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value)) 

writer = tf.io.TFRecordWriter(train_aug_filename)
i = 1;

for num, aug_type, label in augmentation_matrix:
    
    # Carico l'immagine dal dataset di training
    volume_directory = X_train[num]; # 3D ndarray
    # PET volume, DICOM
    reader = sitk.ImageSeriesReader();
    dicom_names = reader.GetGDCMSeriesFileNames(volume_directory);
    reader.SetFileNames(dicom_names);
    volume = reader.Execute();
    volume = np.transpose(sitk.GetArrayFromImage(volume));
    print('Immagine ' + str(i) + '/' + str(n_train*aug_factor));

    # Applico la trasformazione all'immagine
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
    
    # Converto la label in variabile categoriale
    label = to_categorical(label, num_classes=3);
    # Create a feature
    # Nella feature memorizzero l'immagine vettorizzata, la label ed il numero
    # che identifica il tipo di trasformazione applicata all'immagine.
    feature = {'train/image': _float_feature(volume.ravel()),
                'train/label': _float_feature(label)};
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature));
    # Serialize to string and write on the file
    writer.write(example.SerializeToString());
    i+=1;
    
writer.close()

print('Training set salvato!')


""" --- VALIDATION SET --- """

writer = tf.io.TFRecordWriter(val_filename)
i = 1;

for num in range(n_val): 
    
    print('Immagine ' + str(num) + '/' + str(n_val));
    
    label = y_val[num];
    # Carico l'immagine dal dataset di training
    volume_directory = X_val[num]; # 3D ndarray
    # PET volume, DICOM
    reader = sitk.ImageSeriesReader();
    dicom_names = reader.GetGDCMSeriesFileNames(volume_directory);
    reader.SetFileNames(dicom_names);
    volume = reader.Execute();
    volume = np.transpose(sitk.GetArrayFromImage(volume));
    
    # Converto la label in variabile categoriale
    label = to_categorical(label, num_classes=3);
    # Create a feature
    # Nella feature memorizzero l'immagine vettorizzata, la label ed il numero
    # che identifica il tipo di trasformazione applicata all'immagine.
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

for num in range(n_test): 
    
    print('Immagine ' + str(num) + '/' + str(n_test));
    
    label = y_test[num];
    # Carico l'immagine dal dataset di training
    volume_directory = X_test[num]; # 3D ndarray
    # PET volume, DICOM
    reader = sitk.ImageSeriesReader();
    dicom_names = reader.GetGDCMSeriesFileNames(volume_directory);
    reader.SetFileNames(dicom_names);
    volume = reader.Execute();
    volume = np.transpose(sitk.GetArrayFromImage(volume));
    
    # Converto la label in variabile categoriale
    label = to_categorical(label, num_classes=3);
    # Create a feature
    # Nella feature memorizzero l'immagine vettorizzata, la label ed il numero
    # che identifica il tipo di trasformazione applicata all'immagine.
    feature = {'test/image': _float_feature(volume.ravel()),
               'test/label': _float_feature(label)};
    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature));
    # Serialize to string and write on the file
    writer.write(example.SerializeToString());
    i+=1;
    
writer.close()

