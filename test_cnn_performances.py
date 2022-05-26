#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:15:00 2022

@author: lisadesanti


Test Classification's performaces of the trained network in the test set.

The FDG PET scans employed were obtained from the Alzheimer Disease 
Neuroimaging Initiative (ADNI), data can be downloaded at
 http://adni.loni.usc.edu/ after applying for the access.
 
The ADNI was launched in 2003 as a public-private partnership, led by Principal 
Investigator Michael W. Weiner, MD. The primary goal of ADNI has been to test 
whether serial magnetic resonance imaging (MRI), positron emission tomography 
(PET), other biological markers, and clinical and neuropsychological assessment 
can be combined to measure the progression of mild cognitive impairment (MCI) 
and early Alzheimer’s disease (AD).

The model's weight obtained during one of the training session has been 
provided in the folder.

"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import auc, roc_auc_score, roc_curve
from itertools import cycle
from utilities_function import get_model_CNN, test_performance




classes = {0:"CN", 1:"MCI", 2:"AD"};
num_classes = len(classes); # 3
img_depth = 96; # depth
img_width = 160; # width
img_height = 160; # heigth
img_num_test = 511; # images in the test set
test_set_directory = "INSERT_YOUR_OWN_DIRECTORY";
weights_directory = "INSERT_YOUR_OWN_DIRECTORY";



""" Parsing TFRecord files """

# Converto i dati in features
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# Utilizzo i file tfrecord creati per creare il dataset
# Carico i TFRecord
# Decodifico i dati

def decode_test(serialized_example):
    # Decode examples stored in TFRecord
    # NOTE: make sure to specify the correct dimensions for the images
    features = tf.io.parse_single_example(
        serialized_example,
        features = {'test/image': tf.io.FixedLenFeature([img_width*img_height*img_depth,], tf.float32),
                    'test/label': tf.io.FixedLenFeature([3,], tf.float32)})
    # NOTE: No need to cast these features, as they are already `tf.float32` values.
    return features['test/image'], features['test/label']



""" Preprocessing TFRecord dataset """

def dataset_preprocessing(volume, label, w = img_width, h = img_height, 
                        d = img_depth):
    """
    Process training data by:
        - Reshaping vectorized images
        - Adding a channel to perform 3D convolutions on the data
    """
    volume = tf.reshape(volume,[w,h,d]);
    volume = tf.expand_dims(volume, axis = 3);
    return volume, label

          
                                   
""" Build 3DCNN Model """

# Build model
model = get_model_CNN(width = img_width, height = img_height, depth = img_depth);
model.summary();

# Compile model
model.compile(
    loss="categorical_crossentropy",
    metrics=["acc"]
);



""" Test Neural Network Performance """

# Test set
test_dataset = tf.data.TFRecordDataset(test_set_directory);
test_dataset = test_dataset.map(decode_test);
test_dataset = test_dataset.map(dataset_preprocessing);
test_dataset = test_dataset.batch(1);
test_dataset = test_dataset.prefetch(2);

# Load best weights.
model.load_weights(weights_directory);

# Test Set
# Accuracy Test set
test_loss, test_accuracy = model.evaluate((test_dataset));
print("Test accuracy: {:.2f}%".format(test_accuracy*100));


# CNN Prediction 
y_pred_test_prob = model.predict(test_dataset); # label predicted
y_pred_test = np.argmax(y_pred_test_prob, axis = 1); # label predicted: probabilities -> label

# Real Label: Categorical
y_test = np.zeros((img_num_test, 3)); # real label: categorical (read from TFRecord file)
test_label = np.zeros((img_num_test, 3));
iter_test = iter(test_dataset);


for i in range(img_num_test):
    print('Immagine ' + str(i) + '/' + str(img_num_test));
    tensor_volume, tensor_label = next(iter_test);
    img = tensor_volume.numpy();
    print('max: ' + str(img.max()) + ' min: ' + str(img.min()));
    label = tensor_label.numpy();
    img = img[0,:,:,:,0];
    label = tensor_label.numpy();
    y_test[i,:] = label;
    test_label[i,:] = label;
    
freq_class_test = test_label.sum(axis=0).astype(int); # images in each classes 

# AUC
roc_test = roc_auc_score(y_test, y_pred_test_prob, multi_class='ovr', average=None);

# Compute ROC curve and ROC area for each class
fpr = dict();
tpr = dict();
roc_auc = dict();
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_test_prob[:, i]);
    roc_auc[i] = auc(fpr[i], tpr[i]);

# Plot all ROC curves
lw = 2;
plt.figure();
colors = cycle(["aqua", "darkorange", "cornflowerblue"]);
for i, color in zip(range(num_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw = lw,
        label = "ROC curve of class {0} (area = {1:0.2f})".format(classes[i], roc_auc[i]),
    );

plt.plot([0, 1], [0, 1], "k--", lw=lw);
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel("False Positive Rate");
plt.ylabel("True Positive Rate");
plt.title("ROC multiclass curves");
plt.legend(loc="lower right");
plt.grid(True);
plt.show();

# Real Label: Label
y_test = np.argmax(y_test, axis = 1); # real label: categorical -> label

# Confusion Matrix
ConfusionMatrix = confusion_matrix(y_test, y_pred_test);
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test);
performance_cn, performance_mci, performance_ad = test_performance(ConfusionMatrix);

print(roc_test)
print(performance_cn)
print(performance_mci)
print(performance_ad)
