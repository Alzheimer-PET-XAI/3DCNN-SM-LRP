#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:15:44 2022

@author: lisadesanti
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math
from skimage import transform
from sklearn.cluster import KMeans
from skimage import measure



def get_model_CNN1(width, height, depth, num_classes=3):
    
    """  
    Define 3D CNN Network Model
        
    FeaturesExtractor -> Global Avg Pooling -> Fully Connected -> Classifier
    ....
    
    
    Attributes
    --------------------
    width: int
        Input tensor width, conv_dim1
    height: int
        Input tensor height, conv_dim2
    depth: int
        Input tensor depth, conv_dim3
    num_classes: int
        Number of classes
    
    ....

    Output
    ------
    Return: keras.Model()
    
    ....
    
    NOTE: 
        
    Input shape: 
        5D tensor (data_format = "channel_last")
        batch_dim + (conv_dim1, conv_dim2, conv_dim3, channels)
        
    Output classes:
        - CN, Congnitively Normal
        - MCI, Mild Cognitive Impairment
        - AD, Alzheiemer's Disease
    ....
    
    
    """
    
    regL2 = tf.keras.regularizers.l2(l2=1e-2);

    inputs = keras.Input((width, height, depth, 1))
    # ------------------------------------------------------------------------
    x = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding="same", kernel_regularizer=regL2)(inputs)
    x = layers.Conv3D(filters=8, kernel_size=3, activation="relu", padding="same", kernel_regularizer=regL2)(x)
    x = layers.MaxPool3D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)
    # ------------------------------------------------------------------------
    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu", padding="same", kernel_regularizer=regL2)(x)
    x = layers.Conv3D(filters=16, kernel_size=3, activation="relu", padding="same", kernel_regularizer=regL2)(x)
    x = layers.MaxPool3D(pool_size=3)(x)
    x = layers.BatchNormalization()(x)
    # ------------------------------------------------------------------------
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding="same", kernel_regularizer=regL2)(x)
    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu", padding="same", kernel_regularizer=regL2)(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    # ------------------------------------------------------------------------
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding="same", kernel_regularizer=regL2)(x)
    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding="same", kernel_regularizer=regL2)(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    # ------------------------------------------------------------------------
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=64, activation="relu", kernel_regularizer=regL2)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(units=32, activation="relu", kernel_regularizer=regL2)(x)
    x = layers.Dropout(0.3)(x)
    # ------------------------------------------------------------------------
    outputs = layers.Dense(units=num_classes, activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn");
    return model



def plot_slices(data, num_columns=10, cmap="gray", title=False, data_min=False, data_max=False):
    
    """
    Plot a merge of input volume's slices.
    ....
    
    
    Attributes
    --------------------
    data: numpy.ndarray
        3D array to plot
    num_columns: int
        Number of columns of the merged plot
    cmap: String
        Colormap chosen by matplotlib reference
    title: String
        Suptitle of the figure
    data_min, data_max: float
        Colorbar range, 
        If False (default) adapt colorbar's dynamic to that one of the input  
    
    
    """
    
    
    width = data.shape[0];
    height = data.shape[1];
    depth = data.shape[2];
    
    if not(data_min) or not(data_max):
        data_min = data.min();
        data_max = data.max();
    
    r, num_rows = math.modf(depth/num_columns);
    num_rows = int(num_rows);
    if num_rows == 0:
        num_columns = int(r*num_columns);
        num_rows +=1;
        r = 0;
    elif r > 0:
        new_im = int(num_columns-(depth-num_columns*num_rows));
        add = np.zeros((width,height,new_im),dtype=type(data[0,0,0]));
        data = np.dstack((data,add));
        num_rows +=1;
    data = np.rot90(np.array(data))
    data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    );
    
    if title:
        f.suptitle(title, fontsize=30);
        
    for i in range(rows_data):
        for j in range(columns_data):
            if rows_data > 1:
                img = axarr[i, j].imshow(data[i][j], cmap=cmap,
                                         vmin=data_min, vmax=data_max);
                axarr[i, j].axis("off");
            else:
                img = axarr[j].imshow(data[i][j], cmap=cmap,
                                      vmin=data_min, vmax=data_max);
                axarr[j].axis("off");

    plt.subplots_adjust(wspace=0, hspace=0, left=0, right = 0.9, bottom=0, top=0.9);
    cbar_ax = f.add_axes([0.92, 0, 0.015, 0.9])         
    f.colorbar(img, cax=cbar_ax);
    plt.show()
    


def test_performance(CM):
    
    """
    Evaluate some useful evaluation metrics of classifier's performances.
    
    NOTE: This function has been specifically designed for the classification 
    task implemented.
    
    
    Attributes
    --------------------
    CM: numpy.ndarray of shape (n_classes, n_classes)
        Confusion Matrix evaluated
        
    ....
    
    Output
    ------
        tuple of dictionaries
        Each dictionay contains the performances evaluated for the specific 
        class (using a one vs all strategy).
        
        The function computes:
            - Accuracy
            - True Positive Rate (Sensitivity)
            - True Negative Rate (Specificity)
            - False Positive Rate 
            - Precision
            - Negative Predictive Value
            - F1 score
            - Matthew Correlation Coefficient
    
    """
    
    Ntot = CM.sum();
    
    # CN vs. AD, MCI
    tp_cn = CM[0,0];
    fp_cn = CM[1,0] + CM[2,0];
    tn_cn = CM[1,1] + CM[1,2] + CM[2,1] + CM[2,2];
    fn_cn = CM[0,1] + CM[0,2];
    acc_cn = (tp_cn + tn_cn)/Ntot;
    tpr_cn = tp_cn/(tp_cn + fn_cn);
    tnr_cn = tn_cn/(tn_cn + fp_cn);
    fpr_cn = 1 - tnr_cn;
    bal_acc_cn = (tpr_cn + tnr_cn)/2;
    ppv_cn = tp_cn/(tp_cn + fp_cn);
    npv_cn = tn_cn/(tn_cn + fn_cn);
    f1_cn = 2*tpr_cn*ppv_cn/(tpr_cn+ppv_cn);
    mcc_cn = (tp_cn*tn_cn-fp_cn*fn_cn)/math.sqrt((tp_cn+fp_cn)*(tp_cn+fn_cn)*(tn_cn+fp_cn)*(tn_cn+fn_cn));
    performance_cn = {"Accuracy CN":acc_cn,
                      "True Positive Rate (Sensitivity) CN":tpr_cn,
                      "True Negative Rate (Specificity) CN":tnr_cn,
                      "False Positive Rate CN":fpr_cn,
                      "Balanced Accuracy CN":bal_acc_cn,
                      "Precision CN":ppv_cn,
                      "Negative Predictive Value CN":npv_cn,
                      "F1 score":f1_cn,
                      "Matthew Correlation Coefficient":mcc_cn};
    
    # MCI vs. AD, CN
    tp_mci = CM[1,1];
    fp_mci = CM[0,1] + CM[2,1];
    tn_mci = CM[0,0] + CM[0,2] + CM[2,0] + CM[2,2];
    fn_mci = CM[1,0] + CM[1,2];
    acc_mci = (tp_mci + tn_mci)/Ntot;
    tpr_mci = tp_mci/(tp_mci + fn_mci);
    tnr_mci = tn_mci/(tn_mci + fp_mci);
    fpr_mci = 1 - tnr_mci;
    bal_acc_mci = (tpr_mci + tnr_mci)/2;
    ppv_mci = tp_mci/(tp_mci + fp_mci);
    npv_mci = tn_mci/(tn_mci + fn_mci);
    f1_mci = 2*tpr_mci*ppv_mci/(tpr_mci+ppv_mci);
    mcc_mci = (tp_mci*tn_mci-fp_mci*fn_mci)/math.sqrt((tp_mci+fp_mci)*(tp_mci+fn_mci)*(tn_mci+fp_mci)*(tn_mci+fn_mci));
    performance_mci = {"Accuracy MCI":acc_mci,
                      "True Positive Rate (Sensitivity) MCI":tpr_mci,
                      "True Negative Rate (Specificity) MCI":tnr_mci,
                      "False Positive Rate MCI":fpr_mci,
                      "Balanced Accuracy MCI":bal_acc_mci,
                      "Precision MCI":ppv_mci,
                      "Negative Predictive Value MCI":npv_mci,
                      "F1 score":f1_mci,
                      "Matthew Correlation Coefficient":mcc_mci};
    
    # AD vs. MCI, CN
    tp_ad = CM[2,2];
    fp_ad = CM[0,2] + CM[1,2];
    tn_ad = CM[0,0] + CM[0,1] + CM[1,0] + CM[1,1];
    fn_ad = CM[2,0] + CM[2,1];
    acc_ad = (tp_ad + tn_ad)/Ntot;
    tpr_ad = tp_ad/(tp_ad + fn_ad);
    tnr_ad = tn_ad/(tn_ad + fp_ad);
    fpr_ad = 1 - tnr_ad;
    bal_acc_ad = (tpr_ad + tnr_ad)/2;
    ppv_ad = tp_ad/(tp_ad + fp_ad);
    npv_ad = tn_ad/(tn_ad + fn_ad);
    f1_ad = 2*tpr_ad*ppv_ad/(tpr_ad+ppv_ad);
    mcc_ad = (tp_ad*tn_ad-fp_ad*fn_ad)/math.sqrt((tp_ad+fp_ad)*(tp_ad+fn_ad)*(tn_ad+fp_ad)*(tn_ad+fn_ad));
    performance_ad = {"Accuracy AD":acc_ad,
                      "True Positive Rate (Sensitivity) AD":tpr_ad,
                      "True Negative Rate (Specificity) AD":tnr_ad,
                      "False Positive Rate AD":fpr_ad,
                      "Balanced Accuracy AD":bal_acc_ad,
                      "Precision AD":ppv_ad,
                      "Negative Predictive Value AD":npv_ad,
                      "F1 score":f1_ad,
                      "Matthew Correlation Coefficient":mcc_ad};
    
    return performance_cn, performance_mci, performance_ad

















































