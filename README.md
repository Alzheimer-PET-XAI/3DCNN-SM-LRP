# 3DCNN-SM-LRP
In our work we developed, trained and tested a 3D Convolutional Neural Network (CNN) which takes as input volumetric Brain 18F-FDG PET images to predict the clinical diagnosis in Alzheimer's Disease. The CNN perform a multiclass classification task (n° classes = 3); the output returned can be: Cognitive Normal (CN), Mild Cognitive Impairment (MCI) and Alzheimer's Disease (AD).

# Data 
The FDG PET scans employed were obtained from the Alzheimer Disease Neuroimaging Initiative (ADNI), data can be downloaded at http://adni.loni.usc.edu/ after applying for the access.

The ADNI was launched in 2003 as a public-private partnership, led by Principal Investigator Michael W. Weiner, MD. The primary goal of ADNI has been to test whether serial magnetic resonance imaging (MRI), positron emission tomography (PET), other biological markers, and clinical and neuropsychological assessment can be combined to measure the progression of mild cognitive impairment (MCI) and early Alzheimer’s disease (AD).

# Model's weights
The model's weight obtained during one of the training session (kfold #1) has been provided in the folder as 'cnn3D_weights.h5' to test CNN's performances.
