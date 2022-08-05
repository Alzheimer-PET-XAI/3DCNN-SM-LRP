<h1> An Explainable Convolutional Neural Network for the Early Diagnosis of Alzheimer's Disease from 18F-FDG PET </h1>

In our work we developed, trained and tested a 3D Convolutional Neural Network (CNN) which takes as input volumetric Brain 18F-FDG PET images to predict the clinical diagnosis in Alzheimer's Disease. The CNN perform a multiclass classification task (n° classes = 3); the output returned can be: Cognitive Normal (CN), Mild Cognitive Impairment (MCI) and Alzheimer's Disease (AD).
The entire implementation was performed using Python programming language (v. 3.9). As Machine Learning library we employed Keras framework on Tensorflow backend (v. 2.6.0).

<h2> Data </h2> 
The FDG PET scans employed were obtained from the Alzheimer Disease Neuroimaging Initiative (ADNI), data can be downloaded at http://adni.loni.usc.edu/ after applying for the access.

The ADNI was launched in 2003 as a public-private partnership, led by Principal Investigator Michael W. Weiner, MD. The primary goal of ADNI has been to test whether serial magnetic resonance imaging (MRI), positron emission tomography (PET), other biological markers, and clinical and neuropsychological assessment can be combined to measure the progression of mild cognitive impairment (MCI) and early Alzheimer’s disease (AD).

We downloaded an amout of 2552 images pre-processed by ADNI team. We selected collection with the higher level of pre-processing (Coreg, Avg, Std Img and Vox Siz, Uniform Resolution), image size: 160x160x96 (single channel).

<h2> Files </h2> 
With the aim to promote reproducibility and open-accessible results for other researchers and practictioners, we provide all the Python codes and CNN's weights obtained during one training sessions.
<ul>
  <li> <em> cnn3D_weights.h5 </em>: model's weight obtained during one of the training session (kfold #1)</li>
  <li>Second item</li>
  <li>Third item</li>
  <li>Fourth item</li>
</ul>
