# Project: Implementation of Iris Recognition Algorithm 
### Code lib Folder

The lib directory contains various files with function definitions (but only function definitions - no code that actually runs).


1. Logic and Scripts:

- 'Main.ipynb': Final presentation jupyter notebook for checking results of whole project.

- 'PerformanceEvaluation.py': Create two tables and one CRR curve plot function for visualization.

-  'IrisLocalization.py': Localize the Iris and pupil part, and do Iris segmentation.

- 'IrisNormalization.py': Normalized the segmented Iris ring into a 64*512 rectangle image.

-  'ImageEnhancement.py': Extract the background of the normalized image (optional, not used in this project), and do histogram equalization for each 8*8 blocks.

-  'FeatureExtraction.py': Apply designed filter and convolutional method to filter the ROI(region of interest: 48*512), and then extract a vector of features(1536) by calculating mean and absolute standard deviation for each 8*8 block with 2 different channels. 

-  'IrisMatching.py': Get all the labels & filenames for both training and testing data; extracting all the feature vectors for all the training and testing data; define the nearest center classifier by 3 different similarity distance metrics (L1, L2, cosine distance).

-  'drawIrisLocalization.py': Visualize the iris segmentation effects for all the images(both testing and training).
