# cnn-svm-classifier

The images are fed to a TensorFlow implementation of Inception V3 with the classification layer removed in order to produce a set of labelled feature vectors.

Dimensionality reduction is carried out on the 2048-d features using t-distributed stochastic neighbor embedding (t-SNE) to transform them into a 2-d feature which is easy to visualize. Note that t-SNE is used as an informative step. If the same color/label points are mostly clustered together there is a high chance that we could use the features to train a classifier with high accuracy.

The 2048-d labelled features are presented to a number of classifiers.
The comparison has been extended to the following:

* Support Vector Machine (SVM)
* Extra Trees (ET)
* Random Forest (RF)
* K-Nearest Neighbor (KNN)
* Multi-Layer Perceptron (ML)
* Gaussian Naive Bayes (GNB)

Training and validation time, and the accuracy of each classifier is displayed. Most classifiers were run with their default tuning values,however tuning was carried, where possible.


## Quick Start


1. The imagenet directory already has graph.pb. Download it from
http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz,un-zip it, and place classify_image_graph_def.pb in a directory called 'imagenet'.

2. Run main.py using Python 3. The following packages are required: tensorflow, sklearn (scikit-learn), numpy, matplotlib.