# Venice Boat Classification

**Note:** This README provides a simple overview of the homework. The report is available [here](Homework2/report).

### Overview 
The images are fed to a TensorFlow implementation of Inception v3 with the classification layer removed in order to produce a set of labelled feature vectors.

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

### Requirements

The following packages are required:
1. tensorflow
2. sklearn (scikit-learn)
3. numpy
4. matplotlib.

### How to run

1. Download [a pretrained model](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz,un-zip) and place it as 'graph.pb' in a directory called 'pretrained'.

2. Download the [training](http://www.dis.uniroma1.it/~labrococo/MAR/classifier/sc5-2013-Mar-Apr.rar) and [test](http://www.dis.uniroma1.it/~labrococo/MAR/classifier/sc5-2013-Mar-Apr-Test-20130412.rar) images of [MarDCT](http://www.dis.uniroma1.it/~labrococo/MAR) for boat classification. 

3. Run the main program

    main.py

### Results
The output images are stored in the 'output' folder

#### t-SNE
![t-SNE plot](Homework2/code/output/features.png)

#### Multi-Layer Perceptron
![MLP plot](Homework2/code/output/CNN-MLP.png)

#### Support Vector Machine
![SVM plot](Homework2/code/output/CNN-SVM.png)

#### K-Nearest Neighbor
![SVM plot](Homework2/code/output/CNN-KNN.png)