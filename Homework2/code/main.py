import tensorflow as tf
from sklearn import model_selection
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
import matplotlib.pyplot as plt
import pickle
import collections
import itertools
import time
import os
import re

from preprocessing import *
from classification import *

# what and where
images_dir = 'images/'
test_images_dir = 'test_images/'
output_dir = "output/"
graph_file = 'pretrained/graph.pb'
features_file = "features"
labels_file = "labels"
classes_file = "classes"
test_features_file = "test_features"
test_labels_file = "test_labels"
tsne_file = 'tsne_features.npz'

#___ MAIN __#

# extract features both for training and test set
features, labels, classes = init_features(graph_file, images_dir, features_file, labels_file, classes_file)
test_features, test_labels = init_test_features(graph_file, test_images_dir, test_features_file, test_labels_file, classes)

# plot features in 2-D
tsne_features = TSNE_classify(tsne_file, features)
plot_features(labels, tsne_features, output_dir)

# define training and test set
X_train, X_test = features, test_features
y_train, y_test = labels, test_labels

# run classifiers and plot results

# LinearSVC defaults:
# penalty=’l2’, loss=’squared_hinge’, dual=True, tol=0.0001, C=1.0, multi_class=’ovr’, fit_intercept=True,
# intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000

# classify the images with a Linear Support Vector Machine (SVM)
print('Support Vector Machine starting ...')
clf = LinearSVC(max_iter = 3000)
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-SVM", classes, output_dir)

# RandomForestClassifier/ExtraTreesClassifier defaults:
# (n_estimators=10, criterion='gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1,
# min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0,
# min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False,
# class_weight=None)

# classify the images with a Extra Trees Classifier
print('Extra Trees Classifier starting ...')
clf = ExtraTreesClassifier(n_jobs=4,  n_estimators=100, criterion='gini', min_samples_split=10,
                           max_features=50, max_depth=40, min_samples_leaf=4)
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-ET", classes, output_dir)

# classify the images with a Random Forest Classifier
print('Random Forest Classifier starting ...')
clf = RandomForestClassifier(n_jobs=4, criterion='entropy', n_estimators=70, min_samples_split=5)
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-RF", classes, output_dir)


# KNeighborsClassifier defaults:
# n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’minkowski’, metric_params=None,
# n_jobs=1, **kwargs

# classify the images with a k-Nearest Neighbors Classifier
print('K-Nearest Neighbours Classifier starting ...')
clf = KNeighborsClassifier(n_neighbors=1, n_jobs=4)
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-KNN", classes, output_dir)

# MPLClassifier defaults:
# hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’,
# learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None,
# tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False,
# validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08

# classify the image with a Multi-layer Perceptron Classifier
print('Multi-layer Perceptron Classifier starting ...')
clf = MLPClassifier()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-MLP", classes, output_dir)

# GaussianNB defaults:
# priors=None

# classify the images with a Gaussian Naive Bayes Classifier
print('Gaussian Naive Bayes Classifier starting ...')
clf = GaussianNB()
run_classifier(clf, X_train, y_train, X_test, y_test, "CNN-GNB", classes, output_dir)

