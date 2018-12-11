from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.manifold import TSNE
import numpy as np
import time
import os

from plotting import *

#___ Classifier performance ___#

def run_classifier(clfr, x_train_data, y_train_data, x_test_data, y_test_data, clf_name, classes, output_dir):
    start_time = time.time()
    clfr.fit(x_train_data, y_train_data)
    y_pred = clfr.predict(x_test_data)
    print("%f seconds" % (time.time() - start_time))

    acc_str = 'accuracy for {0}: {1:0.1f}%'
    prec_str = 'precision for {0}: {1:0.1f}%'
    # confusion matrix computation and display
    print(acc_str.format(clf_name, accuracy_score(y_test_data, y_pred) * 100))
    print(prec_str.format(clf_name, precision_score(y_test_data, y_pred, average='weighted') * 100))
    plot_confusion_matrix(confusion_matrix(y_test_data, y_pred), classes, output_dir, title = clf_name)


# TSNE defaults:
# n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000,
# n_iter_without_progress=300, min_grad_norm=1e-07, metric=’euclidean’, init=’random’, verbose=0,
# random_state=None, method=’barnes_hut’, angle=0.5

def TSNE_classify(tsne_file, features):
    # t-sne feature plot
    if os.path.exists(tsne_file):
        print('t-sne features found. Loading ...')
        tsne_k = tsne_file.split(".")[0]
        tsne_features = np.load(tsne_file)[tsne_k]
    else:
        print('No t-sne features found. Obtaining ...')
        tsne_features = TSNE().fit_transform(features)
        np.savez(tsne_k, tsne_features= tsne_features)
        print('t-sne features obtained and saved.')
    return tsne_features


