import numpy as np
import matplotlib.pyplot as plt
import collections
import itertools

#___ Graphics ___#

def plot_features(labels, t_sne_features, output_dir):
    """feature plot"""
    plt.figure(figsize=(9, 9), dpi=100)

    uniques = {x: labels.count(x) for x in labels}
    od = collections.OrderedDict(sorted(uniques.items()))

    colors = itertools.cycle(["r", "b", "g", "c", "m", "y",
                              "slategray", "plum", "cornflowerblue",
                              "hotpink", "darkorange", "forestgreen",
                              "tan", "firebrick", "sandybrown"])
    n = 0
    for label in od:
        count = od[label]
        m = n + count
        plt.scatter(t_sne_features[n:m, 0], t_sne_features[n:m, 1], c=next(colors), s=10, edgecolors='none')
        c = (m + n) // 2
        plt.annotate(label, (t_sne_features[c, 0], t_sne_features[c, 1]))
        n = m
    plt.savefig(output_dir + "features.png")

def plot_confusion_matrix(cm, classes, output_dir, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.clf()
    np.set_printoptions(precision=2)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cbar_title = ""
    else:
        cbar_title = "Number of images"

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label(cbar_title, rotation=270, labelpad=30, fontsize=12)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_dir + title + ".png")
