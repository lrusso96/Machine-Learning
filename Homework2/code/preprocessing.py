import pickle
import os
import re

from inception import extract_features

# Read in images and extract features
def init_features(graph_file, images_dir, features_file, labels_file, classes_file):
    # get images - labels are from the subdirectory names
    if os.path.exists(features_file):
        print('Pre-extracted features and labels found. Loading them ...')
        features = pickle.load(open(features_file, 'rb'))
        labels = pickle.load(open(labels_file, 'rb'))
        classes = pickle.load(open(classes_file, 'rb'))
    else:
        print('No pre-extracted features - extracting features ...')
        # get the images and the labels from the sub-directory names
        dir_list = [x[0] for x in os.walk(images_dir)]
        images = []
        for image_sub_dir in dir_list:
            sub_dir_images = [image_sub_dir + '/' + f for f in os.listdir(image_sub_dir) if re.search('jpg|JPG', f)]
            images.extend(sub_dir_images)

        # extract features
        features, labels, classes = extract_features(graph_file, images)

        # save, so they can be used without re-running the last step which can be quite long
        pickle.dump(features, open(features_file, 'wb'))
        pickle.dump(labels, open(labels_file, 'wb'))
        pickle.dump(classes, open(classes_file, 'wb'))    
        print('CNN features obtained and saved.')
    return features, labels, classes

def init_test_features(graph_file, images_dir, features_file, labels_file, classes):
    if os.path.exists(features_file):
        print('Pre-extracted test features and labels found. Loading them ...')
        features = pickle.load(open(features_file, 'rb'))
        labels = pickle.load(open(labels_file, 'rb'))
    else:
        print('No pre-extracted test features - extracting features ...')
        images = []
        ground_truth_file = "ground_truth.txt"
        with open(ground_truth_file, "r") as fin:
            for line in fin:
                elems = line.strip().split(";")
                label = elems[1].replace(" ", "").replace(":", "")
                if label in classes:
                    pair = (label, images_dir + elems[0])
                    images.append(pair) 

        # extract features
        features, labels, _ = extract_features(graph_file, images, test = True)

        # save, so they can be used without re-running the last step which can be quite long
        pickle.dump(features, open(features_file, 'wb'))
        pickle.dump(labels, open(labels_file, 'wb'))
        print('CNN features obtained and saved.')
    return features, labels
