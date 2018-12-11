import tensorflow as tf
import numpy as np
import os

#___TensorFlow inception-v3 feature extraction ___#

# Create the CNN graph
def create_graph(graph_file):
    with tf.gfile.GFile(graph_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

# Extract bottleneck features
def extract_features(graph_file, images, test = False):
    nb_features = 2048
    num_images = len(images)
    features = np.empty((num_images, nb_features))
    labels = []
    classes = []
    create_graph(graph_file)

    progress_str = "Processing {0} out of {1} ({2:0.2f} %)\r"
    

    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048 float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG encoding of the image.
    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(images):
            if test:
                imlabel, image = image
            else:
                imlabel = image.split('/')[1]

            # rough indication of progress
            if ind % 100 == 0:
                print(progress_str.format(ind, num_images, 100*ind/num_images), end = "")
            if not tf.gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = tf.gfile.GFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0': image_data})
            features[ind, :] = np.squeeze(predictions)
            labels.append(imlabel)
            if imlabel not in classes:
                classes.append(imlabel)

    return features, labels, classes
