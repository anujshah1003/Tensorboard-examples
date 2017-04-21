# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

PATH = os.getcwd()

LOG_DIR = PATH + '/mnist-tensorboard/log-1'
metadata = os.path.join(LOG_DIR, 'metadata.tsv')

mnist = input_data.read_data_sets(PATH + "/mnist-tensorboard/data/", one_hot=True)
images = tf.Variable(mnist.test.images, name='images')
#def save_metadata(file):
with open(metadata, 'w') as metadata_file:
    for row in range(10000):
        c = np.nonzero(mnist.test.labels[::1])[1:][0][row]
        metadata_file.write('{}\n'.format(c))



with tf.Session() as sess:
    saver = tf.train.Saver([images])

    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
    
# for calling the tensorboard you should be in that drive and call the entire path
#tensorboard --logdir=/Technical_works/tensorflow/mnist-tensorboard/log-1 --port=6006
