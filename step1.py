from __future__ import absolute_import
from __future__ import division

import os
from collections import namedtuple
from math import sqrt

from sklearn import metrics
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import learn


import cPickle as pickle
import numpy as np

from step1_graph import get_image_feature_small

with open('../data/generated/cont_train.pickle', 'rb') as f:
    save = pickle.load(f)
    X = save['X']
    y = save['y']
    X = X.reshape(-1, X.shape[1], X.shape[2], 1)   
    print 'train: X => ', X.shape, 'y => ', y.shape
    ####one hot encoding
    num_out = [10,10,10,10,10]
    y = np.hsplit(y, len(num_out))
    for i in range(len(num_out)):
    	y[i] = np.equal.outer(y[i], np.arange(num_out[i])).astype(np.float)
        y[i] = y[i].reshape(-1, num_out[i])
    y = np.hstack(y)
    print 'train: X => ', X.shape, 'y => ', y.shape

model_dir = '../models/mnist_reco'

# Restore model if graph is saved into a folder.
if os.path.exists("%s/graph.pbtxt"%(model_dir)):
    classifier = learn.TensorFlowEstimator.restore(model_dir)
else:
    # Create a new resnet classifier.
    classifier = learn.TensorFlowEstimator(
        model_fn=get_image_feature_small, n_classes=0, batch_size=100, steps=10,
        learning_rate=0.001, continue_training=True)

while True:
    # Train model and save summaries into logdir.
    classifier.fit(X, y, logdir=model_dir)

    # Calculate accuracy.
    #score = metrics.accuracy_score(
    #    mnist.test.labels, classifier.predict(mnist.test.images, batch_size=64))
    #print('Accuracy: {0:f}'.format(score))

    # Save model graph and checkpoints.
    classifier.save(model_dir)

