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
	

n_digits= 5

def classify(prob):
    max_pred_digits = []
    cum_max_pred = []
    
    for i in range(n_digits):
	print '=====', str(i)
        log_prob = tf.log(prob[i])
        max_pred_digits.append(tf.argmax(log_prob,1))
        max_pred = tf.reduce_max(log_prob,1)

        if i == 0:
            cum_max_pred.append(max_pred)
        else:
            cum_max_pred.append(tf.add_n([cum_max_pred[i-1], max_pred]))
    
    max_pred_digits = tf.reshape(tf.concat(0, max_pred_digits), [-1, n_digits])
    
    log_prob_len = tf.log(prob[n_digits])
    log_prob_len = tf.split(1,n_digits+1,log_prob_len)
    
    total_max_pred = []
    total_max_pred.append(log_prob_len[0])

    for i in range(n_digits):
        total_max_pred.append(tf.add_n([log_prob_len[i+1], tf.reshape(cum_max_pred[i], [-1,1])]))
    
    total_max_pred = tf.reshape(tf.concat(0, total_max_pred), [-1, len(total_max_pred)])
    total_len = tf.cast(tf.argmax(total_max_pred,1), tf.int32)
   
 
    lengths_transposed = tf.expand_dims(total_len, 1)
    lengths_tiled = tf.tile(lengths_transposed, [1, n_digits])
    
    range_all = tf.range(0, n_digits, 1)
    range_row = tf.expand_dims(range_all, 0)
    batch_size = tf.shape(total_len)[0]
    batch_shape = tf.pack([batch_size, 1])
    range_tiled = tf.tile(range_row, batch_shape)

    mask = tf.less(range_tiled, lengths_tiled)
    zeros_ph = tf.placeholder(tf.int64, shape=[None, n_digits])
    zeros_value = tf.zeros_like(zeros_ph)
  
    result = tf.select(mask, max_pred_digits, zeros_value)
    
    return result

def get_image_feature_small(x, y):
    with tf.variable_scope('conv_layer2'):
        net = learn.ops.conv2d(x, 64, [5, 5], batch_norm=True, padding='SAME',
                                activation=tf.nn.relu, bias=True)
    	net = tf.nn.max_pool(
        	net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	net = learn.ops.dropout(net, 0.5)
    with tf.variable_scope('conv_layer3'):
        net = learn.ops.conv2d(net, 128, [5, 5], batch_norm=True, padding='SAME',
                                activation=tf.nn.relu, bias=True)
    	net = tf.nn.max_pool(
        	net, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
	net = learn.ops.dropout(net, 0.5)
    with tf.variable_scope('conv_layer4'):
        net = learn.ops.conv2d(net, 160, [5, 5], batch_norm=True, padding='SAME',
                                activation=tf.nn.relu, bias=True)
    	net = tf.nn.max_pool(
        	net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	net = learn.ops.dropout(net, 0.5)
    
    net_shape = net.get_shape().as_list()
    net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
    
    net = learn.ops.dnn(net, [3072, 3072], activation=tf.nn.relu, dropout=0.5)
    y_value = tf.split(1,n_digits,y)
    print y.get_shape().as_list()
    print y_value[0].get_shape().as_list()
    preds = []
    losses = 0.0
    for i in range(n_digits):
	labels = tf.reshape(y_value[i], [-1, 10])
    	with tf.variable_scope('logistic_regression' + str(i)):
		pred, loss = learn.models.logistic_regression(net, labels)
		preds.append(pred)
		losses += loss

    ############################## one hot encoded label for number of digits#########
    num_digit_present = tf.cast(tf.reduce_sum(y, [1,2]), np.int32)
    num_labels = 6
    
    table = tf.constant(np.identity(num_labels, dtype=np.float32))
    len_labels = tf.nn.embedding_lookup(table, num_digit_present)

    with tf.variable_scope('logistic_regression6'):
	pred, loss = learn.models.logistic_regression(net, len_labels)
	preds.append(pred)
	losses += loss
	
    return classify(preds), losses 

def get_image_feature(x):
    input_shape = x.get_shape().as_list()

    if len(input_shape) == 2:
        ndim = int(sqrt(input_shape[1]))
        x = tf.reshape(x, [-1, ndim, ndim, 1])
	

    #maxout unit 3 filter per unit
    with tf.variable_scope('conv_layer1'):
        net = learn.ops.conv2d(x, 48, [5, 5], batch_norm=True, padding='SAME',
                                activation=maxnet, bias=True)
    
    with tf.variable_scope('conv_layer2'):
        net = learn.ops.conv2d(net, 64, [5, 5], batch_norm=True, padding='SAME',
                                activation=tf.nn.relu, bias=True)
    	net = tf.nn.max_pool(
        	net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	net = learn.ops.dropout(net, 0.5)
    with tf.variable_scope('conv_layer3'):
        net = learn.ops.conv2d(net, 128, [5, 5], batch_norm=True, padding='SAME',
                                activation=tf.nn.relu, bias=True)
    	net = tf.nn.max_pool(
        	net, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
	net = learn.ops.dropout(net, 0.5)
    with tf.variable_scope('conv_layer4'):
        net = learn.ops.conv2d(net, 160, [5, 5], batch_norm=True, padding='SAME',
                                activation=tf.nn.relu, bias=True)
    	net = tf.nn.max_pool(
        	net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	net = learn.ops.dropout(net, 0.5)
    with tf.variable_scope('conv_layer5'):
        net = learn.ops.conv2d(net, 192, [5, 5], batch_norm=True, padding='SAME',
                                activation=tf.nn.relu, bias=True)
    	net = tf.nn.max_pool(
        	net, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
	net = learn.ops.dropout(net, 0.5)
    with tf.variable_scope('conv_layer6'):
        net = learn.ops.conv2d(net, 192, [5, 5], batch_norm=True, padding='SAME',
                                activation=tf.n.relu, bias=True)
    	net = tf.nn.max_pool(
        	net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	net = learn.ops.dropout(net, 0.5)
    with tf.variable_scope('conv_layer7'):
        net = learn.ops.conv2d(net, 192, [5, 5], batch_norm=True, padding='SAME',
                                activation=tf.nn.relu, bias=True)
   	net = tf.nn.max_pool(
        	net, [1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
	net = learn.ops.dropout(net, 0.5)
    with tf.variable_scope('conv_layer8'):
        net = learn.ops.conv2d(net, 192, [5, 5], batch_norm=True, padding='SAME',
                                activation=tf.nn.relu, bias=True)
    	net = tf.nn.max_pool(
        	net, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	net = learn.ops.dropout(net, 0.5)

    net_shape = net.get_shape().as_list()
    net = tf.reshape(net, [-1, net_shape[1] * net_shape[2] * net_shape[3]])
    
    net = learn.ops.dnn(net, [3072, 3072], activation=tf.nn.relu, dropout=0.5)
     
    preds = []
    losses = 0.0
    for i in range(6):
    	pred, loss = learn.models.logistic_regression(net, y[i])
	preds.append(pred)
	losses += loss
    
    return classify(preds), losses 

