from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import collections
import argparse
import sys

import loader
import logger

tf.logging.set_verbosity(tf.logging.INFO)

class ModelBuilder(object):

    def __init__(self, xtrain, ytrain, xtest, ytest, params=None):
        self.log = logger.Logging()
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        if params is not None:
            self.params = params
            self.log.info('parameter values passed in: %s'%params.keys())

    def set_xtrain(self, data):
        self.xtrain = data

    def set_ytrain(self, data):
        self.ytrain = data

    def set_xtest(self, data):
        self.xtest = data

    def set_xtest(self, data):
        self.xtest = data
    
    def set_parameters(self, params):
        self.params = params
        self.log.info('parameter values set: %s'%params.keys())

    def reshape_vector(self, vec, sizes=[-1,28,28,1]):
        return tf.reshape(vec, sizes)

    def cnn_model_fn(self, features, labels, mode):
        """Model function for CNN."""
        # Input Layer
        he_init = tf.contrib.layers.variance_scaling_initializer()
        input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])
        print(input_layer.shape)
        #print(input_layer.shape)

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(inputs=tf.cast(input_layer, tf.float32),filters=50,kernel_size=[5, 5],padding="same",activation=tf.nn.elu, kernel_initializer=he_init)
        print(conv1.shape)
        # Pooling Layer #1
        conv2 = tf.layers.conv2d(inputs=conv1,filters=110,kernel_size=[3, 3],padding="same",activation=tf.nn.elu, kernel_initializer=he_init)
        # Convolutional Layer #2
        print(conv2.shape)
        pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        print('pool 1',pool1.shape)
        conv3 = tf.layers.conv2d(inputs=conv2,filters=64,kernel_size=[5, 5],strides=[1,1],padding="same",activation=tf.nn.elu, kernel_initializer=he_init)
        print(conv3.shape)
        conv4 = tf.layers.conv2d(inputs=conv3,filters=64,kernel_size=[3, 3],strides=[1,1],padding="same",activation=tf.nn.elu, kernel_initializer=he_init)
        print(conv4.shape)
        # Pooling Layer #2
        pool2 = tf.layers.average_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
        print('pool 2',pool2.shape)
        # dense layer
        pool2_flat = tf.reshape(pool2, [-1, 14*14*64])
        print(pool2_flat.shape)
        dense1 = tf.layers.dense(inputs=pool2_flat, units=1300, activation=tf.nn.elu, kernel_initializer=he_init)
        print(dense1.shape)
        dense2 = tf.layers.dense(inputs=dense1, units=900, activation=tf.nn.elu, kernel_initializer=he_init)
        print(dense2.shape)
        dropout1 = tf.layers.dropout(inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
        print(dropout1.shape)
        dense3 = tf.layers.dense(inputs=dropout1, units=300, activation=tf.nn.elu, kernel_initializer=he_init)
        print(dense3.shape)
        dropout = tf.layers.dropout(inputs=dense3, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
        print(dropout.shape)
        logits = tf.layers.dense(inputs=dropout, units=10)
        print(logits.shape)
        predictions = { 'classes' : tf.argmax(input=logits, axis=1), 'probabilities' : tf.nn.softmax(logits, name='softmax_tensor')}

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)

        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        eval_metric_ops = { 'accuracy' : tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])}
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load data, passing in train and/or test filenames')
    parser.add_argument('-train', dest='train_file', help='training filename to read in - csv')
    parser.add_argument('-test', dest='test_file', help='test filename to read in - csv')
    parser.add_argument('-train_label', dest='train_col', help='training label column')
    parser.add_argument('-test_label', dest='test_col', help='testing label column')
    parser.add_argument('-headless', dest='headless', default=False,action='store_true',help='run loader headless for function       calls')
    parser.add_argument('-shuffle', dest='shuffle', default=False,action='store_true',help='shuffle training data')
    parser.add_argument('-distort', dest='distort', default=False,action='store_true',help='distort data by flipping, rotating, dampening')
    FLAGS = parser.parse_args()
    data = loader.Loader(FLAGS)
    mdl = ModelBuilder(data.train_xvals, data.train_yvals, data.test_xvals, data.test_yvals)
    classifier = tf.estimator.Estimator(model_fn=mdl.cnn_model_fn, model_dir='tmp_convnet_model')
    tensors_to_log = {'probabilities' : 'softmax_tensor'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1000)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':data.train_xvals}, y=data.train_yvals, batch_size=100, num_epochs=None, shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=40000, hooks=[logging_hook])
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={'x':data.test_xvals}, y=data.test_yvals, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)


