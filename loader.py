from __future__ import division
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import collections
import argparse
import sys
import pandas as pd
from sklearn.utils import shuffle

import logger

class Loader(object):

    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.log = logger.Logging()
        if FLAGS.headless:
            self.log.info('running headless - feel free to use functionality')
            pass
        else:
            if FLAGS.train_file and FLAGS.train_col:
                self.train_data = pd.read_csv(FLAGS.train_file)
                if FLAGS.shuffle:
                    self.train_data = shuffle(self.train_data)
                self.train_yvals = self.train_data[FLAGS.train_col]
                self.train_yvals = self.train_yvals.as_matrix()
                self.train_xvals = self.train_data.ix[:,self.train_data.columns != FLAGS.train_col]
                self.train_xvals = self.train_xvals.as_matrix()
                if FLAGS.distort:
                    #print(self.train_xvals[0], np.floor(self.train_xvals[0] * .9))
                    percent = int(len(self.train_xvals) *.95)
                    print('distorting images: ',self.train_xvals[percent:])
                    self.train_xvals[percent:]  = np.floor(self.train_xvals[percent:] * .81)
                    self.train_xvals[:-percent]  = np.floor(self.train_xvals[:-percent:] * .91)
                    #half = int(len(self.train_xvals / 2))
                    #self.train_xvals[:two_percent] = self.dampen_data(self.train_xvals[:two_percent], 15)
                    #self.train_xvals[half:half+two_percent] = self.rotate_data(self.train_xvals[half:half+two_percent])
                self.log.info('read in train matrix %s'%self.train_data.shape[0])
            if FLAGS.test_file and FLAGS.test_col:
                self.test_data = pd.read_csv(FLAGS.test_file)
                self.test_yvals = self.test_data[FLAGS.test_col]
                self.test_yvals = self.test_yvals.as_matrix()
                self.test_xvals = self.test_data.ix[:,self.test_data.columns != FLAGS.test_col]
                self.test_xvals = self.test_xvals.as_matrix()
                self.log.info('read in test matrix %s'%self.test_data.shape[0])

    def get_train_data(self):
        return self.train_data

    def set_train_data(self, data):
        self.train_data = data
        self.log.info('set new train matrix %s'%test_data.shape[0])

    def get_test_data(self):
        return self.test_data

    def set_test_data(self, data):
        self.test_data = data
        self.log.info('set new test matrix %s'%test_data.shape[0])
    
    def rotate_data(self, data):
        return np.rot90(data)

    def flip_data(self, data):
        return np.flip(data,1)

    def dampen_data(self, data, n=10):
        data[data>10] -= 10
        return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load data, passing in train and/or test filenames')
    parser.add_argument('-train', dest='train_file', help='training filename to read in - csv')
    parser.add_argument('-test', dest='test_file', help='test filename to read in - csv')
    parser.add_argument('-train_label', dest='train_col', help='training label column')
    parser.add_argument('-test_label', dest='test_col', help='testing label column')
    parser.add_argument('-headless', dest='headless', default=False,action='store_true',help='run loader headless for function calls')
    parser.add_argument('-shuffle', dest='shuffle', default=False,action='store_true',help='shuffle training data')
    parser.add_argument('-distort', dest='distort', default=False,action='store_true',help='distort data by flipping, rotating, dampening')
    FLAGS = parser.parse_args()
    load = Loader(FLAGS)
