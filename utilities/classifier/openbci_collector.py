#!/usr/bin/env python2

import threading
import csv
from open_bci import *
from feature_extractor import *
import features
import numpy as np
import pickle

"""
A class to make it simple to collect tagged csv data from the OpenBCI board.
It will keep track of data and store it in a csv file.

See OpenBCIBoard and FeatureExtractor for more info.
 """

class OpenBCICollector(object):

    def __init__(self,  feature_samples=125,
                 extract_fun = features.extract_features,
                 extra_process=None,
                 fname = 'collect.csv',
                 port='/dev/ttyACM0', baud=115200):
        self.board = OpenBCIBoard(port, baud)
        self.extractor = FeatureExtractor(self.collect, feature_samples, extract_fun)
        self.fname = fname
        self.bg_thread = None
        self.file = None
        self.csv_writer = None
        self.data = []

        test = features.extract_features(np.reshape(np.random.randn(feature_samples*8), (8, feature_samples)))
        self.num_features = test.size
        self.feature_samples = feature_samples
        self.extra_process = extra_process

    def collect(self, d):
        # assuming that csv_writer has been defined before calling this
        self.data.append(d)

        if self.extra_process:
            self.extra_process(d)

        d2 = dict(d)
        signal = d2.pop('signal')
        features = d2.pop('features')

        # d2['features'] = pickle.dumps(signal.tolist())
        # d2['signal'] = pickle.dumps(features.tolist())
        for i, f in enumerate(features):
            ss = 'f%04d' % i
            d2[ss] = f

        for c, s in enumerate(signal):
            for i, x in enumerate(s):
                ss = 's_%d_%03d' % (c, i)
                d2[ss] = x

        if self.csv_writer:
            self.csv_writer.writerow(d2)
            self.file.flush()

    def stop_bg_collection(self):
        # resolve files and stuff
        self.board.should_stream = False
        self.csv_writer = None
        self.data = []
        self.bg_thread = None
        self.file.close()
        self.file = None

    def disconnect(self):
        self.board.disconnect()

    def start_bg_collection(self):
        if self.bg_thread:
            self.stop_bg_collection()

        self.file = open(self.fname, 'w')

        fieldnames = ['tag', 'start_time', 'end_time']

        for i in range(self.num_features):
            fieldnames.append('f%04d' % i)

        for c in range(8):
            for i in range(self.feature_samples):
                fieldnames.append('s_%d_%03d' % (c, i))

        self.csv_writer = csv.DictWriter(self.file, fieldnames)
        self.csv_writer.writeheader()
        # create a new thread in which the OpenBCIBoard object will stream data
        self.bg_thread = threading.Thread(target=self.board.start_streaming,
                                          args=(self.extractor.receive_sample, ))

        self.bg_thread.start()


    def tag_it(self, tag):
        self.extractor.tag_it(tag)
