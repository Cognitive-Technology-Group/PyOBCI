#!/usr/bin/env python2

import time
import numpy as np

"""A feature extractor, meant to be used with the OpenBCIBoard object"""

class FeatureExtractor(object):

    """ Args:
    callback: a function which will receive a Dict object with these key value pairs:
               start_time: time.time() when first sample of this window was collected
               end_time: time.time() when last sample of this window was collected
               tag: tag for this sample (for labeling samples for machine learning)
               features: numpy array of features extracted from sample
               signal: numpy array of signals at the sample
    threshold: how many samples should I process at a time?
    extract_features: a function that will receive a numpy array
                     of shape (N, M) where N = number of channels
                                           M = THRESHOLD
                    This should output a 1D array of features
                    extracted from the signal.
    """
    def __init__(self, callback, threshold, extract_features):
        self.all_samples = []
        self.callback = callback
        self.threshold = threshold
        self.extract_features = extract_features
        self.start_time = time.time()
        self.tag = None

    """ Takes an OpenBCISample object and stores it.
        Will call CALLBACK when THRESHOLD samples have been received"""
    def receive_sample(self, sample):
        sample = sample.channels
        if len(self.all_samples) == 0:
            self.all_samples = [list() for i in range(len(sample))]
            self.start_time = time.time()

        # store it always, but only use when it's the last sample
        end_time = time.time()

        for i, s in enumerate(sample):
            self.all_samples[i].append(s)

        if len(self.all_samples[0]) >= self.threshold:
            x = np.array(self.all_samples)
            self.all_samples = []
            features = self.extract_features(x)
            d = dict()
            d['start_time'] = self.start_time
            d['end_time'] = end_time
            d['tag'] = self.tag
            d['features'] = features
            d['signal'] = x
            self.callback(d)

    def tag_it(self, tag):
        self.tag = tag
