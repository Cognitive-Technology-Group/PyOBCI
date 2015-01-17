#!/usr/bin/env python2

import gc
import pandas
import numpy as np
import mdp
import Oger
from sklearn import neighbors, datasets
from sklearn.pipeline import Pipeline
import pylab as plt

# from preprocess import *

gc.collect()

d = pandas.read_csv("motor_data_jim_2.csv")
d = d.dropna()
d = d.reset_index(drop=True)

data = np.array(d[:])

# sigs = np.zeros((data.shape[0], 3), dtype='float32')
# sigs[..., 0] = (data[..., 0] + data[..., 1])/2.0
# sigs[..., 1] = (data[..., 2] + data[..., 3])/2.0
# sigs[..., 2] = (data[..., 4] + data[..., 5])/2.0

sigs = np.zeros((data.shape[0], 6))
for i in range(6):
    sigs[..., i] = data[..., i]

ignore = 1000
n_train = 20000

sigs_train = sigs[ignore:n_train]

# get features
ica = mdp.nodes.FastICANode(dtype='float64')
# preprocess = EEGFeatures3(sampling_rate=250, box_width=250)
preprocess = EEGFeatures(sampling_rate=250, box_width=250, wavelets_freqs=())
pre_flow = mdp.Flow([preprocess], verbose=1)

pre_flow.train(sigs_train)

X = pre_flow(sigs)

## classify features
y = np.array(d.tag)[:, np.newaxis]
# y = smooth_out_y(y, 250)

# y = (y == 1) * 1

X_train = X[ignore:n_train]
y_train = y[ignore:n_train]

# embed = mdp.nodes.TimeDelayNode(time_frames=5, gap=1)
# reduction = FisherFeatures(output_dim = 300, labelA = 0, labelB = 1)
# dl = mdp.nodes.MiniBatchDictionaryLearningScikitsLearnNode(n_components = 200)
reduction = FisherFeaturesUncorr(output_dim = 200, labelA = -1, labelB = 1)
# scaling = mdp.nodes.NormalizeNode()
reservoir = Oger.nodes.LeakyReservoirNode(output_dim=300, leak_rate=0.02, input_scaling=100)

pca = mdp.nodes.PCANode(output_dim = 0.98)
# fda = mdp.nodes.FDANode(output_dim = 2)
# reduction = FisherFeaturesUncorr(output_dim=50, threshold=0.99, labelA = 0, labelB = 1)
# kmeans = mdp.nodes.KMeansClassifier(num_clusters = 5)
# classify = mdp.nodes.RBMWithLabelsNode(hidden_dim=20, labels_dim=1)
# classify = mdp.nodes.KNeighborsClassifierScikitsLearnNode(n_neighbors=2)
# classify = mdp.nodes.KNeighborsRegressorScikitsLearnNode(n_neighbors=3)
classify = Oger.nodes.RidgeRegressionNode(ridge_param=0.01)
# classify = mdp.nodes.SVCScikitsLearnNode()

flow = mdp.Flow([reduction, reservoir, classify], verbose=1)

x = [X_train]
xy = [(X_train, y_train)]
xys = [(X_train, y_train[:, 0])]
inp = [xy, x, xy]

flow.train(inp)

# I want labels from you KNN classifier. Yes, labels.
if getattr(flow[-1], 'label', None):
    flow[-1].execute = flow[-1].label

pred = flow(X)

lowpass = LowpassFilter(4, 0.002)
medfilt = MedianFilter(51)
post_flow = mdp.Flow([lowpass])

pred2 = post_flow(pred.astype('float32'))

plt.clf()

plt.subplot(2, 1, 1)
plt.plot(pred)
plt.plot(y)
plt.ylim(-1.2, 1.2)

plt.subplot(2, 1, 2)
plt.plot(pred2)
plt.plot(y)
plt.ylim(-1.2, 1.2)

plt.draw()
plt.show(block=False)

gc.collect()
