#!/usr/bin/env python2

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

import pylab as pl

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

ignore = 1200
n_train = 25000

sigs_train = sigs[ignore:n_train]

delay = 0

y = np.array(d.tag)[:, np.newaxis]
y_train = y[(ignore+delay):(n_train+delay)]
# y_train = (y_train != 0) * 1

x = [sigs_train]
xy = [(sigs_train, y_train)]
xys = [(sigs_train, y_train[:, 0])]

ica = mdp.nodes.FastICANode()
artifacts = RemoveArtifacts(remove_electricity=False)
bandpass = BandpassFilter(7, 30, sampling_rate=250)
csp = CSP(labelA = -1, labelB = 1)
var = LogVarianceWindow(box_width=500)
embed = mdp.nodes.TimeDelayNode(time_frames=15, gap=1)
fda = mdp.nodes.FDANode(output_dim=4)
classify = mdp.nodes.KNeighborsClassifierScikitsLearnNode(n_neighbors=1)
# classify = mdp.nodes.SVCScikitsLearnNode()
# classify = mdp.nodes.LibSVMClassifier()
# classify = Oger.nodes.RidgeRegressionNode(ridge_param=0.01)
flow = mdp.Flow([ica, artifacts, bandpass, csp, var, fda, classify], verbose=1)
inp = [x, x, x, xys, x, xys, xy]

flow.train(inp)

# I want labels from you KNN classifier. Yes, labels.
if getattr(flow[-1], 'label', None):
    flow[-1].execute = flow[-1].label

pred = flow(sigs)

lowpass = LowpassFilter(4, 0.002)
medfilt = MedianFilter(31)
post_flow = mdp.Flow([medfilt, lowpass])

pred2 = post_flow(pred.astype('float32'))

pl.clf()
pl.plot(pred2)
pl.plot(y)
pl.draw()
pl.show(block=False)

