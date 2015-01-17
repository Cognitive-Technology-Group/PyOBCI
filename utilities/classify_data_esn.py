#!/usr/bin/env python2

import pandas
from scipy import signal
import numpy as np
from pprint import pprint
from pylab import *

import Oger
import mdp
import gc

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Ridge

from scipy import signal

gc.collect()
d = pandas.read_csv("motor_data_derrick.csv")
d = d.dropna()
d = d.reset_index(drop=True)

data = np.array(d[:])

sigs = np.zeros((data.shape[0], 3))
sigs[..., 0] = (data[..., 0] + data[..., 1])/2.0
sigs[..., 1] = (data[..., 2] + data[..., 3])/2.0
sigs[..., 2] = (data[..., 4] + data[..., 5])/2.0

# sigs = np.zeros((data.shape[0], 6))
# for i in range(6):
#     sigs[..., i] = data[..., i]

X = np.copy(sigs)
y = np.array(d.tag)[:, np.newaxis]

ignore = 1000
n_train = 20000

X_train = X[ignore:n_train]
y_train = y[ignore:n_train]

xx = [X_train]
xy = zip([X_train], [y_train])
xys = zip([X_train], [y_train[:, 0]])

ica = mdp.nodes.TDSEPNode()
reservoir = Oger.nodes.LeakyReservoirNode(output_dim=200, leak_rate=0.05, spectral_radius=0.9)
# reduction = mdp.nodes.PCANode(output_dim = )
# reservoir = Oger.nodes.ReservoirNode(input_dim=X.shape[1], output_dim=300, input_scaling=0.05)
# readout = Oger.nodes.RidgeRegressionNode()
# readout = mdp.nodes.KNeighborsClassifierScikitsLearnNode()
# readout = mdp.nodes.GaussianClassifier()
# readout = mdp.nodes.KNNClassifier(k=3)
# readout = mdp.nodes.RadiusNeighborsRegressorScikitsLearnNode()
flow = mdp.Flow([ica, reservoir], verbose=1)

readout = Ridge(alpha=0.1)
# readout = KNeighborsClassifier(n_neighbors=10)

# data = [xx, xx, xys]
# flow.train(data)

# train
print("train")
flow_out = flow(X_train)
readout.fit(flow_out, y_train[:, 0])

# test
print("test")
flow_out = flow(X)
out = readout.predict(flow_out)

b, a = signal.butter(4, 0.01, 'lowpass', analog=False)
testout = signal.lfilter(b, a, out)

# testout = out
# testout = signal.medfilt(out, 51)


testout = testout[:, np.newaxis]

print(Oger.utils.nrmse(y[n_train:], testout[n_train:]))

clf()
plot(testout)
plot(y)
show(block=False)

