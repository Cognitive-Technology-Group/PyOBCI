#!/usr/bin/env python2

import pandas
from scipy import signal
import numpy as np
from pprint import pprint
from pylab import *

import Oger
import mdp

d = pandas.read_csv("motor_data_derrick.csv")
d = d.dropna()
d = d.reset_index(drop=True)

data = np.array(d[:])
sigs = np.zeros((data.shape[0], 3))
sigs[..., 0] = (data[..., 0] + data[..., 1])/2.0
sigs[..., 1] = (data[..., 2] + data[..., 3])/2.0
sigs[..., 2] = (data[..., 4] + data[..., 5])/2.0

X = np.copy(sigs)
y = np.array(d.tag)[:, np.newaxis]

ignore = 1000
n_train = 20000

X_train = X[ignore:n_train]
y_train = y[ignore:n_train]

reservoir = Oger.nodes.ReservoirNode(input_dim=X.shape[1], output_dim=300, input_scaling=0.05)
readout = Oger.nodes.RidgeRegressionNode()
flow = mdp.Flow([reservoir, readout], verbose=1)

data = [[X_train], zip([X_train], [y_train])]
flow.train(data)

testout = flow(X)

print(Oger.utils.nrmse(y, testout))

clf()
plot(y)
plot(testout)
show(block=False)
