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

d = pandas.read_csv("motor_data_derrick.csv")
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

ignore = 3000
n_train = 23000

sigs_train = sigs[ignore:n_train]

y = np.array(d.tag)[:, np.newaxis].astype('float32')
y_train = y[(ignore+delay):(n_train+delay)]
# y_train = (y_train != 0) * 1

# x = [sigs_train]
# xy = [(sigs_train, y_train)]
# xys = [(sigs_train, y_train[:, 0])]

delay = -10


n_chunks  = 3
sigs_split, y_split = split_data_by_chunks(sigs[ignore:], y[(ignore+delay):, 0], n_chunks, labels=[-1, 1])
_, y_split_newaxis = split_data_by_chunks(sigs[ignore:], y[(ignore+delay):], n_chunks, labels=[-1, 1])

ica = mdp.nodes.FastICANode()
artifacts = RemoveArtifacts(remove_electricity=False)
bandpass = BandpassFilter(7, 30, sampling_rate=250)
csp = CSP(labelA = -1, labelB = 1)
var = LogVarianceWindow(box_width=100)
embed = mdp.nodes.TimeDelayNode(time_frames=1, gap=1)
fda = mdp.nodes.FDANode(output_dim=2)
knn = mdp.nodes.KNeighborsClassifierScikitsLearnNode(n_neighbors=1)
reservoir = Oger.nodes.LeakyReservoirNode(output_dim=200, leak_rate=0.01, spectral_radius=1.0,
                                          input_scaling=0.1, bias_scaling=0.2)
ridge = Oger.nodes.RidgeRegressionNode(ridge_param=0.001)
# gaussian = mdp.nodes.GaussianClassifier()
gaussian = GaussianClassifierArray()
# gauss_proc = mdp.nodes.GaussianHMMScikitsLearnNode()

medfilt = MedianFilter(151)
# classify = mdp.nodes.KNeighborsRegressorScikitsLearnNode(n_neighbors=1)
# classify = mdp.nodes.SVCScikitsLearnNode()
# classify = mdp.nodes.LibSVMClassifier()
# classify = Oger.nodes.RidgeRegressionNode(ridge_param=0.01)
lowpass = LowpassFilter(4, 0.002)
lowpass2 = LowpassFilter(3, 0.005)

flow = mdp.Flow([ica, artifacts, bandpass,
                 embed, csp, var, fda, gaussian, lowpass], verbose=0)

##I want labels from you classifiers. Yes, labels.
for c in flow:
    if getattr(c, 'label', None):
        c.execute = c.label

x = sigs_split
xy = zip(sigs_split, y_split_newaxis)
xys = zip(sigs_split, y_split)

def get_inp(x, xy, xys):
    inp = [x, x, x,
           x, xys, x, xys, xys, xy]
    return inp

gridsearch_parameters = {
    var: {'box_width': np.arange(50, 250, 50)},
    embed: {'time_frames': np.append(1, np.arange(5, 30, 5))},
#    fda: {'output_dim': np.arange(1, 4, 1)},
    lowpass: {'Wn': np.arange(0.001, 0.01, 0.001)}
}
opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)
opt.grid_search(get_inp(x, xy, xys), flow, cross_validate_function=Oger.evaluation.leave_one_out)
opt_flow = opt.get_optimal_flow(verbose=True)
flow = opt_flow


f = flow.copy()
f.train(get_inp(x, xy, xys))
out = f(sigs)
out = np.array(out)

pl.clf()
# pl.subplot(2, 1, 1)
pl.plot(out[ignore:,])
# pl.subplot(2, 1, 2)
pl.plot(y[ignore:])
pl.draw()
pl.show(block=False)

print(Oger.utils.nrmse(out[ignore:], y[ignore:]))

# errors = Oger.evaluation.validate(inp, flow, Oger.utils.nrmse,
#                                   cross_validate_function=Oger.evaluation.leave_one_out)
# print(errors)
# # flow.train(inp)

data_ixs, test_ixs = Oger.evaluation.leave_one_out(n_chunks)

pred = np.array([])

for data_ix, test_ix in zip(data_ixs, test_ixs):
    print(test_ix, data_ix)
    # data_ix = [1, 2]
    ss = [sigs_split[i] for i in data_ix]
    yy = [y_split[i] for i in data_ix]
    yy_axis = [y_split_newaxis[i] for i in data_ix]

    x = ss
    xy = zip(ss, yy_axis)
    xys = zip(ss, yy)

    inp = get_inp(x, xy, xys)

    tix = test_ix[0]
    f = flow.copy()
    f.train(inp)
    p = f(sigs_split[tix])
    p = np.array(p)
    pred = np.append(pred, p)

    print(Oger.utils.nrmse(p, y_split_newaxis[tix]))

pl.clf()
pl.plot(pred)
pl.plot(np.hstack(y_split))
pl.draw()
pl.ylim(-1.2, 1.2)
pl.show(block=False)
