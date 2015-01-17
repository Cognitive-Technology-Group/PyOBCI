#!/usr/bin/env python2

import gc
import pandas
import numpy as np
import mdp
import Oger
from sklearn import neighbors, datasets
from sklearn.pipeline import Pipeline
import pylab as plt

from preprocess import *

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

ignore = 1200
n_train = 23000

sigs_train = sigs[ignore:n_train]

delay = 0

y = np.array(d.tag)[:, np.newaxis].astype('float32')
y_train = y[(ignore+delay):(n_train+delay)]
# y_train = (y_train != 0) * 1

# x = [sigs_train]
# xy = [(sigs_train, y_train)]
# xys = [(sigs_train, y_train[:, 0])]

n_chunks  = 3
sigs_split, y_split = split_data_by_chunks(sigs[ignore:], y[ignore:, 0], n_chunks, labels=[-1, 1])
_, y_split_newaxis = split_data_by_chunks(sigs[ignore:], y[ignore:], n_chunks, labels=[-1, 1])

y_split = [(yy == 1) * 1.0 for yy in y_split]
y_split_newaxis = [(yy == 1) * 1.0 for yy in y_split_newaxis]

filters = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 28), (28, 32), (32, 36), (36, 40)]

ica = mdp.nodes.FastICANode()
artifacts = RemoveArtifacts(remove_electricity=False)
switchboard = mdp.hinet.Switchboard(input_dim=5, connections=range(5) * len(filters))
bandpass_layer = multi_bandpass_layer(filters, input_dim=5, sampling_rate=250)
# bandpass = BandpassFilter(7, 30, sampling_rate=250)
csp = CSP(labelA = 0, labelB = 1, input_dim=5)
csp_layer = mdp.hinet.Layer([csp.copy() for x in filters])
var = LogVarianceWindow(box_width=300)
embed = mdp.nodes.TimeDelayNode(time_frames=5, gap=1)
fda = mdp.nodes.FDANode(output_dim=2)
fda5 = mdp.nodes.FDANode(input_dim=5, output_dim=1)
fda_layer = mdp.hinet.Layer([fda5.copy() for x in filters])

knn = mdp.nodes.KNeighborsClassifierScikitsLearnNode(n_neighbors=100)
gaussian = mdp.nodes.GaussianClassifier()

forest = mdp.nodes.RandomForestClassifierScikitsLearnNode()
reservoir = Oger.nodes.LeakyReservoirNode(output_dim=200, leak_rate=0.05, spectral_radius=0.8,
                                          input_scaling=0.1, bias_scaling=0.2)

perceptron = Oger.nodes.PerceptronNode(output_dim=1, transfer_func=Oger.utils.SoftmaxFunction)
# var2 = LogVarianceWindow(box_width=50)
ridge = Oger.nodes.RidgeRegressionNode(ridge_param=0.001)
medfilt = MedianFilter(51)
logistic = Oger.nodes.LogisticRegressionNode(Oger.gradient.CGTrainer())
# classify = mdp.nodes.KNeighborsRegressorScikitsLearnNode(n_neighbors=1)
# classify = mdp.nodes.SVCScikitsLearnNode()
# classify = mdp.nodes.LibSVMClassifier()
# classify = Oger.nodes.RidgeRegressionNode(ridge_param=0.01)
signum = mdp.nodes.SignumClassifier()
lowpass = LowpassFilter(4, 0.01)
lowpass2 = LowpassFilter(3, 0.005)

flow = mdp.Flow([ica, artifacts, switchboard,
                 bandpass_layer, csp_layer,
                 var, gaussian], verbose=1)

x = sigs_split
xy = zip(sigs_split, y_split_newaxis)
xys = zip(sigs_split, y_split)

def get_inp(x, xy, xys):
    inp = [x, x, x,
           x, xys,
           x, xys]
    return inp

# I want labels from you classifiers. Yes, labels. 
for c in flow:
    if getattr(c, 'label', None):
        c.execute = c.label

# gridsearch_parameters = {reservoir: {'leak_rate': np.append(np.arange(0.01, 0.1, 0.01), [0.1, 0.2, 0.3, 0.4])}}        
# opt = Oger.evaluation.Optimizer(gridsearch_parameters, Oger.utils.nrmse)
# opt.grid_search(get_inp(x, xy, xys), flow, cross_validate_function=Oger.evaluation.leave_one_out)
# opt_flow = opt.get_optimal_flow(verbose=True)
# flow = opt_flow

x = sigs_split
xy = zip(sigs_split, y_split_newaxis)
xys = zip(sigs_split, y_split)
f = flow.copy()
f.train(get_inp(x, xy, xys))

pred2 = f(np.vstack(sigs_split))
pl.clf()
pl.plot(pred2)
pl.plot(np.hstack(y_split))
pl.ylim(-0.2, 1.2)
pl.draw()
pl.show(block=False)


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

    # print(Oger.utils.nrmse(p, y_split_newaxis[tix]))
    print(Oger.utils.nrmse(p, y_split[tix]))

pl.clf()
pl.plot(pred)
pl.plot(np.hstack(y_split))
pl.draw()
pl.ylim(-0.2, 1.2)
pl.show(block=False)



# if False:
#     pl.clf()
#     pl.plot(pred)
#     pl.plot(y)
#     pl.draw()
#     pl.ylim(-1.2, 1.2)
#     pl.show(block=False)




