#!/usr/bin/env python2

from __future__ import print_function

import pandas
from scipy import signal
import numpy as np
from pprint import pprint
from pylab import *

import Oger
import mdp

from scipy.optimize import minimize

from sklearn.neighbors import KNeighborsClassifier

d = pandas.read_csv("motor_data_tomas_4.csv")
d = d.dropna()
d = d.reset_index(drop=True)

M = 25
r = 250 # sampling rate
f1, f2 = 10, 22

wavelet1 = signal.morlet(M, w=(f1*M)/(2.0*r))
wavelet2 = signal.morlet(M, w=(f2*M)/(2.0*r))

box_width = 250

features_arr = np.zeros( (len(d.index), # rows
                          # cols, FFT len * n_signals * n_wavelets * 1 (abs, no angle)
                          box_width * 3 * 3 * 2)  )

for i in range(box_width, len(d)-1):
    if i % 1000 == 0:
        print(i)

    data = np.array(d[(i - box_width+1):(i+1)])
    sigs = np.zeros((data.shape[0], 3))
    sigs[..., 0] = (data[..., 0] + data[..., 1])/2.0
    sigs[..., 1] = (data[..., 2] + data[..., 3])/2.0
    sigs[..., 2] = (data[..., 4] + data[..., 5])/2.0
    # sigs[..., 0] = data[..., 0]
    # sigs[..., 1] = data[..., 2]
    # sigs[..., 2] = data[..., 4]

    # fft_len = np.fft.rfft(data[..., 0]).shape[0]
    features = np.array([])

    for j in range(3):
        sig = sigs[..., j]
        conv1 = signal.convolve(sig, wavelet1, 'same')
        conv2 = signal.convolve(sig, wavelet2, 'same')
        fourier = np.fft.fft(sig)
        fourier1 = np.fft.fft(conv1)
        fourier2 = np.fft.fft(conv2)
        features = np.hstack([features, np.abs(fourier), np.abs(fourier1), np.abs(fourier2)])
        # not sure if this is a good idea -->
        features = np.hstack([features, np.angle(fourier), np.angle(fourier1), np.angle(fourier2)])


    features_arr[i, ...] = features


feature_names = []
for i in range(3):
    feature_names.extend(['c' + str(i) + '_abs_A_' + str(x)
                          for x in range(box_width)])
    feature_names.extend(['c' + str(i) + '_abs_B_' + str(x)
                          for x in range(box_width)])
    feature_names.extend(['c' + str(i) + '_abs_C_' + str(x)
                          for x in range(box_width)])
    feature_names.extend(['c' + str(i) + '_angle_A_' + str(x)
                          for x in range(box_width)])
    feature_names.extend(['c' + str(i) + '_angle_B_' + str(x)
                          for x in range(box_width)])
    feature_names.extend(['c' + str(i) + '_angle_C_' + str(x)
                          for x in range(box_width)])

feature_names = np.array(feature_names)
    
# clf.fit(features_arr, d.tag)
fftfreq = np.fft.fftfreq(box_width, d=1/250.0)

def fisher_criterion(X, y, a, b):
    X_1 = X[y == a, ]
    X_2 = X[y == b, ]
    top = np.abs(X_1.mean(0) - X_2.mean(0))
    bottom = np.sqrt((X_1.std(0)**2.0 + X_2.std(0)**2) / 2.0)
    return top / bottom

# time embed the features n times, spaced k apart
def time_embed(features, y, k, n):
    total = n*k
    out = features[total:,]
    for i in range(n-1, -1, -1):
        current = i*k
        left = total - current
        out = np.hstack([out, features[current:-left]])
    return out, y[total:]

def fish_good_features(X, y, a, b, n):
    n_fish_features = 100
    fish = fisher_criterion(X, y, a, b)
    cutoff = np.sort(fish)[-n:][0]
    good_features = fish >= cutoff
    return good_features

def remove_corr_good(X, threshold):
    corr = np.corrcoef(X.T)
    c = corr > threshold
    good_features2 = np.ones(c.shape[0], dtype=bool)

    n_fish_features = c.shape[0]

    for i in range(n_fish_features):
        s = sum(c[i][good_features2])
        if s > 1:
            good_features2[i] = False

    return good_features2

import gc
print("collected: %d" % gc.collect())

ignore = 1000
N_train = 20000

X = features_arr[box_width:]
y = np.array(d.tag[box_width:])

good_indexes = np.arange(X.shape[1])

print("fisher features")
# good_features = fish_good_features(X[ignore:N_train,], y[ignore:N_train], -1, 1, 200)
good_features = fish_good_features(X, y, 1, 2, 200)
X_new = X[..., good_features]
good_indexes = good_indexes[good_features]
print(X_new.shape[1])

print("correlation")
good_features2 = remove_corr_good(X_new[ignore:N_train,], 0.95)
X_new = X_new[..., good_features2]
good_indexes = good_indexes[good_features2]
print(X_new.shape[1])

# print("fisher features again")
# good_features = fish_good_features(X_new[ignore:N_train,], y[ignore:N_train], -1, 1, 50)
# X_new = X_new[..., good_features]
# print(X_new.shape[1])

print(feature_names[good_indexes])
# X_new = X_new * 1000

X_new = (X_new - X_new.min(0)) / X_new.max(0)

# X = np.copy(X_new)
X = X_new
# X = np.array(d)[box_width:, :6]
y = np.array(d.tag[box_width:])[:, np.newaxis]
y = y * 1.0

y_hot = np.zeros((len(y), 3))
y_hot[:, 0] = y[:,0] == -1
y_hot[:, 1] = y[:,0] == 1
y_hot[:, 2] = y[:,0] == 0

X_train = X[ignore:N_train, :]
y_train = y[ignore:N_train]
y_hot_train = y_hot[ignore:N_train]

good = y_train != 0
good = good[:, 0]

# X_train = X_train[good, :]
# y_train = y_train[good]

def function(x):
    leak, ridge, output_dim = x
    # reservoir = Oger.nodes.ReservoirNode(input_dim=X.shape[1], output_dim=200)
    reservoir = Oger.nodes.LeakyReservoirNode(input_dim=X.shape[1], output_dim=output_dim, leak_rate=leak)
    readout = Oger.nodes.RidgeRegressionNode(ridge_param=ridge)
    flow = mdp.Flow([reservoir, readout])

    flow.train(data)

    testout = flow(X)

    error = Oger.utils.nrmse(y[N_train:], testout[N_train:])
    
    print(leak, ridge, error)

    return error


# print(Oger.utils.nrmse(y[N_train:][not_blank], np.sign(testout[not_blank])))

class MyBounds(object):
     def __init__(self, xmax=[1.1,1.1], xmin=[-1.1,-1.1] ):
         self.xmax = np.array(xmax)
         self.xmin = np.array(xmin)
     def __call__(self, **kwargs):
         x = kwargs["x_new"]
         tmax = bool(np.all(x <= self.xmax))
         tmin = bool(np.all(x >= self.xmin))
         return tmax and tmin

gc.collect()
bounds = MyBounds(xmin=(0.001, 0, 0.01), xmax=(1.5, 5.0, 2.0))

def test_func(x, disp=True):
    # leak, ridge, output_dim = (0.03, 0.01, 200)
    
    if not bounds(x_new=x):
        # print(' outside range')
        return 100

    if disp:
        x_pr = ', '.join(["{0:0.3f}".format(i) for i in x])
        print('x = (', x_pr, ')')
    # sys.stdout.flush()

    
    leak, ridge, radius = x
    n_neurons = 50
    
    reduction = mdp.nodes.PCANode(input_dim = X_train.shape[1], output_dim=40)
    reservoir = Oger.nodes.LeakyReservoirNode(output_dim=n_neurons, leak_rate=leak, spectral_radius=radius)
    readout2 = Oger.nodes.RidgeRegressionNode(ridge_param=ridge)
    flow = mdp.Flow([reduction, reservoir, readout2])
    
    flow.train(data)
    testout = flow(X[N_train:])

    # error = Oger.utils.nrmse(y[N_train:][not_blank], np.sign(testout[not_blank]))
    error = Oger.utils.nrmse(y[N_train:][not_blank], testout[not_blank])

    if disp:
        print('\terror = {0:0.3f}'.format(error))
    # print('', error)

    
    return error


xmin = (0.001, 0, 0.01)
xmax = (1.5, 2.0, 2.0)
start = (0.03, 0.01, 0.9)

# res = optimize.anneal(test_func, start, schedule='fast', full_output=True, maxiter=5, dwell=5,
#                 lower=xmin, upper=xmax, disp=True)

# res = basinhopping(test_func, (0.03, 0.01, 0.9), disp=True, accept_test=bounds, stepsize=0.1)

# class AlgoAnneal(Annealer):
#     def move(self):
#         self.state += np.random.uniform(-0.5, 0.5, len(self.state))
#         self.state = np.clip(self.state, xmin, xmax)
#     def energy(self):
#         error = test_func(self.state, disp=False)
#         energy = exp(-(1/error))
#         return energy
#     def copy_state(self, state):
#         return np.copy(state)


# algo = AlgoAnneal(start)       

# # auto_schedule = algo.auto(minutes=1, steps=10)
# algo.Tmax = 0.5
# algo.Tmin = 0.01
# algo.steps = 500
# algo.updates = 30
# print()
# state, energy = algo.anneal()



# print(Oger.utils.nrmse(y, testout))
# res = minimize(function, (0.5, 0.5), bounds=[(0,1), (0,1)])
xx = [X_train]
xy = zip([X_train], [y_hot_train])
xys = zip([X_train], [y_train[:, 0]])
# data = [[X_train], [X_train], [X_train], [X_train], zip([X_train], [y_train])]
# data = [[X_train], [X_train],  zip([X_train], [y_train])]
data = [xx, xx, xys]

# leak, ridge, output_dim = (0.03, 0.01, 200)
# leak, ridge, radius = [ 1.9795126 ,  2.1767528 ,  1.79839377]
leak, ridge, radius = [ 0.02, 0.01, 0.9 ]
n_neurons = 100
# reservoir = Oger.nodes.ReservoirNode(input_dim=X.shape[1], output_dim=200)

reduction = mdp.nodes.PCANode(input_dim = X_train.shape[1], output_dim=40)
# filtering = mdp.nodes.SFANode()
# filtering = mdp.nodes.FastICANode()
reservoir = Oger.nodes.LeakyReservoirNode(output_dim=n_neurons, leak_rate=0.1, spectral_radius=radius)
reservoir2 = Oger.nodes.LeakyReservoirNode(output_dim=50, leak_rate=0.05)
# filtering2 = mdp.nodes.SFANode()
readout = mdp.nodes.KNNClassifier(k=3)
# readout = Oger.nodes.RidgeRegressionNode(ridge_param=0.5)
# readout2 = Oger.nodes.RidgeRegressionNode(ridge_param=ridge)
# readout =  mdp.nodes.KNeighborsRegressorScikitsLearnNode()
flow = mdp.Flow([reduction, reservoir, readout], verbose=1)

# vals = reservoir(X_train)

# knn = KNeighborsClassifier()
# knn.fit(vals, y_train.ravel())

# vals = reservoir(X)
# testout = knn.predict(vals)
# testout = testout[:, np.newaxis]

flow.train(data)
testout = flow(X[N_train:])


# error = Oger.utils.nrmse(y[N_train:], testout)
# print(error)

w = 0.005
bf, af = signal.butter(4, w, 'lowpass')

out_f = np.copy(testout)
# out_f[: ,0] = signal.lfilter(bf, af, out_f[:, 0], 0)
out_f[:, 0] = signal.medfilt(out_f[:,0], 501)
out_f[:, 1] = signal.medfilt(out_f[:,1], 501)
out_f[:, 2] = signal.medfilt(out_f[:,2], 501)

# not_blank = (y[N_train:] != 0).ravel()
# print(Oger.utils.nrmse(y[N_train:][not_blank], np.sign(testout[not_blank])))
# print(Oger.utils.nrmse(y[N_train:][not_blank], np.sign(out_f[not_blank])))

ix = 2

clf()

subplot(2,1,1)
plot(y_hot[N_train:, ix])
plot(testout[:, ix])
# vlines(N_train, -1, 1)

subplot(2,1,2)
plot(y_hot[N_train:, ix])
plot(out_f[:, ix])
# vlines(N_train, -1, 1)

show(block=False)


gc.collect()

