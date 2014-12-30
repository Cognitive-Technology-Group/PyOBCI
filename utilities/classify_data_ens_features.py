#!/usr/bin/env python2

import pandas
from scipy import signal
import numpy as np
from pprint import pprint
from pylab import *

import Oger
import mdp

d = pandas.read_csv("motor_data_jim_2.csv")
d = d.dropna()
d = d.reset_index(drop=True)

M = 30
r = 250 # sampling rate
f1, f2 = 10, 22

wavelet1 = signal.morlet(M, w=(f1*M)/(2.0*r))
wavelet2 = signal.morlet(M, w=(f2*M)/(2.0*r))

box_width = 250

features_arr = np.zeros( (len(d.index), # rows
                          # cols, FFT len * n_signals * n_wavelets * 1 (abs, no angle)
                          box_width * 3 * 3 * 1)  ) 

for i in range(box_width, len(d)-1):
    if i % 1000 == 0:
        print(i)
    
    data = np.array(d[(i - box_width+1):(i+1)])
    sigs = np.zeros((data.shape[0], 3))
    sigs[..., 0] = (data[..., 0] + data[..., 1])/2.0
    sigs[..., 1] = (data[..., 2] + data[..., 3])/2.0
    sigs[..., 2] = (data[..., 4] + data[..., 5])/2.0
    
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
        # features = np.hstack([features, np.angle(fourier), np.angle(fourier1), np.angle(fourier2)])


    features_arr[i, ...] = features
        

feature_names = []
for i in range(3):
    feature_names.extend(['c' + str(i) + '_abs_A_' + str(x)
                          for x in range(box_width)])
    feature_names.extend(['c' + str(i) + '_abs_B_' + str(x)
                          for x in range(box_width)])
    feature_names.extend(['c' + str(i) + '_abs_C_' + str(x)
                          for x in range(box_width)])
    # feature_names.extend(['c' + str(i) + '_angle_A_' + str(x)
    #                       for x in range(box_width)])
    # feature_names.extend(['c' + str(i) + '_angle_B_' + str(x)
    #                       for x in range(box_width)])
    # feature_names.extend(['c' + str(i) + '_angle_C_' + str(x)
    #                       for x in range(box_width)])

# clf.fit(features_arr, d.tag)
fftfreq = np.fft.fftfreq(box_width, d=1/250.0)

def fisher_criterion(X, y, a, b):
    X_1 = X[y == a, ]
    X_2 = X[y == b, ]
    top = np.abs(X_1.mean(0) - X_2.mean(0))
    bottom = np.sqrt((X_1.std(0)**2.0 + X_2.std(0)**2) / 2.0)
    return top / bottom

# time embed the features n times, spaced k apart
def time_embed(features, k, n):
    total = n*k
    out = features[total:,]
    for i in range(n-1, -1, -1):
        current = i*k
        left = total - current
        out = np.hstack([out, features[current:-left]])
    return out

X = features_arr[box_width:]
y = np.array(d.tag[box_width:])[:, np.newaxis]

ignore = 500
N_train = 25000
# N_test_end = 20000

# print("first round")
# n_fish_features = 400
# fish = fisher_criterion(X[:N_train,], y[:N_train], -1, 1)
# cutoff = np.sort(fish)[-n_fish_features:][0]
# good_features = fish >= cutoff
# X = X[..., good_features]

# print("embedding")
# k, n = 25, 2
# X = time_embed(X, k, n)
# y = y[(n*k):]

print("fisher features")
# n_boxes = X.shape[1] / box_width
# n_fish_features_per_box = 3
# n_fish_features = n_boxes * n_fish_features_per_box
n_fish_features = 100
fish = fisher_criterion(X[ignore:N_train,], y[ignore:N_train, 0], -1, 1)
# fish = fisher_criterion(X, y, -1, 1)

cutoff = np.sort(fish)[-n_fish_features:][0]
good_features = fish >= cutoff

# for i in range(n_boxes):
#     start = i*box_width
#     end = (i+1)*box_width
#     cutoff = np.sort(fish[start:end])[-n_fish_features_per_box:][0]
#     good_features[start:end] = fish[start:end] >= cutoff
    
X_new = X[..., good_features]

print("correlation")
corr = np.corrcoef(X_new.T)
c = corr > 0.95
good_features2 = np.ones(n_fish_features, dtype=bool)

for i in range(n_fish_features):
    s = sum(c[i][good_features2])
    if s > 1:
        good_features2[i] = False

X_new = X_new[..., good_features2]


# # good_features = ETC.feature_importances_ >= 0.003

# ff = np.array(feature_names)[good_features]
# print(ff)
# print(np.sum(good_features))
# pprint(zip(ff, fish[good_features]))


# n_features = n_fish_features * (n+1)


X_new = X_new * 1000

#n_features = sum(good_features2)
n_features = X_new.shape[1]

print(n_features)

X = np.copy(X_new)
# y = y[:, np.newaxis]
y = np.array(d.tag[box_width:])[:, np.newaxis]


X_train = X[ignore:n_train]
y_train = y[ignore:n_train]

reservoir = Oger.nodes.ReservoirNode(input_dim=X.shape[1], output_dim=100, input_scaling=0.05)
readout = Oger.nodes.RidgeRegressionNode()
flow = mdp.Flow([reservoir, readout], verbose=1)

data = [[X_train], zip([X_train], [y_train])]
flow.train(data)

testout = flow(X)

print(Oger.utils.nrmse(y, testout))

out_f = np.zeros(testout.shape)
out_f[:, 0] = signal.medfilt(testout[:,0], 11)

print(Oger.utils.nrmse(y, out_f))

clf()

subplot(2,1,1)
plot(y)
plot(testout)

subplot(2,1,2)
plot(y)
plot(out_f)

show(block=False)

