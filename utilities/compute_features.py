#!/usr/bin/env python2

import pandas
from scipy import signal
import numpy as np
from pprint import pprint
from pylab import *

in_fname = "motor_data_derrick.csv"
out_fname = "motor_features_derrick.csv"

d = pandas.read_csv(in_fname)
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
    c = corr > 0.92
    good_features2 = np.ones(c.shape[0], dtype=bool)

    n_fish_features = c.shape[0]

    for i in range(n_fish_features):
        s = sum(c[i][good_features2])
        if s > 1:
            good_features2[i] = False

    return good_features2

X = features_arr[box_width:]
y = np.array(d.tag[box_width:])

# data = np.hstack([X, y[:, np.newaxis]])
# feature_names.append("tag")
print("creating data frame")

df = pandas.DataFrame(data=X, columns=feature_names)
df['tag'] = y

print("writing to csv")
df.to_csv(out_fname, index=False)






