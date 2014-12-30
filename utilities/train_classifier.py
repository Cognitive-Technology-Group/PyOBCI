#!/usr/bin/env python2

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from scipy import signal
import pandas
import numpy as np

box_width = 250
M = 25
r = 250
f1, f2 = 7, 14

wavelet1 = signal.morlet(M, w=(f1*M)/(2.0*r))
wavelet2 = signal.morlet(M, w=(f2*M)/(2.0*r))

datafile = 'motor_data.csv'

def extract_features(data):
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

        
    return features

def preprocess_data(d):

    features_arr = np.zeros( (len(d.index), # rows
                              # cols, FFT len * n_signals * n_wavelets * 1 (abs, no angle)
                              box_width * 3 * 3 * 2)  ) 

    for i in range(box_width, len(d)-1):
        if i % 1000 == 0:
            print(i)

        data = np.array(d[i - box_width+1:i+1])
        features = extract_features(data)
        features_arr[i, ...] = features

    return features_arr


print("reading csv...")
d = pandas.read_csv(datafile)
d = d.dropna()
d = d.reset_index(drop=True)

print("preprocessing data...")
features = preprocess_data(d)
X, y = features[box_width:], d.tag[box_width:]

print("finding bad features...")
ETC = ExtraTreesClassifier()
ETC.fit(X, y)
n_features = 20
cutoff = np.sort(ETC.feature_importances_)[-n_features:][0]
good_features = ETC.feature_importances_ >= cutoff
X_new = X[..., good_features]

print("fit neighbors classifier...")
neigh = KNeighborsClassifier(n_neighbors=10, weights='distance')
neigh.fit(X_new, y)

print("dumping model...")
joblib.dump([neigh, good_features], 'neighbors_model.pkl', compress=4)





