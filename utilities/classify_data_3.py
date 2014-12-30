#!/usr/bin/env python2

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn import svm
from sklearn.externals import joblib
import pandas
from scipy import signal
import numpy as np
from pprint import pprint
from pylab import *

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.qda import QDA

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.datasets import SupervisedDataSet

d = pandas.read_csv("motor_data_tomas.csv")
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
y = np.array(d.tag[box_width:])

N_train = 10000
N_test_end = 18000

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

fftfreq = np.fft.fftfreq(box_width, d=1/250.0)

# good_indexes = np.array([ 184, 194, 196, 514, 736, 916, 918, 932, 1224, 1264,
#                           1275, 1276, 1668, 2014, 2024])

good_indexes2 = np.array([ 175, 195, 202, 206, 570, 877, 891, 911, 917,
                          924, 1226, 1273, 1274, 1625, 1631, 1680,
                          1692, 1695, 2026, 2027])

print("fisher features")
good_indexes = np.arange(X.shape[1])
# n_boxes = X.shape[1] / box_width
# n_fish_features_per_box = 3
# n_fish_features = n_boxes * n_fish_features_per_box
n_fish_features = 100
# fish = fisher_criterion(X[:N_train,], y[:N_train], -1, 1)
fish = fisher_criterion(X, y, -1, 1)

cutoff = np.sort(fish)[-n_fish_features:][0]
good_features = fish >= cutoff
good_indexes = good_indexes[good_features]

# for i in range(n_boxes):
#     start = i*box_width
#     end = (i+1)*box_width
#     cutoff = np.sort(fish[start:end])[-n_fish_features_per_box:][0]
#     good_features[start:end] = fish[start:end] >= cutoff

X_new = X[..., good_features]

print("correlation")
corr = np.corrcoef(X_new.T)
c = abs(corr) > 0.92
good_features2 = np.ones(n_fish_features, dtype=bool)
good_indexes = good_indexes[good_features2]

for i in range(n_fish_features):
    s = sum(c[i][good_features2])
    if s > 1:
        good_features2[i] = False

X_new = X_new[..., good_features2]
good_indexes = good_indexes[good_features2]

print(X_new.shape[1])

n_fish_features = 20
# fish = fisher_criterion(X_new[:N_train,], y[:N_train], -1, 1)
fish = fisher_criterion(X_new, y, -1, 1)
cutoff = np.sort(fish)[-n_fish_features:][0]
good_features = fish >= cutoff
X_new = X_new[..., good_features]
good_indexes = good_indexes[good_features]

# # good_features = ETC.feature_importances_ >= 0.003

fs = np.array(feature_names)
ff = np.array(feature_names)[good_indexes]
print(ff)
# print(np.sum(good_features))
# pprint(zip(ff, fish[good_features]))


# n_features = n_fish_features * (n+1)
X_new = X[..., good_indexes]

X_new = X_new * 1000

#n_features = sum(good_features2)
n_features = X_new.shape[1]

print(n_features)
# # clf = svm.SVC()
# # clf.fit(X_new, y)

# neigh = KNeighborsClassifier(n_neighbors=10, weights='distance')
# scores = cross_validation.cross_val_score(neigh, X_new, y, cv=5)
# print(scores)
# # neigh.fit(X_new, y)


# good_features = ETC.feature_importances_ >= 0.0005
# print(np.sum(good_features))
# X_new2 = X[..., good_features]

# n_features = 20
# pca = PCA(n_components = n_features)
# pca.fit(X_new)
# print(pca.explained_variance_)
# X_new = pca.transform(X_new)



print("data")

train_data = ClassificationDataSet(n_features, 1, nb_classes=2)
test_data = ClassificationDataSet(n_features, 1, nb_classes=2)
all_data = ClassificationDataSet(n_features, 1, nb_classes=2)

# train_data = SupervisedDataSet(n_features, 1)
# test_data = SupervisedDataSet(n_features, 1)
# all_data = SupervisedDataSet(n_features, 1)

target = (y == 1) * 1
# target = y + 1
# target = y

for i in xrange(N_train):
    if y[i] != 0:
        train_data.addSample(X_new[i,], [target[i]])

for i in xrange(N_train+1, N_test_end):
    if y[i] != 0:
        test_data.addSample(X_new[i,], [target[i]])

for i in xrange(X_new.shape[0]):
    all_data.addSample(X_new[i,], [target[i]])

train_data._convertToOneOfMany()
test_data._convertToOneOfMany()
all_data._convertToOneOfMany()

print("building")
fnn = buildNetwork( train_data.indim, 5, train_data.outdim, fast=True,
                    outclass = SoftmaxLayer)
trainer = BackpropTrainer( fnn, dataset=train_data, momentum=0.2, verbose=True, learningrate=0.05, lrdecay=1.0)
# trainer = RPropMinusTrainer( fnn, dataset=train_data, momentum=0.1, verbose=True, learningrate=0.01, lrdecay=1.0)

# trainer.trainUntilConvergence()

best = fnn.copy()
best_test = 1

for i in range(5):
    print("training")
    trainer.trainEpochs(1)

    print("testing")
    trnresult = trainer.testOnData()
    tstresult = trainer.testOnData( dataset=test_data )

    if tstresult < best_test:
        best = fnn.copy()
        best_test = tstresult

    print "epoch: %4d" % trainer.totalepochs, \
        "  train error: %.3f" % trnresult, \
        "  test error: %.3f" % tstresult

    # if tstresult <= 0.14:
    #     break

fnn = best
trainer.module = best

print("testing")
trnresult = trainer.testOnData()
tstresult = trainer.testOnData( dataset=test_data )
print("train", trnresult)
print("test", tstresult)

print("testing more")
probs = pred = fnn.activateOnDataset(all_data)

# pred = pred[:, 0]

probs = probs - probs.mean(0)
pred = probs.argmax(1)
pred = pred * 2 - 1
# pred = pred - 1

smoother_len = 1000
# smoother = np.repeat(1.0/smoother_len, smoother_len)
smoother = np.exp(-0.001 * np.arange(0,smoother_len))
smoother = smoother / sum(smoother)

s = pred
# s = signal.wiener(s, 200)
s = signal.convolve(s, smoother, 'same')
# s = signal.medfilt(s, 11)

s = s - s.mean()

error = abs(s - y)
index = np.arange(len(error))
e_train = error[np.logical_and(index < N_train, y != 0)]
e_test = error[np.logical_and(index >= N_train, y != 0)]

print("RMS error on train", np.sqrt(e_train.mean()))
print("RMS error on test", np.sqrt(e_test.mean()))

print("proportion misclass on train", (np.abs(e_train > 1).mean()))
print("proportion misclass on test", (np.abs(e_test > 1).mean()))

# error2 = abs(np.sign(s) - y)
# e2_train = error2[np.logical_and(index < N_train, y != 0)]
# e2_test = error2[np.logical_and(index >= N_train, y != 0)]
# print("proportion misclass on train, sign", (np.abs(e2_train > 1).mean()))
# print("proportion misclass on test, sign", (np.abs(e2_test > 1).mean()))

# plot(signal.convolve(pred, smoother, 'same'))
# plot(signal.medfilt(pred, 121))
# subplot(2,1,1)
# vlines(N_train, -1, 1, linestyles='dotted')
# plot(p)

t = np.arange(len(s)) / 250.0

clf()

subplot(2,1,1)
vlines(N_train / 250.0, -2, 2, linestyles='solid')
vlines(N_test_end / 250.0, -2, 2, linestyles='dashed')
plot(t, s)
plot(t, y)

subplot(2,1,2)
vlines(N_train / 250.0, 0, 2, linestyles='solid')
hlines(1.0, 0, max(t), linestyles='dashed')
plot(t, error)

show(block=False)

#neigh.fit(X_new, y)


# joblib.dump([neigh, good_features], 'neighbors_model.pkl', compress=4)

# neigh, good_features = joblib.load('neighbors_model.pkl')
