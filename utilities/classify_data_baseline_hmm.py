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

import sklearn.cluster

from seqlearn.hmm import MultinomialHMM

from sklearn.hmm import GaussianHMM

d = pandas.read_csv("motor_data_derrick.csv")
# d2 = pandas.read_csv("motor_data_tomas.csv")

# d = d.append(d2)

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
    N = c.shape[0]
    good_features2 = np.ones(N, dtype=bool)

    for i in range(N):
        s = sum(c[i][good_features2])
        if s > 1:
            good_features2[i] = False

    return good_features2

X = features_arr[box_width:]
y = np.array(d.tag[box_width:])

# N_switch = len(d2)
N_train = 20000
# N_test_end = 20000
N_test_end = N_train + 5000

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
good_indexes = np.arange(X.shape[1])

gf1 = fish_good_features(X[:N_train,], y[:N_train], 1, 0, 70)
gf2 = fish_good_features(X[:N_train,], y[:N_train], -1, 0, 70)
gf3 = fish_good_features(X[:N_train,], y[:N_train], -1, 1, 70)
good_features = np.logical_or(gf1, gf2, gf3)
# good_features = fish_good_features(X[:N_train,], y[:N_train], -1, 1, 200)
X_new = X[..., good_features]
good_indexes = good_indexes[good_features]

print("correlation")
good_features2 = remove_corr_good(X_new, 0.92)
X_new = X_new[..., good_features2]
good_indexes = good_indexes[good_features2]

print(X_new.shape[1])

print("fisher features again")
gf1 = fish_good_features(X_new[:N_train,], y[:N_train], 1, 0, 10)
gf2 = fish_good_features(X_new[:N_train,], y[:N_train], -1, 0, 10)
gf3 = fish_good_features(X_new[:N_train,], y[:N_train], -1, 1, 70)
good_features3 = np.logical_or(gf1, gf2, gf3)
# good_features = fish_good_features(X_new[:N_train,], y[:N_train], -1, 1, 15)
X_new = X_new[..., good_features3]
good_indexes = good_indexes[good_features3]

print(X_new.shape[1])
# # good_features = ETC.feature_importances_ >= 0.003

ff = np.array(feature_names)[good_indexes]
pprint(list(ff))
#print(np.sum(good_features))
#pprint(zip(ff, fish[good_features]))




# n_features = n_fish_features * (n+1)


X_new = X_new * 1000

#n_features = sum(good_features2)
n_features = X_new.shape[1]

print(n_features)
# # clf = svm.SVC()
# # clf.fit(X_new, y)


# hmm = MultinomialHMM()
# pos = np.where(np.diff(y) != 0)[0]
# d = np.hstack([0, pos+1, len(y)])
# lens = np.diff(d)
# hmm.fit(X_new, y, lens)

hmm = GaussianHMM(n_components = 20)
hmm.fit([X_new])

clusters = pred = hmm.predict(X_new)

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



# print("unsupervised learning")
# # model = sklearn.cluster.MiniBatchKMeans(n_clusters=30, reassignment_ratio=0.05)
# model = sklearn.cluster.DBSCAN()
# # model = sklearn.cluster.AffinityPropagation()
# # model = sklearn.cluster.AgglomerativeClustering(n_clusters = 3)
# clusters = model.fit_predict(X_new)

print("time embedding")
k, n = 50, 20

# for k in [1, 5, 10, 25, 50, 100]:
#     for n in range(1, 101, 10):

dd = list()
depth = 11
estimators = 4

# for m_features in range(1, 50):
    # print("k", k, "n", n)
print("")
# print("depth", depth)
y_train = (y[:N_train] == 0).astype(float)
c2, y2 = time_embed(clusters[:N_train, np.newaxis], y_train, k, n)

#print("random forest")
# model = KNeighborsClassifier(n_neighbors=2, weights='distance')
rf = RandomForestClassifier(max_depth=depth, n_estimators=estimators, max_features=1)
rf.fit(c2, y2)
print("train", rf.score(c2, y2))


#print("testing")
# clusters = model.predict(X)
y_real = (y == 0).astype(float)
c2, y2 = time_embed(clusters[:, np.newaxis], y_real, k, n)
pred = rf.predict(c2)

test = rf.score(c2, y2)
print("test", test)

# dd.append(test)

# clf()
# plot(dd)
# draw()
    
smoother_len = 400
# # smoother = np.repeat(1.0/smoother_len, smoother_len)
smoother = np.exp(-0.001 * np.arange(0,smoother_len))
smoother = smoother / sum(smoother)

s = pred
# s = signal.wiener(s, 200)
s = signal.convolve(s, smoother, 'same')
# s = s / max(s)
# s = signal.medfilt(s, 11)
clf()
# plot((s - s.mean()) / s.std())
subplot(2,1,1)
plot(s - 1)
plot(y2)
# ylim(-0.5, 1.5)
subplot(2,1,2)
plot(abs(s - y2))
# plot(abs(s - y))
draw()
show(block=False)

# s = s - s.mean()
# s2 = signal.wiener(s, 251)

# target = (y == 0) * 2 - 1
# error = abs(s - target)
# index = np.arange(len(error))
# e_train = error[:N_train]
# e_test = error[N_train:N_test_end]

# # print("RMS error on train", np.sqrt(e_train.mean()))
# # print("RMS error on test", np.sqrt(e_test.mean()))
# p = (np.abs(e_test > 0.9).mean())

# if p < p_best:
#     s_best = s
#     p_best = p

# print("proportion misclass on train", (np.abs(e_train > 0.9).mean()))
# print("proportion misclass on test", p)

# s = s_best
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

# t = np.arange(len(s)) / 250.0

clf()

n_clusters = len(np.unique(clusters))

ys = np.unique(y)

for i, yy in enumerate(ys):
    subplot(3, 1, i+1)
    hist(clusters[np.logical_and(y == yy, clusters != -1)], bins=n_clusters)
    title(yy)

draw()

# vlines(N_train / 250.0, -2, 2, linestyles='solid')
# vlines(N_test_end / 250.0, -2, 2, linestyles='dashed')
# # vlines(N_switch / 250.0, -2, 2, linestyles='dashed', colors='red')
# plot(t, pred)
# # plot(t, (y == 0) * 2 - 1)
# plot(t, y)

# subplot(2,1,2)
# vlines(N_train / 250.0, 0, 2, linestyles='solid')
# hlines(1.0, 0, max(t), linestyles='dashed')
# plot(t, error)

show(block=False)

#neigh.fit(X_new, y)


# joblib.dump([neigh, good_features], 'neighbors_model.pkl', compress=4)

# neigh, good_features = joblib.load('neighbors_model.pkl')
