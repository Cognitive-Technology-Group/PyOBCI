#!/usr/bin/env python2

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn import svm
from sklearn import decomposition
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

from sklearn import manifold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn import linear_model

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.datasets import SupervisedDataSet

from mpl_toolkits.mplot3d import Axes3D
Axes3D

d = pandas.read_csv("motor_data_jisoo.csv")
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
                          box_width * 3 * 3 * 1 + box_width * 3 * 1 * 1)  )

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
        #features = np.hstack([features, np.abs(fourier)])
        # not sure if this is a good idea -->
        features = np.hstack([features, np.angle(fourier)])
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
    feature_names.extend(['c' + str(i) + '_angle_A_' + str(x)
                          for x in range(box_width)])
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

def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

X = features_arr[box_width:18000]
y_orig = np.array(d.tag[box_width:18000])

# X = features_arr[box_width:-1]
# y_orig = np.array(d.tag[box_width:-1])

# indexes = np.random.choice(len(X), 10000, replace=False)
# indexes = np.sort(indexes)

# X = X[indexes]
# y_orig = y_orig[indexes]

y = np.copy(y_orig)

N_train = 12000
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
print("fisher features")
good_features = fish_good_features(X[:N_train,], y[:N_train], -1, 1, 800)
X_new = X[..., good_features]

print("correlation")
good_features = remove_corr_good(X_new, 0.95)
X_new = X_new[..., good_features]

print("fisher features")
good_features = fish_good_features(X_new[:N_train,], y[:N_train], -1, 1, 20)
X_new = X_new[..., good_features]


print(X_new.shape[1])

# print("fisher features again")
# good_features = fish_good_features(X_new[:N_train,], y[:N_train], -1, 1, 30)
# X_new = X_new[..., good_features]

# X_new_features = X_new

print(X_new.shape[1])

print("LLE")
X_new = X_new_features

man = manifold.LocallyLinearEmbedding(method='modified',
                                      n_components=4, n_neighbors=20)
# man = decomposition.MiniBatchDictionaryLearning(n_components=4, alpha=1, batch_size = 5)
# man = manifold.Isomap(n_components=2, n_neighbors=5)
# Y = LLE.fit_transform(X_new[:N_train])
# man = decomposition.MiniBatchSparsePCA(n_components=10)
# man = decomposition.FastICA(n_components=5)
# man = decomposition.KernelPCA(kernel="rbf", gamma=3, n_components=5)
# man = decomposition.NMF(n_components=2)
Y = man.fit_transform(X_new)
# Y = man.fit_transform(X)

vals = [-1, 0, 1]
labels = ["left", "baseline", "right"]
colors = plt.cm.rainbow(np.linspace(0, 1, len(vals)))

# yy = y[:N_train]
yy = y

clf()

# # ax = plt.subplot(111, projection='3d')
ax = plt.subplot(111)

Ls = list()

for c, v, lab in zip(colors, vals, labels):
    p = Y[yy == v, :]
    p = np.copy(p)
    # np.random.shuffle(p)
    h = ax.scatter(rand_jitter(p[:, 0]), rand_jitter(p[:, 1]), # rand_jitter(p[:, 2]),
                   c=c, antialiased=True, label=lab)
    Ls.append(h)

ax.clear()
    
ax.scatter(rand_jitter(Y[:, 0]), rand_jitter(Y[:, 1]), c=colors)
    
legend(Ls, labels)
show(block=False)

t = np.arange(len(y)) / 250.0

subplot(2,1,1)
plot(t, y)

subplot(2,1,2)
plot(t, Y)

show(block=False)

X_new = X_new_features

# print("nearest neighbors regressor")
# neigh = KNeighborsRegressor(n_neighbors=10)
# neigh.fit(X_new[:N_train,], Y)
# Y2 = neigh.predict(X_new)

# model = SVR(kernel='rbf', C=1e3, gamma=0.1)
# model = linear_model.Lasso(alpha = 0.01, max_iter=10000)
# model = linear_model.Ridge(alpha=0.001)
# model.fit(X_new[:N_train,], Y)
# Y2 = model.predict(X_new)

# Y2 = Y

# clf()

# ax = plt.subplot(111, projection='3d')
# # ax = plt.subplot(111)

# for c, v, lab in zip(colors, vals, labels):
#     p = Y2[yy == v, :]
#     ax.scatter(rand_jitter(p[:, 0]), rand_jitter(p[:, 1]), rand_jitter(p[:, 2]),
#             c=c, antialiased=True, label=lab)

# legend()
# show(block=False)

# print("fisher features again")
# good_features = fish_good_features(X_new[:N_train,], y[:N_train], -1, 1, 15)
# X_new = X_new[..., good_features]


# # n_features = n_fish_features * (n+1)
# X_new = X[..., good_indexes]

# X_new = Y
# clf()
# plot(Y)
# show(block=False)

# k, n = 25, 10
# X_new, y = time_embed(Y2, y_orig, k, n)

X_new = Y
X_new = X_new * 1000

#n_features = sum(good_features2)
n_features = X_new.shape[1]

print(n_features)
# # clf = svm.SVC()
# # clf.fit(X_new, y)


model = KNeighborsClassifier(n_neighbors=2)
# model = svm.SVC()
scores = cross_validation.cross_val_score(model, X_new, y, cv=3)
print(scores)
model.fit(X_new[:N_train], y[:N_train])

pred  = model.predict(X_new)

# probs = model.predict_proba(X_new)
# pred = 1.0 * (probs[:, 2] >= 0.8) +  -1.0 * (probs[:, 0] >= 0.8)

# good_features = ETC.feature_importances_ >= 0.0005
# print(np.sum(good_features))
# X_new2 = X[..., good_features]

# n_features = 20
# pca = PCA(n_components = n_features)
# pca.fit(X_new)
# print(pca.explained_variance_)
# X_new = pca.transform(X_new)



# print("data")

# train_data = ClassificationDataSet(n_features, 1, nb_classes=3)
# test_data = ClassificationDataSet(n_features, 1, nb_classes=3)
# all_data = ClassificationDataSet(n_features, 1, nb_classes=3)

# # train_data = SupervisedDataSet(n_features, 1)
# # test_data = SupervisedDataSet(n_features, 1)
# # all_data = SupervisedDataSet(n_features, 1)

# # target = (y == 1) * 1
# target = y + 1
# # target = y

# for i in xrange(N_train):
#     # if y[i] != 0:
#     train_data.addSample(X_new[i,], [target[i]])

# for i in xrange(N_train+1, N_test_end):
#     # if y[i] != 0:
#     test_data.addSample(X_new[i,], [target[i]])

# for i in xrange(X_new.shape[0]):
#     all_data.addSample(X_new[i,], [target[i]])

# train_data._convertToOneOfMany()
# test_data._convertToOneOfMany()
# all_data._convertToOneOfMany()

# print("building")
# fnn = buildNetwork( train_data.indim, train_data.outdim, fast=True,
#                     outclass = SoftmaxLayer)
# trainer = BackpropTrainer( fnn, dataset=train_data, momentum=0.2, verbose=True, learningrate=0.05, lrdecay=1.0)
# # trainer = RPropMinusTrainer( fnn, dataset=train_data, momentum=0.1, verbose=True, learningrate=0.01, lrdecay=1.0)

# # trainer.trainUntilConvergence()

# best = fnn.copy()
# best_test = 1

# for i in range(5):
#     print("training")
#     trainer.trainEpochs(1)

#     print("testing")
#     trnresult = trainer.testOnData()
#     tstresult = trainer.testOnData( dataset=test_data )

#     if tstresult < best_test:
#         best = fnn.copy()
#         best_test = tstresult

#     print "epoch: %4d" % trainer.totalepochs, \
#         "  train error: %.3f" % trnresult, \
#         "  test error: %.3f" % tstresult

#     # if tstresult <= 0.14:
#     #     break

# fnn = best
# trainer.module = best

# print("testing")
# trnresult = trainer.testOnData()
# tstresult = trainer.testOnData( dataset=test_data )
# print("train", trnresult)
# print("test", tstresult)

# print("testing more")
# probs = pred = fnn.activateOnDataset(all_data)

# pred = pred[:, 0]

# probs = probs - probs.mean(0)
# pred = probs.argmax(1)
# pred = pred * 2 - 1
# pred = pred - 1

smoother_len = 250
# smoother = np.ones(smoother_len)
smoother = np.exp(-0.005 * np.arange(0,smoother_len))
smoother = smoother / sum(smoother)

s = pred
# s = signal.wiener(s, 20)
s = signal.convolve(s, smoother, 'same')
# s = signal.medfilt(s, 25)

#s = s - s.mean()

error = abs(s - y)
index = np.arange(len(error))
e_train = error[np.logical_and(index < N_train, y != 0)]
e_test = error[np.logical_and(index >= N_train, y != 0)]

print("RMS error on train", np.sqrt(e_train.mean()))
print("RMS error on test", np.sqrt(e_test.mean()))

print("proportion misclass on train", (np.abs(e_train > 0.95).mean()))
print("proportion misclass on test", (np.abs(e_test > 0.95).mean()))

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

subplot(3,1,1)
vlines(N_train / 250.0, -2, 2, linestyles='solid')
# vlines(N_test_end / 250.0, -2, 2, linestyles='dashed')
plot(t, s)
plot(t, y)

subplot(3,1,2)
vlines(N_train / 250.0, 0, 2, linestyles='solid')
hlines(0.9, 0, max(t), linestyles='dashed')
plot(t, error)

subplot(3,1,3)
plot(t, Y)
# ylim([-0.01, 0.01])

show(block=False)

#neigh.fit(X_new, y)


# joblib.dump([neigh, good_features], 'neighbors_model.pkl', compress=4)

# neigh, good_features = joblib.load('neighbors_model.pkl')
