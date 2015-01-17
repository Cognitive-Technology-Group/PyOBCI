#!/usr/bin/env python2


import gc
import pandas
import numpy as np
import mdp
import Oger

from pylab import *

# from preprocess import *

gc.collect()

d = pandas.read_csv("motor_data_tomas.csv")
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

ignore = 1000
n_train = 14000

sigs_train = sigs[ignore:n_train]

# get features
ica = mdp.nodes.FastICANode(dtype='float64')
remove_artifacts = RemoveArtifacts()
# preprocess = EEGFeatures3(sampling_rate=250, box_width=250)
preprocess = EEGFeatures(sampling_rate=250, box_width=250, wavelets_freqs=())
pre_flow = mdp.Flow([preprocess], verbose=1)

pre_flow.train(sigs_train)

X = pre_flow(sigs)

def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

y = np.array(d.tag)[:, np.newaxis]
# y = smooth_out_y(y, 250)
X_train = X[ignore:n_train]
y_train = y[ignore:n_train]


fisher = FisherFeaturesUncorr(output_dim=50, labelA = -1, labelB = 1)
fda = mdp.nodes.FDANode(output_dim=2)
pca = mdp.nodes.PCANode(output_dim=2)
ica2 = mdp.nodes.FastICANode()

flow = mdp.Flow([fisher, pca], verbose=1)

x = [X_train]
xy = [(X_train, y_train)]
xys = [(X_train, y_train[:, 0])]
inp = [xys, x]

flow.train(inp)

Y = flow(X)


vals = [-1, 0, 1]
labels = ["left", "baseline", "right"]
colors = np.array(["red", "green", "blue"])

# yy = y[:N_train]
yy = y[:, 0]
cc = colors[np.int16(yy + 1)]

clf()

# # ax = plt.subplot(111, projection='3d')
ax = plt.subplot(111)

# Ls = list()

for c, v, lab in zip(colors, vals, labels):
    p = Y[yy == v, :]
    p = np.copy(p)
    # np.random.shuffle(p)
    h = ax.scatter(p[0, 0], p[0, 0], # rand_jitter(p[:, 2]),
                   c=c, antialiased=True, label=lab)
    Ls.append(h)


figure(1)
clf()
arr = Y

a, b = 0, 1

ax = plt.subplot(211)
ix = np.arange(len(cc))[:n_train]
np.random.shuffle(ix)
ax.scatter(rand_jitter(arr[ix, a]), rand_jitter(arr[ix, b]), c=cc[ix], s=20, alpha=0.2)

ax = plt.subplot(212)
ix = np.arange(len(cc))[n_train:]
np.random.shuffle(ix)
ax.scatter(rand_jitter(arr[ix, a]), rand_jitter(arr[ix, b]), c=cc[ix], s=20, alpha=0.2)


legend(Ls, labels)
show(block=False)

figure(1)
clf()
ICs = ica(sigs)
for i in range(ICs.shape[1]):
    subplot(ICs.shape[1]+1, 1, i+1)
    plot(ICs[:, i])
    # specgram(ICs[:, i])
subplot(ICs.shape[1]+1, 1, ICs.shape[1]+1)
plot(yy)
draw()
    
figure(2)
clf()
ICs = ica(sigs)
for i in range(ICs.shape[1]):
    subplot(ICs.shape[1]+1, 1, i+1)
    # plot(ICs[:, i])
    specgram(ICs[:, i], NFFT=32, Fs=250.0, noverlap=16)
subplot(ICs.shape[1]+1, 1, ICs.shape[1]+1)
plot(yy)
draw()
show(block=False)


clf()
n_clusters = 20
arr = Y

kmeans = sklearn.cluster.MiniBatchKMeans(n_clusters=n_clusters)
kmeans.fit(arr[:n_train,])


for i, v in enumerate(vals):
    ix = yy[:n_train] == v
    subplot(len(vals), 2, i*2+1)
    hist(kmeans.predict(arr[ix,:]), normed=True, bins=n_clusters, range=(0, n_clusters-1))
    xlim(0,n_clusters)
    title(v)

    ix = yy[n_train:] == v
    subplot(len(vals), 2, i*2+2)
    hist(kmeans.predict(arr[ix,:]), normed=True, bins=n_clusters, range=(0, n_clusters-1))
    xlim(0,n_clusters)
    title(v)


draw()

