import pandas
from scipy import signal
import numpy as np
from pprint import pprint
from pylab import *

d = pandas.read_csv("motor_data_tomas.csv")
d = d.dropna()
d = d.reset_index(drop=True)

data = np.array(d)

M = 30
r = 250 # sampling rate
f1, f2 = 10, 22

W_n = r / 2.0
b, a = signal.butter(4, [7.0 / W_n, 14. / W_n], 'bandpass')

# 7-14 abs hilbert
# 18-25 abs hilbert

# sig = (data[..., 0] + data[..., 1])/2.0
sig = (data[..., 2] + data[..., 3])/2.0
# sig = (data[..., 4] + data[..., 5])/2.0
x = signal.lfilter(b, a, sig)
x = np.abs(signal.hilbert(x))

clusters = x[box_width:]
y = np.array(d.tag[box_width:])
# y = (y==0) * 1

clf()
l, h = min(clusters), max(clusters)

ys = np.unique(y)

for i, yy in enumerate(ys):
    subplot(4, 1, i+1)
    # hist(clusters[np.logical_and(y == yy, clusters != -1)]) #, bins=n_clusters/5)
    hist(clusters[y == yy], range=(l, h), bins=50)
    title(yy)

draw()
show(block=False)


# time embed the features n times, spaced k apart
def time_embed(features, y, k, n):
    total = n*k
    out = features[total:,]
    for i in range(n-1, -1, -1):
        current = i*k
        left = total - current
        out = np.hstack([out, features[current:-left]])
    return out, y[total:]

N_train = 15000

c = clusters[:, np.newaxis]

k, n = 50, 10
c2, y2 = time_embed(c, y, k, n)
# c2, y2 = c, y

# model = KNeighborsClassifier(n_neighbors=2, weights='distance')
rf = KNeighborsClassifier(n_neighbors=6, weights='distance')
# rf = RandomForestClassifier(max_depth=10, n_estimators=3, max_features=1)
# rf = AdaBoostClassifier(n_estimators=100) # to try
rf.fit(c2[:N_train], y2[:N_train])
train = rf.score(c2[:N_train], y2[:N_train])
test = rf.score(c2[N_train:], y2[N_train:])
print("train", train, "test", test)

pred = rf.predict(c2)

smoother_len = 1000
# smoother = np.repeat(1.0/smoother_len, smoother_len)
smoother = np.exp(-0.001 * np.arange(0,smoother_len))
smoother = smoother / sum(smoother)

s = pred
# s = signal.wiener(s, 200)
s = signal.convolve(s, smoother, 'same')
# s = signal.medfilt(s, 11)

# s = s - s.mean()

error = abs(s - y2)
index = np.arange(len(error))
e_train = error[np.logical_and(index < N_train, y2 != 0)]
e_test = error[np.logical_and(index >= N_train, y2 != 0)]

print("RMS error on train", np.sqrt(e_train.mean()))
print("RMS error on test", np.sqrt(e_test.mean()))

print("proportion misclass on train", (np.abs(e_train > 0.95).mean()))
print("proportion misclass on test", (np.abs(e_test > 0.95).mean()))

print("proportion misclass on train 1", (np.abs(e_train > 0.45).mean()))
print("proportion misclass on test 1", (np.abs(e_test > 0.45).mean()))

subplot(4,1,4)
hold(False)
plot(s)
hold(True)
plot(y2)
vlines(N_train, -2, 2, linestyles='solid')

draw()
show(block=False)
