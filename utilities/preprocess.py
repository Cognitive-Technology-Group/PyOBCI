#!/usr/bin/env python2

import numpy as np
from scipy import signal
import scipy
import mdp

import pyeeg
import eegtools

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

# time embed the features n times, spaced k apart
def time_embed_y(features, y, k, n):
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

def lpf(sig, cutoff, sampling_rate):
    Wn = cutoff / (sampling_rate / 2.0)
    b, a = signal.butter(4, Wn, 'lowpass')
    return signal.lfilter(b, a, sig)

def bp(sig, low, high, sampling_rate):
    if low <= 0:
        return lpf(sig, high, sampling_rate)
    
    w_low = low / (sampling_rate / 2.0)
    w_high = high / (sampling_rate / 2.0)
    b, a = signal.butter(4, (w_low, w_high), 'bandpass')
    return signal.lfilter(b, a, sig)

def rms(signal):
    return np.sqrt(np.sum(signal**2) * 1.0/signal.size)

def get_features(sig, sampling_rate = 250.0):
    #print(signal)

    # freq_cutoffs = [3, 8, 12, 27, 40, 59, 61, 80, 100, 120]

    features = []

    # features.append(rms(sig))

    # s = lpf(sig, freq_cutoffs[0], sampling_rate)
    # features.append(rms(s))

    # for i in range(len(freq_cutoffs)-1):
    #     s = bp(sig, freq_cutoffs[i], freq_cutoffs[i+1], sampling_rate)
    #     features.append(rms(s))

    # fourier = np.fft.rfft(sig * np.hamming(sig.size))
    # features.extend(abs(fourier))

    # features.append(pyeeg.hurst(sig))
    # features.append(pyeeg.hfd(sig, 10))
    # e = pyeeg.spectral_entropy(sig, np.append(0.5, freq_cutoffs), sampling_rate)
    # features.append(e)

    # very fast, but slow overall =/
    features.extend(pyeeg.hjorth(sig))
    features.append(pyeeg.pfd(sig))

    features.append(np.mean(sig))
    features.append(np.std(sig))

    features.append(scipy.stats.skew(sig))
    features.append(scipy.stats.kurtosis(sig))

    #features.extend(sig)

    return features

# for use with EEGFeatures
def smooth_out_y(y, box_width):
    out = np.copy(y).astype('float64')
    
    for start in range(box_width, y.shape[0]):
        if start % 5000 == 0:
            print("{0:.2%}".format(float(start) / y.shape[0]))

        # fft_len = np.fft.rfft(data[..., 0]).shape[0]
        out[start] = np.mean(y[(start-box_width+1):(start+1)])

    return out


class EEGFeatures(mdp.Node):
    def __init__(self, sampling_rate=250, box_width=250, M=25, wavelets_freqs=(10, 22),
                 input_dim=None, dtype=None):
        super(EEGFeatures, self).__init__(input_dim=input_dim, dtype=dtype)

        self.sampling_rate = sampling_rate
        self.box_width = box_width
        self.M = M
        self.wavelets_freqs = wavelets_freqs

        self.wavelets = list()
        for f in wavelets_freqs:
            wavelet = signal.morlet(M, w=(f*M)/(2.0*sampling_rate))
            self.wavelets.append(wavelet)

    def _set_input_dim(self, n):
        n_wavelets = 1 + len(self.wavelets)

        self._input_dim = n
        self.output_dim = self.box_width * n * n_wavelets * 1

    def is_trainable(self):
        return False
    def is_invertible(self):
        return False
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    
    def _execute(self, X):

        n_signals = X.shape[1]
        n_wavelets = 1 + len(self.wavelets)

        features_arr = np.zeros( (X.shape[0], # rows
                                  # cols, FFT len * n_signals * n_wavelets * 1 (abs, no angle)
                                  self.box_width * n_signals * n_wavelets * 1)  )

        ss = [np.zeros(X.shape) for i in range(n_wavelets)]
        ss[0] = np.copy(X)
        for i in range(1, n_wavelets):
            wavelet = self.wavelets[i-1]
            for j in range(n_signals):
                ss[i][:, j] = signal.convolve(X[:, j], wavelet, 'same')


        for start in range(self.box_width, X.shape[0]):
            if start % 5000 == 0:
                print("{0:.2%}".format(float(start) / X.shape[0]))

            # fft_len = np.fft.rfft(data[..., 0]).shape[0]
            features = np.array([])

            for i in range(n_wavelets):
                sigs = np.array(ss[i][(start - self.box_width+1):(start+1)])
                fourier = np.fft.fft(sigs, axis=0)
                for j in range(n_signals):
                    features = np.hstack([features, np.abs(fourier[..., j])])

            features_arr[start, ...] = features

        return features_arr



class EEGFeatures2(mdp.Node):
    def __init__(self, sampling_rate=250, box_width=250, input_dim=None, dtype=None):
        super(EEGFeatures2, self).__init__(input_dim=input_dim, dtype=dtype)

        self.sampling_rate = sampling_rate
        self.box_width = box_width

    def _set_input_dim(self, n):
        test_sig = np.zeros(self.box_width)
        x = get_features(test_sig, self.sampling_rate)
        fnum = len(x)

        self._input_dim = n
        self.output_dim = fnum * n

    def is_trainable(self):
        return False
    def is_invertible(self):
        return False
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, X):

        n_signals = X.shape[1]

        test_sig = np.zeros(self.box_width)
        x = get_features(test_sig, self.sampling_rate)
        fnum = len(x)
        
        features_arr = np.zeros( (X.shape[0], # rows
                                  fnum * n_signals) ) # cols

        for start in range(self.box_width, X.shape[0]):
            if start % 500 == 0:
                print("{0:.2%}".format(float(start) / X.shape[0]))

            features = np.array([])

            sigs = np.array(X[(start - self.box_width+1):(start+1)])

            for i in range(n_signals):
                ff = get_features(sigs[:, i], self.sampling_rate)
                features = np.hstack([features, ff])
            
            features_arr[start, ...] = features

        return features_arr


# TODO: make this node into a filtering + rms node
class EEGFeatures3(mdp.Node):
    def __init__(self, sampling_rate=250, box_width=250, input_dim=None, dtype=None):
        super(EEGFeatures3, self).__init__(input_dim=input_dim, dtype=dtype)

        self.sampling_rate = sampling_rate
        self.box_width = box_width

    def _set_input_dim(self, n):
        test_sig = np.zeros(self.box_width)
        x = get_features(test_sig, self.sampling_rate)
        fnum = len(x)

        self._input_dim = n
        self.output_dim = fnum * n

    def is_trainable(self):
        return False
    def is_invertible(self):
        return False
    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, X):

        n_signals = X.shape[1]

        freq_cutoffs = [0, 3, 8, 12, 27, 40, 59, 61, 80, 100, 120]

        n_bins = len(freq_cutoffs)
        
        ss = [np.zeros(X.shape) for i in range(n_bins)]
        ss[0] = np.copy(X) # lowest bin is no cutoff

        for i in range(1, len(freq_cutoffs)):
            for j in range(n_signals):
                ss[i][:, j] = bp(X[:, j],
                                 freq_cutoffs[i-1], freq_cutoffs[i],
                                 self.sampling_rate)

        n_features = n_signals * n_bins
                
        features_arr = np.zeros( (X.shape[0], # rows
                                  n_features)  )

                
        for start in range(self.box_width, X.shape[0]):
            if start % 5000 == 0:
                print("{0:.2%}".format(float(start) / X.shape[0]))

            features = np.zeros(n_features)

            for i in range(n_bins):
                sigs = np.array(ss[i][(start - self.box_width+1):(start+1)])
                ff = np.sqrt(np.mean(np.square(sigs), axis=0)) # root mean square
                begin = i * n_signals
                end = (i+1) * n_signals
                features[begin:end] = ff
                
            features_arr[start, ...] = features

        return features_arr


    
class FisherFeatures(mdp.Node):
    def __init__(self, output_dim=200, labelA = 0, labelB = 1, input_dim=None, dtype=None):
        super(FisherFeatures, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)

        self.labelA = labelA
        self.labelB = labelB
        self.output_dim = output_dim

        self.indexes = None

    def is_trainable(self):
        return True
    def is_invertible(self):
        return False

    def _train(self, X, labels):
        if len(labels.shape) == 2:
            y = labels[:, 0]
        else:
            y = labels

        good_features = fish_good_features(X, y, self.labelA, self.labelB, self.output_dim)

        indexes = np.arange(X.shape[1])
        self.indexes = indexes[good_features]

    def _execute(self, X):
        return X[:, self.indexes]

class RemoveCorr(mdp.Node):
    def __init__(self, threshold=0.95):
        super(RemoveCorr, self).__init__()

        self.threshold = threshold

        self.indexes = None

    def is_trainable(self):
        return True
    def is_invertible(self):
        return False

    def _train(self, X):
        indexes = np.arange(X.shape[1])
        good_features = remove_corr_good(X, self.threshold)
        self.indexes = indexes[good_features]
        self.output_dim = len(self.indexes)

    def _execute(self, X):
        return X[:, self.indexes]

# uncorrelated fisher features
class FisherFeaturesUncorr(mdp.Node):
    def __init__(self, output_dim=15, labelA = 0, labelB = 1, threshold=0.95):
        super(FisherFeaturesUncorr, self).__init__()

        self.labelA = labelA
        self.labelB = labelB
        self.output_dim = output_dim
        self.threshold = threshold

        self.indexes = None

    def is_trainable(self):
        return True
    def is_invertible(self):
        return False

    def _train(self, X, labels):
        if len(labels.shape) == 2:
            y = labels[:, 0]
        else:
            y = labels

        indexes = np.arange(X.shape[1])

        multiplier = 6
        if self.output_dim >= 250:
            multiplier = 2.5
        elif self.output_dim >= 100:
            multiplier = 3

        good_features = fish_good_features(X, y, self.labelA, self.labelB, self.output_dim * multiplier)
        indexes = indexes[good_features]
        print(len(indexes))

        good_features = remove_corr_good(X[:, indexes], self.threshold)
        indexes = indexes[good_features]
        print(len(indexes))

        good_features = fish_good_features(X[:, indexes], y, self.labelA, self.labelB, self.output_dim)
        indexes = indexes[good_features]
        print(len(indexes))

        self.indexes = indexes
        self.output_dim = len(self.indexes) # just in case

    def _execute(self, X):
        return X[:, self.indexes]



class LowpassFilter(mdp.Node):
    def __init__(self, N, Wn):
        super(LowpassFilter, self).__init__()

        self.Wn = Wn

        b, a = signal.butter(N, Wn, 'lowpass')
        self.b = b
        self.a = a

    def _set_input_dim(self, n):
        self._input_dim = n
        self.output_dim = n

    def is_trainable(self):
        return False
    def is_invertible(self):
        return False

    def _execute(self, X):
        out = signal.lfilter(self.b, self.a, X, axis=0)
        return out


class BandpassFilter(mdp.Node):
    def __init__(self, f_low, f_high, sampling_rate, input_dim=None, N=4):
        super(BandpassFilter, self).__init__(input_dim=input_dim)

        w_low = f_low / (sampling_rate / 2.0)
        w_high = f_high / (sampling_rate / 2.0)

        self.w_low = w_low
        self.w_high = w_high
        self.N = N
        self.sampling_rate = sampling_rate
        
        b, a = signal.butter(N, (w_low, w_high), 'bandpass')
        self.b = b
        self.a = a

    def _set_input_dim(self, n):
        self._input_dim = n
        self.output_dim = n

    def is_trainable(self):
        return False
    def is_invertible(self):
        return False

    def _execute(self, X):
        out = signal.lfilter(self.b, self.a, X, axis=0)
        return out

    
    
class MedianFilter(mdp.Node):
    def __init__(self, n):
        super(MedianFilter, self).__init__()

        self.n = n

    def _set_input_dim(self, n):
        self._input_dim = n
        self.output_dim = n

    def is_trainable(self):
        return False
    def is_invertible(self):
        return False

    def _execute(self, X):
        out = np.copy(X)
        for i in range(X.shape[1]):
            out[:, i] = signal.medfilt(X[:, i], self.n)
        return out

# to be used after filtering with ICA
# will find ICA components with artifacts and EXTERMINATE them.
class RemoveArtifacts(mdp.Node):
    def __init__(self, ignore_first=1000,
                 remove_muscle=True, remove_electricity=True,
                 elec_freq=120, sampling_rate=250):
        super(RemoveArtifacts, self).__init__()
        
        self.remove_muscle = remove_muscle
        self.remove_electricity = remove_electricity
        self.ignore = ignore_first
        
        self.indexes = None

        self.sampling_rate = sampling_rate
        self.elec_freq = elec_freq
        
    def is_trainable(self):
        return True
    def is_invertible(self):
        return False

    def _train(self, X):
        self.indexes = np.arange(X.shape[1])
        
        if self.remove_muscle:
            kurt = scipy.stats.kurtosis(X[self.ignore:], axis=0)
            ix = np.argmax(kurt)
            # print(ix)
            self.indexes = np.delete(self.indexes, ix)

        if self.remove_electricity:
            nrows = X[self.ignore:].shape[0]
            fftfreq = np.fft.rfftfreq(nrows, 1.0/self.sampling_rate)
            fft = np.abs(np.fft.rfft(X[self.ignore:], axis=0))
            e = self.elec_freq
            t = np.logical_and(fftfreq >= e-0.5, fftfreq <= e+0.5)
            m = np.sum(fft[t,], axis=0)
            ix = np.argmax(m)
            # print(ix)
            self.indexes = np.delete(self.indexes, ix)

    def _execute(self, X):
        return X[:, self.indexes]
        

class CSP(mdp.Node):
    def __init__(self, labelA = 0, labelB = 1, input_dim=None, m = None):
        super(CSP, self).__init__(input_dim=input_dim)

        self.labelA = labelA
        self.labelB = labelB
        self.m = m

        self.W = None

        self._covA = mdp.utils.CovarianceMatrix()
        self._covB = mdp.utils.CovarianceMatrix()
        
    def is_trainable(self):
        return True
    def is_invertible(self):
        return False
        
    
    def _train(self, X, y):
        self._covA.update(X[y == self.labelA])
        self._covB.update(X[y == self.labelB])

    def _stop_training(self):
        # print(X.shape[0])
        
        # covA = np.cov(X[y == self.labelA].T)
        # covB = np.cov(X[y == self.labelB].T)
        covA, avgA, tlenA = self._covA.fix()
        covB, avgB, tlenB = self._covB.fix()

        if not self.m:
            self.m = covA.shape[1]
        
        self.W = eegtools.spatfilt.csp(covA, covB, self.m)

        self.output_dim = self.m
        
    
    def _execute(self, X):
        return np.dot(self.W, X.T).T



class VarianceWindow(mdp.Node):
    def __init__(self, box_width=250):
        super(VarianceWindow, self).__init__()

        self.box_width = box_width
        
    def is_trainable(self):
        return False
    def is_invertible(self):
        return False

    def _execute(self, X):
        out = np.zeros(X.shape)

        for start in range(self.box_width):
            sigs = X[:(start+1)]
            out[start, :] = np.var(sigs, axis=0)
        
        for start in range(self.box_width, X.shape[0]):
            sigs = X[(start - self.box_width+1):(start+1)]
            out[start, :] = np.var(sigs, axis=0)
            
        return out
        

class LogVarianceWindow(mdp.Node):
    def __init__(self, box_width=250):
        super(LogVarianceWindow, self).__init__()

        self.box_width = box_width
        
    def is_trainable(self):
        return False
    def is_invertible(self):
        return False

    def _execute(self, X):
        out = np.zeros(X.shape)

        for start in range(1, self.box_width):
            sigs = X[:(start+1)]
            out[start, :] = np.log(np.var(sigs, axis=0))
        
        for start in range(self.box_width, X.shape[0]):
            sigs = X[(start - self.box_width+1):(start+1)]
            out[start, :] = np.log(np.var(sigs, axis=0))
            
        return out


class GaussianClassifierArray(mdp.nodes.GaussianClassifier):
    def label(self, X):
        out = np.array(super(GaussianClassifierArray, self).label(X))
        if len(out.shape) == 1:
            out = out[:, np.newaxis]
        return out

# data is a time series
# this function splits it into chunks by value of y
# optional argument labels specifies which labels are kept
def split_data_by_label(data, y, labels=None):
    out_data = list()
    out_y = list()
    
    start = 0
    curr = y[0]
    for i in range(data.shape[0]):
        if y[i] != curr:
            if (labels == None) or (curr in labels):
                out_data.append(data[start:i])
                out_y.append(curr)
            start = i
            curr = y[i]

    if (labels == None) or (curr in labels):
        out_data.append(data[start:i])
        out_y.append(curr)
    
    return out_data, out_y

# splits DATA and Y evenly into N_CHUNKS
# if LABELS is given, only those labels will be kept in y, and
# corresponding data
def split_data_by_chunks(data, y, n_chunks, labels=None):
    if labels != None:
        yy = y
        if len(yy.shape) > 1:
            yy = yy[:, 0]
            
        ix = yy == labels[0]
        for L in labels:
            ix = np.logical_or(ix, yy == L)

        data = data[ix]
        y = y[ix]
    
    chunk_len = int(data.shape[0] / n_chunks)

    out_data = list()
    out_y = list()
    
    for i in range(n_chunks):
        start = chunk_len * i
        end = start + chunk_len
        out_data.append(data[start:end])
        out_y.append(y[start:end])

    return out_data, out_y

def multi_bandpass_layer(fs, input_dim, sampling_rate=250):
    bps = list()
    for (low, high) in fs:
        bp = BandpassFilter(low, high, input_dim=input_dim, sampling_rate=sampling_rate)
        bps.append(bp)

    return mdp.hinet.Layer(bps)

