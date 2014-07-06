import mne
import scipy
import numpy as np
import pyeeg

SAMPLING_RATE = 250
bp = mne.filter.band_pass_filter
lpf = mne.filter.low_pass_filter

def rms(signal):
    return np.sqrt(np.sum(signal**2) * 1.0/signal.size)

def num_zero_crossings(signal):
    d = np.append(signal[1:], signal[-1])
    out = 0
    out += np.sum(np.logical_and(signal > 0, d < 0))
    out += np.sum(np.logical_and(signal < 0, d > 0))
    return out


def filter_signal(signal):
    low = lpf(signal, SAMPLING_RATE, 55)
    return low

def get_features(signal):
    #print(signal)

    freq_cutoffs = [3, 8, 12, 27, 50]

    features = []

    features.append(rms(signal))

    s = lpf(signal, SAMPLING_RATE, freq_cutoffs[0])
    features.append(rms(s))

    for i in range(len(freq_cutoffs)-1):
        s = bp(signal, SAMPLING_RATE, freq_cutoffs[i], freq_cutoffs[i+1])
        features.append(rms(s))

    fourier = np.fft.rfft(signal * np.hamming(signal.size))
    features.extend(abs(fourier))

    wsize = 64
    X = mne.time_frequency.stft(signal, wsize, verbose=False)
    freqs = np.reshape(abs(X), X.size)
    features.extend(freqs)

    features.append(pyeeg.hurst(signal))
    features.append(pyeeg.hfd(signal, 10))
    e = pyeeg.spectral_entropy(signal, np.append(0.5, freq_cutoffs), SAMPLING_RATE)
    features.append(e)

    features.extend(pyeeg.hjorth(signal))
    features.append(pyeeg.pfd(signal))
    features.append(pyeeg.mean(signal))

    features.append(scipy.stats.skew(signal))
    features.append(scipy.stats.kurtosis(signal))

    #features.extend(signal)

    return features

def get_features_filter(signal):
    return get_features(filter_signal(np.array(signal)))


def extract_features(signals):
    out = np.array([])
    for i in range(signals.shape[0]):
        sig = signals[i]
        out = np.append(out, get_features_filter(sig))
    return out
# signal = np.random.randn(125)
# print(get_features(filter_signal(signal)))
