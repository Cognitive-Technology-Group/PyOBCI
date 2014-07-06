
##John Naulty##
##Modified from http://wiki.scipy.org/Cookbook/ButterworthBandpass ###


from scipy.signal import butter, lfilter
import argparse
from mne import filter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz

    # Sample rate and desired cutoff frequencies (in Hz).
    fs = 250.0
    lowcut = 1.5
    highcut = 60.0


    # grab data
    data = np.genfromtxt('data/worldcup0.csv', delimiter=',')

    # T = time in seconds of sample
    T = 40
    nsamples = T * fs

    #for graphing purposes, space lines evenly
    t = np.linspace(0, T, nsamples, endpoint=False)
    a = 0.02


    #grab channel data
    channel_1 = data [:, 0]

    #figure 1 plots noisy signal and filtered signal
    plt.figure(1)
    plt.clf()
    plt.plot(t, channel_1, label='Noisy signal')

    y = butter_bandpass_filter(channel_1, lowcut, highcut, fs, order=6)
    plt.plot(t, y, label='Filtered signal (%g Hz - %g Hz)' % (lowcut, highcut))
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    #figure 2 plots just the filtered signal
    plt.figure(2)
    plt.plot(t, y)
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')

    #figure 3 plots using the awesome MNE bandpass filter!!!!
    plt.figure(3)
    plt.plot(filter.band_pass_filter(channel_1, fs, lowcut, highcut))
    plt.xlabel('time (seconds)')
    plt.hlines([-a, a], 0, T, linestyles='--')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.show()