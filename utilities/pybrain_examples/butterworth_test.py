# #John Naulty##
##Modified from http://wiki.scipy.org/Cookbook/ButterworthBandpass ###

# This is an example to use.
# This class takes in a single column (channel) of EEG data and performs a bandpass filter
# test_data = np.genfromtxt('data/worldcup0.csv', delimiter=',')
# filtered_stream = Streamfilter(test_data[:, 0])
# y = filtered_stream.butter_bandpass_filter()



from scipy.signal import butter, lfilter
import numpy as np
from mne import filter


class Streamfilter(object):
    def __init__(self, data, lowcut=1, highcut=60, fs=250, order=5):
        self.data = data
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        self.nyq = .5*fs

    def butter_bandpass(self):
        self.low = self.lowcut / self.nyq
        self.high = self.highcut / self.nyq
        b, a = butter(self.order, [self.low, self.high], btype='band')
        return b, a



    def butter_bandpass_filter(self):
        b, a = self.butter_bandpass()
        filtered_data = lfilter(b, a, self.data)
        return filtered_data

    def mne_bandpass_filter(self):
        filter.band_pass_filter(self.data, self.fs, self.lowcut, self.highcut)
        return mne_filter

# test_data = np.genfromtxt('data/worldcup0.csv', delimiter=',')
# for data[:]
#     filtered_stream = Streamfilter(test_data[:, i])
# y = filtered_stream.butter_bandpass_filter()



    #
    # if __name__ == "__main__":
    #     import numpy as np
    #     import matplotlib.pyplot as plt
    #     from scipy.signal import freqz
    #
    #     # Sample rate and desired cutoff frequencies (in Hz).
    #     fs = 250.0
    #     lowcut = 1.5
    #     highcut = 60.0
    #
    #
    #     # grab data
    #     data = np.genfromtxt('data/worldcup0.csv', delimiter=',')
    #
    #     # T = time in seconds of sample
    #     T = 40
    #     nsamples = T * fs
    #
    #     #for graphing purposes, space lines evenly
    #     t = np.linspace(0, T, nsamples, endpoint=False)
    #     a = 0.02
    #
    #
    #     #grab channel data
    #     channel_1 = data [:, 0]
    #
    #     #figure 1 plots noisy signal and filtered signal
    #     plt.figure(1)
    #     plt.clf()
    #     plt.plot(t, channel_1, label='Noisy signal')
    #
    #     y = butter_bandpass_filter(channel_1, lowcut, highcut, fs, order=6)
    #     plt.plot(t, y, label='Filtered signal (%g Hz - %g Hz)' % (lowcut, highcut))
    #     plt.xlabel('time (seconds)')
    #     plt.hlines([-a, a], 0, T, linestyles='--')
    #     plt.grid(True)
    #     plt.axis('tight')
    #     plt.legend(loc='upper left')
    #
    #     #figure 2 plots just the filtered signal
    #     plt.figure(2)
    #     plt.plot(t, y)
    #     plt.xlabel('time (seconds)')
    #     plt.hlines([-a, a], 0, T, linestyles='--')
    #     plt.grid(True)
    #     plt.axis('tight')
    #     plt.legend(loc='upper left')
    #
    #     #figure 3 plots using the awesome MNE bandpass filter!!!!
    #     plt.figure(3)
    #     plt.plot(filter.band_pass_filter(channel_1, fs, lowcut, highcut))
    #     plt.xlabel('time (seconds)')
    #     plt.hlines([-a, a], 0, T, linestyles='--')
    #     plt.grid(True)
    #     plt.axis('tight')
    #     plt.legend(loc='upper left')
    #     plt.show()