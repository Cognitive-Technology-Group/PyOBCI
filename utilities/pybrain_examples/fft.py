
from mne import filter
import numpy as np
import matplotlib.pyplot as plt

def windowed_fft(data, fs):
        """ Applies a Hanning window, calculates FFT, and returns one-sided
        FFT as well as corresponding frequency vector.
        """
        N = len(data)
        window = np.hanning(N)
        win_pow = np.mean(window ** 2)
        windowed_data = np.fft.fft(data * window) / np.sqrt(win_pow)
        pD = np.abs(windowed_data * np.conjugate(windowed_data) / N ** 2)
        freqs = np.fft.fftfreq(N, 1 / float(fs))
        f = freqs[:N / 2 ]
        pD = pD[:N / 2 ]
        pD[1:] = pD[1:] * 2
        return pD, f

#Graphing purposes
#hz_per_bin = float(SAMPLE_RATE) / self.n_fft
#min_psds = []
#max_psds = []
#y = data to process
#y = y - y.mean()
#psd, f = _windowed_fft(y, SAMPLE_RATE)
def mne_bandpass_filter(self):
        filter.band_pass_filter(self.data, self.fs, self.lowcut, self.highcut)
        return mne_filter


# from scipy.signal import lfilter
# class TimeDomainFilter(object):
#         def __init__(self, a, b, signal):
#                 self.a = []
#                 self.b = []
#                 self.signal = signal
   
#         def apply(self, signal):
#                  return lfilter(self.b, self.a, signal)

# from scipy.io import loadmat
# mat = loadmat('bp_filter_coeff.mat')
# filters = [TimeDomainFilter(b=mat['bp_filter_coeff']['b'][0, 0].squeeze(),
#                                 a=mat['bp_filter_coeff']['a'][0, 0].squeeze()),
#         TimeDomainFilter(b=mat['bp_filter_coeff']['b_notch'][0, 0].squeeze(),
#                                  a=mat['bp_filter_coeff']['a_notch'][0, 0].squeeze()), ]
# for filter in filters:
#         y = TimeDomainFilter.apply(y)

if "__name__"=="__main__":
        hz_per_bin = float(250) / 256
        data = np.genfromtxt('data/sample.csv', delimiter=',')
        filt_data = filter.band_pass_filter(data[:, 4], 250, 1, 60)
        psd, fft_data = _windowed_fft(filt_data, 250)
        psd_per_bin = psd / hz_per_bin


        plt.figure(1)
        plt.plot(fft_data, np.sqrt(psd_per_bin))

        plt.figure(2)
        plt.plot(filt_data)
        plt.show() 

                          
