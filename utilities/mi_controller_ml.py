#!/usr/bin/env python2

import time
import threading
import csv
import numpy as np
from multiprocessing import Process
import sys
import serial
from sklearn.externals import joblib
from scipy import signal

sys.path.append('..')
from open_bci import *
from pylab import *

#'/dev/tty.usbmodemfd121'
#Notes for using csv_collector.py
#initiate CSVCollector, takes filename, port, and baud as inputs
#start recording with start() method
#tag recording with tag() method
#what needs to be implemented
#   A if statement for multiprocessing/threading.

OPENBCI_PORT = '/dev/ttyACM0'
TEENSY_PORT = '/dev/ttyACM1'
TEENSY_ENABLED = False

box_width = 250
look_back = 60
M = 25
r = 250
f1, f2 = 7, 14

wavelet1 = signal.morlet(M, w=(f1*M)/(2.0*r))
wavelet2 = signal.morlet(M, w=(f2*M)/(2.0*r))

def extract_features(data):
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
        features = np.hstack([features, np.angle(fourier), np.angle(fourier1), np.angle(fourier2)])

        
    return features


class MIPlotter(object):

    def __init__(self, port=None, baud=115200):
        print("connecting to OpenBCI...")
        self.board = OpenBCIBoard(port, baud)
        
        self.bg_thread = None
        self.bg_draw_thread = None
        self.data = np.array([0]*8)
        self.should_plot = False
        self.control = np.array([0,0,0])
        self.control_f = np.array([0])
        self.out_sig = np.array([0])
        self.controls = np.array([[0]*4])
        self.eye_r = np.array([0])
        self.eye_l = np.array([0])

        model, good_features = joblib.load('neighbors_model.pkl')
        self.eye_l_temp, self.eye_r_temp = joblib.load('eye_blinks.pkl')
        self.good_features = good_features
        self.model = model
        
        print("connecting to teensy...")
        if TEENSY_ENABLED:
            self.teensy = serial.Serial(TEENSY_PORT, 57600)
        
        
    def stop(self):
        # resolve files and stuff
        self.board.should_stream = False
        self.should_plot = False
        #self.bg_thread.join()
        self.bg_thread = None
        self.data = np.array([0]*8)
        
    def disconnect(self):
        self.board.disconnect()

    def plot(self):

        plt.clf()

        hold(True)
        data = np.copy(self.data)

        
        control_arr = np.zeros(look_back)

        for i in range(look_back):
            d = data[i:(box_width+i)]
            features = extract_features(d)
            features = features[self.good_features]
            control_arr[i] = self.model.predict(features)

        control = control_arr.mean()
        control_f = 0.1 * control + 0.8 * self.control[-1] 

        
        # control_f2 = 1 * control_f - 1 * self.control[-1]
        # control_f3 = 0.6 * control_f2 + 0.4 * self.control_f[-1]
        
        self.control = np.append(self.control, control_f)
        # self.control_f = np.append(self.control_f, control_f3)
        # self.controls = np.vstack([self.controls, controls_f])


        # out_sig = control
        
        if control_f < -0.07:
            out_sig = 550
        elif control_f > 0.07:
            out_sig = 450
        else:
            out_sig = 500


        r_max = np.max(signal.correlate(self.eye_r_temp, data[..., 6]))
        l_max = np.max(signal.correlate(self.eye_l_temp, data[..., 7]))

        self.eye_r = np.append(self.eye_r, r_max)
        self.eye_l = np.append(self.eye_l, l_max)
        
        
        self.out_sig = np.append(self.out_sig, out_sig)

        if TEENSY_ENABLED:
            self.teensy.write("0" + str(out_sig) + "\r")
        
        # for i in range(4):
        #     plot(self.controls[-40:, i], label=str(i+1))

        # plot(sig)
        # plot(conv1)
        # plot(conv2)
            
        plot(self.control[-40:])
        plot(self.out_sig[-40:] * 0.001)

        # plot(self.eye_l[-40:])
        # plot(self.eye_r[-40:])
        # plot((self.eye_r[-40:] > 0.00011) * 0.001)
        
        # plot(self.control_f[-40:])
        # plot(self.out_sig[-40:] * 0.001)
        # ylim([-0.005, 0.005])

        # ylim([0, 0.02])

        # legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        #        ncol=4, mode="expand", borderaxespad=0.)

            #plot(freq, np.log(abs(fourier)), label=str(i+1))
            
            #title('channel {0}'.format(i+1))
        # ylim([-0.0005, 0.0005])
        # ylim([-12, 0])
        # xlim([0, 60])
        # legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
        #        ncol=4, mode="expand", borderaxespad=0.)

        # plot(freq, np.log(abs(fourier)))
        
        show(block=False)
        draw()
        
    def background_plot(self):
        while self.should_plot:
            if len(self.data) >= box_width + look_back:
                self.plot()
            time.sleep(0.05)
        
    def receive_sample(self, sample):
        t = time.time()
        idd = sample.id
        sample = sample.channels
        if not np.any(np.isnan(sample)):
            self.data = np.vstack( (self.data[-250-look_back:, ...], sample) )

        
    def start(self):
        
        if self.bg_thread:
            self.stop()

            
        #create a new thread in which the OpenBCIBoard object will stream data
        self.bg_thread = threading.Thread(target=self.board.start, 
                                        args=(self.receive_sample, ))
        # self.bg_thread = Process(target=self.board.start,
        #                         args=(self.receive_sample, ))

        self.bg_thread.start()

        # self.bg_draw_thread = threading.Thread(target=self.background_plot,
        #                                        args=())

        # self.bg_draw_thread.start()
        
        ion()
        figure()
        show(block=False)

        self.should_plot = True
        
        self.background_plot()

if __name__ == '__main__':
    plotter = MIPlotter(port=OPENBCI_PORT)
    plotter.start()

    plt.rc('axes', color_cycle=['red', 'orange', 'yellow', 'green'])






