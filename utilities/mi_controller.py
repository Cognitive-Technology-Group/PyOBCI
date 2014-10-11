#!/usr/bin/env python2

import time
import threading
import csv
import numpy as np
from multiprocessing import Process
import sys
import serial
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

        M = 100
        r = 250
        f1, f2 = 6, 10
        
        self.wavelet1 = signal.morlet(M, w=(f1*M)/(2.0*r))
        self.wavelet2 = signal.morlet(M, w=(f2*M)/(2.0*r))

        
        print("connecting to teensy...")
        self.teensy = serial.Serial(TEENSY_PORT, 9600)
        
        
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

        #offsets = []
        val = [0] * 4

        sigs = np.zeros((data.shape[0], 3))
        sigs[..., 0] = (data[..., 0] + data[..., 1])/2.0
        sigs[..., 1] = (data[..., 2] + data[..., 3])/2.0
        sigs[..., 2] = (data[..., 4] + data[..., 5])/2.0

        sig = sigs[..., 2]
        sig = sig[~np.isnan(sig)]
        c1 = signal.convolve(sig, self.wavelet1, 'same')
        c2 = signal.convolve(sig, self.wavelet2, 'same')
        f1 = np.fft.rfft(c1)
        f2 = np.fft.rfft(c2)
        freq2 = np.fft.rfftfreq(sig.size, d=1.0/250.0)

        print(f1.shape, f2.shape)
                
        for i in range(2):
            #subplot(8, 1, i+1)
            #sig = self.data[..., i]
            sig = sigs[..., i]
            sig = sig[~np.isnan(sig)]
            conv1 = signal.convolve(sig, self.wavelet1, 'same')
            conv2 = signal.convolve(sig, self.wavelet2, 'same')
            #sig = sig * np.hamming(len(sig))
            fourier1 = np.fft.rfft(conv1) 
            fourier2 = np.fft.rfft(conv2)
            freq = np.fft.rfftfreq(sig.size, d=1.0/250.0)
            # plot(sig, label=str(i+1))

            #print(freq)
        
            for f, v1, v2 in zip(freq, fourier1, fourier2):
                if f >= 4 and f <= 30:
                # if f >= 8 and f <= 12:
                    val[i] += abs(v1)
                    val[i+2] += abs(v2)

            # for f, v1, v2 in zip(freq2, f1, f2):
            #     if f >= 4 and f <= 30:
            #     # if f >= 8 and f <= 12:
            #         val[i] -= abs(v1)
            #         val[i+2] -= abs(v2)
            #     # if f >= 20 and f <= 40:
            #     #     val[i+2] += abs(v)


                    
        print(val)
        
        controls_f = 0.3 * np.array(val) + 0.7 * self.controls[-1]

        control = controls_f[0] - controls_f[1]
        
        # if control < -0.002 and self.control_f[-1] < 0:
        #     control = -0.002
        # elif control > 0.002 and self.control_f[-1] > 0:
        #     control = 0.002
            
        control_f = 0.4 * control + 0.6 * self.control[-1]

        
        # if control_f < -0.002:
        #     control_f = -0.002
        # elif control_f > 0.002:
        #     control_f = 0.002
        
        control_f2 = 1 * control_f - 1 * self.control[-1]
        control_f3 = 0.6 * control_f2 + 0.4 * self.control_f[-1]
        
        self.control = np.append(self.control, control_f)
        self.control_f = np.append(self.control_f, control_f3)
        self.controls = np.vstack([self.controls, controls_f])


        if control_f < -0.0001:
            out_sig = 2
        elif control_f > 0.0001:
            out_sig = 1
        else:
            out_sig = 0
            
        self.out_sig = np.append(self.out_sig, out_sig)

        self.teensy.write(str(out_sig))
        
        # for i in range(4):
        #     plot(self.controls[-40:, i], label=str(i+1))

        # plot(sig)
        # plot(conv1)
        # plot(conv2)
            
        plot(self.control[-40:])
        # plot(self.control_f[-40:])
        # plot(self.out_sig[-40:] * 0.001)
        # ylim([-0.005, 0.005])

        # ylim([0, 0.02])

        legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=4, mode="expand", borderaxespad=0.)

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
            if len(self.data) > 10:
                self.plot()
            time.sleep(0.1)
        
    def receive_sample(self, sample):
        t = time.time()
        idd = sample.id
        sample = sample.channels
        self.data = np.vstack( (self.data[-250:, ...], sample) )

        
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






