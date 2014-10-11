#!/usr/bin/env python2

import time
import threading
import csv
import numpy as np
from multiprocessing import Process
import sys
import serial

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

class MIPlotter(object):

    def __init__(self, port=None, baud=115200):
        self.board = OpenBCIBoard(port, baud)
        self.bg_thread = None
        self.bg_draw_thread = None
        self.data = np.array([0]*8)
        self.should_plot = False
        self.control = np.array([0])
        self.out_sig = np.array([0])
        self.controls = np.array([[0]*4])

        
        
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

        #offsets = []
        val = [0] * 4
        
        for i in range(2):
            #subplot(8, 1, i+1)
            signal = self.data[..., i]
            signal = signal[~np.isnan(signal)]
            #signal = signal * np.hamming(len(signal))
            fourier = np.fft.rfft(signal)
            freq = np.fft.rfftfreq(signal.size, d=1.0/250.0)
            # plot(signal, label=str(i+1))

            #print(freq)
            
            for f, v in zip(freq, fourier):
                if f >= 14 and f <= 20:
                    val[i] += abs(v)
                # if f >= 20 and f <= 40:
                #     val[i+2] += abs(v)


                    
        print(val)
        
        controls_f = 0.35 * np.array(val) + 0.65 * self.controls[-1]

        control = controls_f[0] - controls_f[1]
        control_f = 0.1 * control + 0.9 * self.control[-1]

        
        self.control = np.append(self.control, control_f)
        self.controls = np.vstack([self.controls, controls_f])


        if control_f < -0.0001:
            out_sig = 2
        elif control_f > 0.0001:
            out_sig = 1
        else:
            out_sig = 0
            
        self.out_sig = np.append(self.out_sig, out_sig)
        
        # for i in range(2):
        #     plot(self.controls[-40:, i], label=str(i+1))

        plot(self.control[-40:])
        plot(self.out_sig[-40:] * 0.001)
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
            time.sleep(0.05)
        
    def receive_sample(self, sample):
        t = time.time()
        sample = sample.channels
        self.data = np.vstack( (self.data[-125:, ...], sample) )

        
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
    plotter = MIPlotter(port='/dev/ttyACM0')
    plotter.start()

    plt.rc('axes', color_cycle=['red', 'orange', 'yellow', 'green'])






