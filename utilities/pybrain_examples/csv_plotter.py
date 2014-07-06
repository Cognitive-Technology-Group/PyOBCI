
####CSV_Plotter####
####John Naulty####

import csv
import numpy as np
import matplotlib.pyplot as plt
import os.path

# class CSV_plotter(ojbect):

# 	def __init__(self, fname):
# 		self.fname = fname

# 	def loader(fname):
# 		self.data = np.genfromtxt('fname', delimiter=',')

# 	def plotter(data):
# 		self.plt.plot(data)
# 		self.plt.show()

# csv_plotter = CSV_plotter('collect0.csv')
# csv_plotter.plottter

## put csv file into workable python data.

data = np.genfromtxt('motor0.csv', delimiter=',')

## create subplot with 8 channels (each column==channel)
plt.figure(1)
for x in range(8):
	plt.subplot(8,1, x)
	plt.plot(np.fft.fft(data[:x]))

plt.figure(2)
for x in range(8):
	plt.subplot(8,1,x)
	plt.plot(data[:,x])
# plt.plot(np.fft.fft(data[:,0]))
# plt.plot(data[:,0])
plt.show()	

plt.specgram(data[:,0])
plt.show()


