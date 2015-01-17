#!/usr/bin/env python2

import gc
import pandas
import numpy as np
# from preprocess import *

gc.collect()

d = pandas.read_csv("motor_data_derrick.csv")
d = d.dropna()
d = d.reset_index(drop=True)

data = np.array(d[:])

sigs = np.zeros((data.shape[0], 3), dtype='float32')
sigs[..., 0] = (data[..., 0] + data[..., 1])/2.0
sigs[..., 1] = (data[..., 2] + data[..., 3])/2.0
sigs[..., 2] = (data[..., 4] + data[..., 5])/2.0

y = np.array(d.tag)[:, np.newaxis]

preprocess = EEGFeatures()
reduction = FisherFeatures()
rcorr = RemoveCorr()

flow = mdp.Flow([preprocess, reduction, rcorr], verbose=1)

inp = [[sigs], [(sigs, y)], [sigs]]

flow.train(inp)

te = TimeEmbed(n_times = 3, k_spaces = 2)

# out = preprocess(sigs)

# reduction.train(out, y)
# out = reduction(out)

# rcorr.train(out)
# out = rcorr(out)

