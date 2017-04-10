#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 04:53:34 2017

@author: kim
"""

import h5py
import numpy as np

train_filename = 'rnn_data.hdf5'
#0:h 1:e 2:l 3:o
data = np.array((0,1,2,2)).reshape((4,1,1))
clip = np.array((1,2,3,4)).reshape((4,1))
labels = np.array((1, 2, 2, 3))

with h5py.File(train_filename, 'w') as f:
  f['data'] = data.astype(np.float32)
  f['label'] = labels.astype(np.float32)
  f['clip'] = clip.astype(np.float32)
