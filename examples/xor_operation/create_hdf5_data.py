#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 04:53:34 2017

@author: kim
"""

import h5py
import numpy as np

train_filename = 'xor_data.hdf5'
data = np.array(((0, 0), (0, 1), (1, 0), (1, 1)))
labels = np.array((0, 1, 1, 0))

with h5py.File(train_filename, 'w') as f:
  f['data'] = data.astype(np.float32)
  f['label'] = labels.astype(np.float32)