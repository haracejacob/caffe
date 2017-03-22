#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 04:53:34 2017

@author: kim
"""

import h5py
import numpy as np

train_filename = 'iris_data.hdf5'

f = open("iris.txt", 'r')
line = f.readline()
line = line.split(',')
data = [[line[0], line[1], line[2], line[3]]]
line[4] = line[4][:-1]
if line[4] == 'Iris-setosa' :
    labels = [0]
elif line[4] == 'Iris-versicolor' :
    labels = [1]
elif line[4] == 'Iris-virginica' :
    labels = [2]
    
while True:
    line = f.readline()
    if (not line) or (line[0] == '\n'): break

    line = line.split(',')
    data_val = [[line[0], line[1], line[2], line[3]]]
    line[4] = line[4][:-1]
    if line[4] == 'Iris-setosa' :
        label_val = [0]
    elif line[4] == 'Iris-versicolor' :
        label_val = [1]
    elif line[4] == 'Iris-virginica' :
        label_val = [2]

    data = np.append(data, data_val, axis=0)
    labels = np.append(labels, label_val, axis=0)
    
f.close()

np_data = np.array(data)
np_labels = np.array(labels)

with h5py.File(train_filename, 'w') as f:
  f['data'] = np_data.astype(np.float32)
  f['label'] = np_labels.astype(np.float32)
