#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 09:59:16 2017

@author: kim
"""

import pandas
import matplotlib.pyplot as plt
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

dataframe = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

#data = numpy.reshape(dataset, (dataset.shape[0], 1, dataset.shape[1]))   #144*1*1
#clip = numpy.zeros((144,1), numpy.float32)

#import h5py
#train_filename = 'lstm_deploy_data.hdf5'
#with h5py.File(train_filename, 'w') as f:
#  f['data'] = data.astype(numpy.float32)
#  f['clip'] = clip.astype(numpy.float32)

#draw graph from dataset
plt.plot(dataset)
plt.show()

import caffe
caffe.set_mode_cpu()

model = 'lstm_short_deploy_net.prototxt'
weights = 'snapshot_iter_50000.caffemodel'

net=caffe.Net(model, weights, caffe.TEST)

res = net.forward()

#draw graph from dataset and deploy data
plt.plot(res['ip'])
plt.plot(dataset)
plt.savefig('graph.png')
plt.show()