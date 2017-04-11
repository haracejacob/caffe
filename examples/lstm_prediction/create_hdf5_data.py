#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 10:26:26 2017

@author: kim
"""
import pandas
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import h5py

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

# load the dataset
dataframe = pandas.read_csv('international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))   # 94*1*1
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))       # 46*1*1

trainClip = numpy.zeros((94,1), numpy.float32)
testClip = numpy.zeros((46,1), numpy.float32)

# make hdf5 file
train_filename = 'lstm_train_data.hdf5'
with h5py.File(train_filename, 'w') as f:
  f['data'] = trainX.astype(numpy.float32)
  f['label'] = trainY.astype(numpy.float32)
  f['clip'] = trainClip.astype(numpy.float32)

train_filename = 'lstm_test_data.hdf5'
with h5py.File(train_filename, 'w') as f:
  f['data'] = testX.astype(numpy.float32)
  f['label'] = testY.astype(numpy.float32)
  f['clip'] = testClip.astype(numpy.float32)