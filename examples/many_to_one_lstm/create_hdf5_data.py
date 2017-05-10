#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 07:25:35 2017

@author: kim
"""

import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import h5py

def get_norm_data(pos_file_path, neg_file_path) :
    norm_dataset = []
    
    pos_emotion_data = open(pos_file_path, 'r')
    pos_data = pos_emotion_data.read().split()
    pos_dataset = np.array(pos_data).astype('float32')
    pos_dataset = np.reshape(pos_dataset, (100,1))
    pos_dataset = scaler.fit_transform(pos_dataset)
    pos_emotion_data.close()
    
    neg_emotion_data = open(neg_file_path, 'r')
    neg_data = neg_emotion_data.read().split()
    neg_dataset = np.array(neg_data).astype('float32')
    neg_dataset = np.reshape(neg_dataset, (100,1))
    neg_dataset = scaler.fit_transform(neg_dataset)
    neg_emotion_data.close()
    
    for i in range(len(pos_dataset)) :
        norm_dataset.append([pos_dataset[i],neg_dataset[i]])
    
    return np.array(norm_dataset)

DATADIR = './scenario/norm/'
datalist = os.listdir(DATADIR)
datalist.sort()
datalist_pos = [row for row in datalist if row.find('pos') != -1]
datalist_neg = [row for row in datalist if row.find('neg') != -1]

scaler = MinMaxScaler(feature_range=(0, 1))

filelist = open('./many_to_one_lstm_train.filelist.txt', 'w')
clip = np.zeros((100,1), np.float32)
for idx in range(len(datalist_pos)) :
    dataset = get_norm_data(DATADIR+datalist_pos[idx], DATADIR+datalist_neg[idx])
    dataset = np.reshape(dataset, (100,1,2))
    labels = np.array((np.random.random_sample()*5))
    labels = np.reshape(labels, (1,))

    hdf5_filename = './data/' + datalist_pos[idx][:datalist_pos[idx].find('_s_pos')]+'.hdf5'
    with h5py.File(hdf5_filename, 'w') as f:
      f['data'] = dataset.astype(np.float32)
      f['label'] = labels.astype(np.float32)
      f['clip'] = clip.astype(np.float32)
    filelist.write(hdf5_filename+'\n')

filelist.close()
print 'making hdf5 file complete!!'
#data size 100*1*2
#label size 1
#clip size 100*1
#batch size 100
