#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 12:02:04 2017

@author: kim
"""
import os
import numpy as np
import caffe
import h5py

caffe.set_mode_cpu()

model = 'ddos_deploy.prototxt'
weights = 'ddos_detect_iter_15000.caffemodel'

net=caffe.Net(model, weights, caffe.TEST)

TP, FN, FP, TN = 0,0,0,0
SUM_TP, SUM_FN, SUM_FP, SUM_TN = 0,0,0,0
fp = open('./test.filelist.txt', 'r')
while True:
    line = fp.readline()
    if not line: break
    print(line[:-1])
    TP, FN, FP, TN = 0,0,0,0
    with h5py.File(line[:-1], 'r') as f:
        data_n = f['.']['data_n'].value
        data_c = f['.']['data_c'].value
        data_d = f['.']['data_d'].value
        label = f['.']['label'].value
        
        for idx in range(data_n.shape[0]) :
            net.blobs['data_n'].data[...] = data_n[idx]
            net.blobs['data_c'].data[...] = data_c[idx]
            net.blobs['data_d'].data[...] = data_d[idx]
        
            res = net.forward()
            if res['argmax'] == 0 :
                if label[idx] == 0 :
                    TN += 1
                else :
                    FN += 1
            else :
                if label[idx] == 0 :
                    FP += 1
                else :
                    TP += 1
    recall = float(TP)/float((TP+FN))
    precision = float(TP)/float((TP+FP))
    f1 = 2*recall*precision/(recall+precision)
    accuracy = (float(TP+TN))/(float(TP+TN+FN+FP))
    print 'TP : ', TP, 'FN : ', FN
    print 'FP : ', FP, 'TN : ', TN
    print 'RECALL : ', recall, 'PRECISION : ', precision, 'f1 : ', f1, 'Accuracy : ', accuracy
    
    SUM_TP += TP
    SUM_FN += FN
    SUM_FP += FP
    SUM_TN += TN
fp.close()

recall = float(SUM_TP)/float((SUM_TP+SUM_FN))
precision = float(SUM_TP)/float((SUM_TP+SUM_FP))
f1 = 2*recall*precision/(recall+precision)
accuracy = (float(SUM_TP+SUM_TN))/(float(SUM_TP+SUM_TN+SUM_FN+SUM_FP))
print 'TP : ', SUM_TP, 'FN : ', SUM_FN
print 'FP : ', SUM_FP, 'TN : ', SUM_TN
print 'RECALL : ', recall, 'PRECISION : ', precision, 'f1 : ', f1, 'Accuracy : ', accuracy
