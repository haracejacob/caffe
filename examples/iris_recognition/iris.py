#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:31:35 2017

@author: kim
"""
import caffe

#model = 'iris_net.prototxt'
#weights = 'snapshot_iter_2000.caffemodel'
model = 'iris_dropout.prototxt'
weights = 'snapshot_dropout_iter_10000.caffemodel'


caffe.set_mode_cpu()

net=caffe.Net(model, weights, caffe.TEST)
res = net.forward()

for idx in range(150) :
    print res['argmax'][idx], res['label'][idx], res['prob'][idx]