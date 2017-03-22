#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 05:19:10 2017

@author: kim
"""
import caffe

model = 'xor_net.prototxt'
weights = 'snapshot_iter_10000.caffemodel'

caffe.set_mode_cpu()

net=caffe.Net(model, weights, caffe.TEST)
res = net.forward()

for idx in range(4) :
    print res['argmax'][idx], res['label'][idx], res['prob'][idx]