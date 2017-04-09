#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 09:59:16 2017

@author: kim
"""
import caffe
import numpy as np
caffe.set_mode_cpu()

net = caffe.Net('rnn_train.prototxt', caffe.TEST)

print [(k, v.data.shape) for k, v in net.blobs.items()]