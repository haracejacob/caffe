#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 08:42:22 2017

@author: kim
"""
"""
0 : Duration
1 : Service
2 : Source bytes
3 : Destination bytes
4 : Count
5 : Same_srv_rate
6 : Serror_rate
7 : Srv_serror_rate
8 : Dst_host_count
9 : Dst_host_srv_count
10 : Dst_host_same_src_port_rate
11 : Dst_host_serror_rate
12 : Dst_host_srv_serror_rate
13 : Flag

14 : IDS_detection
15 : Malware_detection
16 : Ashula_detection
17 : Label 1 : Normal -1 : Known Attack -2 : Unkown Attack
18 : Source_IP_Address
19 : Source_Port_Number
20 : Destination_IP_Address
21 : Destination_Port_Number
22 : Start_Time
23 : Duration(session)
"""
"""
#Normal_Data
Duration
Source bytes
Destination bytes
Start_Time

#Count_Data
Count
Smae_srv_rate
Serror_rate
Srv_serror_rate

#Dst_host_count_Data
Dst_host_count
Dst_host_srv_count
Dst_host_same_src_port_rate
Dst_host_serror_rate
Dst_host_srv_serror_rate

#Label
1:DDOS
0:Not DDOS
"""
import pandas as pd
import numpy as np
import h5py
import os
import random

#DIRPATH에 hdf5파일로 만들 폴더 설정
DIRPATH = './data/2015/02/'
datalist = os.listdir(DIRPATH)
datalist.sort()
#HDF5 file 저장할 폴더 설정(폴더 있어야함)
HDF5PATH = './hdf5/'

#train용 test용 골라서 주석 해제
#train_filelist = open('./train.filelist.txt', 'w')
test_filelist = open('./test.filelist.txt', 'w')

def create_dataset(datapath, flag) :
    data_normal, data_count, data_dst_host, label = [], [], [], []
    label_data, cnt_pos, cnt_neg = 0,0,0
    dataframe = pd.read_csv(DIRPATH+datapath, sep="\t", header = None)
    dataset = dataframe.values
    
    hdf5_filename = datapath[:-4]			#remove .hdf5
    print 'filename : ', hdf5_filename, len(dataset),
    for d in dataset :
	#값이 0인 데이터 걸러냄
        if d[0] == 0 and d[4] == 0 and d[9] == 0 :
            continue
        if(d[17] != 1) :
	    #subsampling
            #if random.random() > 0.25 :
            #    continue
            label_data = 1
            cnt_neg += 1
        else :
            label_data = 0
            cnt_pos += 1
	#HH:MM:SS -> seconds
        session_time = sum(int(x) * 60 ** i for i,x in enumerate(reversed(d[22].split(":"))))
        
        data_normal.append(np.array((d[0], d[2], d[3], session_time)).astype(np.float32))
        data_count.append(np.array((d[4], d[5], d[6], d[7])).astype(np.float32))
        data_dst_host.append(np.array((d[8], d[9], d[10], d[11], d[12])).astype(np.float32))
        label.append(np.array((label_data)).astype(np.int32))
        
    n_data_normal = np.array(data_normal)
    n_data_normal = np.reshape(n_data_normal, (n_data_normal.shape[0], 1, 1, n_data_normal.shape[1]))
    n_data_count = np.array(data_count)
    n_data_count = np.reshape(n_data_count, (n_data_count.shape[0], 1, 1, n_data_count.shape[1]))
    n_data_dst_host = np.array(data_dst_host)
    n_data_dst_host = np.reshape(n_data_dst_host, (n_data_dst_host.shape[0], 1, 1, n_data_dst_host.shape[1]))
    
    #make hdf5 file
    hdf5_filepath = HDF5PATH + hdf5_filename +'.hdf5'
    print 'hdf5 file : ', hdf5_filepath, len(data_normal)
    with h5py.File(hdf5_filepath, 'w') as f:
        f['data_n'] = n_data_normal
        f['data_c'] = n_data_count
        f['data_d'] = n_data_dst_host
        f['label'] = np.array(label)
    
    if flag == 1 :
        train_filelist.write(hdf5_filepath + '\n')
    else :        
        test_filelist.write(hdf5_filepath + '\n')
    print ' success', 'pos : ', cnt_pos, 'neg : ', cnt_neg



#create_dataset(d, 1) : create training data, create_dateset(d, 0) : create test data
for d in datalist :
    create_dataset(d, 0)
    
#train_filelist.close()
test_filelist.close()

