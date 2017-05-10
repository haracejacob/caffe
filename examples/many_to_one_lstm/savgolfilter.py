import os
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

DIRECTORY = './scenario/'
posScenarioPath = DIRECTORY+os.listdir(DIRECTORY)[0]
negScenarioPath = DIRECTORY+os.listdir(DIRECTORY)[1]
NORMDIR = DIRECTORY+'norm/'
GRAPHDIR = "./Graph/GraphSavitzkyGolay/"

#flag 1:pos 0:neg
def savitzky_golay_filter(filename, flag):
    if flag == 1 :
        datafile = open(posScenarioPath + '/' + filename, 'r')
    else :
        datafile = open(negScenarioPath + '/' + filename, 'r')
    data = datafile.read()
    datafile.close()
    data = data.split('\n')

    # Invert Str to float
    for i in range(0,len(data)):
        try:
            data[i] = float(data[i])
        except ValueError as ve:
            del data[i]
            print ve

    x = range(0, len(data))
    datahat = sps.savgol_filter(data, 13, 7)
    datahat = sps.resample(datahat, 100)
    print len(datahat)
    title = filename.split('.')[0]
    if flag == 1 :
        title = title +'_pos'
    else :
        title = title + '_neg'

    #save normalized data
    f = open(NORMDIR+title+'.txt', 'w')
    for i in range(0,len(datahat)) :
        normdata = str(datahat[i]) +'\n'
        f.write(normdata)
    f.close()
        
        
    # plt.title(title+"_Savgol_Filter.")
    # plt.plot(x, data, color = 'blue')
    # plt.plot(x, datahat, color = 'red')
    # plt.savefig("./Graph/GraphSavitzkyGolay/" + title + ".png")
    # plt.clf()

    plt.figure(figsize=(550/80,700/80))
    plt.suptitle('1D Data Smoothing ('+title+')', fontsize=16)

    plt.subplot(2, 1, 1)
    p1 = plt.plot(data, ".k")
    p1 = plt.plot(data, "-k")
    a = plt.axis()
    plt.axis([a[0], a[1], 0, .12])
    plt.text(2, .11, "raw data", fontsize=14)

    plt.subplot(2, 1, 2)
    p1 = plt.plot(datahat, ".k")
    p1 = plt.plot(datahat, "-k")
    a = plt.axis()
    plt.axis([a[0], a[1], 0, .12])
    plt.text(2, .11, "savitzky-golay filter", fontsize=14)

    figname = GRAPHDIR + title + ".png"
    plt.savefig(figname, dpi=80)

posScenarioPath = DIRECTORY+os.listdir(DIRECTORY)[0]
negScenarioPath = DIRECTORY+os.listdir(DIRECTORY)[1]

posFileList = os.listdir(posScenarioPath)
negFileList = os.listdir(negScenarioPath)

for name in posFileList :
    print 'pos'
    savitzky_golay_filter(name, 1)
    print 'neg'
    savitzky_golay_filter(name, 0)


#file = open("scenelist.txt", 'r')
#list = file.read().split('\n')

#for item in list:
#    savitzky_golay_filter(item)
