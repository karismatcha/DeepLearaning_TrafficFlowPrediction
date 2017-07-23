# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 01:53:12 2017

@author: oob13
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

np.random.seed(1234) # seed random number generator
srng_seed = np.random.randint(2**30)

from keras.models import Sequential, Model
from keras.optimizers import SGD
from keras.layers import Dense

from keras_extensions.logging import log_to_file
from keras_extensions.rbm import GBRBM
from keras_extensions.layers import SampleBernoulli
from keras_extensions.initializations import glorot_uniform_sigm
from numpy import genfromtxt
import math





#read_data = genfromtxt('C:\Users\oob13\Desktop\Internship\TrafficFlowPrediction\clustering2.csv', delimiter=',')[1:,:]

def importdict(filename):#creates a function to read the csv
#create data frame from csv with pandas module
    df=pd.read_csv(filename+'.csv', names=['systemtime', 'Var1', 'var2'],sep=';',parse_dates=[0]) #or:, infer_datetime_format=True)
    fileDATES=df.T.to_dict().values()#export the data frame to a python dictionary
    return fileDATES #return the dictionary to work with it outside the function
fileDATES = importdict('C:\Users\oob13\Desktop\Internship\TrafficFlowPrediction\clustering2')
readbuffer = []
for i in range(1,len(fileDATES)):
    readbuffer.append((fileDATES[i]['systemtime'].split(","))) #append only time into list #A
for j in range(0,len(readbuffer)):
    readbuffer[j].append(j)

header = (fileDATES[0]['systemtime'].split(","))
header.append("run")
    
    
with open('for_smooth.csv', 'wb') as f:
     writer = csv.writer(f, delimiter = ',')
     writer.writerow(header) 
     for row in readbuffer:
         writer.writerow(row) 