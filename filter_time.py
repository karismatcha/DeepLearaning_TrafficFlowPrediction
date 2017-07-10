# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 00:47:09 2017

@author: oob13
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt


def importdict(filename):#creates a function to read the csv
    #create data frame from csv with pandas module
        df=pd.read_csv(filename+'.csv', names=['systemtime', 'Var1', 'var2'],sep=';',parse_dates=[0]) #or:, infer_datetime_format=True)
        fileDATES=df.T.to_dict().values()#export the data frame to a python dictionary
        return fileDATES #return the dictionary to work with it outside the function
fileDATES = importdict('clustering')
timebuffer = []
for i in range(1,len(fileDATES)):
    timebuffer.append((fileDATES[i]['systemtime'].split(","))[2]) #append only time into list #A
#filter data in time range 6.00am to 9.00am
with open('.\\clustering.csv', 'rb') as f:
    reader = csv.reader(f)
    import_data = []
    for row in reader:
        import_data.append(row)
header = import_data[0]


data = []
for i in range(0,len(timebuffer)):
    if timebuffer[i] == '6:00':
        while timebuffer[i] != '9:05':
            data.append(import_data[i+1])
            i+=1
import_data = np.array(import_data)

with open("new_data.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
print("Done")
