"""
==============================================================
Restricted Boltzmann Machine features for digit classification
==============================================================

For greyscale image data where pixel values can be interpreted as degrees of
blackness on a white background, like handwritten digit recognition, the
Bernoulli Restricted Boltzmann machine model (:class:`BernoulliRBM
<sklearn.neural_network.BernoulliRBM>`) can perform effective non-linear
feature extraction.

In order to learn good latent representations from a small dataset, we
artificially generate more labeled data by perturbing the training data with
linear shifts of 1 pixel in each direction.

This example shows how to build a classification pipeline with a BernoulliRBM
feature extractor and a :class:`LogisticRegression
<sklearn.linear_model.LogisticRegression>` classifier. The hyperparameters
of the entire model (learning rate, hidden layer size, regularization)
were optimized by grid search, but the search is not reproduced here because
of runtime constraints.

Logistic regression on raw pixel values is presented for comparison. The
example shows that the features extracted by the BernoulliRBM help improve the
classification accuracy.
"""

from __future__ import print_function

print(__doc__)

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from numpy import genfromtxt
###############################################################################
#Traffic part

#Filter data and time to just 6.00 am to 9.00 am
def importdict(filename):#creates a function to read the csv
    #create data frame from csv with pandas module
    df=pd.read_csv(filename+'.csv', names=['systemtime', 'Var1', 'var2'],sep=';',parse_dates=[0]) #or:, infer_datetime_format=True)
    fileDATES=df.T.to_dict().values()#export the data frame to a python dictionary
    return fileDATES #return the dictionary to work with it outside the function
fileDATES = importdict('clustering')
timebuffer = []
for i in range(1,len(fileDATES)):
    timebuffer.append((fileDATES[i]['systemtime'].split(","))[2]) #append only time into list #Already get list of time

#load speed data
load_data = genfromtxt('.\\clustering.csv', delimiter=',')[1:5185,-3]
#filter data in time range 6.00am to 9.00am
speed = []
for i in range(0,len(timebuffer)):
    if timebuffer[i] == '6:00':
        while timebuffer[i] != '9:05':
            speed.append(load_data[i])
            i+=1
speed = np.array(speed)

#chage 1D array to 2D array because fitting data require 2D array
#Create X
input_dim = 1 #Number of dimension affect to accuracy
assert speed.shape[0]%input_dim == 0 , "incorrect input dimension"
dataset = np.array([[]])
for i in range(0,len(speed),input_dim):
    buffer = np.array([])
    for j in range(0,input_dim):
        buffer = np.append(buffer,round(speed[i+j]))
    buffer2 = np.array([buffer])
    if i == 0:
        dataset = buffer2
    else:
        dataset = np.concatenate((dataset,buffer2))
X = (np.asarray(dataset, 'float32'))[:-1]

buffer = np.array([])
for i in range(input_dim,speed.shape[0],input_dim):
    buffer = np.append(buffer,round(speed[i]))
Y = buffer

X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

#Divide X,Y into train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2,
                                                    random_state=0)

# Models we will use
logistic = linear_model.LogisticRegression()
rbm = BernoulliRBM(random_state=0, verbose=True)

classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])

###############################################################################
# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
# n_component = number of binary hidden unit
rbm.n_components = input_dim
logistic.C = 6000.0

# Training RBM-Logistic Pipeline
classifier.fit(X_train, Y_train)

'''
# Training Logistic regression
logistic_classifier = linear_model.LogisticRegression(C=100.0)
logistic_classifier.fit(X_train, Y_train)
'''
###############################################################################
# Evaluation

print()

#Evaluation part
#set speed difference threshold
threshold = genfromtxt('.\\clustering.csv', delimiter=',')[1:,-5]
predict = classifier.predict(X_test)
check_count = 0
for i in range(0,predict.shape[0]):
    check = abs(Y_test[i] - predict[i]) > abs(threshold[i])
    if check == True:
        check_count+=1
        print("error predict: pred {0} truth {1}" .format(predict[i],Y_test[i]))
accuracy = (float(predict.shape[0]-check_count)/predict.shape[0])*100
print("RBM Accuracy = %.2f %%" % accuracy)


'''
logistic = logistic_classifier.predict(X_test)
check_count = 0
for i in range(0,logistic.shape[0]):
    check = abs(Y_test[i] - logistic[i]) > abs(threshold[i])
    if check == True:
        check_count+=1
        #print("error predict: pred {0} truth {1}" .format(predict[i],Y_test[i]))
accuracy = (float(logistic.shape[0]-check_count)/logistic.shape[0])*100
print("Logistic Accuracy = %.2f %%" % accuracy)

'''
