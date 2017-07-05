from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

np.random.seed(1234) # seed random number generator
srng_seed = np.random.randint(2**30)

from keras.models import Sequential, Model
from keras.optimizers import SGD

from keras_extensions.logging import log_to_file
from keras_extensions.rbm import GBRBM
from keras_extensions.layers import SampleBernoulli
from keras_extensions.initializations import glorot_uniform_sigm
from numpy import genfromtxt
import csv


# configuration
input_dim = 1
hidden_dim = 1
batch_size = 10
nb_epoch = 30
nb_gibbs_steps = 10
lr = 0.001  # small learning rate for GB-RBM

@log_to_file('example.log')
def main():
    # generate dummy dataset
    def importdict(filename):#creates a function to read the csv
    #create data frame from csv with pandas module
        df=pd.read_csv(filename+'.csv', names=['systemtime', 'Var1', 'var2'],sep=';',parse_dates=[0]) #or:, infer_datetime_format=True)
        fileDATES=df.T.to_dict().values()#export the data frame to a python dictionary
        return fileDATES #return the dictionary to work with it outside the function
    fileDATES = importdict('clustering')
    timebuffer = []
    for i in range(1,len(fileDATES)):
        timebuffer.append((fileDATES[i]['systemtime'].split(","))[2]) #append only time into list #A
    load_data = genfromtxt('.\\clustering.csv', delimiter=',')[1:5185,-3]
    #filter data in time range 6.00am to 9.00am
    speed = []
    for i in range(0,len(timebuffer)):
        if timebuffer[i] == '6:00':
            while timebuffer[i] != '9:05':
                speed.append(load_data[i])
                i+=1
    speed = np.array(speed)
    
    #generate 2d array 
    dataset = np.array([[]])
    for i in range(0,len(speed),1):
        buffer = np.array([])
        for j in range(0,1):
            buffer = np.append(buffer,round(speed[i+j]))
        buffer2 = np.array([buffer])
        if i == 0:
            dataset = buffer2
        else:
            dataset = np.concatenate((dataset,buffer2))
    dataset = (np.asarray(dataset, 'float32'))[:-1]
    #transform datset to 0-1 value
    rescale = (dataset - np.min(speed)) / (np.max(speed)-np.min(speed))
    
    #dataset = np.random.normal(loc=np.zeros(input_dim), scale=np.ones(input_dim), size=(nframes, input_dim))
    
    
    # split into train and test portion
    #ntest   = int(0.1*len(speed))
    #X_train = rescale[:-ntest :]     # all but last 1000 samples for training
    
    #X_test  = rescale[-ntest:, :]    # last 1000 samples for testing
    #Y_train = rescale[:-ntest :]
    
    X_train = rescale[:-1]
    X_test = X_train
    Y_train = rescale[1:]
    Y_test = dataset[1:]
    '''
    X_train = rescale[:-1]
    X_test = X_train
    Y_train = dataset[1:]
    Y_test = Y_train
    '''
    '''
    # split into train and test portion
    ntest   = int(0.1*len(speed))
    X_train = rescale[:-ntest]
    X_train = X_train[:-1]
    Y_train = dataset[:-ntest]
    Y_train = Y_train[1:]
    
    X_test = rescale[-ntest:]
    X_test = X_test[:-1]
    Y_test = dataset[-ntest:]
    Y_test = Y_test[1:]
    '''
    
    #assert X_train.shape[0] >= X_test.shape[0], 'Train set should be at least size of test set!'
    # setup model structure
    print('Creating training model...')
    rbm = GBRBM(hidden_dim, input_dim=input_dim,
    		init=glorot_uniform_sigm,
    		activation='relu',
    		nb_gibbs_steps=nb_gibbs_steps,
    		persistent=True,
    		batch_size=batch_size,
    		dropout=0.0)
    
    rbm.srng = RandomStreams(seed=srng_seed)
    
    train_model = Sequential()
    train_model.add(rbm)
    train_model.summary()
    opt = SGD(lr, 0., decay=0.0, nesterov=False)
    loss=rbm.contrastive_divergence_loss
    metrics = [rbm.reconstruction_loss]
    
    # compile theano graph
    print('Compiling Theano graph...')
    train_model.compile(optimizer=opt, loss=loss, metrics=metrics)
     
    # do training
    print('Training...')    
    train_model.fit(X_train, Y_train, batch_size, nb_epoch, 
    		    verbose=1, shuffle=False)
    
    
    
    # generate hidden features from input data
    print('Creating inference model...')
    
    h_given_x = rbm.get_h_given_x_layer(as_initial_layer=True)
    
    inference_model = Sequential()
    inference_model.add(h_given_x)
    #inference_model.add(SampleBernoulli(mode='maximum_likelihood'))
    
    print('Compiling Theano graph...')
    inference_model.compile(opt, loss='mean_squared_error')
    
    print('Doing inference...')
    h = inference_model.predict(X_test)
    for i in range(0,len(dataset)):
        if dataset[i] == round(np.average(dataset)):
            base = dataset[i]
    itemindex = np.where(dataset==base)[0][0]
    
    base_transform = h[itemindex-1]
    float_base_transform = float(base_transform)
    
    diff_ratio = (h[1]-h[0])/(dataset[2]-dataset[1])
    for i in range(0,len(h)) :
        h[i] = round(((h[i]-float_base_transform)/diff_ratio) + np.average(dataset))
    
    
    
    print(dataset)
    
    print(h)
    
    
    #save to csv
    print('Done!')
    with open("dataset.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(dataset[0:100,:])
    with open("houtput.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(h[0:100,:])

       
    
    #Evaluation part
    #set speed difference threshold
    #threshold = genfromtxt('.\\clustering.csv', delimiter=',')[1:,-5]
    threshold = 5
    check_count = 0
    for i in range(0,len(h)):
        #check = abs(Y_test[i] - h[i]) > abs(threshold[i])
        check = abs(Y_test[i] - h[i]) > abs(threshold)
        if check == True:
            check_count+=1
            #print("error predict: pred {0} truth {1} threshold {2}" .format(h[i],Y_test[i],abs(threshold[i])))
            print("error predict: pred {0} truth {1} threshold {2}" .format(h[i],Y_test[i],abs(threshold)))
    accuracy = (float(h.shape[0]-check_count)/h.shape[0])*100
    print("RBM Accuracy = %.2f %%" % accuracy)
    
if __name__ == '__main__':
    main()
