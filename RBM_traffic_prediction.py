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



# configuration
input_dim = 1
hidden_dim = 1
batch_size = 10
nb_epoch = 30
nb_gibbs_steps = 10
lr = 0.001  # small learning rate for GB-RBM

@log_to_file('example.log')
def build_model():
    # generate dummy dataset
    def importdict(filename):#creates a function to read the csv
    #create data frame from csv with pandas module
        df=pd.read_csv(filename+'.csv', names=['systemtime', 'Var1', 'var2'],sep=';',parse_dates=[0]) #or:, infer_datetime_format=True)
        fileDATES=df.T.to_dict().values()#export the data frame to a python dictionary
        return fileDATES #return the dictionary to work with it outside the function
    fileDATES = importdict('C:\Users\user\Desktop\DeepLearaning_TrafficFlowPrediction\clustering2')
    #get time and keep it in list
    #use time to filter data later
    timebuffer = []
    for i in range(1,len(fileDATES)):
        timebuffer.append((fileDATES[i]['systemtime'].split(","))[2]) #append only time into list #A
    
    
    #load any features
    CarsSpeed = genfromtxt('C:\Users\user\Desktop\DeepLearaning_TrafficFlowPrediction\clustering2.csv', delimiter=',')[1:,-3]
    #CarsTotal = genfromtxt('C:\Users\oob13\Desktop\Internship\TrafficFlowPrediction\clustering2.csv', delimiter=',')[1:,4]
    #hol = genfromtxt('C:\Users\oob13\Desktop\Internship\TrafficFlowPrediction\clustering2.csv', delimiter=',')[1:,-7]
    
    
    #use all speed
    speed = np.array(CarsSpeed)
    
    
    '''
    #filter data in time range 6.00am to 9.00am
    #open comment depends on feature used
    #get speed since 6.00 am to 9.00 am
    speed = []
    for i in range(0,len(timebuffer)):
        #print(timebuffer[i])
        if timebuffer[i] == '6:00':
            while timebuffer[i] != '9:05':
                speed.append(CarsSpeed[i])
                i+=1
    speed = np.array(speed)
    
    
    #get number of car since 6.00 am to 9.00 am
    num_car = []
    for i in range(0,len(timebuffer)):
        if timebuffer[i] == '6:00':
            while timebuffer[i] != '9:05':
                num_car.append(CarsTotal[i])
                i+=1
    num_car = np.array(num_car)
    
    #get holiday data since 6.00 am to 9.00 am
    holiday = []
    for i in range(0,len(timebuffer)):
        if timebuffer[i] == '6:00':
            while timebuffer[i] != '9:05':
                holiday.append(hol[i])
                i+=1
    holiday = np.array(holiday)

    '''
    
    #combine speed and number of car into dataset 2d array
    #get dataset = [100] -> [[100]]
    dataset = np.array([[]])
    for i in range(0,len(speed)):
        buffer = np.array([])
        buffer = np.append(buffer,(speed[i]))
        #buffer = np.append(buffer,round(num_car[i]))
        #buffer = np.append(buffer,round(holiday[i]))
        buffer2 = np.array([buffer])
        if i == 0:
            dataset = buffer2
        else:
            dataset = np.concatenate((dataset,buffer2))
    dataset = (np.asarray(dataset, 'float32'))
    
    
    #change all value from real value to 0-1
    #use equation z = (x-avg(x))/sd(x)
    rescale = np.array([[]])
    for i in range(0,dataset.shape[0]):
        buffer = np.array([])
        buffer = np.append(buffer,(dataset[i,0] - np.mean(speed)) / (np.std(speed)))
        #buffer = np.append(buffer,(dataset[i,1] - np.mean(num_car)) / (np.std(num_car)))
        #buffer = np.append(buffer,(dataset[i,2] - np.mean(holiday)) / (np.std(holiday)))
        buffer2 = np.array([buffer])
        if i == 0:
            rescale = buffer2
        else:
            rescale = np.concatenate((rescale,buffer2))
    
    #divide training data and testing data
    train_ratio = 0.75
    divider = int(round(train_ratio*rescale.shape[0]))
    
    #set future minutes for predicting
    pred_minutes = 60
    
    #divide data into train and test
    X_train = rescale[:divider-int(pred_minutes/5)]
    X_test = rescale[divider:-int(pred_minutes/5)]
    Y_train = rescale[int(pred_minutes/5):divider]
    Y_test = dataset[divider+int(pred_minutes/5):]
     
    
    # setup model structure
    print('Creating training model...')
    rbm = GBRBM(hidden_dim, input_dim=input_dim,
    		init=glorot_uniform_sigm,
    		activation='sigmoid',
    		nb_gibbs_steps=nb_gibbs_steps,
    		persistent=True,
    		batch_size=batch_size,
    		dropout=0.0)
    
    rbm.srng = RandomStreams(seed=srng_seed)
    
    train_model = Sequential()
    
    train_model.add(rbm)
    
    
    #set optimizer as Stochastic gradient descent
    #set loss fnction as contrastive divergence loss
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
    
    #add output layer to model
    inference_model = Sequential()
    inference_model.add(h_given_x)
    
    print('Compiling Theano graph...')
    inference_model.compile(opt, loss='mean_squared_error')
    
    #predicting result
    #get 0-1 values
    print('Doing inference...')
    h = inference_model.predict(X_test)
    print(h)
    
    #convert result to real speed
    #use invert of the same equation
    #speed_result var is the predicted speed after values are transformed 
    speed_result = []
    for i in range(0,len(h)):
        speed_result.append(round((h[i,0]*(np.std(speed)) + np.mean(speed)))) #transfrom all predicted value into speed value
    speed_result = np.array(speed_result)
    print(speed_result)
    
    
    
    #Evaluation part
    #find accuracy of model by using threshold
    #set speed difference threshold
    threshold = 5
    check_count = 0
    for i in range(0,speed_result.shape[0]):
        check = abs(Y_test[i,0] - speed_result[i]) > abs(threshold)
        if check == True:
            check_count+=1
    accuracy = (float(speed_result.shape[0]-check_count)/speed_result.shape[0])*100
    print("RBM Prediction Accuracy = %.2f %%" % accuracy)
    print('Done!')
    
    #find root mean square error
    def rmse(predictions,targets):
        return (np.sqrt(((predictions-targets)**2).mean()))
    print ("MSE")
    print (rmse(speed_result, Y_test[:,0]))
    
    #find mean absolute error
    def mae(predictions,targets):
        return ((np.absolute(predictions-targets)).mean())
    
    print ("MAE")
    print (mae(speed_result, Y_test[:,0]))
      
    #save ground-truth value and predicted value to csv
    with open('speed.csv', 'wb') as f:
         writer = csv.writer(f, delimiter = ',')
         for row in Y_test:
             writer.writerow([row[0]])    
    with open('result.csv', 'wb') as f:
         writer = csv.writer(f, delimiter = ',')
         for row in speed_result:
             writer.writerow([row])        
    
    
    


if __name__ == '__main__':
    build_model()
    