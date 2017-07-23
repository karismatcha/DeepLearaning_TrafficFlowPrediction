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
    fileDATES = importdict('C:\Users\oob13\Desktop\Internship\TrafficFlowPrediction\clustering2')
    timebuffer = []
    for i in range(1,len(fileDATES)):
        timebuffer.append((fileDATES[i]['systemtime'].split(","))[2]) #append only time into list #A
    #load_data = genfromtxt('.\\clustering.csv', delimiter=',')[1:5185,-3]
    
    CarsSpeed = genfromtxt('C:\Users\oob13\Desktop\Internship\TrafficFlowPrediction\clustering2.csv', delimiter=',')[1:,-3]
    #CarsTotal = genfromtxt('C:\Users\oob13\Desktop\Internship\TrafficFlowPrediction\clustering2.csv', delimiter=',')[1:,4]
    #hol = genfromtxt('C:\Users\oob13\Desktop\Internship\TrafficFlowPrediction\clustering2.csv', delimiter=',')[1:,-7]
    #week_read = genfromtxt('C:\Users\oob13\Desktop\Internship\TrafficFlowPrediction\clustering2.csv', delimiter=',')[1:,-8]
    
    
    
    speed = np.array(CarsSpeed)
    
    '''
    
    #filter data in time range 6.00am to 9.00am
    
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
    
    #get holiday data since 6.00 am to 9.00 am
    week = []
    for i in range(0,len(timebuffer)):
        if timebuffer[i] == '6:00':
            while timebuffer[i] != '9:05':
                week.append(week_read[i])
                i+=1
    week = np.array(week)
    '''
    #combine speed and number of car into dataset 2d array
    #get dataset = [speed,num_car,holiday]
    dataset = np.array([[]])
    for i in range(0,len(speed)):
        buffer = np.array([])
        buffer = np.append(buffer,(speed[i]))
        #buffer = np.append(buffer,round(num_car[i]))
        #buffer = np.append(buffer,round(holiday[i]))
        #buffer = np.append(buffer,round(week[i]))
        buffer2 = np.array([buffer])
        if i == 0:
            dataset = buffer2
        else:
            dataset = np.concatenate((dataset,buffer2))
    dataset = (np.asarray(dataset, 'float32'))
    
    '''
    rescale = np.array([[]])
    for i in range(0,dataset.shape[0]):
        buffer = np.array([])
        buffer = np.append(buffer,(dataset[i,0] - np.min(speed)) / (np.max(speed)-np.min(speed)))
        buffer = np.append(buffer,(dataset[i,1] - np.min(num_car)) / (np.max(num_car)-np.min(num_car)))
        buffer = np.append(buffer,(dataset[i,2] - np.min(holiday)) / (np.max(holiday)-np.min(holiday)))
        buffer = np.append(buffer,(dataset[i,3] - np.min(week)) / (np.max(week)-np.min(week)))
        buffer2 = np.array([buffer])
        if i == 0:
            rescale = buffer2
        else:
            rescale = np.concatenate((rescale,buffer2))
    '''        
    rescale = np.array([[]])
    for i in range(0,dataset.shape[0]):
        buffer = np.array([])
        buffer = np.append(buffer,(dataset[i,0] - np.mean(speed)) / (np.std(speed)))
        #buffer = np.append(buffer,(dataset[i,1] - np.mean(num_car)) / (np.std(num_car)))
        #buffer = np.append(buffer,(dataset[i,2] - np.mean(holiday)) / (np.std(holiday)))
        #buffer = np.append(buffer,(dataset[i,3] - np.mean(week)) / (np.std(week)))
        buffer2 = np.array([buffer])
        if i == 0:
            rescale = buffer2
        else:
            rescale = np.concatenate((rescale,buffer2))
    
    train_ratio = 0.75
    divider = int(round(train_ratio*rescale.shape[0]))
    
    pred_minutes = 5
    
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
    #train_model.add(Dense(1, activation='sigmoid'))
    
    
    #train_model.summary()
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
    #inference_model.add(Dense(6, input_dim = 2, activation='relu'))
    inference_model.add(h_given_x)
    #inference_model.add(Dense(8, activation='relu'))
    #inference_model.add(SampleBernoulli(mode='maximum_likelihood'))
    
    print('Compiling Theano graph...')
    inference_model.compile(opt, loss='mean_squared_error')
    
    print('Doing inference...')
    h = inference_model.predict(X_test)
    print(h)
    
    '''
    #convert result to real speed
    speed_result = []
    for i in range(0,len(h)):
        speed_result.append(round((h[i,0]*(np.max(speed)-np.min(speed)) + np.min(speed)))) #transfrom all predicted value into speed value
    speed_result = np.array(speed_result)
    print(speed_result)
    '''
    
    #convert result to real speed
    speed_result = []
    for i in range(0,len(h)):
        speed_result.append(round((h[i,0]*(np.std(speed)) + np.mean(speed)))) #transfrom all predicted value into speed value
    speed_result = np.array(speed_result)
    print(speed_result)
    
    
    '''
    
    range_sd = genfromtxt('.\\clustering.csv', delimiter=',')[1:,-5]
    check_count = 0
    ##############################################
    for i in range(0,len(timebuffer)-1):
        sd = []
        j=0
        if timebuffer[i] == '6:00': #This is 1 day data
            while timebuffer[i+j] != '9:05':
                print(i)
                sd.append(abs(range_sd[i+j]))
                j+=1
            sd = np.array(sd)
            threshold = np.mean(sd)
            threshold = math.ceil(threshold)
            if(threshold <= 5):
                threshold = 5
            #print(threshold)
        for j in range(0,37):
            check = abs(Y_test[i+j,0] - speed_result[i+j]) > abs(threshold)
            if check == True:
                    check_count+=1
    accuracy = (float(speed_result.shape[0]-check_count)/speed_result.shape[0])*100
    print("RBM Prediction Accuracy = %.2f %%" % accuracy)
    
    ##############################################
    '''
    
    
    #Evaluation part
    #set speed difference threshold
    #threshold = genfromtxt('.\\clustering.csv', delimiter=',')[1:,-5]
    threshold = 5
    mse_buffer = 0
    mae_buffer = 0
    check_count = 0
    for i in range(0,speed_result.shape[0]):
        check = abs(Y_test[i,0] - speed_result[i]) > abs(threshold)
        mse_buffer += (Y_test[i,0] - speed_result[i])*(Y_test[i,0] - speed_result[i])
        mae_buffer += abs(Y_test[i,0] - speed_result[i])
        if check == True:
            check_count+=1
    accuracy = (float(speed_result.shape[0]-check_count)/speed_result.shape[0])*100
    print("RBM Prediction Accuracy = %.2f %%" % accuracy)
    
    
    mse = math.sqrt(mse_buffer/speed_result.shape[0])
    mae = mae_buffer/speed_result.shape[0]
    print("MSE = %.2f %%" % mse)
    print("MAE = %.2f %%" % mae)
    
    print('Done!')
    
    '''
    
    less_than_5 = 0
    less = 0
    more = 0
    more_than_5 = 0
    equal = 0
    for i in range(0,speed_result.shape[0]):
        if (speed_result[i] - Y_test[i,0]) < -threshold and (speed_result[i] - Y_test[i,0]) < 0:
            less_than_5 = less_than_5 +1
        elif (speed_result[i] - Y_test[i,0]) >= -threshold and (speed_result[i] - Y_test[i,0]) < 0:
            less = less +1
        elif (speed_result[i] - Y_test[i,0]) == 0:
            equal = equal +1
        elif (speed_result[i] - Y_test[i,0]) <= threshold and (speed_result[i] - Y_test[i,0]) > 0:
            more = more +1
        elif (speed_result[i] - Y_test[i,0]) > threshold and (speed_result[i] - Y_test[i,0]) > 0:
            more_than_5 = more_than_5 +1
    
    less_than_5 = (less_than_5/speed_result.shape[0])*100
    less = (less/speed_result.shape[0])*100
    equal = (equal/speed_result.shape[0])*100
    more = (more/speed_result.shape[0])*100
    more_than_5 = (more_than_5/speed_result.shape[0])*100
    
    print("outbound lower = %.2f %% " % less_than_5)
    print("lower = %.2f %%" % less)
    print("equal = %.2f %%" % equal)
    print("higher = %.2f %%" % more)
    print("outbound higher = %.2f %%" % more_than_5)
    
    
    print('Done!')
    '''
    
    
    
    with open('speed.csv', 'wb') as f:
         writer = csv.writer(f, delimiter = ',')
         for row in Y_test:
             writer.writerow([row[0]])    
    with open('result.csv', 'wb') as f:
         writer = csv.writer(f, delimiter = ',')
         for row in speed_result:
             writer.writerow([row])        
    
    
    
    
    
    
    
    '''
    #get user input and predict the next speed
    min_speed = float(np.min(speed))
    max_speed = float(np.max(speed))
    min_numcar = float(np.min(num_car))
    max_numcar = float(np.max(num_car))
    min_hol = float(np.min(holiday))
    max_hol =  float(np.max(holiday))
    min_week = float(np.min(week))
    max_week =  float(np.max(week))
    
    def get_input():
        input_speed = input('Enter speed: ')
        input_numcar = input('Enter number of car: ')
        input_hol = input('Enter holday(0 = no, 1 = yes): ')
        input_week = input('Enter week(1 = Monday): ')
        if(input_speed != -1 and input_numcar != -1 and input_hol!=-1 and input_week != -1):
            input_speed = (input_speed - min_speed) / (max_speed-min_speed)
            input_numcar = (input_numcar - min_numcar) / (max_numcar-min_numcar)
            input_hol = (input_hol - min_hol) / (max_hol-min_hol)
            input_week = (input_week - min_week) / (max_week-min_week)
            buffer = np.array([[input_speed,input_numcar,input_hol,input_week]])
            h = inference_model.predict(buffer)
            
            result = round(h[0,0]*(max_speed-min_speed) + min_speed)
            print ('Next Speed is ' + str(result))
            return 1
        else:
            return -1
    out = 1
    while(out!=-1):
        print()
        print("Enter speed = -1 , number of car = -1 , holiday = -1 and week = -1 to exit")
        out = get_input()
    '''

if __name__ == '__main__':
    build_model()
    