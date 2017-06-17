from __future__ import division

import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

np.random.seed(1234) # seed random number generator
srng_seed = np.random.randint(2**30)

from keras.models import Sequential, Model
from keras.optimizers import SGD

from keras_extensions.logging import log_to_file
from keras_extensions.rbm import GBRBM
from keras_extensions.layers import SampleBernoulli
from keras_extensions.initializations import glorot_uniform_sigm


# configuration
input_dim = 100
hidden_dim = 200
batch_size = 10
nb_epoch = 15
nb_gibbs_steps = 10
lr = 0.001  # small learning rate for GB-RBM

@log_to_file('example.log')
def main():
    # generate dummy dataset
    nframes = 10000

	#Draw random samples from a normal (Gaussian) distribution.
	#Mean = 0 with input_dim size, sd = 0 with input_dim size, size = nframes * input_dim
	#Get dataset which has value near 0 (bcz sd = 1) size 10000 * 100
	#[[],[],...]
    dataset = np.random.normal(loc=np.zeros(input_dim), scale=np.ones(input_dim), size=(nframes, input_dim)) 

    # split into train and test portion
    ntest   = 1000
    X_train = dataset[:-ntest :]     # all but last 1000 samples for training
    X_test  = dataset[-ntest:, :]    # last 1000 samples for testing

	#check the condition. Trigger an error if the condition is false.
    assert X_train.shape[0] >= X_test.shape[0], 'Train set should be at least size of test set!'
    
    # setup model structure
	# Gaussian-Bernoulli Restricted Boltzmann Machines
    print('Creating training model...')
    rbm = GBRBM(hidden_dim, input_dim=input_dim,
		init=glorot_uniform_sigm,
		activation='sigmoid',
		nb_gibbs_steps=nb_gibbs_steps,
		persistent=True,
		batch_size=batch_size,
		dropout=0.0)
	#output = <keras_extensions.rbm.GBRBM object at 0x0000000010C595F8>

    rbm.srng = RandomStreams(seed=srng_seed)
	#output = <theano.sandbox.rng_mrg.MRG_RandomStreams object at 0x0000000010C6D6A0>


    #--------------------------train model part----------------------------
    
    train_model = Sequential() #make sequential model
	#output = <keras.models.Sequential object at 0x0000000010C6D588>
    train_model.add(rbm) 

    opt = SGD(lr, 0., decay=0.0, nesterov=False)
    loss=rbm.contrastive_divergence_loss
    metrics = [rbm.reconstruction_loss]



    # compile theano graph
    print('Compiling Theano graph...')
    train_model.compile(optimizer=opt, loss=loss, metrics=metrics)
     
    # do training, training occur here
    print('Training...')    
    tm = train_model.fit(X_train, X_train, batch_size, nb_epoch, verbose=1, shuffle=False)
	#Epoch 1/15 5000/9000 [========>..............]
 

    #--------------------------Inference model part----------------------------
    # generate hidden features from input data
    print('Creating inference model...')
    
    # Generates a new Dense Layer that computes mean of Bernoulli distribution p(h|x), ie. p(h=1|x).
    h_given_x = rbm.get_h_given_x_layer(as_initial_layer=True)
    
    inference_model = Sequential() #make sequential model
    inference_model.add(h_given_x)
    #inference_model.add(SampleBernoulli(mode='maximum_likelihood'))

    print('Compiling Theano graph...')
    inference_model.compile(opt, loss='mean_squared_error')
	#Epoch 1/15 5000/9000 [========>..............]

    #Generates output predictions for the input samples,processing the samples in a batched way.
    # Returns a Numpy array of predictions.
    print('Doing inference...')
    h = inference_model.predict(dataset)
	
    print("dataset")
    print(dataset)
    
    print("train")
    print(tm)

    print("result")
    print(h)

    print("num dataset = ")
    print(dataset.shape[0])
    print(dataset.shape[1])
    print("num result = ")
    print(h.shape[0])
    print(h.shape[1])
    
    #print("corr")
    #print np.corrcoef(dataset,h)

    print('Done!')

if __name__ == '__main__':
    main()
