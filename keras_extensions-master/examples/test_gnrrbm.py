from __future__ import division

import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

np.random.seed(1234) # seed random number generator
srng_seed = np.random.randint(2**30)

from keras.models import Sequential, Model
from keras.optimizers import SGD

from keras_extensions.logging import log_to_file
from keras_extensions.rbm import GNRRBM
from keras_extensions.layers import SampleBernoulli
from keras_extensions.initializations import glorot_uniform_sigm


# configuration
input_dim = 100
hidden_dim = 200
batch_size = 10
nb_epoch = 15
nb_gibbs_steps = 1 # should be 1 for Gausian-NRelu units
lr = 0.001  # small learning rate for GB-RBM

@log_to_file('example.log')
def main():
    # generate dummy dataset
    nframes = 10000
    dataset = np.random.normal(loc=np.zeros(input_dim), scale=np.ones(input_dim), size=(nframes, input_dim))

    # split into train and test portion
    ntest   = 1000
    X_train = dataset[:-ntest :]     # all but last 1000 samples for training
    X_test  = dataset[-ntest:, :]    # last 1000 samples for testing

    assert X_train.shape[0] >= X_test.shape[0], 'Train set should be at least size of test set!'
    
    # setup model structure
    print('Creating training model...')
    rbm = GNRRBM(hidden_dim, input_dim=input_dim,
		init=glorot_uniform_sigm,
		activation='sigmoid',
		nb_gibbs_steps=nb_gibbs_steps,
		persistent=False,
		batch_size=batch_size,
		dropout=0.0)

    rbm.srng = RandomStreams(seed=srng_seed)

    train_model = Sequential()
    train_model.add(rbm)

    opt = SGD(lr, 0., decay=0.0, nesterov=False)
    loss=rbm.contrastive_divergence_loss
    metrics = [rbm.reconstruction_loss]

    # compile theano graph
    print('Compiling Theano graph...')
    train_model.compile(optimizer=opt, loss=loss, metrics=metrics)
     
    # do training
    print('Training...')    
    train_model.fit(X_train, X_train, batch_size, nb_epoch, 
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
    h = inference_model.predict(dataset)

    print(h)

    print('Done!')

if __name__ == '__main__':
    main()
