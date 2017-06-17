from keras import backend as K
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def random_binomial(shape, n=0, p=0.5, dtype=K.floatx(), seed=None):
    if seed is None:
        seed = np.random.randint(10e6)
    rng = RandomStreams(seed=seed)
    return rng.binomial(shape, n=n, p=p, dtype=dtype)

def random_normal(shape, avg=0, std=1.0, dtype=K.floatx(), seed=None):
    if seed is None:
        seed = np.random.randint(10e6)
    rng = RandomStreams(seed=seed)
    return rng.normal(shape, std=std, dtype=dtype)
