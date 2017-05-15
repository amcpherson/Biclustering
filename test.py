#!/usr/bin/env python
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import pymc3 as pm
import theano as t
from theano import tensor as tt
from pymc3 import Model
from pymc3.distributions.continuous import Gamma,Beta
from pymc3.distributions.discrete import Categorical
from pymc3.backends import Text

from data_generator import generate_data

def test():
    model = Model()
    with model:
        p = Gamma("p",mu=1,sd=1)
        db = Text('trace_test')
        trace = pm.sample(100000, n_init=500,njobs=2,trace=db)
    return model



if __name__=="__main__":
    #x = generate_data()
    #plt.plot(x[:,0],x[:,1],"o")
    #plt.show()
    print("HI")
    test()
    print("DONE")
