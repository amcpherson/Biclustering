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
from pymc3 import Deterministic
from pymc3.backends import Text

from data_generator import generate_data

MAX_AXIS_CLUSTERS = 15 
MAX_CLUSTERS = 15

def build_model(data):
    n,dim = data.shape
    bc_model = Model()
    with bc_model:
        axis_alpha = 1
        axis_beta = 1
        axis_dp_alpha = Gamma("axis_dp_alpha",mu=1,sd=1)
        cluster_dp_alpha = Gamma("cluster_dp_alpha",mu=1,sd=1)
        cluster_std_dev = Gamma("cluster_std_dev",mu=1,sd=1)
        betas = Beta("axis_betas",alpha=axis_alpha,beta=axis_beta,shape=(dim,MAX_AXIS_CLUSTERS))
        axis_cluster_magnitude = tt.extra_ops.cumprod(1-betas,axis=1)/(1-betas)*betas
        axis_cluster_magnitude = tt.set_subtensor(
            axis_cluster_magnitude[:,-1],
            1-tt.sum(axis_cluster_magnitude[:,:-1],axis=1))

        axis_cluster_locations = Beta(
            "axis_cluster_locations",alpha=axis_alpha, beta=axis_beta, shape=(dim,MAX_AXIS_CLUSTERS))

        betas = Beta("cluster_betas",1,cluster_dp_alpha,shape=(MAX_CLUSTERS))
        cluster_magnitude = tt.extra_ops.cumprod((1-betas)/(1-betas)*(betas))
        cluster_magnitude = tt.set_subtensor(
            cluster_magnitude[-1],
            1-tt.sum(cluster_magnitude[:-1]))

        #spawn axis clusters
        cluster_locations = tt.zeros((MAX_CLUSTERS,dim))
        cluster_indicies = Categorical("cluster_indicies",shape=(MAX_CLUSTERS),p=axis_cluster_magnitude.T)
        for d in range(dim):
            #TODO:find a cleaner way of doing this
            cluster_locations = tt.set_subtensor(
                cluster_locations[:,d],
                axis_cluster_locations[d,cluster_indicies])

        loc = Deterministic("cluster_locations",cluster_locations)

        data_expectation = tt.zeros((n,dim))
        location_indicies = Categorical("location_indicies",shape=(n),p=cluster_magnitude)
        for d in range(dim):
            data_expectation = tt.set_subtensor(
                data_expectation[:,d],
                cluster_locations[location_indicies,d])

        x = Beta("data".format(d),shape=(n,d),mu=data_expectation,sd=cluster_std_dev,observed=data)
        db = Text('trace')
        trace = pm.sample(100000, n_init=500, njobs=5 ,trace=db)
        print(list(trace["cluster_locations"]))
    return bc_model


def main():
    data,locations = generate_data()
    model = build_model(data)
    parameters = infer_parameters(model)
    print(locations)

if __name__=="__main__":
    #x = generate_data()
    #plt.plot(x[:,0],x[:,1],"o")
    #plt.show()
    print("START")
    main()
    print("DONE")
