#!/usr/bin/env python
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
DIM = 2
N  = 100
MAX_AXIS_CLUSTERS = 15
MAX_CLUSTERS = 15
AXIS_ALPHA = 1
AXIS_BETA = 1
AXIS_DP_ALPHA = 2
CLUSTER_DP_ALPHA = 3
CLUSTER_VARIANCE = 0.01

def generate_data():
    #make axis count prior
    betas = rnd.beta(1,AXIS_DP_ALPHA,size=(DIM,MAX_AXIS_CLUSTERS))
    axis_cluster_magnitude = np.cumprod(1-betas,axis=1)/(1-betas)*betas
    axis_cluster_magnitude[:,-1] = 1-np.sum(axis_cluster_magnitude[:,:-1],axis=1)

    axis_cluster_locations = rnd.beta(AXIS_ALPHA,AXIS_BETA,size=(DIM,MAX_AXIS_CLUSTERS))

    betas = rnd.beta(1,CLUSTER_DP_ALPHA,size=(MAX_CLUSTERS))
    cluster_magnitude = np.cumprod(1-betas)/(1-betas)*betas
    cluster_magnitude[-1] = 1-np.sum(cluster_magnitude[:-1])

    #spawn axis clusters
    cluster_locations = np.zeros((MAX_CLUSTERS,DIM))
    for dim in range(DIM): 
        cluster_indicies = rnd.choice(np.arange(MAX_AXIS_CLUSTERS),MAX_CLUSTERS,p=axis_cluster_magnitude[dim,:])
        cluster_locations[:,dim] = axis_cluster_locations[dim,cluster_indicies]

    data_expectations = np.zeros((N,DIM))
    for dim in range(DIM): 
        location_indicies = rnd.choice(np.arange(MAX_CLUSTERS),N,p=cluster_magnitude)
        data_expectations[:,dim] = cluster_locations[location_indicies,dim]
    x = rnd.normal(data_expectations,CLUSTER_VARIANCE)
    return x,cluster_locations

if __name__=="__main__":
    x,_ = generate_data()
    plt.plot(x[:,0],x[:,1],"o")
    plt.show()
