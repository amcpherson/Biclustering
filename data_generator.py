#!/usr/bin/env python
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
DIM = 4
N  = 100
MAX_AXIS_CLUSTERS = 5
MAX_CLUSTERS = 20
AXIS_ALPHA = 1
AXIS_BETA = 1
AXIS_DP_ALPHA = 2
CLUSTER_DP_ALPHA = 2
CLUSTER_CLUSTERING = 500

def generate_data():
    #make axis count prior
    axis_betas= rnd.beta(1,AXIS_DP_ALPHA,size=(DIM,MAX_AXIS_CLUSTERS))
    axis_cluster_magnitudes = np.cumprod(1-axis_betas,axis=1)/(1-axis_betas)*axis_betas
    axis_cluster_magnitudes[:,-1] = 1-np.sum(axis_cluster_magnitudes[:,:-1],axis=1)

    axis_cluster_locations = rnd.beta(AXIS_ALPHA,AXIS_BETA,size=(DIM,MAX_AXIS_CLUSTERS))

    cluster_betas = rnd.beta(1,CLUSTER_DP_ALPHA,size=(MAX_CLUSTERS))
    cluster_magnitudes = np.cumprod(1-cluster_betas)/(1-cluster_betas)*cluster_betas
    cluster_magnitudes[-1] = 1-np.sum(cluster_magnitudes[:-1])

    #spawn axis clusters
    cluster_locations = np.zeros((MAX_CLUSTERS,DIM))
    cluster_indicies = np.zeros((MAX_CLUSTERS,DIM),dtype=np.int64)
    for dim in range(DIM): 
        cluster_indicies[:,dim] = rnd.choice(np.arange(MAX_AXIS_CLUSTERS),MAX_CLUSTERS,p=axis_cluster_magnitudes[dim,:])
        cluster_locations[:,dim] = axis_cluster_locations[dim,cluster_indicies[:,dim]]

    data_expectations = np.zeros((N,DIM))
    location_indicies = rnd.choice(np.arange(MAX_CLUSTERS),N,p=cluster_magnitudes)
    for dim in range(DIM): 
        data_expectations[:,dim] = cluster_locations[location_indicies,dim]
        a = CLUSTER_CLUSTERING*data_expectations
        b = CLUSTER_CLUSTERING*(1-data_expectations)
    x = rnd.beta(a,b)
    state = {
        "axis_betas": axis_betas,
        "axis_cluster_locations": axis_cluster_locations,
        "cluster_betas": cluster_betas,
        "cluster_indicies": cluster_indicies,
        "location_indicies": location_indicies
    }
    return x,state

if __name__=="__main__":
    x,_ = generate_data()
    plt.plot(x[:,0],x[:,1],"o")
    plt.show()
