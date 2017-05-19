#!/usr/bin/env python
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import pymc3 as pm
import theano as t
from theano import tensor as tt
from theano import printing as tp
from pymc3 import Model
from pymc3.distributions.continuous import Gamma,Beta
from pymc3.distributions.discrete import Categorical
from pymc3 import Deterministic
from pymc3.backends import Text

from data_generator import generate_data
from scipy import stats
import seaborn as sns

MAX_AXIS_CLUSTERS = 5 
MAX_CLUSTERS = 20

def build_model(data):
    n,dim = data.shape
    bc_model = Model()
    with bc_model:
        axis_alpha = 1
        axis_beta = 1
        axis_dp_alpha = 2#Gamma("axis_dp_alpha",mu=2,sd=1)
        cluster_dp_alpha = 2#Gamma("cluster_dp_alpha",mu=2,sd=1)
        #how clustered the clusters are, more means tighter clusters
        cluster_clustering = 500#Gamma("cluster_std_dev",mu=500,sd=250)
        betas = Beta("axis_betas",alpha=axis_alpha,beta=axis_beta,shape=(dim,MAX_AXIS_CLUSTERS))
        axis_cluster_magnitude = tt.extra_ops.cumprod(1-betas,axis=1)/(1-betas)*betas
        axis_cluster_magnitude = tt.set_subtensor(
            axis_cluster_magnitude[:,-1],
            1-tt.sum(axis_cluster_magnitude[:,:-1],axis=1))
        axis_cluster_locations = Beta(
            "axis_cluster_locations",alpha=axis_alpha, beta=axis_beta, shape=(dim,MAX_AXIS_CLUSTERS))

        betas2 = Beta("cluster_betas",1,cluster_dp_alpha,shape=(MAX_CLUSTERS))
        cluster_magnitude = tt.extra_ops.cumprod(1-betas2)/(1-betas2)*(betas2)
        cluster_magnitude = tt.set_subtensor(
            cluster_magnitude[-1],
            1-tt.sum(cluster_magnitude[:-1]))


        #spawn axis clusters
        cluster_locations = tt.zeros((MAX_CLUSTERS,dim))
        #one = tp.Print()(tt.sum(axis_cluster_magnitude,axis=1))
        #axis_cluster_magnitude=axis_cluster_magnitude*tt.sum(one/2)
        cluster_indicies = Categorical("cluster_indicies",shape=(MAX_CLUSTERS,dim),p=axis_cluster_magnitude)
        for d in range(dim):
            #TODO:find a cleaner way of doing this
            cluster_locations = tt.set_subtensor(
                cluster_locations[:,d],
                axis_cluster_locations[d,cluster_indicies[:,d]])

        loc = Deterministic("cluster_locations",cluster_locations)
        loc = Deterministic("cluster_magnitudes",cluster_magnitude)

        data_expectation = tt.zeros((n,dim))
        #one = tp.Print()(tt.sum(cluster_magnitude))
        #cluster_magnitude=cluster_magnitude*one
        location_indicies = Categorical("location_indicies",shape=(n),p=cluster_magnitude)
        for d in range(dim):
            data_expectation = tt.set_subtensor(
                data_expectation[:,d],
                cluster_locations[location_indicies,d])
        #x = Beta("data".format(d),shape=(n,d),mu=data_expectation,sd=cluster_std_dev,observed=data)
        a=data_expectation*cluster_clustering
        b=(1-data_expectation)*cluster_clustering
        x = Beta("data",shape=(n,dim),alpha=a,beta=b,observed=data)
        db = Text('trace')
        
        steps1 = pm.CategoricalGibbsMetropolis(vars=[location_indicies],proposal='uniform')
        steps2 = pm.CategoricalGibbsMetropolis(vars=[cluster_indicies],proposal='uniform')
        steps3 = pm.step_methods.HamiltonianMC(vars=[betas,betas2,axis_cluster_locations],step_scale=0.02,path_length=1)
        #steps2 = pm.Metropolis(vars=[cluster_B])
        trace = pm.sample(3000,tune=40000,n_init=10000, njobs=1,step=[steps1,steps2,steps3]) #,trace=db)

    return bc_model,trace
    
def get_map_trace(trace,model):
    pass

def plot_posterior_predictive(trace,data):
    samples = pm.sample_ppc(trace,len(trace)*10,)
    pass
    
    

def plot_max_n(trace,n,last,spacing):
    cl = trace["cluster_locations"]
    cm = trace["cluster_magnitudes"]
    print(cl)

    #find indicies of the largest clusters
    sort = np.argsort(cm,axis=1)
    #biggest cluster
    for i in range(n):
        indicies = sort[-last::spacing,i]
        values = cl[-last::spacing,indicies,:]
        sns.set_style('whitegrid')
        sns.plt.ylim(0,1)
        sns.plt.xlim(0,1)
        sns.kdeplot(values[:,0],values[:,0], bw='scott')
def infer_parameters(model):
    pass

def main():
    data,locations,mag,_,_ = generate_data()
    model,trace = build_model(data)
    plot_ppd(trace,data)
    sns.regplot(x=data[:,0], y=data[:,1], marker="+",fit_reg=False)
    parameters = infer_parameters(model)
    plt.show()

if __name__=="__main__":
    print("START")
    main()
    print("DONE")
