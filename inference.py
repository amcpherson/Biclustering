#!/usr/bin/env python
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
import theano as t
from theano import tensor as tt
from theano import printing as tp
from pymc3 import Model,Dirichlet
from pymc3.distributions.continuous import Gamma,Beta,Exponential
from pymc3.distributions.discrete import Categorical,BetaBinomial
from pymc3 import Deterministic
from pymc3.backends import Text

from data_generator import generate_data
from scipy import stats
import seaborn as sns
import scipy.optimize as opt

MAX_AXIS_CLUSTERS = 10
MAX_CLUSTERS = 200
#TODO:Remove magic numbers
def build_model(ref,alt,tre,tcnt,iter_count=5000,start=None):
    """Returns a model of the data along with samples from it's posterior.
    
    Creates a pymc3 model object that represents a heirachical dirchelet 
    process model with N+1 dirchelet processes, where N is the number of 
    dimensions for the data. The first N dirchelet processes specifiy the 
    location of clusters along each axis and have a uniform beta base 
    distribution. The second dirchelet process uses the cartesian product of 
    the axis dirchelet processes as it's base distribuiton to generate
    cluster means, which are in turn used to put a gamma likelyhood 
    distribuiton over datapoints. 

    Args:
        data:the ground truth data used to train the model
        iter_count
        start:The starting point for the markov chains
    Returns:
        (model,trace): The pymc3 model object along with the sampled trace.

    """
    n,dim = ref.shape

    # Ensure cluster indices is the correct shape
    if 'cluster_indicies' in start:
        full_cluster_indices = np.zeros((MAX_CLUSTERS,dim)) + 2
        full_cluster_indices[:start['cluster_indicies'].shape[0], :] = start['cluster_indicies']
        start['cluster_indicies'] = full_cluster_indices

    #TODO:Add assert statement
    #assert(ref.shape == alt.shape,"ref and alt have different shapes!")
    bc_model = Model()
    with bc_model:
        axis_alpha = 1
        axis_beta = 1
        axis_dp_alpha = 0.001#Gamma("axis_dp_alpha",mu=2,sd=1)
        cluster_dp_alpha = 0.1#Gamma("cluster_dp_alpha",mu=2,sd=1)
        #concentration parameter for the clusters
        cluster_clustering = Gamma("cluster_clustering",mu=500.,sd=1.)
        #cluster_sd = Gamma("cluster_std_dev",mu=0.15,sd=0.04)#0.2

        #per axis DP
        axis_dp_betas = Beta("axis_dp_betas",alpha=1,beta=axis_dp_alpha,shape=(dim,MAX_AXIS_CLUSTERS))
        axis_cluster_magnitudes = tt.extra_ops.cumprod(1-axis_dp_betas,axis=1)/(1-axis_dp_betas)*axis_dp_betas
        axis_cluster_magnitudes = tt.set_subtensor(
            axis_cluster_magnitudes[:,-1],
            1-tt.sum(axis_cluster_magnitudes[:,:-1],axis=1))
        axis_cluster_locations = Beta(
            "axis_cluster_locations",alpha=axis_alpha, beta=axis_beta, shape=(dim,MAX_AXIS_CLUSTERS))

        #second DP
        cluster_betas = Beta("cluster_betas",1,cluster_dp_alpha,shape=(MAX_CLUSTERS))
        cluster_magnitudes = tt.extra_ops.cumprod(1-cluster_betas)/(1-cluster_betas)*(cluster_betas)
        cluster_magnitudes = tt.set_subtensor(
            cluster_magnitudes[-1],
            1-tt.sum(cluster_magnitudes[:-1]))

        #spawn axis clusters
        cluster_locations = tt.zeros((MAX_CLUSTERS,dim))
        cluster_indicies = Categorical("cluster_indicies",shape=(MAX_CLUSTERS,dim),p=axis_cluster_magnitudes)
        for d in range(dim):
            #TODO:find a cleaner way of doing this
            cluster_locations = tt.set_subtensor(
                cluster_locations[:,d],
                axis_cluster_locations[d,cluster_indicies[:,d]])

        data_expectation = tt.zeros((n,dim))
        location_indicies = Categorical("location_indicies",shape=(n),p=cluster_magnitudes)
        for d in range(dim):
            data_expectation = tt.set_subtensor(
                data_expectation[:,d],
                cluster_locations[location_indicies,d])

        a=data_expectation*cluster_clustering
        b=(1-data_expectation)*cluster_clustering

        f = Deterministic("f_expected",data_expectation)
        t = tre
        c = Categorical("tumour_copies",shape=(n,dim),p=np.array([0.33, 0.33, 0.34])) + 1

        vaf = f * c * tcnt / (2 * (1 - tcnt) + tre * tcnt)

        x = BetaBinomial("x",alpha=a,beta=b,n=alt+ref,observed=alt)

        #Log useful information
        Deterministic("f",data_expectation)
        Deterministic("cluster_locations",cluster_locations)
        Deterministic("cluster_magnitudes",cluster_magnitudes)
        Deterministic("logP",bc_model.logpt)
        
        #assign step methods for the sampler
        steps1 = pm.CategoricalGibbsMetropolis(vars=[location_indicies],proposal='uniform')
        steps2 = pm.CategoricalGibbsMetropolis(vars=[cluster_indicies],proposal='uniform')
        steps3 = pm.CategoricalGibbsMetropolis(vars=[c],proposal='uniform')
        steps4 = pm.step_methods.HamiltonianMC(
            vars=[cluster_clustering,axis_dp_betas,cluster_betas,axis_cluster_locations],step_scale=0.002,path_length=0.2)
        steps = [steps4,steps1,steps2,steps3]
        #steps3 = pm.step_methods.Metropolis(vars=[betas,betas2,axis_cluster_locations])

        #Save data to csv
        # db = Text('trace_output')
        trace = pm.sample(iter_count,start=start,init=None,tune=40000,n_init=10000, njobs=1,step=steps)#,trace=db)

    return bc_model,trace

def plot_hard_clustering(model,trace,data,truth=None):
    """Plot the hard clustering generated by the MAP estimate of the trace
    along with the true clustering of the data.
    
    Takes N dimesensional data, a model, a trace,and the
    true data generating parameters to produce 2 N by N grids of 
    2D plots showing a scatter plot along each pair of axes.

    Args:
        model:the model of the data
        trace:trace generated by sampling from the model
        data:the ground truth data used to train the model
        truth:the ground truth of the latent variables that
        generated the data. Must contain a "location_indicies"
        index with the data's true clustering
    Returns:
        None

    """
    #extract true indicies and extra indicies
    is_truth = truth is not None
    indicies = get_map_item(model,trace,"location_indicies")
    if is_truth:
        true_indicies = truth["location_indicies"]

    g = gen_plot(data,'CLUSTERING',indicies)
    
    if is_truth:
        h = gen_plot(data,'GROUND TRUTH',true_indicies)

def show_plots():
    plt.show()

def get_map_item(model,trace,index):
    """Aquire the MAP estimate of the value
    of a variable in a model.

    Args:
        model:the model of the data
        trace:trace generated by sampling from the model
        index:the name of the variable of interest 
    Returns:
        The map estimate of the variable specified by index
    """
    with model:
        map_index = np.argmax(trace["logP"])
        item = trace[index][map_index]
    return item

def compute_cluster_means(data,clustering,cluster_names):
    """Compute the cluster means of some data.
    """
    cluster_count = len(cluster_names)
    dim = data.shape[1]
    cluster_means = np.ndarray((cluster_count,dim),dtype=np.float32)
    for i in range(cluster_count):
        cluster_means[i,:] = np.mean(np.squeeze(data[np.where(clustering==cluster_names[i]),:]),axis=0)
    return cluster_means


def gen_plot(data,subtitle,indicies):
    """Generates a grid plot with a given
    subtitle with coloring specified by 
    indicies"""
    def cluster_plot(x,y,**kwargs):
        sns.set_style('whitegrid')
        sns.plt.ylim(0,3)
        sns.plt.xlim(0,3)
        plt.scatter(x,y,**kwargs)

    df = pd.DataFrame(data)
    dim = data.shape[1]
    df = df.assign(location_indicies = indicies)
    g = sns.PairGrid(df,hue="location_indicies",vars=range(dim))
    g.fig.suptitle(subtitle)
    g.map_lower(cluster_plot)
    g.map_diag(plt.hist)
    g.add_legend(fontsize=14)
    return g

def plot_cluster_means(data,clustering,subtitle):
    """Plots cluster means of a given dataset with a
    given clustering"""
    indicies = list(set(clustering))
    cluster_means = compute_cluster_means(data,clustering,indicies)
    gen_plot(cluster_means,subtitle,indicies)


def display_map_axis_mapping(model,trace):
    """Creates a lookup table showing each cluster and which cluster
    means are used for each dimension."""
    mapping = get_map_item(model,trace,"cluster_indicies")
    print(pd.DataFrame(mapping))
    fig, ax = plt.subplots(1)
    ax.table(cellText=mapping,fontsize=10,rowLabels=range(MAX_CLUSTERS),loc='center',bbox=[0.1, 0.1, 0.9, 0.9])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_ppd(model,trace,data):
    """Plot the posterior predictive distribution over the data.
    
    Takes N dimesensional data, a model, and a trace to produce 
    An N by N grid of 2d plots. Showing a scatter plot along each pair
    of axes.

    Args:
        model:the model of the data
        trace:trace generated by sampling from the model
        data:the ground truth data used to train the model
    Returns:
        None

    """
    n_predictions = 1000    
    burn_in = 500
    
    #generate array of predictions
    with model:
        samples = pm.sample_ppc(trace)
    predictions = samples["data"]
    predictions = predictions[burn_in:,:,:]
    t,n,d = predictions.shape
    predictions = np.reshape(predictions,(t*n,d))

    #grab a random sample of predictions
    np.random.shuffle(predictions)
    predictions = predictions[:n_predictions,:]

    def ppd_plot(x,y,**kwargs):
        """Plots kde if kwargs[source]="s" 
            or a scatter plot if kwargs[source]="o" """
        source = kwargs["source"]
        del kwargs["source"]
        sns.set_style('whitegrid')
        sns.plt.ylim(0,1)
        sns.plt.xlim(0,1)
        if source == "s":
            kwargs["cmap"] = "Oranges"
            sns.kdeplot(x,y,n_levels=20,**kwargs)
            #plt.scatter(x,y,**kwargs)
        elif source == "o":
            kwargs["cmap"] = "Blues"
            plt.scatter(x,y,**kwargs)
     
    df_predictive = pd.DataFrame(predictions)
    df_predictive = df_predictive.assign(source= lambda x: "s")
    df_observed = pd.DataFrame(data)
    df_observed = df_observed.assign(source= lambda x: "o")
    #merge observed and predicted data into one dataframe
    #and distinguishg them by the value of the "source" column
    df = pd.concat([df_predictive,df_observed],ignore_index=True)
    
    #Map ppd_plot onto the data in a pair grid to visualize predictive density 
    g = sns.PairGrid(df,hue="source",hue_order=["s","o"],hue_kws={"source":["s","o"]})
    g.map_offdiag(ppd_plot)
    plt.show()
    
    

def plot_max_n(trace,n,last,spacing):
    """Plot the first 2 dimensions of position of the 
    largest n cluster means in a 2 dimensional data set.

    Args:
        trace:trace generated by sampling from the model
        last:the last n draws to plot from the trace
        spacing:the spacing between trace samples
    Returns:
        None

    """
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

def main():
    """Generates sample data, clusters it, and plots
    the resulting clustering along with the ground truth.
    """
    print("START")
    data,state = generate_data()
    #model,trace = build_model(data,start=state)
    model,trace = build_model(data,start=None)
    #plot_ppd(model,trace,data)
    plot_hard_clustering(model,trace,data,state)
    print("DONE")

if __name__=="__main__":
    main()
