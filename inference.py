#!/usr/bin/env python
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import mpld3
import pymc3 as pm
import pandas as pd
import theano
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
import pickle
import os

MAX_AXIS_CLUSTERS = 20
MAX_CLUSTERS = 100
THINNING = 2
#TODO:Remove magic numbers
def preprocess_panel(panel):

    ref = get_array(panel,"ref_counts")
    alt = get_array(panel,"alt_counts")
    tre = get_array(panel,"total_raw_e")
    maj = get_array(panel,"major")
    tcnt = get_array(panel,"tumour_content")
    data = get_array(panel,"ccf")
    vaf = get_array(panel,"vaf")

    #Replace uncounted mutations with a count of 1
    #Ideally we would keep a nan mask and
    #Infer the unobserved datapoints
    ref[np.where(ref+alt == 0)] = 1
    #ref[np.where(np.logical_not((np.isfinite(data))))] = 0

    return ref,alt,tre,tcnt,maj

def get_array(panel, col):
    return np.array(panel[col])

def build_model(panel, iter_count, tune, trace_location, start=None, cluster_params="one"):
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
    x = preprocess_panel(panel)
    ref,alt,tre,tcnt,maj = x

    n,dim = ref.shape

    #TODO:Add assert statement
    #assert(ref.shape == alt.shape,"ref and alt have different shapes!")
    bc_model = Model()
    with bc_model:
        axis_alpha = 1
        axis_beta = 1
        axis_dp_alpha = Gamma("axis_dp_alpha", mu=1, sd=1)
        cluster_dp_alpha = 1#Gamma("cluster_dp_alpha",mu=2,sd=1)
        #concentration parameter for the clusters
        if cluster_params == "samplecluster":
            shape = (MAX_CLUSTERS,dim)
        elif cluster_params == "sample":
            shape = (dim,)
        elif cluster_params == "one":
            shape = ()
        else:
            raise Exception("invalid clustering")

        cluster_clustering = Gamma("cluster_clustering", mu=500., sd=250,shape=shape)
        #cluster_sd = Gamma("cluster_std_dev",mu=0.15,sd=0.04)#0.2

        #per axis DP
        axis_dp_betas = Beta("axis_dp_betas", alpha=1, beta=axis_dp_alpha, shape=(dim,MAX_AXIS_CLUSTERS))

        axis_cluster_magnitudes = tt.extra_ops.cumprod(1-axis_dp_betas, axis=1)

        # Shift
        axis_cluster_magnitudes = tt.set_subtensor(
            axis_cluster_magnitudes[:, 1:],
            axis_cluster_magnitudes[:, :-1])
        axis_cluster_magnitudes = tt.set_subtensor(
            axis_cluster_magnitudes[:, 0],
            1.)

        # beta * cumprod(1-beta) 
        axis_cluster_magnitudes = axis_cluster_magnitudes * axis_dp_betas

        # Normalize with final elements
        axis_cluster_magnitudes = tt.set_subtensor(
            axis_cluster_magnitudes[:,-1],
            1-tt.sum(axis_cluster_magnitudes[:,:-1],axis=1))

        # Impose ordering
        #axis_cluster_magnitudes_flat = axis_cluster_magnitudes.reshape(shape=(dim*MAX_AXIS_CLUSTERS,))

        axis_cluster_locations = Beta(
            "axis_cluster_locations", alpha=axis_alpha, beta=axis_beta, shape=(dim,MAX_AXIS_CLUSTERS))

        #second DP
        cluster_betas = Beta("cluster_betas", 1, cluster_dp_alpha, shape=(MAX_CLUSTERS))
        cluster_magnitudes = tt.extra_ops.cumprod(1-cluster_betas)/(1-cluster_betas)*(cluster_betas)
        cluster_magnitudes = tt.set_subtensor(
            cluster_magnitudes[-1],
            1-tt.sum(cluster_magnitudes[:-1]))

        #spawn axis clusters
        cluster_locations = tt.zeros((MAX_CLUSTERS,dim))
        cluster_indicies = Categorical("cluster_indicies", shape=(MAX_CLUSTERS,dim), p=axis_cluster_magnitudes)
        for d in range(dim):
            #TODO:find a cleaner way of doing this
            cluster_locations = tt.set_subtensor(
                cluster_locations[:,d],
                axis_cluster_locations[d,cluster_indicies[:,d]])

        data_expectation = tt.zeros((n,dim))
        dispersion_factors = tt.zeros((n,dim))
        location_indicies = Categorical("location_indicies", shape=(n), p=cluster_magnitudes)
        for d in range(dim):
            data_expectation = tt.set_subtensor(
                data_expectation[:,d],
                cluster_locations[location_indicies,d])

            if cluster_params == "samplecluster":
                sub_tensor = cluster_clustering[location_indicies,d]
            elif cluster_params == "sample":
                sub_tensor = cluster_clustering[d]
            elif cluster_params == "one":
                sub_tensor = cluster_clustering
            else:
                raise Exception("This should never happen!")

            dispersion_factors = tt.set_subtensor(
                dispersion_factors[:,d],
                sub_tensor)

        
        dispersion = dispersion_factors
        mutation_ccf = data_expectation
################################################################################

        #Account for private mutations but we don't know wthe private mutations
        """
        # Neutral evolution
        private_frac = pm.Uniform('private_frac', lower=0, upper=1., shape=len(mutation_cluster))
        
        mutation_ccf_2 = mutation_ccf * (
            private_frac * is_private +
            np.ones(len(is_private)) - is_private) # only for privates
        """
        mutation_ccf_2 = data_expectation 

        alt_counts = alt
        total_counts = alt+ref
        major_cn = maj
        snv_count = len(major_cn)
        tumour_content = tcnt
        variable_tumour_copies = True
        
        tcn_vars = []
        tcns = theano.tensor.zeros((n,dim))
        for cn in np.unique(major_cn):
            idx = np.where(major_cn == cn)
            idx_len = len(idx[0])
            if cn == 0:
                print(idx)
                raise Exception("Major copy number should not be zero.")
            elif cn == 1:
                tcn = pm.Deterministic('tcn_1', theano.tensor.zeros((idx_len,))) + 1
            elif cn == 2:
                tcn = pm.Bernoulli('tcn_2', p=np.array([0.5] * idx_len), shape=idx_len) + 1
            else:
                tcn = pm.Categorical('tcn_'+str(cn),shape=idx_len,p=np.ones(cn)) + 1
            tcn_vars.append(tcn)
            idx_0 = theano.tensor.as_tensor_variable(idx[0])
            idx_1 = theano.tensor.as_tensor_variable(idx[1])
            tcns = theano.tensor.set_subtensor(tcns[(idx_0,idx_1)],tcn)

        pm.Deterministic("tcns", tcns)
        """
        if variable_tumour_copies:
            mean_tumour_copies = pm.Uniform('mean_tumour_copies', lower=0, upper=1., shape=snv_count)
            #mean_tumour_copies_v = mean_tumour_copies_v + 4.
            mean_tumour_copies_v = mean_tumour_copies*(np.repeat(5., snv_count) - tcns) + tcns
            #mean_tumour_copies_v = pm.Deterministic('mean_tumour_copies_v', mean_tumour_copies_v)
            #print(mean_tumour_copies.tag.test_value)
            #slow as molasses
            #mean_tumour_copies_v = [pm.Uniform('mean_tumour_copies', lower=tcns[i], upper=5.) for i in range(snv_count)]
        else:
            #mean_tumour_copies = pm.Deterministic('mean_tumour_copies', mean_tumour_copies)
            mean_tumour_copies_v = tre 
        """
        mean_tumour_copies_v = tre 
        
        #Account for normal contamination but we don't know the normal average contamination
        """
        average_normal_contamination = float(normal_alt_counts.sum())/float(normal_total_counts.sum())
        
        nalpha = average_normal_contamination * normal_dispersion
        nbeta = (1.-average_normal_contamination) * normal_dispersion
        
        norm_p = pm.Beta(
            'np', alpha=nalpha, beta=nbeta,
            shape = len(mutation_cluster))
        
        norm_alt_counts = pm.Binomial(
            'nx', n=normal_total_counts, p = norm_p,
            observed = normal_alt_counts)
        """

        norm_p = 0

        #vaf = (
        #    mutation_ccf_2 * tcns * tumour_content / 
        #    (2 * (1 - tumour_content) + mean_tumour_copies_v * tumour_content))
        
        ## Incorporates noise from the normal. 
        #tp.Print('vector', attrs = [ 'shape' ])(mutation_ccf_2)
        vaf = (
            (mutation_ccf_2 * tcns * tumour_content + (1-tumour_content) * norm_p * 2)/ 
            (2 * (1 - tumour_content) + mean_tumour_copies_v * tumour_content))
        vaf = pm.Deterministic("vaf",vaf)
        alpha = vaf * dispersion
        beta = (1 - vaf) * dispersion

        alt_counts = pm.BetaBinomial(
            'x', alpha=alpha, beta=beta,
            n=total_counts, observed=alt_counts)
################################################################################
        #t = tre
        #c = Categorical("tumour_copies",shape=(n,dim),p=np.array([0.33, 0.33, 0.34])) + 1
        #c = 2#<-- TODO REMOVE LATER!!!

        #vaf = data_expectation * c * tcnt / (2 * (1 - tcnt) + tre * tcnt)
        #Deterministic("vaf",vaf)

        #a=vaf*cluster_clustering
        #b=(1-vaf)*cluster_clustering

        #x = BetaBinomial("x",alpha=a,beta=b,n=alt+ref,observed=alt)

################################################################################

        #Log useful information
        Deterministic("f_expected", data_expectation)
        Deterministic("cluster_locations", cluster_locations)
        Deterministic("cluster_magnitudes", cluster_magnitudes)
        Deterministic("axis_cluster_magnitudes",axis_cluster_magnitudes)
        Deterministic("logP",bc_model.logpt)
        Deterministic("model_evidence", alt_counts.logpt)

        #assign lower step methods for the sampler
        steps1 = pm.CategoricalGibbsMetropolis(vars=tcn_vars, proposal='uniform')
        """
        if variable_tumour_copies:
            steps2 = pm.step_methods.HamiltonianMC(vars=[
                #b, 
                #frac, 
                #private_frac,
                mean_tumour_copies,
                #norm_p,
                ], step_scale=0.002, path_length=0.2)
        else:
            steps2 = pm.step_methods.HamiltonianMC(vars=[
                #b, 
                #frac, 
                #private_frac, 
                #norm_p
                ], 
                step_scale=0.002, path_length=0.2)
        """

        
        #assign upper step methods for the sampler
        steps3 = pm.CategoricalGibbsMetropolis(vars=[location_indicies], proposal='uniform')
        steps4 = pm.CategoricalGibbsMetropolis(vars=[cluster_indicies], proposal='uniform')
        #steps3 = pm.CategoricalGibbsMetropolis(vars=[c],proposal='uniform')
        """steps5 = pm.step_methods.HamiltonianMC(
            vars=[cluster_clustering,axis_dp_betas,cluster_betas,axis_cluster_locations,axis_dp_alpha],step_scale=0.002,path_length=0.2)"""
        steps = [steps1,
            #steps2,
            steps3,
            steps4,
            #steps5
            ]
        #steps3 = pm.step_methods.Metropolis(vars=[betas,betas2,axis_cluster_locations])

        #Save data to csv
        # db = Text('trace_output')

        """
        if start is not None:
            with open(start,"rb") as f:
                start_trace = pickle.load(f)
                print(list(start_trace))
        """

        if not os.path.isfile(trace_location):
            trace = pm.sample(iter_count, start=None, init=None,
                nuts_kwargs={"target_accept":0.9},
                tune=tune, n_init=10000, njobs=1, step=steps)[::THINNING]
            with open(trace_location, "wb") as f:
                pickle.dump(trace, f)
        else:
            with open(trace_location, "rb") as f:
                trace = pickle.load(f)
        #trace = pm.sample(iter_count,start=start,init=None,nuts_kwargs={"target_accept":0.9},tune=500,n_init=10000, njobs=1,step=steps)#,trace=db)

    return bc_model, trace

def plot_hard_clustering(model, trace, data, truth=None):
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
    indicies = get_map_item(model, trace, "location_indicies")
    if is_truth:
        true_indicies = truth["location_indicies"]

    g = gen_plot(data, 'CLUSTERING', indicies)
    
    if is_truth:
        h = gen_plot(data, 'GROUND TRUTH', true_indicies)

def show_plots():
    mpld3.show()

def get_map_item(model, trace, index):
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

def compute_cluster_means(data, clustering, cluster_names):
    """Compute the cluster means of some data.
    """
    cluster_count = len(cluster_names)
    dim = data.shape[1]
    cluster_means = np.ndarray((cluster_count,dim), dtype=np.float32)
    for i in range(cluster_count):
        cluster_means[i,:] = np.mean(np.squeeze(data[np.where(clustering==cluster_names[i]),:]), axis=0)
    return cluster_means


def gen_plot(data, subtitle, indicies):
    """Generates a grid plot with a given
    subtitle with coloring specified by 
    indicies"""
    def cluster_plot(x,y,**kwargs):
        sns.set_style('whitegrid')
        #sns.plt.ylim(0,3)
        #sns.plt.xlim(0,3)
        plt.scatter(x,y,**kwargs)

    df = pd.DataFrame(data)
    dim = data.shape[1]
    df = df.assign(location_indicies = indicies)
    g = sns.PairGrid(df, hue="location_indicies",vars=range(dim))
    g.fig.suptitle(subtitle)
    g.map_lower(cluster_plot)
    g.map_diag(plt.hist)
    g.add_legend(fontsize=14)
    return g

def plot_axis(model, trace):
    pass

def plot_cluster_means(data, clustering, subtitle):
    """Plots cluster means of a given dataset with a
    given clustering"""
    indicies = list(set(clustering))
    cluster_means = compute_cluster_means(data, clustering, indicies)
    gen_plot(cluster_means, subtitle, indicies)


def display_map_axis_mapping(model, trace):
    """Creates a lookup table showing each cluster and which cluster
    means are used for each dimension."""
    mapping = get_map_item(model, trace, "cluster_indicies")
    print(pd.DataFrame(mapping))
    fig, ax = plt.subplots(1)
    ax.table(cellText=mapping, fontsize=10, rowLabels=range(MAX_CLUSTERS), loc='center', bbox=[0.1, 0.1, 0.9, 0.9])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_ppd(model, trace, data):
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
    predictions = np.reshape(predictions, (t*n,d))

    #grab a random sample of predictions
    np.random.shuffle(predictions)
    predictions = predictions[:n_predictions,:]

    def ppd_plot(x, y, **kwargs):
        """Plots kde if kwargs[source]="s" 
            or a scatter plot if kwargs[source]="o" """
        source = kwargs["source"]
        del kwargs["source"]
        sns.set_style('whitegrid')
        sns.plt.ylim(0,1)
        sns.plt.xlim(0,1)
        if source == "s":
            kwargs["cmap"] = "Oranges"
            sns.kdeplot(x, y, n_levels=20, **kwargs)
            #plt.scatter(x,y,**kwargs)
        elif source == "o":
            kwargs["cmap"] = "Blues"
            plt.scatter(x, y, **kwargs)
     
    df_predictive = pd.DataFrame(predictions)
    df_predictive = df_predictive.assign(source= lambda x: "s")
    df_observed = pd.DataFrame(data)
    df_observed = df_observed.assign(source= lambda x: "o")
    #merge observed and predicted data into one dataframe
    #and distinguishg them by the value of the "source" column
    df = pd.concat([df_predictive,df_observed],ignore_index=True)
    
    #Map ppd_plot onto the data in a pair grid to visualize predictive density 
    g = sns.PairGrid(df,hue="source", hue_order=["s","o"], hue_kws={"source":["s","o"]})
    g.map_offdiag(ppd_plot)
    plt.show()
    
    

def plot_max_n(trace, n, last, spacing):
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
    model,trace = build_model(data, start=None)
    #plot_ppd(model,trace,data)
    plot_hard_clustering(model, trace, data, state)
    print("DONE")

if __name__=="__main__":
    main()
