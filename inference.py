#!/usr/bin/env python
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
#import mpld3
import pymc3 as pm
import pandas as pd
import theano
import theano as t
import theano.sparse.basic as sp
from theano import tensor as tt
from theano import printing as tp
from pymc3 import Model,Dirichlet
from pymc3.distributions.continuous import Gamma,Beta,Exponential
from pymc3.distributions.discrete import Categorical,BetaBinomial
from pymc3 import Deterministic
from pymc3.backends import Text

from data_generator import generate_data
from scipy import stats
from functools import reduce
from pprint import pprint
import seaborn as sns
import scipy.optimize as opt
import pickle
import os
import time

MAX_AXIS_CLUSTERS = 14
MAX_CLUSTERS = 60
MAX_CN = 5
THINNING = 2
TREE_DEPTH = 10
#TODO:Remove magic numbers
def preprocess_panel(panel):

    ref = get_array(panel,"ref_counts")
    alt = get_array(panel,"alt_counts")
    tre = get_array(panel,"total_raw_e")
    maj = get_array(panel,"major")
    tcnt = get_array(panel,"tumour_content")[1,:]
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

def build_model(x, iter_count, tune, trace_location, start=None, cluster_params="one"):
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
    #x = preprocess_panel(panel)
    ref,alt,tre,tcnt,maj = x

    n,dim = ref.shape

    #TODO:Add assert statement
    #assert(ref.shape == alt.shape,"ref and alt have different shapes!")
    bc_model = Model()
    with bc_model:
        
        #number of clusters
        c_count = 2**(TREE_DEPTH+1)-1
        split_prob= pm.Beta("split_prob",alpha=1,beta=1)
        is_split = 1#pm.Bernoulli("split_prob",shape=(c_count,dim),p = split_prob)
        split_factor = pm.Beta("split_factor",shape=(c_count,dim),alpha=1,beta=1)
        eff_split_factor = split_factor*is_split

        prior_cluster_locations = tt.zeros((c_count,dim))
        prior_cluster_locations = tt.set_subtensor(
            prior_cluster_locations[0],
            eff_split_factor[0])
        old_start,old_end = 0,1
        prior_cluster_magnitudes = tt.zeros((c_count,))
        prior_cluster_magnitudes = tt.set_subtensor(
            prior_cluster_magnitudes[0],
            1)
        relative_magnitude = 1
        for i in range(TREE_DEPTH):
            new_start,new_end = old_end,(old_end+1)*2-1
            left_magnitude = prior_cluster_locations[old_start:old_end]*split_factor[old_start:old_end]
            right_magnitude = 1-left_magnitude
            prior_cluster_locations = tt.set_subtensor(
                prior_cluster_locations[new_start:new_end:2],
                left_magnitude)
            prior_cluster_locations = tt.set_subtensor(
                prior_cluster_locations[new_start+1:new_end:2],
                right_magnitude)
            prior_cluster_magnitudes = tt.set_subtensor(
                prior_cluster_magnitudes[new_start:new_end:2],
                prior_cluster_magnitudes[old_start:old_end]/2)
            prior_cluster_magnitudes = tt.set_subtensor(
                prior_cluster_magnitudes[new_start+1:new_end:2],
                prior_cluster_magnitudes[old_start:old_end]/2)

            old_start,old_end = new_start,new_end

        #spawn tree clusters
        cluster_locations = tt.zeros((MAX_CLUSTERS,dim))
        cluster_indicies = Categorical("cluster_indicies", shape=(MAX_CLUSTERS), p=prior_cluster_magnitudes)
        for d in range(dim):
            #TODO:find a cleaner way of doing this
            cluster_locations = tt.set_subtensor(
                cluster_locations[:,d],
                prior_cluster_locations[cluster_indicies[:],d])


        #second DP
        cluster_dp_alpha = 1#Gamma("cluster_dp_alpha",mu=2,sd=1)
        cluster_betas = Beta("cluster_betas", 1, cluster_dp_alpha, shape=(MAX_CLUSTERS))
        cluster_magnitudes = tt.extra_ops.cumprod(1-cluster_betas)/(1-cluster_betas)*(cluster_betas)
        cluster_magnitudes = tt.set_subtensor(
            cluster_magnitudes[-1],
            1-tt.sum(cluster_magnitudes[:-1]))

        location_indicies = Categorical("location_indicies", shape=(n), p=cluster_magnitudes)
        
        #specify clustering parameter count
        if cluster_params == "samplecluster":
            shape = (MAX_CLUSTERS,dim)
        elif cluster_params == "sample":
            shape = (dim,)
        elif cluster_params == "one":
            shape = ()
        else:
            raise Exception("invalid clustering")
        cluster_clustering = Gamma("cluster_clustering", mu=500., sd=250,shape=shape)

        data_expectation = tt.zeros((n,dim))
        dispersion_factors = tt.zeros((n,dim))
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
        tcnt_precision = pm.Gamma("tcnt_precision",mu=2000,sd=2000)

        mutation_ccf_2 = data_expectation 
        alt_counts = alt
        total_counts = alt+ref
        major_cn = maj
        snv_count = len(major_cn)
        tumour_content = pm.Beta("tumour_content",alpha=tcnt*tcnt_precision,beta=(1-tcnt)*tcnt_precision,shape=(dim,))
        variable_tumour_copies = True
        max_cn = np.max(major_cn)
        cn_iterator = range(1,max_cn+1)
        

        cn_error = 0.05#pm.Beta('cn_error',alpha=1,beta=1)
        maj_p = np.zeros((MAX_CN,n,dim))
        for i in range(MAX_CN):
            cn = i+1
            array = np.zeros(MAX_CN)
            array[:cn] = 1/cn
            maj_p[:,major_cn == cn] = array[:,np.newaxis]
            
        p = maj_p*(1-cn_error)+np.ones((MAX_CN,n,dim))/MAX_CN*cn_error
        tcn_var = TensorCategorical('tcn_var',shape=(n,dim),p=p)
        tcns = pm.Deterministic('tcns',tcn_var+1)

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
        tumour_content = tumour_content[np.newaxis,:]
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
        #Deterministic("axis_cluster_magnitudes",axis_cluster_magnitudes)
        Deterministic("logP",bc_model.logpt)
        Deterministic("model_evidence", alt_counts.logpt)

        #assign lower step methods for the sampler
        #steps1 = pm.CategoricalGibbsMetropolis(vars=tcn_vars, proposal='uniform')
        #steps1 = pm.Metropolis(vars=tcn_vars, proposal='uniform')
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
        proposals = [
            None,
            CatProposal(MAX_CLUSTERS),
            CatProposal(MAX_CN)
            ]

        steps3 = IndependentVectorMetropolis(
            variables=[alt_counts,location_indicies,tcn_var],
            axes=[(1,),(),(1,)],
            proposals = 
                proposals,
            mask =[1,0,0])
        steps4 = IndependentClusterMetropolis(
            [alt_counts], 
            [(1,)], 
            location_indicies, 
            [cluster_indicies], 
            [()], 
            [lambda x: np.random.choice(MAX_AXIS_CLUSTERS,size = x.shape)], 
            [0], MAX_CLUSTERS, n)
        steps5 = pm.step_methods.HamiltonianMC(
            vars=[cluster_clustering,cluster_betas,split_prob,split_factor,tcnt_precision,tumour_content],
            step_scale=0.002,path_length=0.02)
        #steps5 = [pm.step_methods.Metropolis(
            #vars=[cluster_clustering,axis_dp_betas,cluster_betas,
            #axis_cluster_locations,axis_dp_alpha])]*10
        steps = [#steps1,
            #steps2,
            steps3,
            steps4,
            #steps5
            ]

        for rv in bc_model.basic_RVs:
            print(rv.name, rv.logp(bc_model.test_point))
            #pprint(bc_model.named_vars["tcns"].tag.test_value)#[32])
        pprint(bc_model.named_vars["vaf"].tag.test_value)

        if not os.path.isdir(trace_location):
            db = Text(trace_location)
            trace = pm.sample(iter_count, start=None, init=None,
                #nuts_kwargs={"target_accept":0.90,"integrator":"two-stage","step_scale":0.03},
                tune=tune, n_init=10000, njobs=1, step=steps,trace=db)
        else:
            trace = pm.backends.text.load(trace_location)
        #trace = pm.sample(iter_count,start=start,init=None,nuts_kwargs={"target_accept":0.9},tune=500,n_init=10000, njobs=1,step=steps)#,trace=db)

    return bc_model, trace

class CatProposal:
    def __init__(self,k):
        self.k = k
    def __call__(self,x):
        return np.random.choice(self.k,size = x.shape)
        
    

class TensorCategorical(pm.Discrete):
    def __init__(self, p, *args, **kwargs):
        super(TensorCategorical, self).__init__(*args, **kwargs)
        self.p = p/tt.sum(p, axis=0)
        self.mode = tt.max(p, axis=0)

    def logp(self,value):
        mask = tt.eq(tt.arange(self.p.shape[0])[:,np.newaxis,np.newaxis],value)
        vals = tt.sum(mask*self.p, axis=0)
        return tt.log(vals)

class IndependentVectorMetropolis(object):
    def __init__(self, variables=[],axes=[], proposals=[], mask=[]):
        #mandatory list of variables
        self.vars = variables
        #mandatory property
        self.generates_stats = False

        self.proposals = proposals
        self.axes = axes
        self.mask = mask

        self.sequence = list(self._get_indicies_iterator())

        model = pm.model.modelcontext(None)
        log_p = 0
        for var_index in range(len(self.vars)):
            var = self.vars[var_index]
            axes = self.axes[var_index]
            log_p += tt.sum(var.logp_elemwiset,axis=axes)

        self.log_p = model.fastfn(log_p)

    def step(self, point):
        new_point = point.copy()
        for var_index,np_slice in self.sequence:
            var = self.vars[var_index]
            axes = self.axes[var_index]
            proposal = self.proposals[var_index]
            name = var.name

            proposed_vals = proposal(new_point[name][np_slice])
            mr = self._metropolis_ratio(proposed_vals, name, np_slice, new_point)
            new_point[name][np_slice] = self._metropolis_switch(mr, proposed_vals, new_point[name][np_slice])

        return new_point


    def _get_indicies_iterator(self):
        #iterate through all non-independent variables
        for var_index in range(len(self.vars)):
            #do not sample from masked variables
            if self.mask[var_index]:
                continue

            axes,var = self.axes[var_index],self.vars[var_index]
            shape =  var.dshape
            axes = list(axes)
            axes.sort()

            if axes == []:
                yield var_index,(Ellipsis,)
                continue
            try:
                dims = [shape[axis] for axis in axes]
            except IndexError:
                raise Exception(
                "Axes index at position {} to large for variable with shape {}".format(var_index,shape))
            nmax = reduce(lambda x,y : x*y, dims, 1)
            for index_num in range(nmax):
                index = []
                for i in range(axes[-1]+1):
                    if i in axes:
                        n = index_num % shape[i]
                        index_num //= shape[i]
                        index.append(n)
                    else:
                        index.append(slice(None))
                        pass
                index.append(Ellipsis)
                yield var_index,tuple(index)

    def _metropolis_ratio(self, proposed_vals, name, np_slice, new_point):
        old_vals = np.copy(new_point[name][np_slice])

        new_point[name][np_slice] = proposed_vals
        logp_prop = self._eval_point(new_point)

        new_point[name][np_slice] = old_vals
        logp_init = self._eval_point(new_point)

        mr = logp_prop - logp_init
        return mr

    def _eval_point(self,point):
        return self.log_p(point)

    def _metropolis_switch(self, mr, proposed_vals, curr_vals):
        shape = mr.shape
        mask = mr > np.log(np.random.uniform(size=shape))
        new_vals = np.where(mask, proposed_vals, curr_vals)
        return new_vals

class IndependentClusterMetropolis(IndependentVectorMetropolis):
    def __init__(self, d_vars, d_var_axes, clustering, c_vars, c_var_axes, c_var_proposals, c_var_mask, num_clusters, num_data):
        self.generates_stats = False

        self.vars = c_vars
        self.axes = c_var_axes
        self.proposals = c_var_proposals
        self.mask = c_var_mask

        self.d_vars = d_vars
        self.d_var_axes = d_var_axes
        self.d_var_axes = d_var_axes

        self.sequence = list(self._get_indicies_iterator())

        log_p = 0
        for var_index in range(len(self.vars)):
            var = self.vars[var_index]
            axes = self.axes[var_index]
            log_p += tt.sum(var.logp_elemwiset,axis=axes)

        log_p_d = 0
        for var_index in range(len(self.d_vars)):
            var = self.d_vars[var_index]
            axes = self.d_var_axes[var_index]
            log_p_d += tt.sum(var.logp_elemwiset,axis=axes)

        log_p += self.add_by_clustering(log_p_d,clustering,num_clusters,num_data)

        model = pm.model.modelcontext(None)
        self.log_p = model.fastfn(log_p)

    def add_by_clustering(self, data, clustering, num_clusters, num_data):
        a = np.ones(num_data)
        b = np.arange(0,num_data+1)
        data = tt.shape_padleft(data)
        sparse_matrix = sp.CSR(a,clustering,b,(num_data,num_clusters))
        out_matrix = sp.structured_dot(data,sparse_matrix)
        output = tt.squeeze(out_matrix)
        return output

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
    x = [
        np.random.randint(5000,size =(200,6)),
        np.random.randint(5001,10000,size=(200,6)),
        np.random.uniform(2,4,size=(200,6)),
        np.random.uniform(size=(1,6)),
        np.random.randint(1,5,size=(200,6))
        ]
    #data,state = generate_data()
    #model,trace = build_model(data,start=state)
    model,trace = build_model(x,500,500,"trace/trace_null",start=None)
    #plot_ppd(model,trace,data)
    #plot_hard_clustering(model, trace, data, state)
    print("DONE")

if __name__=="__main__":
    main()
