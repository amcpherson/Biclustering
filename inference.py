#!/usr/bin/env python
import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
import mpld3
import pymc3 as pm
import pandas as pd
import theano
import theano as t
import theano.sparse.basic as sp
from theano import tensor as tt
from theano import printing as tp
from pymc3 import Model,Dirichlet,Deterministic
from pymc3.math import logsumexp
from pymc3.distributions.dist_math import bound
from pymc3.distributions.continuous import Gamma,Beta,Exponential
from pymc3.distributions.discrete import Categorical,BetaBinomial

from data_generator import generate_data
from scipy import stats
from functools import reduce
from pprint import pprint
import scipy
import seaborn as sns
import scipy.optimize as opt
import pickle
import os
import time

MAX_AXIS_CLUSTERS = 14
MAX_CLUSTERS = 50
MAX_CN = 5
THINNING = 2
#TODO:Remove magic numbers
def preprocess_panel(panel):

    ref = get_array(panel,"ref_counts")
    alt = get_array(panel,"alt_counts")
    tre = get_array(panel,"total_raw_e")
    maj = get_array(panel,"major")
    tcnt = get_array(panel,"tumour_content")[1,:]

    #Replace uncounted mutations with a count of 1
    #Ideally we would keep a nan mask and
    #Infer the unobserved datapoints
    ref[np.where(ref+alt == 0)] = 1

    return ref,alt,tre,tcnt,maj

def get_array(panel, col):
    return np.array(panel[col])

def build_model(panel, tune, iter_count, trace_location, 
    prev_trace=None, 
    cluster_params="one", 
    thermodynamic_beta=1,
    sampler="NUTS",
    infer_cn_error=False):
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
        panel: a pandas panel with reference counts, alternate counts, tumour content,
            major copy number and mean tumour copy number
        tune: number of steps to tune the samplers
        iter_count: number of steps to sample from the sampler
        trace_location: path to the current hd5 trace
        prev_trace: A previous pymc3 hd5 trace object
        cluster_params: The number of clustering parameters
        thermodynamic_beta: the inverse sampling temperature of the system higher values
        bias the likelyhood toward points in the space with higher likelyhoods
        sampler: Which kind of sampler to use for the continous variables
        infer_cn_error: wheter or not to infer the number of incorrect tumour copy number calls
    Returns:
        (model,trace): The pymc3 model object along with the sampled trace object.

    """
    ref,alt,tre,tcnt,maj = preprocess_panel(panel)

    n,dim = ref.shape

    bc_model = Model()
    with bc_model:
        
        #Parameters of the per sample prior distribution over cluster 
        #locations
        #both are set to 1 to give a uniform distribution over
        #the space 
        axis_alpha = 1
        axis_beta = 1

        #Alpha parameters of all dp's alpha
        #The model is relatively insensitive to these parameters as
        #the amount of data increases
        axis_dp_alpha = Gamma("axis_dp_alpha", mu=1, sd=1)
        cluster_dp_alpha = 1

        #concentration parameter for the clusters
        if cluster_params == "samplecluster":
            shape = (MAX_CLUSTERS,dim)
        elif cluster_params == "sample":
            shape = (dim,)
        elif cluster_params == "one":
            shape = ()
        else:
            raise ValueError("invalid cluster parameters")

        #Concentration parameter for all clusters
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

        #Assign per datapoint expected values and clustering parameters
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
                raise Exception("Incorrect cluster parameter argument")

            dispersion_factors = tt.set_subtensor(
                dispersion_factors[:,d],
                sub_tensor)
        
        dispersion = dispersion_factors
        mutation_ccf = data_expectation


        #TODO: bring variables from  allens likelyhood under 
        alt_counts = alt
        total_counts = alt+ref
        major_cn = maj
        snv_count = len(major_cn)
        variable_tumour_copies = True
        max_cn = np.max(major_cn)
        cn_iterator = range(1,max_cn+1)

        #Uncertainty in tumour content, values were set under the assumption that
        #The reported tumour content is mostly correct
        tcnt_precision = pm.Gamma("tcnt_precision",mu=2000,sd=2000)
        tumour_content = pm.Beta("tumour_content",alpha=tcnt*tcnt_precision,beta=(1-tcnt)*tcnt_precision,shape=(dim,))

        #Account for private mutations but we don't know wthe private mutations
        # Neutral evolution

        #Check to see if a mutation is private
        test = np.vectorize(lambda x,y: (scipy.stats.binom_test(x,y,p=0.001,alternative='greater') < 0.01))
        is_private = np.sum(test(alt_counts,total_counts),axis=1) == 1

        #Account for private mutations but we don't know wthe private mutations
        # Neutral evolution
        private_frac = pm.Uniform('private_frac', lower=0, upper=1., shape=(n,dim))
        
        #assign a uniform probability to all private mutations, independent of their clustering
        mutation_ccf_2 = (mutation_ccf * (
            private_frac * is_private[:,np.newaxis] +
            (1 - is_private[:,np.newaxis])))[:,:,np.newaxis]
        
        if infer_cn_error is False:
            cn_error = 0.05
        else:
            cn_error = pm.Beta('cn_error',alpha=1,beta=1)


        #Place probabilty distribution over mutation copy number
        maj_p = np.zeros((n,dim,MAX_CN))
        for i in range(MAX_CN):
            cn = i+1
            array = np.zeros(MAX_CN)
            array[:cn] = 1/cn
            maj_p[major_cn == cn,:] = array
            
        p = maj_p*(1-cn_error)+np.ones((n,dim,MAX_CN))/MAX_CN*cn_error
        tcns = np.array(range(1,MAX_CN+1))

        
        #Assign expected vaf to each data_point
        mean_tumour_copies = tre[:,:,np.newaxis]
        tumour_content = tumour_content[np.newaxis,:,np.newaxis]
        vaf = (
            (mutation_ccf_2 * tcns * tumour_content)/ 
            (2 * (1 - tumour_content) + mean_tumour_copies * tumour_content))
        vaf = pm.Deterministic("vaf",vaf)

        alpha = vaf * dispersion[:,:,np.newaxis]
        beta = (1 - vaf) * dispersion[:,:,np.newaxis]

        alt_count_components = pm.BetaBinomial.dist(alpha=alpha, beta=beta,
            n=total_counts[:,:,np.newaxis],shape=(n,dim,MAX_CN))
        alt_counts = pm.Mixture('x',w=p,comp_dists=alt_count_components,observed=alt_counts[:,:,np.newaxis],shape=(n,dim,1))
        #Properly define logp_elemwiset for a mixture distribution
        dist = alt_counts.distribution
        comp_log_prob = tt.switch(vaf >= 1,-1e5,dist._comp_logp(alt_counts))
        alt_counts.logp_elemwiset = logsumexp(tt.log(dist.w)+comp_log_prob,axis=-1)

        #Log useful information
        Deterministic("f_expected", data_expectation)
        Deterministic("cluster_locations", cluster_locations)
        Deterministic("cluster_magnitudes", cluster_magnitudes)
        Deterministic("axis_cluster_magnitudes",axis_cluster_magnitudes)
        Deterministic("logP",bc_model.logpt)

        #TODO: Determine why this isn't getting logged in the trace after upgrading
        #to pymc3 3.2
        Deterministic("model_evidence", alt_counts.logpt)

        #Potential correction factor for simulated annealing
        pm.Potential("extra_potential",bc_model.logpt*(thermodynamic_beta-1))

        #Assign step methods for all variables
        steps1 = IndependentVectorMetropolis(
            variables=[alt_counts,location_indicies],
            axes=[(1,2),()],
            proposals = [None,CatProposal(MAX_CLUSTERS)],
            mask=[1,0], t_b=thermodynamic_beta)
        steps2 = IndependentClusterMetropolis(
            [alt_counts], 
            [(1,2)], 
            location_indicies, 
            [cluster_indicies], 
            [(1,)], 
            [CatProposal(MAX_AXIS_CLUSTERS)], 
            [0], MAX_CLUSTERS, n, t_b=thermodynamic_beta)
        steps3 = pm.step_methods.Metropolis(
            vars=[cluster_clustering,axis_dp_betas,cluster_betas,axis_cluster_locations,
            axis_dp_alpha,tcnt_precision,tumour_content,private_frac],tune_interval=50)

        if sampler == "NUTS":
            #Multiplication by 5 to partialy account for MH sampling inefficiency
            steps = [
                steps1,
                steps2
                ]*5
        elif samper == "Metropolis":
            steps = [
                steps1,
                steps2,
                steps3
                ]
        else:
            raise ValueError("Invalid sampler")

        #Create trace object
        if not os.path.isfile(trace_location):
            #If no trace is already present create the trace
            print("Starting sampling...")
            trace = pm.backends.hdf5.HDF5(name=trace_location)
            pm.sample(iter_count,
                #nuts_kwargs={"target_accept":0.80,"max_treedepth":5},
                tune=tune, n_init=10000, njobs=1, step=steps,trace=trace)
        elif prev_trace is not None:
            #If trace is present and a previous trace object is already given
            #Append the current run to the trace object
            print("Continuing sampling...")
            trace = prev_trace 
            #TODO: With hd5 traces, this operation is extremely memory inefficient
            pm.sample(iter_count,
                tune=tune, n_init=10000, njobs=1, step=steps,trace=trace)
        else:
            #If trace is present and a previous trace object is not given then
            #Reload the trace object as a multitrace for postprocessing
            print("Reloading trace...")
            trace = pm.backends.hdf5.load(trace_location)

    return bc_model, trace

class CatProposal:
    """Categorical proposal distribution object"""
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
    def __init__(self, variables=[],axes=[], proposals=[], mask=[],t_b = 1):
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

        #set thermodynamic_beta
        self.t_b = t_b

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

        mr = (logp_prop - logp_init)*self.t_b
        return mr

    def _eval_point(self,point):
        return self.log_p(point)

    def _metropolis_switch(self, mr, proposed_vals, curr_vals):
        shape = mr.shape
        mask = mr > np.log(np.random.uniform(size=shape))
        new_vals = np.where(mask, proposed_vals, curr_vals)
        return new_vals

class IndependentClusterMetropolis(IndependentVectorMetropolis):
    def __init__(self, d_vars, d_var_axes, clustering, c_vars, c_var_axes, c_var_proposals, c_var_mask, num_clusters, num_data,t_b=1):
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

        #set thermodynamic_beta
        self.t_b = t_b

    def add_by_clustering(self, data, clustering, num_clusters, num_data):
        a = np.ones(num_data)
        b = np.arange(0,num_data+1)
        data = tt.shape_padleft(data)
        sparse_matrix = sp.CSR(a,clustering,b,(num_data,num_clusters))
        out_matrix = sp.structured_dot(data,sparse_matrix)
        output = tt.squeeze(out_matrix)
        return output

def plot_clustering(model, trace, panel):
    pass

def get_map_item(model, trace, index):
    """Aquire the MAP estimate of the value
    of a variable in a trace.

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




