#!/usr/bin/env python

from inference import build_model,\
                      plot_hard_clustering,\
                      plot_cluster_means,\
                      display_map_axis_mapping,\
                      show_plots
from data_generator import generate_data
import numpy as np
import pandas as pd

def main():
    path = "data.tsv"
    print("Parsing Data...")
    data,ref,alt,tre,sample_names,node = parse_data(path)
    #data,_ = generate_data()
    print("Generating Model...")
    model,trace = build_model(ref,alt,tre,120000,start=node)
    print(trace["logP"])
    print(trace["cluster_clustering"])
    print("Plotting data...")
    #display_map_axis_mapping(model,trace)
    #plot_hard_clustering(model,trace,data,node)
    #plot_cluster_means(data,node["location_indicies"],"Origin Node Clusters")
    #show_plots()
    print("Done!")

def parse_data(path):
    """Reads a tsv file and converts it to clusterable data for
    the model.
    
    Args:
        path: the path to the data
    Returns:
        (data,sample_names,node):The post processed data along with
        the names of each sample and the origin node of each data point
    """
    df = pd.DataFrame.from_csv(path,sep='\t')
    df = df[["sample_id","ccf","Origin Node","alt_counts","ref_counts","total_raw_e"]]
    entries = len(df)
    sample_names = list(set(df["sample_id"]))
    sample_count = len(sample_names)

    epsilon = 1E-5

    alt = np.ndarray((entries//sample_count,sample_count),dtype=np.int32)
    ref = np.ndarray((entries//sample_count,sample_count),dtype=np.int32)
    tre = np.ndarray((entries//sample_count,sample_count),dtype=np.float32)
    data = np.ndarray((entries//sample_count,sample_count),dtype=np.float32)
    node = {}
    node["location_indicies"] = np.array(df.loc[lambda df: df.sample_id == sample_names[0],"Origin Node"])
    for i in range(sample_count):
        alt[:,i] = df.loc[lambda df: df.sample_id == sample_names[i],"alt_counts"]
        ref[:,i] = df.loc[lambda df: df.sample_id == sample_names[i],"ref_counts"]
        tre[:,i] = df.loc[lambda df: df.sample_id == sample_names[i],"total_raw_e"]
        data[:,i] = df.loc[lambda df: df.sample_id == sample_names[i],"ccf"]

    #Replace uncounted mutations with a count of 1
    #Ideally we would keep a nan mask and 
    #Infer the unobserved datapoints
    ref[np.where(ref+alt == 0)] = 1
    ref[np.where(np.logical_not((np.isfinite(data))))] = 0

    return data,ref,alt,tre,sample_names,node


if __name__=="__main__":
    main()
