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
    data,sample_names,node = parse_data(path)
    #data,_ = generate_data()
    print("Generating Model...")
    model,trace = build_model(data,1500,start=node)
    print(trace["logP"])
    print("Plotting data...")
    display_map_axis_mapping(model,trace)
    plot_hard_clustering(model,trace,data,node)
    plot_cluster_means(data,node["location_indicies"],"Origin Node Clusters")
    show_plots()
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
    df = df[["sample_id","ccf","Origin Node"]]
    entries = len(df)
    sample_names = list(set(df["sample_id"]))
    sample_count = len(sample_names)

    epsilon = 1E-5

    data = np.ndarray((entries//sample_count,sample_count),dtype=np.float32)
    node = {}
    node["location_indicies"] = np.array(df.loc[lambda df: df.sample_id == sample_names[0],"Origin Node"])
    for i in range(sample_count):
        data[:,i] = df.loc[lambda df: df.sample_id == sample_names[i],"ccf"]

    #Replace zeros with epsilons so that the data becomes part of the model support 
    data[np.where(data == 0.0)] = epsilon


    #Replace missing values with epsilon
    #Ideally we would keep a nan mask and 
    #Infer the unobserved datapoints
    data[np.where(np.logical_not((np.isfinite(data))))] = epsilon

    return data,sample_names,node


if __name__=="__main__":
    main()
