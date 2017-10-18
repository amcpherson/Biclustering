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
    data,ref,alt,tre,maj,tcnt,sample_names,node = parse_data(path)
    #data,_ = generate_data()
    print("Generating Model...")
    model,trace = build_model(ref,alt,tre,tcnt,maj,"trace_output.pkl",30,30,start=node)
    print(trace["logP"])
    #print(trace["cluster_clustering"])
    #print(trace["cluster_magnitudes"][-1])
    #print("Plotting data...")
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
    df = df[["sample_id","ccf","Origin Node","alt_counts","ref_counts","total_raw_e","major","tumour_content"]]
    entries = len(df)
    sample_names,sample_count = get_sample_names(df)

    epsilon = 1E-5

    alt = np.ndarray((entries//sample_count,sample_count),dtype=np.int32)
    ref = np.ndarray((entries//sample_count,sample_count),dtype=np.int32)
    tre = np.ndarray((entries//sample_count,sample_count),dtype=np.float32)
    maj = np.ndarray((entries//sample_count,sample_count),dtype=np.int32)
    tcnt = np.ndarray((entries//sample_count,sample_count),dtype=np.float32)
    data = np.ndarray((entries//sample_count,sample_count),dtype=np.float32)
    node = {}
    node["location_indicies"] = np.array(df.loc[lambda df: df.sample_id == sample_names[0],"Origin Node"])
    for i in range(sample_count):
        alt[:,i] = df.loc[lambda df: df.sample_id == sample_names[i],"alt_counts"]
        ref[:,i] = df.loc[lambda df: df.sample_id == sample_names[i],"ref_counts"]
        tre[:,i] = df.loc[lambda df: df.sample_id == sample_names[i],"total_raw_e"]
        maj[:,i] = df.loc[lambda df: df.sample_id == sample_names[i],"major"]
        tcnt[:,i] = df.loc[lambda df: df.sample_id == sample_names[i],"tumour_content"]
        data[:,i] = df.loc[lambda df: df.sample_id == sample_names[i],"ccf"]

    #Replace uncounted mutations with a count of 1
    #Ideally we would keep a nan mask and 
    #Infer the unobserved datapoints
    ref[np.where(ref+alt == 0)] = 1
    ref[np.where(np.logical_not((np.isfinite(data))))] = 0

    return data,ref,alt,tre,maj,tcnt,sample_names,node

def get_sample_names(df):
    sample_names = list(sorted(set(df["sample_id"])))
    return sample_names,len(sample_names)

def add_column_to_df(df,column,data):
    sample_names,sample_count = get_sample_names(df)
    for i in range(sample_count):
        try:
            df.loc[lambda df: df.sample_id == sample_names[i],column] = data[:,i]
        except Exception as e:
            print(type(e))
            df.loc[lambda df: df.sample_id == sample_names[i],column] = data[:]
    return df


if __name__=="__main__":
    main()
