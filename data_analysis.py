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
    sample_names,sample_count = get_sample_names(df)
    df = df.reset_index("event_id").set_index(["event_id","sample_id"])
    panel = df.to_panel()
    return panel,sample_names

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
