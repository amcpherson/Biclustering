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
    df = pd.DataFrame.from_csv(path,sep='\t',index_col=None)
    sample_names,sample_count = get_sample_names(df)
    df = df.set_index(["event_id","sample_id"])
    panel = df.to_panel()

    #Filter out anything with a major copy number of zero
    panel = filter_panel(panel)
    return panel,sample_names

def filter_panel(panel):
    masks = []
    masks.append((panel["major"] == 0).any(axis=1))
    masks.append(np.logical_not(np.isfinite(panel["total_raw_e"]).all(axis=1)))

    bad_snv_mask = False
    for mask in masks:
        bad_snv_mask = np.logical_or(bad_snv_mask,mask)
    bad_snvs = panel[:,bad_snv_mask,:].axes[1]

    panel = panel.drop(bad_snvs,axis=1)
    return panel

def get_sample_names(df):
    sample_names = list(sorted(set(df["sample_id"])))
    return sample_names,len(sample_names)

def add_column_to_panel(pn,column,data):
    """
    sample_names,sample_count = get_sample_names(df)
    for i in range(sample_count):
        try:
            df.loc[lambda df: df.sample_id == sample_names[i],column] = data[:,i]
        except Exception as e:
            print(type(e))
            df.loc[lambda df: df.sample_id == sample_names[i],column] = data[:]
    df = df.set_index(["event_id","sample_id"])
    pn = df.to_panel()
    """
    columns = pn.axes[2]
    df = pd.DataFrame(columns=columns,index=pn.axes[1])
    if len(data.shape) == 2:
        df[:,:] = data
    elif len(data.shape) == 1:
        for col in columns:
            df[col] = data
    else:
        raise Exception("Invalid data")
    pn[column] = df

    return pn



if __name__=="__main__":
    main()
