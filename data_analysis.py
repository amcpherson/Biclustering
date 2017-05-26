#!/usr/bin/env python

from inference import build_model
from inference import plot_hard_clustering 
from data_generator import generate_data
import numpy as np
import pandas as pd

def main():
    path = "data.tsv"
    print("Parsing Data...")
    data,sample_names = parse_data(path)
    #data,_ = generate_data()
    print("Generating Model...")
    model,trace = build_model(data,2000)
    print(trace["logP"])
    print("Plotting data...")
    plot_hard_clustering(model,trace,data)
    print("Done!")

def parse_data(path):
    df = pd.DataFrame.from_csv(path,sep='\t')
    df = df[["sample_id","ccf"]]
    entries = len(df)
    sample_names = list(set(df["sample_id"]))
    sample_count = len(sample_names)

    epsilon = 1E-5

    data = np.ndarray((entries//sample_count,sample_count),dtype=np.float32)
    for i in range(sample_count):
        data[:,i] = df.loc[lambda df: df.sample_id == sample_names[i],"ccf"]

    data[np.where(data == 0.0)] = epsilon
    #data[np.where(data > 1)] = 1-epsilon
    data[np.where(np.logical_not((np.isfinite(data))))] = epsilon
    print(data)

    return data,sample_names
    

if __name__=="__main__":
    main()
