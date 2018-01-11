#!/usr/bin/env python

import inference as inf
import run_lichee as rl
from data_generator import generate_data
import numpy as np
import pandas as pd
import os.path
import sys
import json

def load_config(config):
    with open(config,"r") as conf:
        config = json.load(conf)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="cluster parameters")
    args = parser.parse_args()

def main():
    path,trace_path,lichee,config_path = sys.argv[1:5]

    config = load_config(config_path)
    
    panel,sample_names = parse_data(path)
    panel["major"] = panel["major"].astype(np.int64)
    annealing_schedule = config["annealing_schedule"]
    trace = None
    for tune,sample,t_b in annealing_schedule:
        _,trace = inf.build_model(panel, tune, sample, trace_path,
            prev_trace=trace, thermodynamic_beta=t_b, sampler=config["sampler"],
            marg_tcn=config["marg_tcn"])

    if config["lichee"] is False:
        return
        
    try:
        os.makedirs(lichee)
    except:
        pass

    sub_trace = trace[BURNIN::SUBSAMPLE]
    location_indicies_list = list(sub_trace["location_indicies"])
    cluster_indicies_list = list(sub_trace["cluster_indicies"])
    axis_cluster_locations_list = list(sub_trace["axis_cluster_locations"])
    l = list(zip(location_indicies_list,cluster_indicies_list,axis_cluster_locations_list))
    data = os.path.join(lichee,"c_count.txt")
    with open(data,"w") as f:
        for i in range(len(l)):
            index = i*SUBSAMPLE+BURNIN
            location_indicies,cluster_indicies,axis_cluster_locations = l[i]
            pre_active_clusters,counts = np.unique(location_indicies,return_counts=True)
            active_clusters = pre_active_clusters[counts > THRESH]
            max_n = len(active_clusters)
            ssnv_input,cluster_input = rl.build_lichee_inputs(
                lichee,panel,location_indicies,cluster_indicies,axis_cluster_locations,treshold=THRESH)
            output = os.path.join(lichee,"output_{}".format(index))
            tree = rl.run_lichee(ssnv_input,cluster_input,output)
            count = check_lichee_output(tree)
            f.write("{}:{}:{}\n".format(index,count,max_n))

def check_lichee_output(tree):
    with open(tree,"r") as f:
        f.readline()
        count = 0
        while(True):
            line =  f.readline()
            if line is "\n":
                break
            count += 1
        return count

def parse_data(path):
    """Reads a tsv file and converts it to clusterable data for
    the model.
    
    Args:
        path: the path to the data
    Returns:
        (data,sample_names,node):The post processed data along with
        the names of each sample and the origin node of each data point
    """
    df = pd.read_csv(path,sep='\t',index_col=None)
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
    if len(data.shape) == 2:
        pn[column] = data
    elif len(data.shape) == 1:
        columns = pn.axes[2]
        df = pd.DataFrame(columns=columns,index=pn.axes[1])
        for col in columns:
            df[col] = data
        pn[column] = df
    else:
        raise Exception("Invalid data")

    return pn



if __name__=="__main__":
    main()
