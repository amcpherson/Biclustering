#!/usr/bin/env python

import inference as inf
import run_lichee as rl
from data_generator import generate_data
import numpy as np
import pandas as pd
import os.path
import sys

BURNIN = 4000
SUBSAMPLE = 50
THRESH = 5
def main():
    path,trace_path,lichee = sys.argv[1:4]
    
    panel,sample_names = parse_data(path)
    #print((panel["major"] == 0).to_string())
    panel["major"] = panel["major"].astype(np.int64)
    annealing_schedule = [
        (10000,2000,1,None),
        (1500,500,1.02,None),
        (1500,500,1.05,None),
        (1500,500,1.1,None),
        (1500,500,1.2,None),
        (1500,500,1.3,None),
        (1500,500,10,None),
        (1500,500,100,None),
        (1500,500,1000,None)
        ]
    trace = None
    for tune,sample,t_b,start in annealing_schedule:
        _,trace = inf.build_model(panel, tune, sample, trace_path, prev_trace=trace, thermodynamic_beta=t_b,start=start)
    try:
        os.makedirs(lichee)
    except:
        pass

    """
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
    """
    print("Done!")

def load_prev(path,panel):
    _,trace = inf.build_model(panel, 1, 1, path)
    print(trace[2000])
    return trace[2000]

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
