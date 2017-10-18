import subprocess
import pandas as pd
import numpy as np
import os.path
import subprocess
from inference import get_array
#path to LICHeE
LICHEE="/Users/chuebner/development/python/biclustering/lichee/LICHeE/release/lichee.jar"
ABSENT = 0.01
PRESENT = 0.05

def run_lichee(ssnv_input,cluster_input):
    subprocess.check_call([
        "java", "-jar", LICHEE,
        "-build",
        "-i", ssnv_input,
        "-clustersFile", cluster_input,
        "-n", "0",
        "--absent", str(ABSENT),
        "--present", str(PRESENT),
        "--tree", "1"])

def build_lichee_inputs(dest,df,location_indicies,cluster_indicies,axis_cluster_locations):
    df = df.reset_index(["event_id","sample_id"])
    df["vaf"].loc[np.isnan(df["vaf"])] = 0
    df = df[["event_id","sample_id","chrom","coord","vaf"]]
    df = df.set_index(["event_id","sample_id"]).unstack("sample_id")
    #Note make this op more efficient later
    #print(df)
    df = df.T.drop_duplicates().T.reset_index("event_id")
    vaf = pd.DataFrame(build_location_matrix(location_indicies,cluster_indicies,axis_cluster_locations))
    vaf.columns = df["vaf"].columns
    chrom = df["chrom"]
    coord = df["coord"]
    event_id = df["event_id"]
    vaf.insert(0,"#chr",chrom)
    vaf.insert(1,"position",coord)
    vaf.insert(2,"description",event_id)
    vaf.insert(3,"normal",0)
    vaf.to_csv("ssnv_input.tsv", sep="\t",index=False)

    #list of clusters with > 0 datapoints
    active_clusters = np.unique(location_indicies)
    location_matrix = build_location_matrix(active_clusters,cluster_indicies,axis_cluster_locations)
    profile_table = build_profile_table(location_matrix)
    snv_index_string_table = build_snv_index_string_table(active_clusters,location_indicies)

    #list of clusters with > 0 datapoints
    active_clusters = np.unique(location_indicies)

    location_matrix = build_location_matrix(active_clusters,cluster_indicies,axis_cluster_locations)
    profile_table = build_profile_table(location_matrix)

    df = build_snv_clustering(profile_table, location_matrix, snv_index_string_table)

    df.to_csv("cluster_input.tsv",sep="\t",index=False,header=False)
    return os.path.join(dest,"ssnv_input.tsv"),os.path.join(dest,"cluster_input.tsv")

def build_snv_clustering(profile_table, location_matrix, snv_index_string_table):
    df = pd.DataFrame(location_matrix)
    df.insert(0,"profile",pd.Series(profile_table))
    df.insert(1,"Normal",0)
    df.insert(len(df.columns),"snvs",pd.Series(snv_index_string_table))
    return df
    

def build_snv_index_string_table(active_clusters,location_indicies,):
    strings = []
    for cluster in active_clusters:
        array = np.nonzero(location_indicies == cluster)[0]+1#+1 for one based indexing
        f = np.vectorize(str)
        string = ",".join(f(array))
        strings.append(string)
    string_table = np.array(strings)
    return string_table

        

def build_location_matrix(indices,cluster_indicies,axis_cluster_locations):
    """build matrix representing coordinates of each cluster"""
    relevant_cluster_indicies = cluster_indicies[indices]
    location_matrix = np.zeros_like(relevant_cluster_indicies,dtype=np.float32)
    for i in range(relevant_cluster_indicies.shape[1]):
        location_matrix[:,i] = axis_cluster_locations[i,:][relevant_cluster_indicies[:,i]]

    return location_matrix


def build_profile_table(location_matrix):
    """build table of profile strings from per cluster locations"""
    clusters,samples = location_matrix.shape
    profile_matrix = location_matrix > 0.1
    #TODO:Vectorize operation
    table_list = []
    for i in range(clusters):
        #init string to 0 for normal
        string = ["0"]
        for j in range(samples):
            if profile_matrix[i,j]:
                string.append("1")
            else:
                string.append("0")
        table_list.append("".join(string))
    table = np.array(table_list)
    return table

def build_lichee_inputs_test():
    pass




