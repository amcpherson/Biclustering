#! /usr/bin/env python
import subprocess
import argparse
import os
from pandas import read_csv

CMD = "OMP_NUM_THREADS=4 THEANO_FLAGS='openmp=True,openmp_elemwise_minsize=100' python data_analysis.py {0} {1} {2} {3}\n"

def parse_args():
    parser = argparse.ArgumentParser(description="Cluster a set of snvs and merge their components together.")
    parser.add_argument("patient_file", help="path to patients file")
    parser.add_argument("--run", help="run all generated scripts", action="store_true" )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    df = read_csv(args.patient_file)

    ps = []
    for paths in df.to_dict('records'):
        data = paths["data"]
        trace = paths["trace"]
        lichee = paths["lichee"]
        config = paths["config"]
        script = paths["script"]
        cmd = CMD.format(data,trace,lichee,config)

        head,_ = os.path.split(script)

        try:
            os.makedirs(head)
        except:
            pass

        with open(script,"w") as f:
            f.write(cmd)

        if args.run is True:
            ps.append(subprocess.Popen(["bash",script]))

    for p in ps:
        p.wait()


if __name__ == "__main__":
    main()
