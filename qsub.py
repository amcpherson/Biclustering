#! /usr/bin/env python
import subprocess
PATIENT_IDS = [1,2,3,4,7,9,10,11,12,13,14,15,16,17]
#PATIENT_IDS = range(15)
DATA = "data/wgs/patient_{}.tsv"
#DATA = "data/dseq/patient_1.tsv"
TRACE = "traces/trace_{}.pkl"
LICHEE = "lichee_out/lichee_{}"
SCRIPT = "scripts/run_{}.sh"
CMD = "OMP_NUM_THREADS=4 THEANO_FLAGS='openmp=True,openmp_elemwise_minsize=100' python data_analysis.py {0} {1} {2}\n"
#CMD = "OMP_NUM_THREADS=2 THEANO_FLAGS='compute_test_value=ignore,floatX=float32,device=cuda0,lib.cnmem=1' python data_analysis.py {0} {1} {2}\n"
RUN_CMD = lambda x: ["qsub","-cwd","-q","shahlab.q","-pe","ncpus","2","-l","h_vmem=10G",x]

def main():
    ps = []
    for id in PATIENT_IDS:
        data = DATA.format(id)
        traces = TRACE.format(id)
        lichee = LICHEE.format(id)
        script = SCRIPT.format(id)
        cmd = CMD.format(data,traces,lichee)
        with open(script,"w") as f:
            #f.write("source activate ml\n")
            f.write(cmd)
        #run_cmd = RUN_CMD(script)
        #subprocess.check_call(run_cmd)
        ps.append(subprocess.Popen(["bash",script]))
    for p in ps:
        p.wait()


if __name__ == "__main__":
    main()
