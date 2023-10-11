from UI.get_args import run as get_args
from main.interval import run as int_run
from main.individual import run_all
from main.solvers import run_all as robust_run
from main.sampler import get_samples
import datetime
import sys

def main():
    num_repeats = 10
    sys.argv += ['--model','expander','--inst','1']
    state_times = {"Interval":[],"Individual":[],"MNE":[],"FSP":[],"subgradient":[]}
    sample_times = {"Interval":[],"Individual":[],"MNE":[],"FSP":[],"subgradient":[]}
    for num_s in range(100):
        state_times["Interval"].append([])
        state_times["Individual"].append([])
        state_times["MNE"].append([])
        state_times["FSP"].append([])
        state_times["subgradient"].append([])
        for repeats in range(num_repeats):
            sys.argv[-1] = str(num_s+2)
            args = get_args()
            samples = get_samples(args)
            int_time = int_run(args, samples)
            ind_time = run_all(args, samples)
            MNE_time, FSP_time, sub_time = robust_run(args, samples)
            state_times["Interval"][-1].append(int_time)
            state_times["Individual"][-1].append(ind_time)
            state_times["MNE"][-1].append(MNE_time)
            state_times["FSP"][-1].append(FSP_time)
            state_times["subgradient"][-1].append(sub_time)
    sys.argv[-1] = 3
    sys.argv += ['-N',100]
    for num_samples in range(100,10000,100):
        sample_times["Interval"].append([])
        sample_times["Individual"].append([])
        sample_times["MNE"].append([])
        sample_times["FSP"].append([])
        sample_times["subgradient"].append([])
        for repeats in range(num_repeats):
            sys.argv[-1] = str(num_samples)
            args = get_args()
            samples = get_samples(args)
            int_time = int_run(args, samples)
            ind_time = run_all(args, samples)
            MNE_time, FSP_time, sub_time = robust_run(args, samples)
            sample_times["Interval"][-1].append(int_time)
            sample_times["Individual"][-1].append(ind_time)
            sample_times["MNE"][-1].append(MNE_time)
            sample_times["FSP"][-1].append(FSP_time)
            sample_times["subgradient"][-1].append(sub_time)
    import pdb; pdb.set_trace()

if __name__=="__main__":
    main()

