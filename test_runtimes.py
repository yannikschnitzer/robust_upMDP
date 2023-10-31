from UI.get_args import run as get_args
from main.interval import run as int_run
from main.individual import run_all
from main.solvers import run_all as robust_run
from main.sampler import get_samples
import datetime
import sys
import pickle

def main():
    num_repeats = 5
    
    min_samples = 100
    max_samples = 1000
    samples_step = 500

    min_states = 3
    max_states = 50
    states_step = 3


    sys.argv += ['--model','expander','--inst','1']
    state_times = {"Interval":[],"Individual":[],"MNE":[],"FSP":[],"det":[],"subgradient":[]}
    sample_times = {"Interval":[],"Individual":[],"MNE":[],"FSP":[],"det":[],"subgradient":[]}
    
    MNE = True
    FSP = False 
    sub = False
    det = False

    #FSP = True
    #sub = True
    #det = True

    for num_s in range(min_states, max_states, states_step):
        state_times["Interval"].append([])
        state_times["Individual"].append([])
        state_times["MNE"].append([])
        state_times["FSP"].append([])
        state_times["det"].append([])
        state_times["subgradient"].append([])
        for repeats in range(num_repeats):
            sys.argv[-1] = str(num_s+2)
            args = get_args()
            samples = get_samples(args)
            int_time = int_run(args, samples)
            ind_time = run_all(args, samples)
            MNE_time, FSP_time, sub_time, det_time = robust_run(args, samples, MNE, FSP, sub, det)
            state_times["Interval"][-1].append(int_time)
            state_times["Individual"][-1].append(ind_time)
            if MNE:
                state_times["MNE"][-1].append(MNE_time)
            if FSP:
                state_times["FSP"][-1].append(FSP_time)
            if det:
                state_times["det"][-1].append(det_time)
            if sub:
                state_times["subgradient"][-1].append(sub_time)
        if sum(state_times["MNE"][-1] == -num_repeats):
            MNE = False
        if sum(state_times["FSP"][-1] == -num_repeats):
            FSP = False
        if sum(state_times["det"][-1] == -num_repeats):
            det = False
        if sum(state_times["subgradient"][-1] == -num_repeats):
            sub = False
    MNE = True
    FSP = False 
    sub = False
    det = False
    #MNE = True
    #FSP = True
    #sub = True
    #det = True
    sys.argv[-1] = 3
    sys.argv += ['-N',100]
    for num_samples in range(min_samples,max_samples,samples_step):
        sample_times["Interval"].append([])
        sample_times["Individual"].append([])
        sample_times["MNE"].append([])
        sample_times["FSP"].append([])
        sample_times["det"].append([])
        sample_times["subgradient"].append([])
        for repeats in range(num_repeats):
            sys.argv[-1] = str(num_samples)
            args = get_args()
            samples = get_samples(args)
            int_time = int_run(args, samples)
            ind_time = run_all(args, samples)
            MNE_time, FSP_time, sub_time, det_time = robust_run(args, samples, MNE, FSP, sub, det)
            sample_times["Interval"][-1].append(int_time)
            sample_times["Individual"][-1].append(ind_time)
            if MNE:
                sample_times["MNE"][-1].append(MNE_time)
            if FSP:
                sample_times["FSP"][-1].append(FSP_time)
            if det:
                sample_times["det"][-1].append(det_time)
            if sub:
                sample_times["subgradient"][-1].append(sub_time)
        if sum(state_times["MNE"][-1] == -num_repeats):
            MNE = False
        if sum(state_times["FSP"][-1] == -num_repeats):
            FSP = False
        if sum(state_times["det"][-1] == -num_repeats):
            det = False
        if sum(state_times["subgradient"][-1] == -num_repeats):
            sub = False
    with open('runtime_res.pkl','wb') as f:
        pickle.dump([state_times, sample_times], f)

if __name__=="__main__":
    main()

