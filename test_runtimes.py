from UI.get_args import run as get_args
from main.solver_base import solver
from main.MNE_algos import PNS_algo, FSP_algo
from main.solvers import *
import datetime
import sys
import pickle
import time

def main():
    num_repeats = 1
    
    min_samples = 100
    max_samples = 1000
    samples_step = 50

    min_states = 1  
    max_states = 50
    states_step = 3


    sys.argv += ['--model','expander','--inst','1']
    state_times = {"Interval":[],"Individual":[],"MNE":[],"FSP":[],"det":[],"subgradient":[]}
    sample_times = {"Interval":[],"Individual":[],"MNE":[],"FSP":[],"det":[],"subgradient":[]}
    
    args = get_args()


    sub = solver(subgrad, args)
    det_solver = solver(det, args)
    PNS_solver = solver(MNE, args, [PNS_algo])
    FSP_solver = solver(MNE, args, [FSP_algo])
    iMDP_solver = solver(interval, args)
    thom_discard_solver = solver(thom_discard, args)
    thom_relax_solver = solver(thom_relax, args)
    
    MNE_on = True
    FSP_on = True
    sub_on = True
    det_on = True
    #FSP = False 
    #sub = False
    #det = False


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
            model = args["model"]
            
            if sub_on:
                start = time.perf_counter()
                sub.solve(samples, model)
                sub_time = time.perf_counter()-start
           
            if det_on:
                start = time.perf_counter()
                det_solver.solve(samples, model)
                det_time = time.perf_counter()-start
            if MNE_on:    
                start = time.perf_counter()
                PNS_solver.solve(samples, model)
                MNE_time = time.perf_counter()-start
            if FSP_on:
                start = time.perf_counter()
                FSP_solver.solve(samples, model)
                FSP_time = time.perf_counter()-start
            
            start = time.perf_counter()
            iMDP_solver.solve(samples, model)
            int_time = time.perf_counter()-start
            
            start = time.perf_counter()
            thom_discard_solver.solve(samples, model)
            ind_time = time.perf_counter()-start
            
            state_times["Interval"][-1].append(int_time)
            
            state_times["Individual"][-1].append(ind_time)
            if MNE_on:
                state_times["MNE"][-1].append(MNE_time)
            if FSP_on:
                state_times["FSP"][-1].append(FSP_time)
            if det_on:
                state_times["det"][-1].append(det_time)
            if sub_on:
                state_times["subgradient"][-1].append(sub_time)
        if sum(state_times["MNE"][-1]) >= num_repeats*args["timeout"]:
            MNE_on = False
        if sum(state_times["FSP"][-1]) >= num_repeats*args["timeout"]:
            FSP_on = False
        if sum(state_times["det"][-1]) >= num_repeats*args["timeout"]:
            det_on = False
        if sum(state_times["subgradient"][-1]) >= num_repeats*args["timeout"]:
            sub_on = False
    MNE_on = True
    FSP_on = True
    sub_on = True
    det_on = True
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
            model = args["model"]
            if sub_on:
                start = time.perf_counter()
                sub.solve(samples, model)
                sub_time = time.perf_counter()-start
           
            if det_on:
                start = time.perf_counter()
                det_solver.solve(samples, model)
                det_time = time.perf_counter()-start
            if MNE_on:    
                start = time.perf_counter()
                PNS_solver.solve(samples, model)
                MNE_time = time.perf_counter()-start
            if FSP_on:
                start = time.perf_counter()
                FSP_solver.solve(samples, model)
                FSP_time = time.perf_counter()-start
            
            start = time.perf_counter()
            iMDP_solver.solve(samples, model)
            int_time = time.perf_counter()-start
            
            start = time.perf_counter()
            thom_discard_solver.solve(samples, model)
            ind_time = time.perf_counter()-start
            
            sample_times["Interval"][-1].append(int_time)
            sample_times["Individual"][-1].append(ind_time)
            if MNE_on:
                sample_times["MNE"][-1].append(MNE_time)
            if FSP_on:
                sample_times["FSP"][-1].append(FSP_time)
            if det_on:
                sample_times["det"][-1].append(det_time)
            if sub_on:
                sample_times["subgradient"][-1].append(sub_time)
        if sum(state_times["MNE"][-1])  >= num_repeats*args["timeout"]:
            MNE_on = False
        if sum(state_times["FSP"][-1])  >= num_repeats*args["timeout"]:
            FSP_on = False
        if sum(state_times["det"][-1])  >= num_repeats*args["timeout"]:
            det_on = False
        if sum(state_times["subgradient"][-1])  >= num_repeats*args["timeout"]:
            sub_on = False
    with open('runtime_res.pkl','wb') as f:
        pickle.dump([state_times, sample_times], f)

if __name__=="__main__":
    main()

