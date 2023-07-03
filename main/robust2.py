from multiprocessing import Pool
import sys
import numpy as np
import cvxpy as cp
import Markov.writer as writer
import Markov.models
from PAC.funcs import *
import itertools
from functools import partial
import logging
from tqdm import tqdm
import time
import copy
import sparse
import pickle
from scipy.spatial import ConvexHull
import os

NUM_POOLS = os.cpu_count()

def load_samples(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data[0], data[1]

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def get_samples(model, N):
    samples = []
    num_states = len(model.States)
    num_acts = len(model.Actions)
    for j in range(N):
        samples.append(model.sample_MDP().Transition_probs) 
    if N == 2:
        samples[0][1][0] = [0.01, 0.99]
        samples[0][2][0] = [0.4, 0.6]
        samples[0][2][1] = [0.9, 0.1]
        samples[1][1][0] = [1.0, 0.0]
        samples[1][2][0] = [0.8, 0.2]
        samples[1][2][1] = [0.3, 0.7]
    return samples

def calc_reach_sets(model):
    backward_reach = [[] for s in model.States]
    for state in model.States:
        successors = set()
        for elem in model.trans_ids[state]:
            successors.update(elem)
        for succ in successors:
            backward_reach[succ].append(state)
    return backward_reach

def test_pol(model, samples, pol=None):
    tic = time.perf_counter_ns()
    #test_MDP = Markov.models.MDP(model.sample_MDP()) # this is slow for drone - find a better way
    num_states = len(model.States)
    num_acts = len(model.Actions)
    j = 0
    true_probs = []
    pols = []
    if model.opt == "max":
        wc = 1
    else:
        wc = 0
    toc = time.perf_counter_ns()
    for ind, sample in enumerate(samples):
        #test_MDP.Transition_probs = sample
        test_MDP = model.fix_params(sample)
        if pol is not None:
            test_model = test_MDP.fix_pol(pol)
        else:
            test_model = test_MDP
        IO = writer.stormpy_io(test_model)
        IO.write()
        res, all_res, sol_pol = IO.solve()
        pols.append(sol_pol)
        if model.opt == "max":
            if wc>res[0]:
                wc = res[0]
        else:
            if wc<res[0]:
                wc = res[0]
        true_probs.append(all_res[0])
    tac = time.perf_counter_ns()
    time_for_mdp_sample = ((toc-tic))
    avg_time_for_solving = ((tac-toc))
    #print("sampling {:.3f}ns, solving {:.3f}ns".format(time_for_mdp_sample, avg_time_for_solving))
    true_probs = np.array(true_probs)
    return wc, true_probs, pols

def solve_subgrad(samples, model, max_iters=500):
    print("--------------------\nStarting subgradient descent")
    grad_dec = True
    avging = False

    num_states = len(model.States)
    num_acts = len(model.Actions)
    pol = np.zeros((num_states, num_acts))
    for s in model.States:
        for a in model.Enabled_actions[s]:
            pol[s,a] = 1/len(model.Enabled_actions[s])

    projected = cp.Variable(num_acts)
    point = cp.Parameter(num_acts)

    obj = cp.Minimize(cp.norm(projected-point))
    wc_hist = []
    for i in tqdm(range(max_iters)):
        wc, true_probs, _ = test_pol(model, samples, pol)
        step = 1/(i+1)
        #step = 0.1
        step = 0.1
        worst = np.argwhere(true_probs[:,model.Init_state]==wc)
        worst = np.random.choice(worst.flatten())
        #if worst.size == 1:
        #    worst = worst[0][0]
        #else:
        #    #worst = np.random.choice(
        #    import pdb; pdb.set_trace()
        if avging:
            # We expect this could fail 
            # e.g. if the true optimal is not a combination of individual optimums
            # use test with test_val = 0.52 to see! 

            best_worst_pol = test_pol(model, [samples[worst]])[2][0]
            new_pol = ((pol*(i))+best_worst_pol)/(i+1)
            pol = new_pol
        elif grad_dec:
            grad = np.zeros_like(pol)
            for s in model.States:
                if len(model.Enabled_actions[s]) > 1:
                    for a in model.Enabled_actions[s]:
                        grad_finder = np.copy(pol)
                        grad_finder[s] = 0
                        grad_finder[s,a] = 1
                        grad[s,a] = test_pol(model, [samples[worst]], grad_finder)[0]
            pol += step*grad
        
        for s in model.States:
            if len(model.Enabled_actions[s]) <= 1:
                pass
            elif len(model.Enabled_actions[s]) == 2:
                act_0 = model.Enabled_actions[s][0]
                act_1 = model.Enabled_actions[s][1]
                diff = pol[s,act_0]-pol[s,act_1]
                if diff > 1:
                    pol[s,act_0] = 1
                    pol[s,act_1] = 0
                elif diff < -1:
                    pol[s, act_0] = 0
                    pol[s, act_1] = 1
                else:
                    pol[s,act_0] = (1+diff)/2
                    pol[s,act_1] = (1-diff)/2
            else:
                cons = [cp.norm(projected,1) <= 1, projected >= 0]
                for a in range(num_acts):
                    if a not in model.Enabled_actions[s]:
                        cons += [projected[a] == 0]
                point.value = pol[s]
                prob = cp.Problem(obj, cons)
                res = prob.solve()
                pol[s] = projected.value
        #print(worst)
        #print(pol)
        #print(wc)
        wc_hist.append(wc)
    return wc, pol
    #import matplotlib.pyplot as plt
    #plt.plot(wc_hist)
    #plt.show()
    #import pdb; pdb.set_trace()
        #grad = 

def run_all(args):
    print("Running code for robust optimal policy \n --------------------")
    model = args["model"]
    #test_support_num(args)
    #a_priori_max_supports = sum([len(acts) for acts in model.Enabled_actions])
    a_priori_max_supports = model.max_supports
    #calc_max_path(model)
    a_priori_eps = calc_eps(args["beta"], args["num_samples"], a_priori_max_supports)
    
    print("A priori upper bound on number of support constraints is " + str(a_priori_max_supports))

    print("A priori bound on violation probability is {:.3f} with confidence {:.3f}".format(a_priori_eps, args["beta"]))

    if args["sample_load_file"] is not None:
        base, samples = load_samples(args["sample_load_file"])
    else:
        samples = get_samples(model, args["num_samples"])
    if args["sample_save_file"] is not None:
        save_data(args["sample_save_file"], (base, samples))
    
    if args["prob_load_file"] is not None:
        warm_probs = load_data(args["prob_load_file"])
    else:
        warm_probs = None
    num_states = len(model.States)
   
    solve_subgrad(samples, model)
    return 0
    if args["result_save_file"] is not None:
        save_data(args["result_save_file"], {"worst": wc, "probs":all_p, "pol":pol, "supports":supports})

    [a_post_eps_L, a_post_eps_U] = \
        calc_eps_risk_complexity(args["beta"], N, a_post_support_num)
    
    print("A posteriori, found " + str(a_post_support_num) + " support constraints")

    print("A posteriori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
          .format(a_post_eps_L, a_post_eps_U, args["beta"]))

    print("Optimal satisfaction probability is found to be {:.3f}".format(wc))

    if pol.size < 50:
        print("Calculated robust policy is:")
        print(pol)

    thresh = a_priori_eps

    if args["MC"]:
        emp_violation = MC_sampler(model, args["MC_samples"], opt_prob, thresh, pol) 
        print("Empirical violation rate is found to be {:.3f}".format(emp_violation))
    print("\n\n")

def test_support_num(args):
    model = args["model"]
    num_states = len(model.States)
    num_acts = len(model.Actions)
    for i in range(10):
        base, samples = gen_samples(model, args["num_samples"], args["batch_size"])

        probs, pol, supports, a_post_support_num, all_p = calc_probs_policy_iteration(model, base, samples)
         
        print("Calculated supports: " + str(supports))
        actual_supports = []
        #for j in tqdm(range(args["num_samples"])):
        for j in range(args["num_samples"]):
            samples_new = remove_sample(j, samples, num_states, num_acts)
            test_probs, test_pol, _, _, all_p_test = calc_probs_policy_iteration(model, base, samples_new)
            if model.opt == "max":
                gap = abs(min(probs) - min(test_probs))
            else:
                gap = abs(max(probs) - max(test_probs))
            if gap >= 1e-4:
                actual_supports.append(j)
            #    if j not in supports:
            #        import pdb; pdb.set_trace()
            #elif j in supports:
            #    import pdb; pdb.set_trace()
        print("Emprical supports: " + str(actual_supports))
        if len(actual_supports) > model.max_supports:
            print("ERROR, a priori max sc was wrong!!!")
            import pdb; pdb.set_trace()
        for sc in actual_supports:
            if sc not in supports:
                print("ERROR, found an empirical SC not in a posteriori support set")
                import pdb; pdb.set_trace()

def remove_sample(pos, samples, num_states, num_acts):
    max_sample = 0
    new_samples = copy.copy(samples)
    for batch_id, batch in enumerate(samples):
        batch_size = int(batch.shape[0]/num_states)
        max_sample += batch_size
        if max_sample > pos:
            ind_samples = [batch[i*num_states:(i+1)*num_states, 
                                   i*num_acts:(i+1)*num_acts,
                                   i*num_states:(i+1)*num_states] for i in range(batch_size)]
            batch_pos = pos - (max_sample - batch_size)
            ind_samples.pop(batch_pos)
            data = []
            coords = [[] for i in range(3)]
            for j, ind in enumerate(ind_samples):
                data += ind.data.tolist()
                ind_coords = ind.coords.tolist()
                ind_coords = [[elem+j*num_states for elem in ind_coords[0]],
                              [elem+j*num_acts for elem in ind_coords[1]],
                              [elem+j*num_states for elem in ind_coords[2]]]
                coords[0] += ind_coords[0]
                coords[1] += ind_coords[1]
                coords[2] += ind_coords[2]
            batch_mat = sparse.COO(coords, data, shape=((batch_size-1)*num_states,
                                                         (batch_size-1)*num_acts,
                                                         (batch_size-1)*num_states))
            new_samples[batch_id] = batch_mat
            return new_samples
