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

def build_sample_mat(num_states, num_acts, enabled_acts, trans_ids, samples):
    return sample_mat

def get_batch(model, N):
    coords = [[] for i in range(3)]
    data = []
    num_states = len(model.States)
    num_acts = len(model.Actions)
    for j in range(N):
        elem = model.sample_MDP().Transition_probs 
        for s in range(num_states):
            for a_num, a in enumerate(model.Enabled_actions[s]):
                for s_prime_num, s_prime in enumerate(model.trans_ids[s][a]):
                    coords[0].append(j*num_states+s)
                    coords[1].append(j*num_acts+a)
                    coords[2].append(j*num_states+s_prime)
                    data.append(elem[s][a_num][s_prime_num])
    sample_mat = sparse.COO(coords, data, \
                            shape=(N*num_states, N*num_acts, N*num_states))
    return sample_mat

def gen_samples(model, N, batch_size):
    num_states = len(model.States)
    num_acts = len(model.Actions)
    num_batches = int(np.ceil(N/batch_size))
    samples = []
    sizes = [batch_size for i in range(num_batches)]
    sizes[-1] -= sum(sizes) - N
    for i in tqdm(range(num_batches)):
        start = i*batch_size
        end = min((i+1)*batch_size, N)
        curr_size = end-start
        samples.append(get_batch(model, curr_size))
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

def calc_max_path(model):
    start = model.Init_state
    paths = [[start]]
    all_paths = [[start]]
    states_to_check = [start]
    max_supports = 0
    goals = model.Labelled_states[model.Labels.index("reached")]
    for i in range(len(model.States)):
        prev_paths = paths
        paths = []
        next_states_to_check = set()
        for state in states_to_check:
            successors = set()
            for elem in model.trans_ids[state]:
                successors.update(elem)
            next_paths = []
            for succ in successors:
                for path in prev_paths:
                    if succ not in path and path[-1] == state:
                        new_path = path+[succ]
                        next_paths.append(new_path)
                        if succ in goals:
                            path_supports = sum([len(model.Enabled_actions[s]) for s in new_path])
                            if path_supports > max_supports:
                                max_supports = path_supports
            if len(next_paths) > 0:
                all_paths.append(next_paths)
            paths += next_paths
            next_states_to_check.update(successors)
        if states_to_check == next_states_to_check:
            break
        states_to_check = next_states_to_check
    return len(backward_reach)

def solve_opt(s, N, num_states, num_acts, trans_mat, probs, tol):
    pi = cp.Variable(num_acts)
    new_prob = cp.Variable(N)
    worst_prob = cp.Variable(1)
    objective = cp.Maximize(worst_prob)
    
    constraints = [new_prob <= 1, new_prob >= probs[s], \
            np.ones(num_acts)@pi == 1, pi >= 0, worst_prob <= new_prob, \
            new_prob == pi@trans_mat[s,:,:]]
    
    program = cp.Problem(objective, constraints)
    result = program.solve(ignore_dpp=True)
    changed = np.any(abs(probs.value[s]-new_prob.value)>=tol)
    return changed, new_prob.value

def test_pol(model, samples, pol, probs, tol):
    tol *= 50
    test_MDP = Markov.models.MDP(model.sample_MDP())
    num_states = len(model.States)
    num_acts = len(model.Actions)
    j = 0
    for batch in samples:
        batch_size = int(batch.shape[0]/num_states)
        mat_batch = [batch[i*num_states:(i+1)*num_states,
                           i*num_acts:(i+1)*num_acts,
                           i*num_states:(i+1)*num_states] for i in range(batch_size)]
        for sample in mat_batch:
            for s in model.States:
                for a_num, a in enumerate(model.Enabled_actions[s]):
                    for s_prime_num, s_prime in enumerate(model.trans_ids[s][a]):
                        test_MDP.Transition_probs[s][a_num][s_prime_num] = sample[s,a,s_prime]
            test_MC = test_MDP.fix_pol(pol)
            IO = writer.stormpy_io(test_MC)
            IO.write()
            res, all_res = IO.solve()
            if abs(probs[model.Init_state, j]-res) > tol:
                import pdb; pdb.set_trace()
            j += 1

        

def test_probs(probs, samples, pol, tol):
    N = probs.shape[-1]
    num_states = pol.shape[0]
    num_acts = pol.shape[1]
    batch_size = int(samples[0].shape[-1]/num_states)
    num_batches = int(np.ceil(N/batch_size))
    trans_mat = np.zeros((num_states, num_acts, N))
    for j in range(num_batches):
        start = j*batch_size
        end = min((j+1)*batch_size, N)
        sample_batch = samples[j]
        #sample_batch = samples[start*num_states:end*num_states, 
        #                       start*num_acts:end*num_acts,
        #                       start*num_states:end*num_states]
        prob_batch = probs[:, start:end]
        res = sample_batch@prob_batch.T.reshape(sample_batch.shape[-1])
        new_shape = (end-start, num_states, num_acts)
        new_strides = (num_states*res.strides[0]+num_acts*res.strides[1], res.strides[0], res.strides[1])
        batch_trans_mat = np.lib.stride_tricks.as_strided(res, new_shape, new_strides)
        batch_trans_mat = np.swapaxes(batch_trans_mat.T, 0, 1)
        trans_mat[:,:,start:end] = batch_trans_mat
        trans_mat[:,:,start:end] = batch_trans_mat
    check = True
    for s in range(num_states):
        curr_check = np.linalg.norm(probs[s,:]-pol[s,:]@trans_mat[s,:,:], ord=np.inf) <= tol
        check = check and curr_check 
    return check

def calc_probs_policy_iteration(model, samples, max_iters=10000, tol=1e-3):
   
    back_set = calc_reach_sets(model)
    num_states = len(model.States)
    num_acts = len(model.Actions)
    batch_size = int(samples[0].shape[-1]/num_states)
    N = sum([int(sample.shape[-1]/num_states) for sample in samples])
    
    states_to_update = set()
    probs = cp.Parameter((num_states,N))
    if model.opt == "max":
        prob_init = np.zeros((num_states,N))
        for reached in model.Labelled_states[model.Labels.index("reached")]:
            prob_init[reached, :] = 1.0
            states_to_update.update(back_set[reached])
    else:
        prob_init = np.ones((num_states,N))
        if 'finished' in model.Labels:
            for deadend in model.Labelled_states[model.Labels.index("finished")]:
                if deadend not in model.Labelled_states[model.Labels.index("reached")]:
                    prob_init[deadend, :] = 0.0
                    states_to_update.update(back_set[deadend])
    probs.value = prob_init 

    trans_mat = np.empty((num_states, num_acts, N))

    pol = np.ones((num_states, num_acts))/num_acts
    pi = cp.Variable(num_acts)
    new_prob = cp.Variable(N)
    worst_prob = cp.Variable(1)
    if model.opt == "max":
        objective = cp.Maximize(worst_prob)
    else:
        objective = cp.Minimize(worst_prob)
    converged=False
    total_time = 0
    updates = [0 for s in model.States]
    for i in range(max_iters):
        tic = time.perf_counter()
        next_states_to_update = set()
        logging.info(("Beginning construction of Q matrix"))
        num_batches = int(np.ceil(N/batch_size))
        trans_mat = np.zeros((num_states, num_acts, N))
        for j in range(num_batches):
            start = j*batch_size
            end = min((j+1)*batch_size, N)
            sample_batch = samples[j]
            prob_batch = probs.value[:, start:end]
            res = sample_batch@prob_batch.T.reshape(sample_batch.shape[-1])
            new_shape = (end-start, num_states, num_acts)
            new_strides = (num_states*res.strides[0]+num_acts*res.strides[1], res.strides[0], res.strides[1])
            batch_trans_mat = np.lib.stride_tricks.as_strided(res, new_shape, new_strides)
            batch_trans_mat = np.swapaxes(batch_trans_mat.T, 0, 1)
            trans_mat[:,:,start:end] = batch_trans_mat
        mat_time = time.perf_counter() - tic

        logging.info(("Completed construction of Q matrix in {:.3f}s").format(mat_time))
        converged=True
        #for s in tqdm(states_to_update):
        for s in states_to_update:
            if s not in model.Labelled_states[model.Labels.index("reached")]:
                enabled = model.Enabled_actions[s]
                pi_mat = np.zeros(num_acts)
                pi_mat[enabled] = 1
                temp = 1/(updates[s]+1)
                if model.opt == "max":
                    constraints = [worst_prob <= new_prob, new_prob >= 0, \
                                   new_prob >= probs[s]-temp, new_prob <= 1]
                else:
                    constraints = [worst_prob >= new_prob, \
                                   new_prob <= probs[s]+temp, new_prob <= 1,new_prob >= 0 ]
                constraints += [pi_mat@pi == 1, pi >= 0, new_prob == pi@trans_mat[s,:,:]]
                
                program = cp.Problem(objective, constraints)
                result = program.solve(ignore_dpp=True)
                if program.status == cp.OPTIMAL:
                    changed = np.any(abs(probs.value[s]-new_prob.value)>=tol) 
                    probs.value[s] = np.maximum(0, np.minimum(new_prob.value,1))
                    pol[s] = np.maximum(np.minimum(pi.value, 1), 0)
                    if changed or not updates[s]:
                        next_states_to_update.update(back_set[s])
                        converged=False
                        updates[s] += 1
                else:
                    converged = False
                    next_states_to_update.add(s)
                    print("Found infeasible problem")
                    updates[s] -= 1
                    updates[s] = max(0, updates[s])
        states_to_update = next_states_to_update
        if model.opt == "max":
            worst = np.min(probs.value, axis=1)
        else:
            worst = np.max(probs.value, axis=1)
        logging.info("Current worst case probabilities are {}".format(worst))
        if converged:
            break
        toc = time.perf_counter()
        total_time += toc-tic
        logging.info("iteration {} completed in {:.3f}s".format(i, toc-tic))
    print("Entire optimization finished in {:.3f}s".format(total_time))
    test_pol(model, samples, pol, probs.value, tol)
    print("Policy verified with tolerance " + str(tol*50))
    
    # sometimes we find an additional support sample
    num_supports = 0
    support_samples = set()
    for s in range(num_states):
        max_sc = len(model.Enabled_actions[s])
        if model.opt == "max":
            found_sc = np.argwhere(probs.value[s] <= np.min(probs.value[s])+tol)
        else:
            found_sc = np.argwhere(probs.value[s] >= np.max(probs.value[s])-tol)
        if found_sc.size <= max_sc:
            support_samples.update([int(elem) for elem in found_sc])
    num_supports = len(support_samples)
    if converged:
        return probs.value[model.Init_state], pol, support_samples, num_supports
    else:
        return -1, -1, -1, -1

def discard(lambda_val, probs):
    min_prob = 1
    discarded=0
    undiscarded = [p_val for p_val in probs if p_val > lambda_val]

    if len(undiscarded)>0:
        min_prob = min(undiscarded)
    else:
        min_prob = lambda_val
    discarded = len(probs)-len(undiscarded)

    return min_prob, discarded

def with_relaxation(rho, probs):
    N = len(probs)
    x_s = cp.Variable(1+N)
    c = -np.ones((N+1,1))*rho
    c[0] = 1

    b = np.array([0]+probs)
    A = np.eye(N+1)
    A[:, 0] = -1
    A[0,0] = 1
    A = -A

    objective = cp.Maximize(c.T@x_s)
    constraints = [A@x_s <= b, x_s >= 0]
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    etas = x_s.value[1:]
    tau = x_s.value[0]
    
    return tau, etas

def run_all(args):
    print("Running code for robust optimal policy \n --------------------")
    model = args["model"]
    
    a_priori_max_supports = sum([len(acts) for acts in model.Enabled_actions])
    #calc_max_path(model)
    a_priori_eps = calc_eps(args["beta"], args["num_samples"], a_priori_max_supports)
    
    print("A priori upper bound on number of support constraints is " + str(a_priori_max_supports))

    print("A priori bound on violation probability is {:.3f} with confidence {:.3f}".format(a_priori_eps, args["beta"]))

    samples = gen_samples(model, args["num_samples"], args["batch_size"])

    probs, pol, supports, a_post_support_num  = calc_probs_policy_iteration(model, samples, tol=args["tol"])
    
    [a_post_eps_L, a_post_eps_U] = \
        calc_eps_risk_complexity(args["beta"], args["num_samples"], a_post_support_num)
    
    print("A posteriori, found " + str(a_post_support_num) + " support constraints")

    print("A posteriori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
          .format(a_post_eps_L, a_post_eps_U, args["beta"]))

    if model.opt == "max":
        opt_prob = min(probs)
    else:
        opt_prob = max(probs)

    print("Optimal satisfaction probability is found to be {:.3f}".format(opt_prob))

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
    for i in range(100):
        samples = gen_samples(model, args["num_samples"], args["batch_size"])

        probs, pol, supports, a_post_support_num  = calc_probs_policy_iteration(model, samples)
        
        print("Calculated supports: " + str(supports))
        actual_supports = []
        for i, sample in enumerate(tqdm(samples)):
                samples.pop(i)
                test_probs, test_pol, _, _ = calc_probs_policy_iteration(model, samples)
                if abs(min(probs) - min(test_probs)) >= 0.01:
                    actual_supports.append(i)
                samples.insert(i, sample)
        print("Emprical supports: " + str(actual_supports))
        if len(actual_supports) > a_post_support_num:
            print("ERROR")
            import pdb; pdb.set_trace()
