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

def load_samples(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data[0], data[1]

def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

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
                    if model.paramed[s][a_num][s_prime_num]:
                        coords[0].append(j*num_states+s)
                        coords[1].append(j*num_acts+a)
                        coords[2].append(j*num_states+s_prime)
                        data.append(elem[s][a_num][s_prime_num])
    sample_mat = sparse.COO(coords, data, \
                            shape=(N*num_states, N*num_acts, N*num_states))
    return sample_mat

def get_base(model):
    num_states = len(model.States)
    num_acts = len(model.Actions)
    base_coords = [[] for i in range(3)]
    base_data = []
    elem = model.sample_MDP().Transition_probs 
    for s in range(num_states):
        for a_num, a in enumerate(model.Enabled_actions[s]):
            for s_prime_num, s_prime in enumerate(model.trans_ids[s][a]):
                if not model.paramed[s][a_num][s_prime_num]:
                    base_coords[0].append(s)
                    base_coords[1].append(a)
                    base_coords[2].append(s_prime)
                    base_data.append(elem[s][a_num][s_prime_num])
    base_mat = sparse.COO(base_coords, base_data, shape=(num_states, num_acts, num_states)) 
    return base_mat

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
    return get_base(model), samples

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

def test_pol(model, base, samples, pol, probs, tol):
    tol *= 50
    test_MDP = Markov.models.MDP(model.sample_MDP())
    num_states = len(model.States)
    num_acts = len(model.Actions)
    j = 0
    for batch in samples:
        batch_size = int(batch.shape[0]/num_states)
        mat_batch = [batch[i*num_states:(i+1)*num_states,
                           i*num_acts:(i+1)*num_acts,
                           i*num_states:(i+1)*num_states]+base for i in range(batch_size)]
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

def calc_probs_policy_iteration(model, base, samples, max_iters=10000, tol=1e-3):
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
        for state in model.States:
            deadend = True
            for trans in model.trans_ids[state]:
                if len(trans) > 1 or state not in trans:
                    deadend = False
            if deadend and state not in model.Labelled_states[model.Labels.index("reached")]:
                prob_init[state, :] = 0.0
                states_to_update.update(back_set[state])
                            
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
    for i in range(max_iters):
        tic = time.perf_counter()
        next_states_to_update = set()
        logging.info(("Beginning construction of Q matrix"))
        num_batches = int(np.ceil(N/batch_size))
        trans_mat = np.zeros((num_states, num_acts, N))
        for j in range(num_batches):
            start = j*batch_size
            end = min((j+1)*batch_size, N)
            
            curr_size = end-start

            sample_batch = samples[j]
            indexer = np.zeros((curr_size, curr_size, curr_size))
            indexer[np.diag_indices(curr_size,ndim=3)] = 1
            base_mat = np.kron(indexer, base)
            sample_mat = sample_batch + base_mat
            prob_batch = probs.value[:, start:end]
            res = sample_mat@prob_batch.T.reshape(sample_mat.shape[-1])
            new_shape = (end-start, num_states, num_acts)
            new_strides = (num_states*res.strides[0]+num_acts*res.strides[1], res.strides[0], res.strides[1])
            batch_trans_mat = np.lib.stride_tricks.as_strided(res, new_shape, new_strides)
            batch_trans_mat = np.swapaxes(batch_trans_mat.T, 0, 1)
            trans_mat[:,:,start:end] = batch_trans_mat
        mat_time = time.perf_counter() - tic

        logging.info(("Completed construction of Q matrix in {:.3f}s").format(mat_time))
        logging.info("Optimizing over {} states".format(len(states_to_update)))
        converged=True
        #for s in tqdm(states_to_update):
        for s in states_to_update:
            if s not in model.Labelled_states[model.Labels.index("reached")]:
                if len(model.Enabled_actions[s]) == 1:
                    pol[s] = np.zeros(num_acts)
                    pol[s, model.Enabled_actions[s][0]] = 1
                    new_p = trans_mat[s, model.Enabled_actions[s][0], :]
                    changed = np.any(abs(probs.value[s]-new_p)>=tol) 
                    probs.value[s] = new_p
                else:
                    enabled = model.Enabled_actions[s]
                    pi_mat = np.zeros(num_acts)
                    pi_mat[enabled] = 1
                    if model.opt == "max":
                        constraints = [worst_prob <= new_prob, new_prob >= probs[s], 
                                       new_prob >= 0, \
                                        new_prob <= 1]
                    else:
                        constraints = [worst_prob >= new_prob, \
                                       new_prob <= probs[s], \
                                       new_prob <= 1,new_prob >= 0 ]
                    constraints += [pi_mat@pi == 1, pi >= 0, new_prob == pi@trans_mat[s,:,:]]
                    
                    program = cp.Problem(objective, constraints)
                    try:
                        result = program.solve(ignore_dpp=True, solver=cp.CLARABEL)
                        opt_failed = program.status != cp.OPTIMAL
                    except:
                        opt_failed = True
                    if not opt_failed:
                        changed = np.any(abs(probs.value[s]-new_prob.value)>=tol) 
                        probs.value[s] = np.maximum(0, np.minimum(new_prob.value,1))
                        pol[s] = np.maximum(np.minimum(pi.value, 1), 0)
                    else:
                        constraints.pop(1)
                        logging.info("Found infeasible problem, resolving")
                        program = cp.Problem(objective, constraints)
                        result = program.solve(ignore_dpp=True, solver=cp.CLARABEL)
                        if program.status != cp.OPTIMAL and program.status != cp.OPTIMAL_INACCURATE:
                            import pdb; pdb.set_trace()
                        else:
                            changed = np.any(abs(probs.value[s]-new_prob.value)>=tol) 
                            probs.value[s] = np.maximum(0, np.minimum(new_prob.value,1))
                            pol[s] = np.maximum(np.minimum(pi.value, 1), 0)
                        #changed = False
                        #converged = False
                        #next_states_to_update.add(s)
                if changed:
                    next_states_to_update.update(back_set[s])
                    converged=False
        states_to_update = next_states_to_update
        if model.opt == "max":
            worst = np.min(probs.value, axis=1)
        else:
            worst = np.max(probs.value, axis=1)
        logging.info("Current worst case probabilities are {}".format(worst))
        logging.info("Current worst case init probability is {}".format(worst[model.Init_state]))
        if converged:
            break
        toc = time.perf_counter()
        total_time += toc-tic
        logging.info("iteration {} completed in {:.3f}s".format(i, toc-tic))
    print("Entire optimization finished in {:.3f}s".format(total_time))
    test_pol(model, base, samples, pol, probs.value, tol)
    print("Policy verified with tolerance " + str(tol*50))
    
    # sometimes we find an additional support sample
    num_supports = 0
    support_samples = set()
    for s in range(num_states):
        if True:
        #if not any([any(elem) for elem in model.paramed[s]]):
        #if any([any(elem) for elem in model.paramed[s]]):
            #max_sc = sum([any(elem) for elem in model.paramed[s]])
            max_sc = len(model.Enabled_actions[s])
            if model.opt == "max":
                found_sc = np.argwhere(probs.value[s] <= np.min(probs.value[s])+tol)
            else:
                found_sc = np.argwhere(probs.value[s] >= np.max(probs.value[s])-tol)
            if found_sc.size <= max_sc:
                support_samples.update([int(elem) for elem in found_sc])
            else:
                sc_list = found_sc.T.tolist()[0]
                for i in range(max_sc):
                    sc_vals = probs.value[s,sc_list]
                    if model.opt=="max":
                        worst_ind = np.argmin(sc_vals)
                    else:
                        worst_ind = np.argmax(sc_vals)
                    next_sc = sc_list.pop(worst_ind)
                    support_samples.add(next_sc)
    num_supports = len(support_samples)
    if converged:
        return probs.value[model.Init_state], pol, support_samples, num_supports, probs.value
    else:
        return -1, -1, -1, -1, -1

def run_all(args):
    print("Running code for robust optimal policy \n --------------------")
    model = args["model"]
    test_support_num(args)
    #a_priori_max_supports = sum([len(acts) for acts in model.Enabled_actions])
    a_priori_max_supports = model.max_supports
    #calc_max_path(model)
    a_priori_eps = calc_eps(args["beta"], args["num_samples"], a_priori_max_supports)
    
    print("A priori upper bound on number of support constraints is " + str(a_priori_max_supports))

    print("A priori bound on violation probability is {:.3f} with confidence {:.3f}".format(a_priori_eps, args["beta"]))

    if args["sample_load_file"] is not None:
        base, samples = load_samples(args["sample_load_file"])
    else:
        base, samples = gen_samples(model, args["num_samples"], args["batch_size"])
    if args["sample_save_file"] is not None:
        save_data(args["sample_save_file"], (base, samples))
    
    probs, pol, supports, a_post_support_num, all_p  = calc_probs_policy_iteration(model, base, samples, tol=args["tol"])
    
    if args["result_save_file"] is not None:
        save_data(args["result_save_file"], {"probs":probs, "pol":pol, "supports":supports})

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
            if gap >= 1e-5:
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
