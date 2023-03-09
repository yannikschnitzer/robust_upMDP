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

def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

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

def batch_sample_batch(model, sizes):
    batch = []
    for size in sizes:
        batch.append(get_batch(model, size))
    return batch

def gen_samples(model, N, batch_size):
    num_states = len(model.States)
    num_acts = len(model.Actions)
    num_batches = int(np.ceil(N/batch_size))
    samples = []
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
            if model.opt == "max":
                if probs[model.Init_state, j]-res > tol:
                    import pdb; pdb.set_trace()
            else:
                if res-probs[model.Init_state, j] > tol:
                    import pdb; pdb.set_trace()
            j += 1

def single_solve(probs, pol, trans_mat, tol, reached_states, enabled, num_acts, num_states, opt, s):
    changed = False
    if s not in reached_states:#model.Labelled_states[model.Labels.index("reached")]:
        if len(enabled[s]) == 1:
            act = enabled[s][0]
            pol[s] = np.zeros(num_acts)
            pol[s, act] = 1
            new_p = trans_mat[s, act, :]
            changed_samples = np.where(abs(probs.value[s]-new_p)>=tol)[0]
            changed = (changed_samples.size) > 0
            old_p = np.copy(probs.value[s])
            probs.value[s] = new_p
        else:
            enabled = enabled[s]
            pi_mat = np.zeros(num_acts)
            pi_mat[enabled] = 1
            variable_a = []
            test_tic = time.perf_counter()
            for a in enabled:
                overlap = False
                if not np.all(trans_mat[s,a,:]==trans_mat[s,a,0]):
                    for a_prime in variable_a:
                        if  np.linalg.norm(trans_mat[s,a,:]-trans_mat[s,a_prime,:],np.inf) < tol:
                            overlap = True
                            break
                    if not overlap:
                        variable_a.append(a)
            if len(variable_a) > 1:
                try:
                    hull = ConvexHull(trans_mat[s,variable_a,:].T, qhull_options='QJ')
                    points = hull.vertices
                except:
                    print("Convex hull error")
                    points = np.linspace(0,N-1, N)
            else:
                if len(variable_a) == 0:
                    points = np.array([0])
                else:
                    points = np.array([np.argmin(trans_mat[s,variable_a,:]),\
                                       np.argmax(trans_mat[s,variable_a,:])])
            new_trans = trans_mat[s,:,points].T
            pi = cp.Variable(num_acts)
            hull_new_prob = cp.Variable(points.size)
            if opt == "max":
                cons = [hull_new_prob >= probs[s,points],
                             hull_new_prob >= 0,
                             hull_new_prob <= 1]
                obj = cp.Minimize(cp.norm(1-hull_new_prob, 'inf'))
            else:
                cons = [hull_new_prob <= probs[s,points],
                             hull_new_prob >= 0,
                             hull_new_prob <= 1]
                obj = cp.Minimize(cp.norm(hull_new_prob, 'inf'))
            cons += [pi_mat@pi == 1, pi >= 0, hull_new_prob == pi@new_trans[:,:]]
            program = cp.Problem(obj, cons)
            try:
                result = program.solve(ignore_dpp=True, solver=cp.CLARABEL)
                opt_failed = program.status != cp.OPTIMAL and program.status != cp.OPTIMAL_INACCURATE
            except:
                opt_failed = True
            if opt_failed:
                cons.pop(0)
                logging.debug("Found infeasible problem, resolving")
                program = cp.Problem(obj, cons)
                result = program.solve(ignore_dpp=True, solver=cp.CLARABEL)
                if program.status != cp.OPTIMAL and program.status != cp.OPTIMAL_INACCURATE:
                    import pdb; pdb.set_trace()
            new_pi = pi.value
            changed_samples = np.where(abs(probs.value[s]-new_pi@trans_mat[s,:,:])>=tol)[0]
            changed = (changed_samples.size) > 0
            old_p = np.copy(probs.value[s])
            probs.value[s] = np.maximum(0, np.minimum(new_pi@trans_mat[s,:,:],1))
            pol[s] = np.maximum(np.minimum(new_pi, 1), 0)
    return probs, pol, changed

def batch_solve(probs, pol, trans_mat, tol, reached_states, enabled, num_acts, num_states, opt, states):
    results = []
    for s in states:
        results.append(single_solve(probs, pol, trans_mat, tol, reached_states, enabled, num_acts, num_states, opt, s))
    return results

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def build_trans_mat(sample_mat, prob_batch, num_states, num_acts, size):
    res = sample_mat@prob_batch.T.reshape((-1,1))
    new_shape = (size, num_states, num_acts)
    new_strides = (num_states*res.strides[0]+num_acts*res.strides[1], res.strides[0], res.strides[1])
    batch_trans_mat = np.lib.stride_tricks.as_strided(res, new_shape, new_strides)
    batch_trans_mat = np.swapaxes(batch_trans_mat.T, 0, 1)
    return batch_trans_mat

def calc_probs_policy_iteration(model, base, samples, max_iters=10000, tol=1e-3, init_probs=None, savefile=None):
    if savefile is not None: 
        save_freq = 10
    else:
        save_freq = max_iters
    back_set = calc_reach_sets(model)
    num_states = len(model.States)
    num_acts = len(model.Actions)
    batch_size = int(samples[0].shape[-1]/num_states)
    N = sum([int(sample.shape[-1]/num_states) for sample in samples])
    states_to_update = set()
    just_updated = set()
    probs = cp.Parameter((num_states,N))
    if init_probs is None:
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
    else:
        prob_init = init_probs
        states_to_update = model.States
    probs.value = prob_init 

    pol = np.ones((num_states, num_acts))/num_acts
    
    converged=False
    total_time = 0
    num_batches = int(np.ceil(N/batch_size))
    
    if model.opt == "max":
        trans_mat = np.zeros((num_states, num_acts, N))
    else:
        trans_mat = np.ones((num_states, num_acts, N))
    
    for i in range(max_iters):
        tic = time.perf_counter()
        next_states_to_update = set()
        logging.info(("Beginning construction of Q matrix"))
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

            batch_trans_mat = build_trans_mat(sample_mat, prob_batch, num_states, num_acts, curr_size)
            trans_mat[:,:,start:end] = batch_trans_mat
        mat_time = time.perf_counter() - tic
        just_updated = set()

        logging.info(("Completed construction of Q matrix in {:.3f}s").format(mat_time))
        logging.info("Optimizing over {} states".format(len(states_to_update)))
        converged=True
        partial_solve = partial(batch_solve, probs, pol, trans_mat, tol, \
                                model.Labelled_states[model.Labels.index("reached")], model.Enabled_actions,
                                num_acts, num_states, model.opt)
        states_to_update = list(states_to_update)
        if len(states_to_update) > NUM_POOLS:
            batches = split(states_to_update, NUM_POOLS)
        else:
            batches = [[elem]for elem in states_to_update]
    
        with Pool() as p:
            out = p.map(partial_solve, batches)
        result = []
        for elem in out:
            result += elem
        for s, elem in zip(states_to_update, result):
            pol[s] = elem[1][s]
            probs.value[s] = elem[0].value[s]
            if elem[-1]:
                next_states_to_update.update(back_set[s])
                converged=False
        states_to_update = next_states_to_update
        if model.opt == "max":
            worst = np.min(probs.value, axis=1)
        else:
            worst = np.max(probs.value, axis=1)
        logging.info("Current worst case probabilities are {}".format(worst))
        logging.info("Current worst case init probability is {}".format(worst[model.Init_state]))
        if (i+1)%save_freq == 0:
            logging.info("Storing current probability value")
            save_data(savefile, probs.value)
        if converged:
            break
        toc = time.perf_counter()
        total_time += toc-tic
        logging.info("iteration {} completed in {:.3f}s".format(i, toc-tic))
    print("Entire optimization finished in {:.3f}s".format(total_time))
    test_pol(model, base, samples, pol, probs.value, tol)
    print("Policy verified with tolerance " + str(tol*50))
    
    # We very often find extra supports
    # sometimes to take us over a priori number for small tests
    # actual supports are always smaller than priori
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
        base, samples = gen_samples(model, args["num_samples"], args["batch_size"])
    if args["sample_save_file"] is not None:
        save_data(args["sample_save_file"], (base, samples))
    num_states = len(model.States)
    N = sum([int(sample.shape[-1]/num_states) for sample in samples])
    
    probs, pol, supports, a_post_support_num, all_p  = calc_probs_policy_iteration(model, base, samples, savefile=args["result_save_file"], tol=args["tol"])
    
    if args["result_save_file"] is not None:
        save_data(args["result_save_file"], {"probs":probs, "pol":pol, "supports":supports})

    [a_post_eps_L, a_post_eps_U] = \
        calc_eps_risk_complexity(args["beta"], N, a_post_support_num)
    
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
