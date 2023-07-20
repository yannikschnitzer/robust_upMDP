import numpy as np
import cvxpy as cp
import Markov.writer as writer
import Markov.models
from PAC.funcs import *
import itertools
from tqdm import tqdm
import time
import copy
import pickle
import logging
import os
import matplotlib.pyplot as plt
import pycarl

import base64
def b64(s):
    s = int(s)
    if s > 4095:
        start = -4
    elif s > 63:
        start = -3
    else:
        start = -2
    return str(base64.b64encode(s.to_bytes(length=6,byteorder="big")))[start:-1]

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
    samples = [model.param_sampler() for j in range(N)]
    return samples

def test_pol(model, samples, pol=None, paramed_models = None):
    num_states = len(model.States)
    num_acts = len(model.Actions)
    
    if paramed_models is not None:
        test_MDP = model.fix_params(samples[0])

    true_probs = []
    pols = []
    if model.opt == "max":
        wc = 1
    else:
        wc = 0
    for ind, sample in enumerate(samples):
        time_start = time.perf_counter()
        if paramed_models is None:
            test_MDP = model.fix_params(sample)
        else:
            test_MDP.Transition_probs = paramed_models[ind]
        time_fix_params = time.perf_counter()
        if pol is not None:
            test_model = test_MDP.fix_pol(pol)
        else:
            test_model = test_MDP
        time_fix_pol = time.perf_counter()
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

        time_end = time.perf_counter()
        time_for_fixing_param = time_fix_params -time_start
        time_for_fixing_pol = time_fix_pol -time_fix_params
        time_for_solving = ((time_end-time_fix_pol))
        logging.debug("solving {:.3f}s".format(time_for_solving))
        logging.debug("fixing params {:.3f}s".format(time_for_fixing_param))
        logging.debug("fixing pol {:.3f}s".format(time_for_fixing_pol))
    true_probs = np.array(true_probs)
    return wc, true_probs, pols

def test_pol2(model, pol, paramed_models, trans_arr):
    # this is v slow, but has the potential for parallelisation...

    num_states = len(model.States)
    num_acts = len(model.Actions)
    MC_arr = np.zeros((num_states,num_states))
    for s in model.paramed_states:
        for a_id, a in enumerate(model.Enabled_actions[s]):
            for s_prime_id, s_prime in enumerate(model.trans_ids[s][a]):
                trans_arr[s][a][s_prime] = model.Transition_probs[s][a_id][s_prime_id]
    for s in model.States:
        MC_arr[s] = pol[s]@trans_arr[s]
    test_vec = np.zeros(num_states)
    test_vec[model.Init_state] = 1
    ss = test_vec @ np.linalg.matrix_power(MC_arr, 100000)
    return sum(ss[model.Labelled_states[-1]])

def calc_param_grad(model, sample, paramed_model = None):
    print("Calculating parameterised gradient")
    num_states = len(model.States)
    num_acts = len(model.Actions)
    grad = {}

    tic = time.perf_counter()
    
    if paramed_model is not None:
        test_MDP = model.fix_params(sample)

    true_probs = []
    pols = []
    if model.opt == "max":
        wc = 1
    else:
        wc = 0
    time_start = time.perf_counter()
    if paramed_model is None:
        test_MDP = model.fix_params(sample)
    else:
        test_MDP.Transition_probs = paramed_model
    
    time_fix_params = time.perf_counter()
    
    time_for_fixing_param = time_fix_params -time_start
    logging.debug("fixing params {:.3f}s".format(time_for_fixing_param))

    IO = writer.PRISM_grad(test_MDP)
    IO.write()
    
    time_write = time.perf_counter()
    time_for_writing = time_write-time_fix_params
    logging.debug("Writing {:.3f}s".format(time_for_writing))

    param_res, all_res, sol_pol = IO.solve()

    time_end = time.perf_counter()
    time_for_solving = ((time_end-time_write))
    logging.debug("solving {:.3f}s".format(time_for_solving))

    toc = time.perf_counter()
    logging.debug("Time to find gradient: " + str(toc-tic))
    return param_res

def find_grad(paramed, pol, enabled):
    grad = np.zeros_like(pol)
    if type(paramed) == tuple:
        for s, row in enumerate(pol):
            if len(enabled[s]) > 1:
                for a in enabled[s]:
                    numerical = ([],[])
                    for i, elem in enumerate(paramed):
                        numerical[i].append(1)
                        for val in elem:
                            if "_" in val:
                                split = val.split("_")
                                if len(split[1]) == 1:
                                    s_b64 = "AAA" + split[1]
                                elif len(split[1]) == 2:
                                    s_b64 = "AA" + split[1]
                                else:
                                    s_b64 = "A" + split[1]
                                state = int.from_bytes(base64.b64decode(s_b64),byteorder="big")
                                
                                b64s = str(base64.b64encode(bytes([s])))[2:4]
                                act = int(split[2])
                                if state == s:
                                    if act != a:
                                        numerical[i][-1] = 0
                                else:
                                    numerical[i][-1] *= pol[state, act]
                            elif val == "+":
                                numerical[i].append(1)
                            else:
                                numerical[i][-1] *= int(val)
                    grad[s,a] = sum(numerical[0])/sum(numerical[1])
    else:
        instantiators = {}
        one = pycarl.cln.Rational(1)
        zero = pycarl.cln.Rational(0)
        for s, row in enumerate(pol):
            for a_i, a_val in enumerate(row):
                instantiators["_{}_{}".format(b64(s),a_i)] = pycarl.cln.Rational(a_val)
        for s, row in enumerate(pol):
            if len(enabled[s]) > 1:
                for a in enabled[s]:
                   func = paramed
                   func_vars = func.gather_variables()
                   var_dict = {}
                   for var in func_vars:
                        split = var.name.split("_")
                        if len(split[1]) == 1:
                            s_b64 = "AAA" + split[1]
                        elif len(split[1]) == 2:
                            s_b64 = "AA" + split[1]
                        else:
                            s_b64 = "A" + split[1]
                        state = int.from_bytes(base64.b64decode(s_b64),byteorder="big")
                        act = int(split[2])
                        if state == s:
                            if act == a:
                               var_dict[var] = one
                            else:
                               var_dict[var] = zero
                        else:
                            var_dict[var] = instantiators[var.name]
                   grad[s,a] = float(func.evaluate(var_dict))
    return grad


def solve_subgrad(samples, model, max_iters=500):
    solve_grad_param = True
    print("--------------------\nStarting subgradient descent")
    
    sample_trans_probs = []
    for sample in samples:
        new_MDP = model.fix_params(sample)
        sample_trans_probs.append(copy.copy(new_MDP.Transition_probs))

    arr = model.get_trans_arr()

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
    best_hist = []
    wc, true_probs, _ = test_pol(model, samples, pol, paramed_models = sample_trans_probs)
    worst = np.argwhere(true_probs[:,model.Init_state]==wc)
    worst = np.random.choice(worst.flatten())
   
    param_grads = {}
    best_worst_pol = test_pol(model, [samples[worst]])[2][0]
    test_wc, test_probs, _ = test_pol(model, samples, best_worst_pol, paramed_models = sample_trans_probs)
    test_worst = np.argwhere(test_probs[:,model.Init_state]==test_wc).flatten()
    
    if model.opt == "max":
        best = 0
    else:
        best = 1

    if worst in test_worst:
        print("Worst case holds with deterministic policy, deterministic is optimal")
        return test_wc, best_worst_pol, test_wc
    for i in tqdm(range(max_iters)):
        time_start = time.perf_counter()
        
        old_pol = np.copy(pol)
        step = 1/(i+2)
        #step = 10
        if solve_grad_param:
            if worst not in param_grads:
                param_grads[worst] = calc_param_grad(model, samples[worst], sample_trans_probs[worst]) 
            grad = find_grad(param_grads[worst], pol, model.Enabled_actions)
            
            nonzero_grad = grad[np.nonzero(grad)]
            min_elem = min(nonzero_grad)
            max_elem = max(nonzero_grad)

        else:
            grad = np.zeros_like(pol)
            min_elem = 1 
            max_elem = 0
            for s in model.States:
                if len(model.Enabled_actions[s]) > 1:
                    for a in model.Enabled_actions[s]:
                        grad_finder = np.copy(pol)
                        grad_finder[s] = 0
                        grad_finder[s,a] = 1
                        tic = time.perf_counter()
                        new = test_pol(model, 
                                             [samples[worst]], 
                                             grad_finder, 
                                             paramed_models = [sample_trans_probs[worst]])[0] 
                        grad[s,a] = new 
                        toc = time.perf_counter()
                        logging.debug("Time to find gradient: " + str(toc-tic))
                        min_elem = min(new, min_elem)
                        max_elem = max(new, max_elem)
                        #test = test_pol2(model, grad_finder, fixed_MDPs[worst], arr)
                        #tac = time.perf_counter()
                        #print("new version: " + str(tac-toc))
            #grad_norm = np.linalg.norm(grad, ord=np.inf)
            ##import pdb; pdb.set_trace()
        nonzero_inds = np.nonzero(grad)
        scaled_grad = np.copy(grad)
        scaled_grad[nonzero_inds] -= min_elem
        scaled_grad /= max_elem-min_elem
        
        time_grads = time.perf_counter()-time_start
        logging.debug("Total time for finding gradients: {:.3f}".format(time_grads))
        if model.opt == "max":
            pol += step*scaled_grad
        else:
            pol -= step*scaled_grad 

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
        time_proj = time.perf_counter()-time_start-time_grads
        logging.debug("Time for projection step: {:.3f}".format(time_proj))
        wc, true_probs, _ = test_pol(model, samples, pol, paramed_models = sample_trans_probs)
        worst = np.argwhere(true_probs[:,model.Init_state]==wc)
        #import pdb; pdb.set_trace()
        worst = np.random.choice(worst.flatten())
        wc_hist.append(wc)
        if model.opt == "max":
            if wc > best:
                best = wc
                best_pol = pol
        else:
            if wc < best:
                best = wc
                best_pol = pol
        best_hist.append(best)
        logging.info("Current value: {:.3f}, with sample {}".format(wc, worst))
        logging.info("Policy inf norm change: {:.3f}".format(np.linalg.norm(pol-old_pol, ord=np.inf)))
    return best, best_pol, best_hist

def find_all_pols(model):
    base_vec = [False for i in model.Actions]
    act_vecs = []
    
    print("--------------------\nFinding policies")
    for a in model.Actions:
        vec = copy.copy(base_vec)
        vec[a] = True
        act_vecs.append(vec)
    act_poss = [[copy.copy(act_vecs[a]) for a in model.Enabled_actions[s]] for s in model.States]
    print("Built list, now building full policies")
    pols = [np.array(pol) for pol in itertools.product(*act_poss)]
    return pols

def build_init_payoff(samples, model):
    pols = find_all_pols(model)
    all_probs = []
    non_dommed_pols = []
    print("--------------------\nBuilding initial payoff matrix")
    best_wc = 0
    for j, pol in enumerate(tqdm(pols)):
        # could also find best deterministic policy here
        probs = test_pol(model, samples, pol)[1][:,model.Init_state]
        if model.opt == "max":
            wc = min(probs)
            if wc > best_wc:
                best_wc = wc
                det_pol = pol
        else:
            wc = max(probs)
            if wc < best_wc:
                best_wc = wc
                det_pol = pol
                
        dommed = False
        for i, elem in enumerate(all_probs):
            if model.opt == "max":
                if np.all(probs <= elem):
                    dommed = True
                elif np.all(probs >= elem):
                    all_probs.pop(i)
                    non_dommed_pols.pop(i)
            else:
                if np.all(probs >= elem):
                    dommed = True
                elif np.all(probs <= elem):
                    all_probs.pop(i)
                    non_dommed_pols.pop(i)
        if not dommed:
            all_probs.append(probs)
            non_dommed_pols.append(j)
    pols = [pols[i] for i in non_dommed_pols]
    payoffs = np.vstack(all_probs)
    return payoffs, pols

def calc_payoff_mat(samples, model):
    payoffs, pols = build_init_payoff(samples, model)

    print("--------------------\nRemoving dominated samples")
    
    non_domed_samples = []
    for i in tqdm(range(len(samples))):
        if model.opt == "max":
            test_arr = np.all(payoffs[:,i][:,np.newaxis] > payoffs, axis=0)
        else:
            test_arr = np.all(payoffs[:,i][:,np.newaxis] < payoffs, axis=0)
        if not np.any(test_arr):
            non_domed_samples.append(i)
    non_domed_payoffs = payoffs[:, non_domed_samples]
    return non_domed_payoffs, pols, non_domed_samples

def MNE_solver(samples, model):
    payoffs, pols, rel_samples = calc_payoff_mat(samples, model)
    best = 0
    print("---------------------\nIterating through MNE combinations")
    combs = list(itertools.combinations(range(len(rel_samples)), len(pols)))
    for elem in tqdm(combs):
        test_non = elem
        test_non_domed_payoffs = payoffs[:, test_non]
        br_mixer = np.ones((1, len(pols)))@np.linalg.inv(test_non_domed_payoffs)
        br_mixer = np.maximum(0, br_mixer)
        br_mixer /= np.sum(br_mixer)
        res = (br_mixer@payoffs).flatten()
        if min(res) > best:
            best = min(res)
            best_samples = elem
            best_pol = np.copy(br_mixer)
    return best, best_pol, pols 

def FSP_solver(samples, model, max_iters = 100000):
    update_parallel = True
    payoffs, pols, rel_samples = calc_payoff_mat(samples, model)
    pol_dist = np.ones((1,len(pols)))/len(pols)
    sample_dist = np.ones((len(rel_samples),1))/len(rel_samples)

    sum_step = 0
    print("---------------------\nStarting FSP")
    for i in tqdm(range(max_iters)):
        #step = 1/(i+1)
        step = 1
        sum_step += step
        new = step/sum_step
        old = 1-new
        if update_parallel:
            sample_vec = pol_dist@payoffs
            pol_vec = payoffs@sample_dist
            
            pol_dist *= old
            pol_dist[0, np.argmax(pol_vec)] += new
            
            sample_dist *= old
            sample_dist[np.argmin(sample_vec), 0] += new
            
    res = min((pol_dist@payoffs).flatten())
    return res, pol_dist, pols, np.sum(sample_dist >= 1e-4)

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

    N = args["num_samples"]
  
    res_sg, pol_sg, sg_hist = solve_subgrad(samples, model)
    
    plt.plot(sg_hist)
    
    wc, probs, _ = test_pol(model, samples, pol_sg)
    if model.opt == "max":
        active_sg = np.sum(probs[:, model.Init_state] <= wc+1e-3)
    else:
        active_sg = np.sum(probs[:, model.Init_state] >= wc-1e-3)

    print("Using subgradient methods found " + str(active_sg) + " active constraints a posteriori")
    [a_post_eps_L, a_post_eps_U] = \
        calc_eps_risk_complexity(args["beta"], N, active_sg)

    print("Hence, a posteriori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
            .format(a_post_eps_L, a_post_eps_U, args["beta"]))

    print("Optimal satisfaction probability is found to be {:.3f}".format(res_sg))
    
    if len(model.States)**len(model.Actions) < 100:
        res_MNE, pol_MNE, pols = MNE_solver(samples, model)
        res_FSP, pol_FSP, pols, a_post_support_num = FSP_solver(samples, model)
        import pdb; pdb.set_trace()
        assert active_sg == a_post_support_num
        print("----------------\nResult comparison:\nmatrix solver: {:.3f}\nFSP: {:.3f}\nSubgradient: {:.3f}".format(res_MNE, res_FSP, res_sg))
        [a_post_eps_L, a_post_eps_U] = \
            calc_eps_risk_complexity(args["beta"], N, a_post_support_num)
        print("Using game thoeretic methods found " + str(a_post_support_num) + " support constraints a posteriori")

        print("Hence, a posteriori, violation probability is in the range [{:.3f}, {:.3f}], with confidence {:.3f}"
            .format(a_post_eps_L, a_post_eps_U, args["beta"]))
        print("Optimal satisfaction probability is found to be {:.3f}".format(res_MNE))
    if args["result_save_file"] is not None:
        save_data(args["result_save_file"], {"worst": wc, "probs":probs, "pol":pol_sg, "supports":active_sg})

    #if pol.size < 50:
    #    print("Calculated robust policy is:")
    #    print(pol)

    #thresh = a_priori_eps

    if args["MC"]:
        emp_violation = MC_sampler(model, args["MC_samples"], res_sg, pol_sg) 
        print("Empirical violation rate is found to be {:.3f}".format(emp_violation))
    print("\n\n")
    
    plt.show()

def test_support_num(args):
    model = args["model"]
    num_states = len(model.States)
    num_acts = len(model.Actions)
    for i in range(10):
        base, samples = gen_samples(model, args["num_samples"], args["batch_size"])

        probs, pol, supports, a_post_support_num, all_p = calc_probs_policy_iteration(model, base, samples)
         
        print("Calculated supports: " + str(supports))
        actual_supports = []
        for j in range(args["num_samples"]):
            samples_new = remove_sample(j, samples, num_states, num_acts)
            test_probs, test_pol, _, _, all_p_test = calc_probs_policy_iteration(model, base, samples_new)
            if model.opt == "max":
                gap = abs(min(probs) - min(test_probs))
            else:
                gap = abs(max(probs) - max(test_probs))
            if gap >= 1e-4:
                actual_supports.append(j)
        print("Emprical supports: " + str(actual_supports))
        if len(actual_supports) > model.max_supports:
            print("ERROR, a priori max sc was wrong!!!")
            import pdb; pdb.set_trace()
        for sc in actual_supports:
            if sc not in supports:
                print("ERROR, found an empirical SC not in a posteriori support set")
                import pdb; pdb.set_trace()

def remove_sample(pos, samples):
    new_samples = copy.copy(samples)
    new_samples.pop(pos)
    return new_samples
